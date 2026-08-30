"""Unit tests for the gated / trust-aware QGFD experiments.

These run on CPU with synthetic tensors and a tiny synthetic Llama -- no
checkpoint download, no GPU. What they pin down is the part of the design that a
bug would silently invalidate:

  * the gate is EXACTLY zero at init, so training starts at exact softmax;
  * the gate still has a LIVE GRADIENT at zero (the reason for tanh over sigmoid);
  * a signed gate never emits a non-distribution;
  * the trust graph's rows are proper distributions over exactly top-k edges;
  * the paired statistics do not call a straddling difference significant.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from scripts.gated_experiments import (
    CORRUPTIONS,
    GatedConfig,
    GatedQGFDOperator,
    TrustGatedOperator,
    TOKEN_PRESERVING,
    paired_delta,
    perturb_ocr,
    perturb_repeat_token,
    perturb_token_replace,
    token_weighted_ce,
    _auc,
    _t95,
)


B, H, L, D = 2, 4, 16, 8


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    scores = torch.randn(B, H, L, L)
    keys = torch.randn(B, H, L, D)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    mask = torch.zeros(1, 1, L, L).masked_fill(~causal[None, None], -1e9)
    return scores, keys, mask


def _op(**kw):
    kw.setdefault("num_heads", H)
    return GatedQGFDOperator(**kw)


# --------------------------------------------------------------------------- #
# Exactness and gradient liveness at init
# --------------------------------------------------------------------------- #
def test_gate_is_exactly_zero_at_init():
    op = _op()
    assert torch.equal(op.w_entropy, torch.zeros(H))
    assert torch.equal(op.w_margin, torch.zeros(H))
    assert torch.equal(op.gate_bias, torch.zeros(H))


def test_init_output_is_bit_identical_to_softmax_under_no_grad():
    scores, keys, mask = _inputs()
    op = _op().eval()
    with torch.no_grad():
        got = op(scores, keys, mask)
        want = F.softmax(scores + mask, dim=-1)
    assert torch.equal(got, want), "alpha=0 equivalence must be bit-exact"


def test_gradient_is_alive_at_zero_gate():
    """The whole reason for tanh. A one-sided sigmoid gate can reach g=0 only at
    a boundary, where the gradient vanishes and training never starts."""
    scores, keys, mask = _inputs()
    op = _op()
    p = op(scores, keys, mask)
    p.sum().backward()
    grads = [op.gate_bias.grad, op.w_entropy.grad, op.w_margin.grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().max() > 0 for g in grads), (
        "a zero-initialised gate with no gradient cannot be trained")


def test_gate_l1_is_differentiable_and_zero_at_init():
    scores, keys, mask = _inputs()
    op = _op()
    op(scores, keys, mask)
    assert op.gate_l1 is not None and op.gate_l1.requires_grad
    assert op.gate_l1.item() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# The gate as a distribution-preserving operation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bias", [-4.0, -1.0, 1.0, 4.0])
def test_output_is_a_distribution_for_any_gate_sign(bias):
    """A signed gate drives individual entries negative wherever the graph routes
    mass to a key p0 gave ~0 to; the operator must clamp and renormalise."""
    scores, keys, mask = _inputs()
    op = _op(g_max=0.1)
    with torch.no_grad():
        op.gate_bias.fill_(bias)
    with torch.no_grad():
        p = op(scores, keys, mask)
    assert torch.isfinite(p).all()
    assert (p >= 0).all(), "negative probability emitted"
    assert torch.allclose(p.sum(-1), torch.ones(B, H, L), atol=1e-4)


def test_negative_gate_sharpens_and_positive_gate_smooths():
    scores, keys, mask = _inputs()
    ent = {}
    for name, bias in (("sharpen", -4.0), ("neutral", 0.0), ("smooth", 4.0)):
        op = _op(g_max=0.1)
        with torch.no_grad():
            op.gate_bias.fill_(bias)
            p = op(scores, keys, mask).clamp_min(1e-12)
        ent[name] = float(-(p * p.log()).sum(-1).mean())
    assert ent["sharpen"] < ent["neutral"] < ent["smooth"]


def test_gate_magnitude_is_bounded_by_g_max():
    scores, keys, mask = _inputs()
    op = _op(g_max=0.05)
    with torch.no_grad():
        op.gate_bias.fill_(50.0)          # saturate tanh
        op.start_tracking()
        op(scores, keys, mask)
        log = op.stop_tracking()
    assert max(abs(v) for v in log["per_head"]) <= 0.05 + 1e-6


def test_disabling_the_operator_returns_plain_softmax():
    scores, keys, mask = _inputs()
    op = _op(enable_qgfd=False)
    with torch.no_grad():
        op.gate_bias.fill_(3.0)           # would fire if it were enabled
        got = op(scores, keys, mask)
    assert torch.equal(got, F.softmax(scores + mask, dim=-1))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_entropy_feature_is_normalised_by_the_causal_prefix():
    """Raw entropy is not comparable across query positions: position t of a
    causal row holds at most log(t+1) nats. A uniform causal row must therefore
    read as 1.0 at every position, not as a ramp."""
    op = _op()
    p0 = torch.zeros(1, 1, L, L)
    for t in range(L):
        p0[0, 0, t, :t + 1] = 1.0 / (t + 1)
    counts = op._valid_counts(p0, None)
    p = p0.clamp_min(1e-12)
    ent_hat = (-(p * p.log()).sum(-1)) / counts.log()
    assert torch.allclose(ent_hat[0, 0, 1:], torch.ones(L - 1), atol=1e-5)


def test_margin_feature_separates_peaked_from_flat_rows():
    op = _op(use_entropy=False, g_max=0.1)
    with torch.no_grad():
        op.w_margin.fill_(10.0)
    peaked = torch.zeros(1, 1, 2, 4)
    peaked[..., 0, 0] = 1.0                       # margin 1
    peaked[..., 1, :] = 0.25                      # margin 0
    with torch.no_grad():
        g = op._gate(peaked, None)
    assert g[0, 0, 0, 0] > g[0, 0, 1, 0]


# --------------------------------------------------------------------------- #
# The trust graph
# --------------------------------------------------------------------------- #
def test_trust_graph_rows_are_causal_distributions():
    _, keys, _ = _inputs()
    op = TrustGatedOperator(num_heads=H, top_k=4)
    with torch.no_grad():
        P = op.build_transition_from_keys(keys, target_heads=H, is_causal=True,
                                          dtype=torch.float32)
    assert P.shape == (B, H, L, L)
    assert torch.allclose(P.sum(-1), torch.ones(B, H, L), atol=1e-5)
    upper = ~torch.tril(torch.ones(L, L, dtype=torch.bool))
    assert P[:, :, upper].abs().max() == 0.0, "mass leaked into the future"


def test_top_k_actually_sparsifies_and_keeps_short_prefixes_intact():
    _, keys, _ = _inputs()
    op = TrustGatedOperator(num_heads=H, top_k=4)
    with torch.no_grad():
        P = op.build_transition_from_keys(keys, target_heads=H, is_causal=True,
                                          dtype=torch.float32)
    nnz = (P > 1e-8).sum(-1)
    # A row whose causal prefix is shorter than k must keep every edge it has.
    assert int(nnz[0, 0, 2]) == 3
    # A long row must be cut down to k.
    assert int(nnz[0, 0, L - 1]) <= 4


def test_reliability_and_self_loop_start_inert_but_are_wired():
    _, keys, _ = _inputs()
    plain = TrustGatedOperator(num_heads=H, top_k=0, self_loop=0.0,
                              reliability=True)
    assert plain.w_trust.item() == 0.0
    with torch.no_grad():
        P0 = plain.build_transition_from_keys(keys, H, True, torch.float32)
        plain.w_trust.fill_(2.0)
        P1 = plain.build_transition_from_keys(keys, H, True, torch.float32)
    assert not torch.allclose(P0, P1), "w_trust is not reaching the logits"


def test_self_loop_increases_diagonal_mass():
    _, keys, _ = _inputs()
    with torch.no_grad():
        a = TrustGatedOperator(num_heads=H, top_k=0, self_loop=0.0
                               ).build_transition_from_keys(
            keys, H, True, torch.float32)
        b = TrustGatedOperator(num_heads=H, top_k=0, self_loop=3.0
                               ).build_transition_from_keys(
            keys, H, True, torch.float32)
    idx = torch.arange(1, L)
    assert (b[:, :, idx, idx] > a[:, :, idx, idx]).all()


def test_p_structure_controls_are_reachable_through_the_gate():
    scores, keys, mask = _inputs()
    outs = {}
    for structure in ("real", "uniform", "shuffled"):
        op = _op(g_max=0.1, p_structure=structure)
        with torch.no_grad():
            op.gate_bias.fill_(4.0)
            outs[structure] = op(scores, keys, mask)
    assert not torch.allclose(outs["real"], outs["uniform"], atol=1e-6)
    assert not torch.allclose(outs["real"], outs["shuffled"], atol=1e-6)
    for p in outs.values():
        assert torch.allclose(p.sum(-1), torch.ones(B, H, L), atol=1e-4)


def test_t_gt_1_is_refused():
    with pytest.raises(AssertionError):
        GatedQGFDOperator(num_heads=H, diffusion_steps=2)


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def _w(vals):
    return [(v, 100) for v in vals]


def test_paired_delta_signs_and_significance():
    a = _w([2.0, 2.1, 1.9, 2.0, 2.05] * 8)
    b = _w([2.5, 2.6, 2.4, 2.5, 2.55] * 8)
    d = paired_delta(a, b)
    assert d["mean"] < 0 and d["sig"] and d["win_frac"] == 1.0
    assert d["ppl_a"] == pytest.approx(math.exp(d["ce_a"]))


def test_paired_delta_calls_a_straddling_difference_not_significant():
    a = _w([2.0 + (0.5 if i % 2 else -0.5) for i in range(40)])
    b = _w([2.0] * 40)
    d = paired_delta(a, b)
    assert abs(d["mean"]) < d["ci95"] and not d["sig"]


def test_paired_delta_needs_two_windows():
    assert paired_delta(_w([1.0]), _w([2.0]))["mean"] is None


def test_paired_delta_truncates_to_the_shorter_arm():
    d = paired_delta(_w([1.0] * 10), _w([2.0] * 6))
    assert d["n"] == 6


def test_token_weighted_ce_weights_by_length():
    assert token_weighted_ce([(1.0, 1), (3.0, 3)]) == pytest.approx(2.5)


def test_t95_falls_back_to_the_normal_quantile_for_large_n():
    assert _t95(2) > _t95(10) > _t95(200) == 1.96


def test_auc_is_half_for_identical_distributions():
    assert _auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)
    assert _auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _auc([], [1.0]) is None


# --------------------------------------------------------------------------- #
# Corruption families
# --------------------------------------------------------------------------- #
def test_new_families_are_registered_and_classified():
    for name in ("token_replace", "repeat_token", "ocr"):
        assert name in CORRUPTIONS
    assert "token_replace" in TOKEN_PRESERVING
    assert "ocr" not in TOKEN_PRESERVING


def test_token_preserving_families_only_reuse_existing_words():
    import random
    text = "alpha beta gamma delta epsilon zeta eta theta"
    words = set(text.split())
    for fn in (perturb_token_replace, perturb_repeat_token):
        out = fn(text, 0.9, random.Random(0)).split()
        assert set(out) <= words, f"{fn.__name__} invented a token"


def test_corruptions_are_identity_at_rate_zero():
    import random
    text = "The quick brown fox jumps over the lazy dog"
    for name in ("word_drop", "word_swap", "token_replace", "repeat_token",
                 "char", "ocr"):
        assert CORRUPTIONS[name](text, 0.0, random.Random(0)) == text


def test_ocr_changes_characters_without_touching_word_count():
    import random
    text = "Ill-lit OSlo Building 588"
    out = perturb_ocr(text, 1.0, random.Random(0))
    assert out != text and len(out.split()) == len(text.split())


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_defaults_are_the_t4_settings():
    cfg = GatedConfig()
    assert cfg.dtype == "float16", "bf16 is emulated on Turing; do not default to it"
    assert cfg.seq_len == 128
    assert cfg.signed is True
    assert cfg.lora_targets == ("q_proj", "k_proj")
    assert cfg.train_corruption in TOKEN_PRESERVING


def test_eval_config_propagates_the_sequence_length():
    cfg = GatedConfig(seq_len=64)
    ec = cfg.eval_config()
    assert ec.ppl_max_length == 64 and ec.ppl_stride == 64
    assert ec.diffusion_steps == 1
