"""
Unit tests for scripts/mechanism_experiments.py.

Each mechanism experiment is a *control*, so the thing that must be tested is
that the control is actually controlled. A temperature arm that silently differs
from softmax at tau=1, or a "uniform P" that quietly moves a different amount of
mass than the real P, would produce a clean-looking number that answers no
question at all — and unlike a crash, nothing downstream would notice.

So these tests pin the invariants the comparisons rest on:
  * TemperatureOperator at tau=1 is bit-identical to SoftmaxOperator.
  * Every P variant is row-stochastic and strictly causal, and the total
    variation displacement bound ||p - p0|| <= alpha holds for all of them —
    that is what makes "same alpha, different structure" a fair comparison.
  * The shuffled P preserves each row's multiset of probabilities exactly.
  * fit_power_law recovers a known exponent.
  * The word-level corruptions leave surviving words' tokenisation intact,
    which is the entire point of E6.
"""
import math
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("transformers")

from torchdire.nn.attention_operators import QGFDOperator, SoftmaxOperator  # noqa: E402

from scripts.mechanism_experiments import (  # noqa: E402
    CORRUPTIONS,
    DEFAULT_ALPHAS,
    MECHANISM_EXPERIMENTS,
    StructuredPOperator,
    TemperatureOperator,
    apply_quick,
    fit_power_law,
    perturb_word_drop,
    perturb_word_swap,
    precompute_corruptions,
)
from scripts.review_experiments import ExperimentConfig  # noqa: E402
B, H, L, D = 2, 4, 16, 8


@pytest.fixture
def scores():
    torch.manual_seed(0)
    s = torch.randn(B, H, L, L)
    mask = torch.full((L, L), torch.finfo(s.dtype).min).triu(1)
    return s + mask[None, None]


@pytest.fixture
def keys():
    torch.manual_seed(1)
    return torch.randn(B, H, L, D)


# --------------------------------------------------------------------------- #
# E3 — the temperature control must BE softmax at tau = 1
# --------------------------------------------------------------------------- #
def test_temperature_at_tau_one_is_bit_identical_to_softmax(scores):
    ref = SoftmaxOperator()(scores)
    got = TemperatureOperator(tau=1.0)(scores)
    assert torch.equal(ref, got), (
        "tau=1 must reproduce softmax exactly, otherwise the E3 baseline and the "
        f"matched arm differ for a second reason; max |d| = {(ref - got).abs().max()}"
    )


@pytest.mark.parametrize("tau", [0.25, 0.5, 2.0, 8.0])
def test_temperature_is_monotone_in_entropy(scores, tau):
    """Bisection in match_temperature is only valid if entropy is monotone in tau."""
    def entropy(t):
        p = TemperatureOperator(tau=t)(scores).clamp_min(1e-12)
        return float((-(p * p.log()).sum(-1)).mean())

    if tau > 1.0:
        assert entropy(tau) > entropy(1.0)
    else:
        assert entropy(tau) < entropy(1.0)


def test_temperature_survives_a_fully_masked_row():
    """
    A row of all-min sentinels must not become NaN. Scaling before masking is
    what prevents min/tau from overflowing to -inf and softmaxing to NaN.
    """
    s = torch.zeros(1, 1, 2, 3)
    bad = torch.full((1, 1, 2, 3), torch.finfo(s.dtype).min)
    out = TemperatureOperator(tau=0.25)(s + bad)
    assert torch.isfinite(out).all(), "fully-masked row produced non-finite probabilities"
# --------------------------------------------------------------------------- #
# E4 — every P variant must be an equally fair control
# --------------------------------------------------------------------------- #
def _op(structure, alpha=0.05, steps=1):
    return StructuredPOperator(diffusion_steps=steps, target_alpha=alpha, mode="full",
                               max_full_seq_len=64, detach_P=True, is_causal=True,
                               p_structure=structure, max_alpha=1.0).eval()


@pytest.mark.parametrize("structure", ["real", "uniform", "shuffled"])
def test_all_P_variants_are_row_stochastic(keys, structure):
    P = _op(structure).build_transition_from_keys(keys, target_heads=H)
    sums = P.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"{structure} P is not row-stochastic; max deviation "
        f"{(sums - 1).abs().max():.2e}. Non-stochastic P breaks the alpha bound "
        "and the arms stop being comparable."
    )


@pytest.mark.parametrize("structure", ["real", "uniform", "shuffled"])
def test_all_P_variants_are_strictly_causal(keys, structure):
    P = _op(structure).build_transition_from_keys(keys, target_heads=H)
    future = ~torch.tril(torch.ones(L, L, dtype=torch.bool))
    leak = P[:, :, future].abs().max()
    # The operator adds a uniform jitter of _eps(P)/L to every entry by design.
    assert leak <= 1e-6 / L * 2, (
        f"{structure} P puts {leak:.2e} mass on future keys. Diffusing onto a "
        "future key leaks the token being predicted and deflates perplexity."
    )


def test_shuffled_P_preserves_each_rows_probability_multiset(keys):
    real = _op("real").build_transition_from_keys(keys, target_heads=H)
    shuf = _op("shuffled").build_transition_from_keys(keys, target_heads=H)
    assert torch.allclose(real.sort(-1).values, shuf.sort(-1).values, atol=1e-6), (
        "shuffled P must be a permutation of the real P within each row — that is "
        "what holds entropy, sparsity and max-mass fixed while varying only which "
        "key receives which mass"
    )
    assert not torch.allclose(real, shuf), "shuffle changed nothing"


def test_uniform_P_is_actually_uniform_over_the_causal_prefix(keys):
    P = _op("uniform").build_transition_from_keys(keys, target_heads=H)
    for i in (1, 5, L - 1):
        row = P[0, 0, i, : i + 1]
        assert torch.allclose(row, row.mean().expand_as(row), atol=1e-6), (
            f"row {i} of uniform P is not uniform: min {row.min():.3e} "
            f"max {row.max():.3e}"
        )
@pytest.mark.parametrize("structure", ["real", "uniform", "shuffled"])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_every_variant_moves_at_most_alpha_of_the_mass(scores, keys, structure, steps):
    """
    The displacement bound that makes E4 a fair comparison.

    Every iterate is an explicit convex combination (1-a)*p0 + a*(p^t P), and
    p^t P is itself a distribution, so ||p^t - p0||_TV <= a for every t. All
    three structures therefore reallocate the SAME budget of probability mass;
    only the destination differs. Without this, "uniform P helped less" could
    just mean "uniform P moved less".
    """
    alpha = 0.05
    op = _op(structure, alpha=alpha, steps=steps)
    p0 = SoftmaxOperator()(scores).float()
    p = op(scores, key_states=keys).float()
    tv = 0.5 * (p - p0).abs().sum(-1).max()
    assert tv <= alpha + 1e-4, (
        f"{structure} at T={steps} displaced {tv:.4f} > alpha={alpha}; the arms "
        "no longer share a mass budget"
    )
    assert tv > 0, f"{structure} at T={steps} changed nothing — the arm is a no-op"


# --------------------------------------------------------------------------- #
# E2 — the curvature fit must recover a known exponent
# --------------------------------------------------------------------------- #
def test_fit_power_law_recovers_a_quadratic():
    alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    deltas = [0.0] + [3.7 * a ** 2 for a in alphas[1:]]
    fit = fit_power_law(alphas, deltas)
    assert fit["exponent_k"] == pytest.approx(2.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["consistent_with_quadratic"] is True


def test_fit_power_law_distinguishes_linear_from_quadratic():
    alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    fit = fit_power_law(alphas, [0.0] + [0.9 * a for a in alphas[1:]])
    assert fit["exponent_k"] == pytest.approx(1.0, abs=1e-6), (
        "a linear cost must not be reported as quadratic — that distinction is the "
        "entire content of E2"
    )


def test_fit_power_law_refuses_to_fit_too_few_points():
    fit = fit_power_law([0.0, 0.05], [0.0, 1.0])
    assert fit["exponent_k"] is None and "note" in fit
# --------------------------------------------------------------------------- #
# E6 — the word-level corruptions must leave tokenisation alone
# --------------------------------------------------------------------------- #
TEXT = ("The quick brown fox jumps over the lazy dog while attention mechanisms "
        "propagate information across key neighborhoods in a single layer.")


def test_word_drop_removes_words_and_keeps_the_rest_verbatim():
    out = perturb_word_drop(TEXT, 0.3, random.Random(0))
    kept, orig = out.split(), TEXT.split()
    assert len(kept) < len(orig), "nothing was dropped"
    assert set(kept) <= set(orig), "word_drop invented a word it should only delete"
    # Order of survivors is preserved — this is deletion, not a shuffle.
    it = iter(orig)
    assert all(any(w == o for o in it) for w in kept)


def test_word_swap_preserves_the_exact_token_multiset():
    out = perturb_word_swap(TEXT, 0.5, random.Random(0))
    assert sorted(out.split()) == sorted(TEXT.split()), (
        "word_swap must permute words, not alter them — otherwise it stops being a "
        "pure word-order corruption"
    )
    assert out != TEXT, "nothing was swapped"


def test_word_corruptions_do_not_refragment_surviving_words():
    """
    The whole reason E6 exists: character noise re-tokenises a word into subword
    fragments, so a QGFD win under it is confounded with fragmentation repair.
    Word-level corruption must not introduce that confound itself.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    n_clean = len(tok(TEXT)["input_ids"])

    for fn in (perturb_word_drop, perturb_word_swap):
        out = fn(TEXT, 0.4, random.Random(1))
        for w in out.split():
            assert w in TEXT.split(), f"{fn.__name__} produced a novel word {w!r}"
        # No word was altered, so no word can have gained subword fragments; the
        # count may fall (deletion) but must never rise.
        assert len(tok(out)["input_ids"]) <= n_clean, (
            f"{fn.__name__} increased the token count, which means it re-fragmented "
            "something and reintroduces the confound E6 exists to remove"
        )

    noisy = CORRUPTIONS["char"](TEXT, 0.4, random.Random(1))
    assert len(tok(noisy)["input_ids"]) > n_clean, (
        "character noise is expected to INCREASE token count via fragmentation; if "
        "it does not, the premise of E6 needs re-checking on this tokenizer"
    )


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_zero_rate_is_the_identity(name):
    assert CORRUPTIONS[name](TEXT, 0.0, random.Random(0)) == TEXT


def test_precompute_corruptions_is_byte_identical_across_calls():
    """The pairing in every sweep depends on this: same seed -> same bytes."""
    texts = [TEXT, TEXT[::-1]]
    a = precompute_corruptions(texts, [0.0, 0.15], "char", seed=7)
    b = precompute_corruptions(texts, [0.0, 0.15], "char", seed=7)
    assert a == b
    assert a[0.15] != a[0.0], "the 15% variant is identical to clean"
# --------------------------------------------------------------------------- #
# Suite wiring
# --------------------------------------------------------------------------- #
def test_quick_config_is_small_enough_to_be_a_cpu_smoke():
    cfg = apply_quick(ExperimentConfig())
    assert cfg.device == "cpu" and cfg.dtype == "float32"
    assert cfg.robustness_num_texts <= 8 and cfg.ppl_max_length <= 128
    assert cfg.max_full_seq_len <= 128


def test_default_alphas_include_the_softmax_baseline():
    assert 0.0 in DEFAULT_ALPHAS, (
        "alpha=0 is the in-sweep softmax baseline every delta is measured against"
    )
    assert sorted(DEFAULT_ALPHAS) == list(DEFAULT_ALPHAS)


def test_every_experiment_declares_what_a_negative_result_kills():
    for key, (title, kills) in MECHANISM_EXPERIMENTS.items():
        assert title and kills, f"{key} has no falsification statement"


def test_alpha_sweep_rejects_a_missing_zero_baseline():
    from scripts.mechanism_experiments import alpha_noise_sweep
    with pytest.raises(ValueError, match="alpha=0.0 must be in"):
        alpha_noise_sweep(apply_quick(ExperimentConfig()), alphas=(0.02, 0.05))


def test_qgfd_at_alpha_zero_is_still_bit_exact_softmax(scores, keys):
    """
    Every sweep here uses alpha=0 as its softmax arm instead of loading a second
    model. That shortcut is only legitimate while this holds.
    """
    op = QGFDOperator(diffusion_steps=2, target_alpha=0.0, mode="full",
                      max_full_seq_len=64, detach_P=True, is_causal=True).eval()
    ref = SoftmaxOperator()(scores)
    got = op(scores, key_states=keys)
    assert torch.equal(ref.float(), got.float()), (
        f"alpha=0 is no longer softmax (max |d| = {(ref - got).abs().max():.3e}); "
        "the in-sweep baselines in E2/E4/E6/E7 are invalid"
    )
