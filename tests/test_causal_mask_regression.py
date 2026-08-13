"""
Regression tests for the self-contained, cache_position-driven causal mask and
the unconditional diffusion renormalization.

Covers:
  1. Prefill causal strictness (future keys must have exactly zero mass)
  2. Chunked prefill with an existing cache (the case the old diagonal=... construction inverted)
  3. Decode q_len==1 steps (previously skipped masking entirely, trusting HF)
  4. Cached decode vs full-recompute agreement
  5. Left-padded batched generation (the stress case for causal-mask bugs)
  6. Kernel diffusion renormalization consistency with masked inputs
"""

import torch

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from torchdire.nn.qgfd_kernel import QGFDKernel
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd


def _tiny_config(**overrides):
    cfg = dict(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    cfg.update(overrides)
    return LlamaConfig(**cfg)


def _patched_model(num_layers=1, **cfg_overrides):
    model = LlamaForCausalLM(_tiny_config(num_hidden_layers=num_layers, **cfg_overrides))
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)
    model.eval()
    return model


def test_prefill_causal_strictness():
    """At prefill, every query row must have exactly zero attention mass on future keys."""
    model = _patched_model()
    torch.manual_seed(0)
    input_ids = torch.randint(1, 100, (2, 16))

    with torch.no_grad():
        attentions = model(input_ids, output_attentions=True).attentions[0]

    L = input_ids.shape[1]
    future = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    future_mass = attentions.masked_select(future[None, None, :, :])
    assert future_mass.numel() > 0
    assert (future_mass == 0.0).all(), f"Future attention mass leaked at prefill: {future_mass.abs().max().item()}"

    row_sums = attentions.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Prefill rows not normalized"

    # Sanity: self/past positions DO carry mass (not accidentally over-masked)
    total_mass = attentions.sum()
    allowed_mass = attentions.masked_select(~future[None, None, :, :])
    assert allowed_mass.sum().item() > 0.7 * total_mass.item(), "Causal mask is over-masking allowed positions"


def test_chunked_prefill_causal_with_cache():
    """
    Chunked prefill: q_len > 1 WITH an existing cache (past > 0).

    The old `diagonal = k_len - q_len + 1` construction inverted the mask in this
    case, allowing only strictly-future keys. Queries must only attend to keys at
    absolute positions <= their own.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    attn = LlamaQGFDAttention(config, layer_idx=0, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)

    cache = DynamicCache()
    chunk1 = torch.randn(1, 8, 64)
    pos1 = torch.arange(8).unsqueeze(0)
    cp1 = torch.arange(8)

    out1, probs1, cache = attn(
        chunk1, position_ids=pos1, past_key_value=cache, cache_position=cp1, output_attentions=True
    )
    assert probs1.shape == (1, 4, 8, 8)
    future1 = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert (probs1.masked_select(future1[None, None, :, :]) == 0.0).all(), "Chunk-1 future leak"

    chunk2 = torch.randn(1, 4, 64)
    pos2 = torch.arange(8, 12).unsqueeze(0)
    cp2 = torch.arange(8, 12)

    out2, probs2, cache = attn(
        chunk2, position_ids=pos2, past_key_value=cache, cache_position=cp2, output_attentions=True
    )
    assert probs2.shape == (1, 4, 4, 12), f"Expected (1, 4, 4, 12), got {probs2.shape}"

    past_seen = 8
    for i in range(4):
        q_abs = past_seen + i
        # keys with absolute position > q_abs must be strictly zero
        bad_start = q_abs + 1
        bad_cols = probs2[0, 0, i, bad_start:] if bad_start < 12 else None
        if bad_cols is not None:
            assert (bad_cols == 0.0).all(), f"Chunked prefill future leak for query at abs pos {q_abs}"
        # keys up to and including q_abs must carry mass
        assert probs2[0, 0, i, : bad_start].sum().item() > 0.9, f"Chunked prefill over-masking query at abs pos {q_abs}"

    row_sums = probs2.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Chunked rows not normalized"

    # Cache growth sanity
    assert cache.get_seq_length() == 12


def test_decode_step_q1_attends_history():
    """
    Decode (q_len == 1): a single row attending over the whole history.
    This is the step where the old code skipped masking entirely; it must not be
    over-masked either — the current token attends to all past + itself.
    """
    torch.manual_seed(0)
    config = _tiny_config()
    attn = LlamaQGFDAttention(config, layer_idx=0, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)

    cache = DynamicCache()
    prompt = torch.randn(1, 8, 64)
    out, probs_prompt, cache = attn(
        prompt, position_ids=torch.arange(8).unsqueeze(0), past_key_value=cache,
        cache_position=torch.arange(8), output_attentions=True,
    )

    x = torch.randn(1, 1, 64)
    out, probs, cache = attn(
        x, position_ids=torch.tensor([[8]]), past_key_value=cache,
        cache_position=torch.tensor([8]), output_attentions=True,
    )

    assert probs.shape == (1, 4, 1, 9)
    assert not torch.isnan(probs).any()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5), "Decode row not normalized"
    # Must be attending to the history, not collapsed on one key
    assert (probs[0, :, 0, :8].sum() > 0.5), "Decode step not attending to past history"
    assert (probs[0, :, 0, :8].min() >= 0.0)


def test_cached_decode_matches_full_recompute():
    """Step-by-step cached decoding must match recomputing the full sequence every step."""
    model = _patched_model(num_layers=2)
    torch.manual_seed(42)
    prompt = torch.randint(1, 100, (2, 6))

    with torch.no_grad():
        # Reference: full recompute each step
        ref_tokens = []
        ref_cur = prompt.clone()
        for _ in range(5):
            ref_logits = model(ref_cur).logits
            nxt = ref_logits[:, -1].argmax(dim=-1, keepdim=True)
            ref_tokens.append(nxt)
            ref_cur = torch.cat([ref_cur, nxt], dim=1)

        # Cached decode: prefill once, then 1 token per step
        cached_tokens = []
        cache_cur = prompt
        past = None
        seq_len = prompt.shape[1]
        for step in range(5):
            if step == 0:
                in_ids = cache_cur
                current_pos = torch.arange(seq_len)
            else:
                in_ids = cache_cur[:, -1:]
                current_pos = torch.tensor([seq_len + step - 1])
            out = model(
                in_ids,
                use_cache=True,
                past_key_values=past,
                cache_position=current_pos,
                position_ids=current_pos.unsqueeze(0).expand(in_ids.shape[0], -1),
            )
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cached_tokens.append(nxt)
            cache_cur = torch.cat([cache_cur, nxt], dim=1)

    for i in range(5):
        assert cached_tokens[i].tolist() == ref_tokens[i].tolist(), (
            f"Decode step {i} mismatch: cached={cached_tokens[i].tolist()} ref={ref_tokens[i].tolist()}"
        )


def test_left_padded_batch_generation():
    """Left-padded batched generation: pad columns must stay masked, output sane and diverse."""
    model = _patched_model(num_layers=2, pad_token_id=0)
    torch.manual_seed(1)

    seq_a = torch.randint(1, 100, (1, 6))
    seq_b = torch.randint(1, 100, (1, 10))
    pad = 10 - seq_a.shape[1]
    seq_a_padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), seq_a], dim=1)
    batch = torch.cat([seq_a_padded, seq_b], dim=0)
    attn_mask = (batch != 0).long()

    # Prefill strictness on padding columns: zero mass at all pads for every query
    with torch.no_grad():
        attentions = model(batch, attention_mask=attn_mask, output_attentions=True).attentions[0]
    pad_mass = attentions[0, :, :, :pad]
    assert (pad_mass == 0.0).all(), f"Attention leaked onto left-padding keys: {pad_mass.abs().max().item()}"

    with torch.no_grad():
        out_cached = model.generate(
            batch, attention_mask=attn_mask, max_new_tokens=6, min_new_tokens=6,
            do_sample=False, use_cache=True,
        )
        out_nocache = model.generate(
            batch, attention_mask=attn_mask, max_new_tokens=6, min_new_tokens=6,
            do_sample=False, use_cache=False,
        )
        # Single-row references: correct pad masking must make each batch row behave
        # exactly as if generated alone.
        ref_a = model.generate(seq_a, max_new_tokens=6, min_new_tokens=6, do_sample=False, use_cache=True)
        ref_b = model.generate(seq_b, max_new_tokens=6, min_new_tokens=6, do_sample=False, use_cache=True)

    assert out_cached.shape == (2, 16)
    assert out_nocache.shape == (2, 16)
    assert not torch.isnan(out_cached.float()).any()
    assert not torch.isnan(out_nocache.float()).any()

    # Padded row (0) and unpadded row (1) must match their single-row generations
    assert out_cached[0, 10:].tolist() == ref_a[0, 6:].tolist(), "Padded-batch row 0 diverged from single-row reference"
    assert out_cached[1, 10:].tolist() == ref_b[0, 10:].tolist(), "Padded-batch row 1 diverged from single-row reference"

    # Cached and non-cached must agree token-for-token even with padding
    assert out_cached.tolist() == out_nocache.tolist(), "Padded cached vs non-cached generation mismatch!"


def test_kernel_renormalization_consistency():
    """Renormalization must keep rows normalized and masked columns exactly zero in both modes."""
    torch.manual_seed(2)
    L = 12
    for mode in ("full", "conv"):
        kernel = QGFDKernel(diffusion_steps=4, target_alpha=0.04, warmup_steps=0, mode=mode)
        scores = torch.randn(1, 2, L, L)
        keys = torch.randn(1, 2, L, L)
        causal = torch.triu(torch.full((L, L), -1e9), diagonal=1)
        masked_scores = scores + causal[None, None, :, :]

        p_masked = kernel(masked_scores, keys)
        assert not torch.isnan(p_masked).any()
        assert (p_masked >= 0.0).all()
        assert torch.allclose(p_masked.sum(dim=-1), torch.ones_like(p_masked.sum(dim=-1)), atol=1e-5), f"mode={mode} rows not normalized"

        future = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        future_mass = p_masked.masked_select(future[None, None, :, :])
        assert (future_mass == 0.0).all(), f"mode={mode}: kernel diffused mass into masked positions"

        # No-mask control: unconditional renormalization must be a no-op on already-normalized rows
        p_plain = kernel(scores, keys)
        assert torch.allclose(p_plain.sum(dim=-1), torch.ones_like(p_plain.sum(dim=-1)), atol=1e-5), f"mode={mode} unmasked rows drifted"


if __name__ == "__main__":
    test_prefill_causal_strictness()
    test_chunked_prefill_causal_with_cache()
    test_decode_step_q1_attends_history()
    test_cached_decode_matches_full_recompute()
    test_left_padded_batch_generation()
    test_kernel_renormalization_consistency()
    print("All causal-mask regression tests passed successfully!")