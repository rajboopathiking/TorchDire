# 🔒 Self-Contained Causal Masking & Diffusion Renormalization Fix

> **Scope:** `torchdire/nn/llama_qgfd.py`, `torchdire/nn/qgfd_kernel.py`, `tests/test_causal_mask_regression.py`
> **Status:** Merged — `1258e15` · **Validation:** 33/33 tests passing

---

## 1. Problem: The `q_len > 1` Trust Gap

The "Bulletproof Causal Masking" block in `LlamaQGFDAttention.forward` guarded
its causal mask with `if q_len > 1`. Consequences:

| Call type | `q_len` | Old behavior |
|---|---|---|
| Prefill (no cache) | `q_len > 1` | Custom `triu(diagonal=k_len - q_len + 1)` mask applied — correct only when `past_len == 0` |
| **Decode (every step after prefill)** | `q_len == 1` | **Mask skipped entirely** — the layer trusts whatever HF delivers in `attention_mask` |
| **Chunked prefill with live cache** | `q_len > 1, past > 0` | `diagonal = past + 1` **inverts** the mask: only strictly-future keys unmasked, all past keys + self blocked |

During decode, correctness depended on HF's `_update_causal_mask` /
`_ignore_causal_mask_sdpa` decisions for the exact transformers version, cache
type (`DynamicCache` vs `StaticCache`), and compilation path. One violated
trust step writes an unmasked / wrongly-masked attention output into the KV
cache — and every subsequent step attends through that poisoned entry with no
way to self-correct ("correct through prefill + first decode step, then
permanent garbage").

## 2. Fix: `cache_position`-Driven Mask via `masked_fill`

The mask is now built inside the layer from the query's **absolute positions**,
independent of `q_len` and of HF's mask delivery:

```python
k_len    = key_states.shape[-2]
past_seen = k_len - q_len
q_abs    = cache_position        # fallback: position_ids, then arange(past_seen, k_len)
key_abs  = torch.arange(k_len)
future   = key_abs[None, :] > q_abs[:, None]          # (q_len, k_len)
attn_weights = attn_weights.masked_fill(future[None, None, :, :],
                                        torch.finfo(attn_weights.dtype).min)
```

- **Prefill** (`past = 0`): `future = j > i` — classic strict causality.
- **Decode** (`q_len == 1`): the single row is at the max absolute position,
  `future.all() == False` — no fill, all cached keys remain attendable.
- **Chunked prefill** (`past > 0`): query at absolute position `p` blocks keys
  `> p` only — fixes the inversion.
- Single `masked_fill` (no additive stacking) — the mask value never
  accumulates across the layer mask + HF padding mask.
- `torch.finfo(dtype).min` is always `< mask_threshold (-1e4)`, so the QGFD
  kernel's `_build_valid_mask` excludes masked keys from diffusion.

HF's `attention_mask` (4D or 2D) is still added afterwards for **padding**
only — causality no longer depends on it.

## 3. Fix: Unconditional Diffusion Renormalization

`QGFDKernel` renormalized the diffused term only `if valid_mask is not None`.
Decode-only calls with an all-valid mask (the unmasked failure mode) skipped
renormalization while `p0` was always softmax-normalized over all keys —
asymmetric normalization support. Both the `full` and `conv` diffusion loops
now always normalize:

```python
p_next = (1 - alpha) * p0 + alpha * (p @ P)   # or conv variant
if valid_mask is not None:
    p_next = p_next * valid_mask
Z = p_next.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_next))
p_next = p_next / Z
```

With no mask this is a no-op (`Z == 1`), so prefill / decode / padded calls
always normalize against the same support — cheap insurance against drift.

## 4. Verification (`tests/test_causal_mask_regression.py`)

| Test | Assertion |
|---|---|
| `test_prefill_causal_strictness` | Future keys carry **exactly zero** mass; rows normalized |
| `test_chunked_prefill_causal_with_cache` | Queries at abs pos `p` attend keys `<= p` only (the previously-inverted path) |
| `test_decode_step_q1_attends_history` | `q_len == 1` row attends the full history, normalized, no NaN |
| `test_cached_decode_matches_full_recompute` | Step-by-step cached decode == full recompute, token-for-token |
| `test_left_padded_batch_generation` | Pad columns exactly zero; each batch row equals its single-row generation; cached == non-cached |
| `test_kernel_renormalization_consistency` | Both modes: masked positions exactly zero, rows normalized, unmasked rows don't drift |

Run: `python tests/test_causal_mask_regression.py` or
`python -m pytest tests/ -k "not integration"`

**Suite result:** 33 passed, 0 failed (incl. `test_llama_qgfd.py`, `test_kernels.py`, `test_qgfd.py`, `test_replacer.py`).