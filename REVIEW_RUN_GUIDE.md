# QGFD Milestone-1 Review — Run Guide

Everything you need to produce review numbers/plots, then record the video.
**All experiments are zero-shot (no training)** — the full suite is a few minutes on one Colab/Kaggle GPU.

---

## TL;DR — what to run

**Option A · Colab (recommended for the video):**
1. Open `QGFD_Review_Experiments.ipynb` in Colab, set runtime to **GPU**.
2. Run cells top-to-bottom. **No Hugging Face token needed** — the default model is ungated,
   so you can ignore Colab's `HF_TOKEN` warning.
3. Artifacts land in `qgfd_review_results/`: `results.json`, `robustness_curve.png`, `attention_stats.png`.

**Option B · Command line (local GPU / Kaggle):**
```bash
pip install -e .            # from the repo root
python -m scripts.review_experiments --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device auto
# fast smoke test on CPU with a tiny model:
python -m scripts.review_experiments --model_id JackFram/llama-160m --device cpu --dtype float32 --quick
```

---

## Which models actually work (read this before choosing)

QGFD is only genuinely wired up for the **Llama-family adapters** —
`LlamaAttentionAdapter` and its `Qwen2`/`Mistral` subclasses. Verified on CPU:

| model | result |
|---|---|
| `JackFram/llama-160m` | ✅ `LlamaAttentionAdapter` on 12 layers, logits change |
| `gpt2` | ❌ patched forward raises `TypeError` — `GPT2AttentionAdapter` is a `pass` stub |
| `facebook/opt-125m` | ❌ patched forward raises `TypeError` — `OPTAttentionAdapter` is a `pass` stub |
| `EleutherAI/gpt-neo-125m` | ❌ **0 layers patched — QGFD silently has no effect** |

> **This contradicts the current README/slide claim** of drop-in support for "OPT and GPT-2".
> Either fix those adapters or drop them from the claim before the panel asks for a demo.

Ungated models that do work (no token required):

| model | notes |
|---|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama, 1.1B, 32 heads / 4 KV → **real GQA**, default |
| `HuggingFaceTB/SmolLM2-135M` | llama, 135M, fast |
| `JackFram/llama-160m` | llama, 160M, fastest smoke test |
| `Qwen/Qwen2.5-0.5B` | qwen2, 0.5B, 14 heads / 2 KV |

`meta-llama/Llama-3.2-1B` also works but is **gated** (needs a token + approved licence).

**Safety net:** `make_model()` calls `verify_patch()`, which raises if zero layers were patched,
if any layer fell back to `GenericAttentionAdapter`, or if the operator is never invoked during a
forward pass. A silent no-op can therefore never be reported as a QGFD result.

---

## What each experiment gives you (maps to your slides)

| Experiment | Output | Review claim it supports |
|---|---|---|
| Perplexity (WikiText-2) | `clean_ppl` per arm | Quality parity — QGFD does not degrade LM quality |
| Noise robustness | `robustness_curve.png`, Δ% table | **Core claim** — QGFD degrades less under input noise |
| Attention entropy / sink mass | `attention_stats.png` | Mechanism works — smoother, less-collapsed attention |
| Compute overhead | `prefill_ms`, `tokens_per_s`, `Nx baseline` | Honest cost accounting |
| Generation samples | printed side-by-side | Coherent text with QGFD enabled (demo in video) |

Read the printed **SUMMARY** table for the headline numbers; drop the two PNGs straight into slides.

---

## Two correctness fixes applied (verified on CPU, `JackFram/llama-160m`)

1. **`is_causal=True` is mandatory for QGFD on a causal LM.**
   With the default `is_causal=False` (as in the older copy-paste snippet), editing a *future*
   token changed logits at *earlier* positions by ~4e-3 — a causality violation: the key-graph
   diffusion `p0 @ P` spreads mass onto future keys, which deflates teacher-forced perplexity.
   With `is_causal=True` the leak is exactly `0.0`, matching the base model.
   → `build_operator()` in `scripts/review_experiments.py` forces `is_causal=True`.
   Use the same in any hand-written QGFD code.

2. **Return-arity is now transformers-version-aware** (`torchdire/nn/attention_adapters.py`).
   `LlamaAttentionAdapter` (and Qwen2/Mistral, which subclass it) returned a 2-tuple
   unconditionally, which crashes on `transformers < 4.48` (expects a 3-tuple incl. the KV cache).
   It now returns a 3-tuple on `<4.48` and a 2-tuple on `>=4.48`. All 20 operator/causal/
   integration tests pass on the local `transformers 4.47.1`.

> On Colab (modern transformers ≥4.48) the previously-pushed code already works; fix #2 mainly
> helps local/older environments. Fix #1 matters everywhere.

---

## Troubleshooting

**`DatasetNotFoundError: Dataset 'wikitext-2-raw-v1' doesn't exist on the Hub`**
The legacy `wikitext` repo is script-based, and `datasets>=4.0` removed loading-script
support. `load_wikitext()` now tries `Salesforce/wikitext` (parquet-backed, the canonical
mirror) first, then falls back to `wikitext` and `mikasenghaas/wikitext-2`, and raises a
message listing every attempt if all fail. If you hit this, **re-copy the harness cell
from the current `QGFD_Review_Experiments.ipynb`** — an older copy called
`load_dataset("wikitext-2-raw-v1")` with the config passed as the repo name.

To use your own corpus instead:
```python
texts = load_wikitext(200, sources=[("<hf-repo>", "<config-or-None>")])
```

---

## Tuning knobs (Cell 4 / CLI)

- `target_alpha` (0.05): diffusion blend fraction. Sweep {0.02, 0.05, 0.10} for an ablation slide.
- `diffusion_steps` (1): number of diffusion iterations. 1 is the validated default.
- `ppl_num_texts` / `robustness_num_texts`: lower these to run faster; raise for tighter estimates.
- `noise_rates` (0, 5%, 10%, 15%): input corruption levels for the robustness curve.

---

## Phase-2 (optional, NOT needed for this review)

A trained comparison (equal-budget LoRA SFT A/B, multi-seed variance, α/step ablation) lives in
`scripts/compare_softmax_vs_qgfd.py` and the protocol in `qgfd_experimentation_plan.md`.
That is thesis-phase work; the zero-shot suite above is sufficient for Milestone 1.
