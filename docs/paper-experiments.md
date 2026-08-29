# Running the paper experiments

End-to-end recipe for the QGFD introductory paper: three experiment tracks plus an
ablation, producing `paper/REPORT.md` with populated tables. Sized for a **free
T4 / P100 (~16 GB)**; the whole paper is a few GPU-hours.

---

## TL;DR

**Notebook (recommended).** Open `QGFD_Paper_Experiments.ipynb`, set the runtime to
GPU, run top to bottom. It clones the repo if needed, detects the dtype, runs all
four tracks with per-model `try/except` so one failure does not abort the run, and
finishes by rendering the report inline.

Set `QUICK = True` in the config cell for a ~5-minute rehearsal on one tiny model
before committing to the full run. Do that first — it exercises every code path.

**Command line.** Each track is independently runnable and writes its own
aggregate, so you can stop after any track and still build a coherent report:

```bash
pip install -e .

python -m scripts.review_experiments --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seeds 0,1,2 --out_dir results/zeroshot/tinyllama
python -m scripts.finetune_qgfd     --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seeds 0,1,2 --max_steps 300 --out_dir results/finetune/tinyllama
python -m scripts.eval_synthetic    --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seeds 0,1,2 --out_dir results/synthetic/tinyllama
python -m scripts.build_report --scan results --out paper/REPORT.md
```

CPU smoke test of the same pipeline (no GPU, no large download, minutes):

```bash
python -m scripts.review_experiments --model_id JackFram/llama-160m --device cpu \
    --dtype float32 --quick --seeds 0,1
python -m scripts.finetune_qgfd  --model_id JackFram/llama-160m --device cpu --quick
python -m scripts.eval_synthetic --model_id JackFram/llama-160m --device cpu --quick
```

---

## Hardware and dtype — read this before quoting bf16

**bf16 requires compute capability ≥ 8.0.** A T4 is `sm_75` and a P100 is `sm_60`,
so on free-tier hardware the run resolves to **fp16, not bf16**. The notebook
detects this and prints what actually ran:

```python
DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
```

Both arms share the dtype, so it is not a confound — but the paper must state which
one executed. Earlier drafts of the plan claimed bf16 on a T4; that was wrong.

---

## Models

QGFD is only genuinely wired for the **Llama-family adapters** —
`LlamaAttentionAdapter` and its `Qwen2` / `Mistral` subclasses. `verify_patch()`
raises rather than let an unsupported architecture report plain-softmax numbers
under a QGFD label, so an unsupported model fails loudly, never silently.

| Model | Family | Params | Role in the paper |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M` | llama | 135M | Cheapest full sweep; the ablation runs here |
| `Qwen/Qwen2.5-0.5B` | qwen2 | 0.5B | Cross-family check (different adapter subclass) |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama | 1.1B | **Headline** — 32 heads / 4 KV, so real GQA |
| `JackFram/llama-160m` | llama | 160M | Smoke tests only, not a reported result |

All four are ungated — no Hugging Face token needed. GPT-2, OPT and GPT-Neo do
**not** work: their adapters are stubs or patch zero layers.

---

## Budget

| Track | Per model | Notes |
| --- | --- | --- |
| 1 · Zero-shot | 5–15 min | 3 seeds; perplexity, noise sweep, attention stats, latency |
| 2 · LoRA A/B | 20–60 min | 3 seeds × 2 arms at `max_steps=300` |
| 3 · Synthetic | 3–10 min | 3 seeds; induction + passkey |
| 4 · Ablation | ~10 min | Smallest model only, 8 configs, 1 seed each |

Tight on VRAM? Lower `batch_size` and raise `grad_accum` — the product is what
matters, and the effective batch stays identical.

---

## Artefacts

```
<out_dir>/
  results.json                 # one zero-shot seed
  results_aggregated.json      # zero-shot, mean ± std + paired CIs   → Tables 1, 2
  robustness_curve.png         # single seed
  robustness_aggregated.png    # multi-seed, with error bars          → embedded in report
  attention_stats.png
  finetune_results.json        # one fine-tuning seed
  finetune_aggregated.json     # fine-tuning A/B                      → Table 3
  synthetic_results.json       # one synthetic seed
  synthetic_aggregated.json    # induction + passkey                  → Table 4
  ablation/ablation.json       # the (T, α, detach_P) grid
  overhead.json                # prefill latency + peak VRAM at L ∈ {128, 256, 512}
paper/REPORT.md                # built from every *_aggregated.json found
```

`build_report.py --scan` accepts directories *or* direct JSON paths, dedupes by real
path (so overlapping roots are safe), and buckets each file by its
`meta["track"]` field rather than by filename.

Plots need `matplotlib`. If it is missing, plotting is skipped with a warning and
every JSON is still written — the report simply omits the figure.

---

## What each track measures

### Track 1 — zero-shot

Both arms run through the *same* adapter; the only difference is whether the
probability vector comes from `SoftmaxOperator` or `QGFDOperator`. Measures clean
WikiText-2 perplexity, perplexity at character-noise rates {0, 5, 10, 15}%,
attention entropy and sink mass at position 0, and prefill latency.

### Track 2 — LoRA fine-tuning A/B

Equal budget, identical LoRA adapters on `q/k/v/o`, identical seed, data, learning
rate, schedule and step count. The only difference is the probability operator, so
the comparison isolates QGFD rather than adapter capacity.

Two guards run automatically because both failure modes used to be silent:

* `verify_lora_live()` does a probe forward+backward and refuses to report a result
  unless a `lora_B` tensor actually receives gradient. See
  [architecture.md](architecture.md#the-dead-lora-bug) for why this was necessary.
* `report_alpha()` checks `step_count ≥ warmup_steps`, i.e. that α genuinely warmed
  up. Warmup is driven by a `TrainerCallback`; mutating step state inside `forward()`
  diverges on gradient-checkpoint recompute.

### Track 3 — synthetic multi-hop

**Induction.** A random sequence `S` is shown twice; at each position of the second
copy the model must emit whatever followed the same token in the first copy — the
canonical two-hop circuit. Setting `induction_noise_rates` replaces a fraction of
the second copy with unrelated tokens; corrupted positions are **excluded from
scoring** rather than counted as failures, so the remaining positions stay well-posed
but must route through a garbled context. That is the same axis as the headline
robustness claim.

`control_acc` scores the *first* copy, where the answer is genuinely unpredictable.
It is the chance-level floor: if induction accuracy is not far above it, the row is
uninformative no matter which arm wins.

**Passkey.** A 5-digit key at a controlled depth in filler text, retrieved at the
end. Decoding is an explicit greedy loop with `use_cache=False`, so the probe
exercises exactly the operator the model was patched with.

> **Dataset deviation, stated plainly.** Induction sequences are built from a vetted
> single-token vocabulary so that "one word = one token" holds exactly and target
> alignment is provable. This is a cleaner and more artificial probe than
> natural-text induction, and results should not be read as natural-language
> multi-hop performance.

### Track 4 — ablation, and the α=0 equivalence check

The equivalence check is the important half: `QGFDOperator(α=0)` must produce
**bit-identical** logits to `SoftmaxOperator`. It costs one forward pass and it is
falsifiable — measured `max |Δlogit| = 0.000e+00`. If it ever fails, the drop-in
claim is false and nothing else in the report should be believed. The notebook
asserts on it.

Then a grid over `T ∈ {1,2}` × `α ∈ {0.02, 0.05}` × `detach_P ∈ {True, False}`, one
seed each. **Direction only** — one seed cannot separate settings that differ by ~1%.

---

## Troubleshooting

**`DatasetNotFoundError: Dataset 'wikitext-2-raw-v1' doesn't exist on the Hub`**
`datasets ≥ 4.0` dropped loading-script support, and the legacy `wikitext` repo is
script-based. `load_wikitext()` tries `Salesforce/wikitext` (parquet-backed) first,
then two fallbacks, and raises listing every attempt if all fail. To use your own
corpus: `load_wikitext(200, sources=[("<hf-repo>", "<config-or-None>")])`.

**`RuntimeError: no training loss was logged`**
`logging_steps` exceeded `max_steps`, so `trainer.state.log_history` never got a
`loss` entry. Fixed at the source — `train_arm` now clamps
`logging_steps = max(1, min(cfg.logging_steps, cfg.max_steps // 2 or 1))`, so no
caller can hit it.

**`RuntimeError: ... corrupted every scored position`**
`induction_noise_rates` included `1.0`. At full corruption no position has a correct
answer, so the metric is undefined and the track refuses to emit a number.

**Both synthetic arms report identical accuracy.**
Expected at small `n`, and *not* evidence the patch was inactive. At α=0.05 the arms
disagree on only ~1.4% of argmax predictions, so over ~56 predictions you expect
under one flip — and a flip on an already-wrong prediction does not move accuracy.
Check the `operator` block in the JSON: it records `qgfd_active`, `n_modules` and
`alpha_eval_mode` next to every score precisely so you can distinguish a coarse
metric from a dead patch. For headroom, raise `induction_noise_rates`.

**A `GenericAttentionAdapter` warning appeared.**
The architecture is unsupported. Treat the run as void.

---

## Verification checklist

Before believing any number in the report:

1. **Did the patch apply?** Every track calls `verify_patch()`, which raises on a
   no-op. A green run is the evidence.
2. **Did LoRA reach the live projections?** Track 2 prints
   `LoRA live: N trainable tensors, M with non-zero grad on the probe`. At
   initialisation `M ≈ N/2` is correct — `lora_B` starts at zero, so `lora_A` sees no
   gradient on the first probe. `M = 0` would have raised.
3. **Did α warm up?** The `alpha` block must show `step_count ≥ warmup_steps` and
   `alpha_train_mode == target_alpha`.
4. **Is n ≥ 3?** At n=2 the t-critical value is 12.7 and essentially nothing reaches
   significance. Two seeds is a rehearsal, not a result. The report flags this itself.
5. **Read the paired column.** Between-seed corpus variance dwarfs the effect; only
   the within-seed difference has the resolution to say anything.
6. **Did α=0 equivalence pass?** If not, stop.

---

## See also

* [interpreting-results.md](interpreting-results.md) — what the numbers license
* [results-schema.md](results-schema.md) — the JSON contract
* [architecture.md](architecture.md) — how QGFD is wired in



