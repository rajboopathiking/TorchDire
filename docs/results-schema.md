# Result JSON schema

The contract between the experiment scripts and anything that consumes their output,
including `scripts/build_report.py`. Stable enough to write your own report builder
or plotting code against.

---

## The `stat` block

Every aggregated number is a `stat` object, never a bare float:

```json
{
  "mean":   20.134,
  "std":     0.512,
  "sem":     0.296,
  "ci95":    1.272,
  "n":       3,
  "values": [19.71, 20.02, 20.67]
}
```

| Field | Meaning |
| --- | --- |
| `mean`, `std` | Sample mean and **sample** standard deviation (`n−1` denominator) |
| `sem` | `std / √n` |
| `ci95` | **Half-width** of the two-sided 95% CI: `t₀.₉₅(n−1) · sem`. Not `1.96·sem` |
| `n` | Number of seeds |
| `values` | Every per-seed value, so you can re-derive any statistic |

At `n = 1`, `std`/`sem`/`ci95` are all `0.0` — that is a marker for "no dispersion
information", not a claim of zero variance. At `n = 0`, `mean` is `NaN`.

Significance convention used throughout: `*` when `|mean| > ci95 > 0`, `ns` when the
CI includes zero, `n/a` when `n < 2`.

Noise-rate and depth keys are **strings** with fixed precision (`"0.0500"`,
`"0.15"`, `"0.90"`), because JSON object keys cannot be floats. Use the `_rk()`
helper in `scripts/review_experiments.py` if you need tolerant lookup.

---

## Discovery

`discover(roots)` walks each root (directories *or* direct JSON file paths),
deduplicates by `os.path.realpath`, and buckets each aggregate by
`meta["track"]`, falling back to the filename:

| Filename | Track |
| --- | --- |
| `results_aggregated.json` | `zeroshot` |
| `finetune_aggregated.json` | `finetune` |
| `synthetic_aggregated.json` | `synthetic` |

Per-seed files (`results.json`, `finetune_results.json`, `synthetic_results.json`)
are **not** ingested by the report — they are kept for auditing individual seeds.

---

## `results_aggregated.json` — zero-shot (Tables 1 and 2)

```jsonc
{
  "meta": {
    "seeds": [0, 1, 2], "n_seeds": 3,
    "model_id": "...", "device": "cuda", "dtype": "float16",
    "diffusion_steps": 1, "target_alpha": 0.05,
    "noise_rates": ["0.0000", "0.0500", "0.1000", "0.1500"],
    "ci_method": "two-sided t, 95%",
    "baseline_note": "baseline is eager materialised softmax ..."
  },
  "arms": {
    "softmax": {
      "clean_ppl":            { /* stat */ },
      "robustness":           { "0.0500": { /* stat */ }, ... },   // absolute PPL
      "robustness_delta_pct": { "0.0500": { /* stat */ }, ... },   // % vs own clean
      "attention": { "mean_attention_entropy_nats": {}, "mean_sink_mass_pos0": {} },
      "latency":   { "prefill_ms": {}, "tokens_per_s": {} }
    },
    "qgfd": { /* identical shape */ }
  },
  "paired": {
    "clean_ppl_qgfd_minus_softmax": { /* stat */ },  // positive => QGFD costs PPL
    "latency_overhead_x":           { /* stat */ },  // qgfd_ms / softmax_ms
    "robustness_gap_pct": { "0.0500": { /* stat */ }, ... }
  }
}
```

**`robustness_gap_pct` is the headline.** Per seed it is
`Δ%_softmax − Δ%_qgfd`, so **positive means QGFD degraded less**, i.e. QGFD wins.
The `0.0` rate is omitted — its gap is zero by construction.

`robustness_delta_pct` is computed against **that seed's own** clean perplexity, not
against a pooled mean, which is what makes the pairing valid.

`attention` and `latency` are present only when *every* run has them, so the
fine-tuning track (which reuses this aggregator without them) omits both keys rather
than emitting nulls. Consumers must treat them as optional.

---

## `finetune_aggregated.json` — LoRA A/B (Table 3)

Same aggregator as the zero-shot track, so `arms.*.clean_ppl`,
`arms.*.robustness`, `arms.*.robustness_delta_pct` and
`paired.{clean_ppl_qgfd_minus_softmax, robustness_gap_pct}` are identical in shape —
but measured **after** fine-tuning. `attention`, `latency` and
`paired.latency_overhead_x` are absent.

Extra keys:

```jsonc
{
  "meta": {
    "track": "finetune",
    "backend": "operator",          // or "kernel"
    "max_steps": 300,
    "lora": { "r": 16, "alpha": 32, "targets": ["q_proj","k_proj","v_proj","o_proj"] }
  },
  "train": {
    "softmax": { "final_loss": { /* stat */ }, "seconds": { /* stat */ } },
    "qgfd":    { "final_loss": { /* stat */ }, "seconds": { /* stat */ } }
  }
}
```

Because both arms get identical budget, data order, LoRA rank and target modules,
a difference in `train.*.final_loss` is attributable to the operator rather than to
adapter capacity. `seconds` is wall-clock, so it is only comparable within one run
on one machine.

If you pass a single arm (`--arms qgfd`), no pairing is possible and the function
returns `{"runs": [...]}` with no `meta`/`arms`/`paired` — `discover()` will not
bucket it, and the report will show the track as not run. That is deliberate: a
one-armed run cannot support an A/B claim.

---

## `synthetic_aggregated.json` — induction and passkey (Table 4)

```jsonc
{
  "meta": {
    "track": "synthetic", "seeds": [0,1,2], "n_seeds": 3,
    "post_lora": false,                     // true => probes ran on a LoRA checkpoint
    "model_id": "...", "backend": "operator", "device": "cuda",
    "diffusion_steps": 1, "target_alpha": 0.05,
    "induction_seq_len": 48,
    "induction_predictions_per_seed": 2944,  // sample size behind each accuracy
    "passkey_context_tokens": 384,
    "passkey_n_per_seed": 72,
    "ci_method": "t-based 95% CI (small n); std is the sample std",
    "operator": {                            // per arm — the provenance record
      "softmax": { "qgfd_active": false, "n_modules": 22 },
      "qgfd":    { "qgfd_active": true,  "n_modules": 22,
                   "alpha_eval_mode": 0.05, "diffusion_steps": 1,
                   "mode": "full", "is_causal": true }
    },
    "coarse_metric_note": "...",
    "control_note": "..."
  },
  "arms": {
    "softmax": {
      "induction_acc":         { /* stat */ },   // clean (noise 0.00)
      "induction_control_acc": { /* stat */ },   // chance-level FLOOR
      "induction_by_noise":    { "0.00": {}, "0.20": {}, "0.40": {} },
      "passkey_acc":           { /* stat */ },   // strict prefix match
      "passkey_by_depth":      { "0.10": {}, "0.50": {}, "0.90": {} }
    },
    "qgfd": { /* identical shape */ }
  },
  "paired": {
    "induction_gap":           { /* stat */ },   // qgfd − softmax, positive = QGFD better
    "induction_gap_by_noise":  { "0.00": {}, "0.20": {}, ... },
    "passkey_gap":             { /* stat */ },
    "passkey_gap_by_depth":    { "0.10": {}, ... },
    "note": "qgfd - softmax, computed within each seed on identical prompts"
  }
}
```

Three fields exist specifically to stop a misreading, and a consumer should surface
all three:

* **`induction_control_acc`** — the floor. If `induction_acc` is not well above it,
  no in-context induction happened and the arm comparison is meaningless.
* **`meta.operator`** — proof the patch was live. This is what distinguishes
  "identical scores because the metric is coarse" from "identical scores because
  QGFD never ran".
* **`meta.coarse_metric_note`** — the limitation, carried inside the artefact.

Per-seed `synthetic_results.json` additionally holds the lenient `contains` score
alongside strict `acc` for each depth; strict is always ≤ lenient.

---

## Notebook-only artefacts

These are written by `QGFD_Paper_Experiments.ipynb` rather than by a script, and are
**not** ingested by `scripts/build_report.py` — read them directly.

`ablation/ablation.json` — a flat list, one entry per grid cell, one seed each:

```json
[{"T": 1, "alpha": 0.02, "detach_P": true,
  "clean_ppl": 20.13, "noise_rate": 0.15, "degradation_pct": 41.2}]
```

`overhead.json` — prefill latency and peak VRAM per arm per sequence length:

```json
{"model_id": "...", "dtype": "float16",
 "rows": [{"arm": "softmax", "seq_len": 128, "prefill_ms": 10.0, "peak_vram_mb": 812.0}]}
```

`peak_vram_mb` is `null` on CPU. Both arms are measured in the same process with
`torch.cuda.reset_peak_memory_stats()` between sequence lengths.

---

## Consuming the output yourself

```python
from scripts.build_report import discover

found = discover(["qgfd_paper_results"])          # dirs and/or JSON paths
for path, agg in found["zeroshot"]:
    gap = agg["paired"]["robustness_gap_pct"]
    worst = max(gap, key=float)                    # keys are strings
    s = gap[worst]
    verdict = "*" if abs(s["mean"]) > s["ci95"] > 0 else "ns"
    print(f"{agg['meta']['model_id']} @ {float(worst):.0%} noise: "
          f"{s['mean']:+.2f} pp ± {s['ci95']:.2f} ({verdict}, n={s['n']})")
```

Two rules for any consumer:

1. **Treat `attention`, `latency` and `paired.latency_overhead_x` as optional.**
   They are omitted, not nulled, on the fine-tuning track.
2. **Never render a missing track as `0` or as an empty cell.** The report builder
   emits `_Not yet run._`; a number that was never measured must never look measured.

---

## See also

* [interpreting-results.md](interpreting-results.md) — what these fields license
* [paper-experiments.md](paper-experiments.md) — how to produce them




