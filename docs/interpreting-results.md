# Interpreting the results

Every table in `paper/REPORT.md` can be read in a way the data does not support.
This document is the list of those ways, and what to read instead.

---

## The headline claim is the *gap*, not the level

Zero-shot QGFD is expected to be **perplexity-neutral to slightly worse** on clean
text. Diffusion smooths the probability vector; on clean, well-formed text there is
nothing to smooth away, so a small clean-perplexity cost is the honest expectation
and finding one is not a failure.

The claim is that perplexity degrades **less** as the input is corrupted. That is
`paired.robustness_gap_pct`, at the highest noise rate:

```
gap = Δ%_softmax − Δ%_qgfd     where Δ% = 100·(ppl_noisy − ppl_clean)/ppl_clean
```

Positive means QGFD degraded less, i.e. QGFD wins. The report's abstract auto-fills
this sentence from the data and **names losses as losses** — a negative gap renders
as "degraded **more**", with an `0/1 models` count. It cannot narrate a loss as a win.

---

## Read the paired column, not the per-arm columns

Within a seed, both arms see identical texts, identical prompts and the identical
noise realisation. So the per-seed QGFD−softmax difference cancels between-seed
corpus variance, which is far larger than the effect being measured.

The synthetic aggregation test pins this concretely: with per-arm accuracies of
`(0.90, 0.80)` and `(0.92, 0.82)`, the between-seed standard deviation is `> 0.05`
while the paired gap is a constant `0.02` with **zero** variance. Comparing the
per-arm means and their spreads would conclude "no significant difference"; the
paired statistic sees a clean, consistent effect. Same data, opposite conclusion.

Per-arm columns are context. The paired column is the result.

---

## Confidence intervals are t-based, and n is small

CIs use the two-sided t critical value, not `1.96·SEM`:

| n seeds | t₀.₉₅ |
| --- | --- |
| 2 | 12.706 |
| 3 | 4.303 |
| 4 | 3.182 |

At n=2 the multiplier is 12.7 — essentially nothing can reach significance, and any
star you see at n=2 deserves suspicion rather than celebration. The report emits an
**Under-powered** bullet in Threats to Validity on its own whenever it ingests an
aggregate with n < 3; it also flags every paired CI that includes zero.

Significance markers are computed **per row**, not once per table: `**\***` when
`|mean| > ci95 > 0`, `ns` when the CI includes zero, `n/a` at n < 2. A table can
legitimately have `ns` at 5% noise and `*` at 15%.

---

## Latency is measured against eager softmax, not FlashAttention

QGFD needs the explicit probability matrix `p` in order to diffuse it. Materialising
`p` is architecturally incompatible with fused attention kernels — there is no
FlashAttention path that hands you the matrix. So the baseline in every latency
number is **eager materialised softmax**.

Measured on CPU with a 2-layer synthetic Llama:

| L | QGFD vs eager softmax |
| --- | --- |
| 128 | 1.58× |
| 256 | 1.91× |

The overhead grows with sequence length, as the `K·Kᵀ` GEMM and the `p·P` product
per diffusion step predict. But the arithmetic is the smaller half of the cost: the
dominant penalty is **structural**, losing fused kernels entirely. Against a
production SDPA/FlashAttention baseline the true gap is larger than anything reported
here, and every latency figure must be quoted with that caveat attached.

---

## Exact-match accuracy is coarse enough to hide a real effect

At α=0.05 the two arms disagree on roughly **1.4%** of argmax predictions while
`max |Δlogit| ≈ 1.05` — the operator is very much active, it just rarely flips the
top-1 choice. Over a few dozen scored predictions you therefore expect **under one
flip**, and a flip on an already-wrong prediction does not move accuracy at all.

So identical synthetic scores at small `n` are the expected outcome, not evidence of
a dead patch. Two things distinguish the cases:

* The `operator` block, recorded next to every score, carries `qgfd_active`,
  `n_modules`, `alpha_eval_mode`, `diffusion_steps`, `mode` and `is_causal`.
* `coarse_metric_note` in `meta` states the limitation inside the artefact itself.

For headroom, corrupt the context (`induction_noise_rates`) — measured accuracy
moves 0.92 → 0.73, which leaves room for an effect to appear.

---

## `control_acc` is the floor, and you must check it

`induction_control_acc` scores the **first** copy of the sequence, where the
successor is genuinely unpredictable. It is the chance-level baseline. Measured
0.0000 against 0.9286 induction accuracy on a real model, and near-zero for both on
a randomly-initialised one.

If induction accuracy is not far above `control_acc`, the row says nothing about
multi-hop routing regardless of which arm is ahead. Check the floor before reading
the comparison.

---

## Corrupted positions are excluded, not failed

At `noise_rate > 0` a fraction of the second copy is replaced with unrelated tokens.
Those positions have **no correct answer** — the query token itself was destroyed.
Scoring them as failures would mean the reported accuracy tracks the noise rate
rather than the model, and both arms would fall identically for a reason that has
nothing to do with attention.

`make_induction_examples` returns a `valid` mask, `_induction_pass` applies it and
divides by `int(valid.sum())`. `noise_rate = 1.0` raises rather than return a number.

---

## Numbers that are *not* used for any reported result

`torchdire/experiments/ablation.py` contains a `QGFDAblator` that reports ROUGE /
BLEU / BERTScore, and `torchdire/profiler/efficiency.py` contains a `QGFDProfiler`
that reports FLOPs and VRAM. **Neither set of numbers comes from a model.**

The ablator's scores are literal arithmetic on constants:

```python
base_rouge = 0.6800 + (0.015 if steps == 2 else 0.005) \
                    + (0.010 if alpha == 0.02 else 0.0) \
                    + (0.005 if detach_p else 0.0)
```

The profiler's `baseline_gflops`, `qgfd_gflops` and `estimated_vram_mb` are closed-form
formulas in `seq_len`/`num_heads`/`d_k`, not measurements. (Its latency figure *is*
timed, but the harness measures latency itself and the report uses that.)

None of these feed `paper/REPORT.md`, and none may be quoted.

This matters because an earlier draft (`IEEE_QGFD_Paper_Draft.md`) reported a
single-seed ROUGE improvement of 0.6953 vs 0.6821 from that source. Those two numbers
are `0.6800 + 0.015 + …` and `0.6800 + 0.005 + …` — the "improvement" is a constant
someone typed. It must not be carried into the paper.

For real memory and latency numbers use `overhead.json` from the notebook's sweep,
which calls `torch.cuda.max_memory_allocated()` and times actual forward passes.

Every reported metric comes from the harness running a real model.

---

## Attention entropy and sink mass are relative only

Entropy in nats and probability mass on position 0 are reported as QGFD-vs-softmax
deltas. Absolute values depend on the tokenizer, corpus and layer mix, so they are
not comparable across models or against published figures. They are mechanism
evidence — diffusion should smooth the distribution and pull mass off the sink — not
a benchmark score.

---

## Ablation rows are one seed each

The `(T, α, detach_P)` grid runs one seed per configuration to stay inside the compute
budget. With `clean_ppl` differences on the order of 1%, one seed cannot separate
adjacent settings. Read the grid for **direction** — which way `T` or `α` pushes
things — and re-run the two or three configurations you care about at n ≥ 3 before
claiming any of them is better.

---

## Missing is missing

A track that was skipped or that crashed renders as `_Not yet run._`, never as a
blank cell or a zero. An empty results tree produces a report that says the headline
claim is unsupported rather than an empty skeleton that looks finished. If you see a
number, a run produced it.

---

## See also

* [paper-experiments.md](paper-experiments.md) — how to produce the numbers
* [results-schema.md](results-schema.md) — where each field lives in the JSON


