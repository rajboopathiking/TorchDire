# QGFD vs. Softmax Baseline — Experimentation Plan (Kaggle GPU only)

**Context:** A single-seed, 300-step equal-budget A/B test showed QGFD attention
(α=0.10, diffusion_steps=3) matching softmax on eval loss (Δ = -0.0008, not
distinguishable from noise at this sample size) while costing ~1.19x compute.
This plan turns that one data point into a defensible claim, using only Kaggle's
free GPU tier.

**Kaggle constraints this plan is built around:** ~30 GPU-hours/week free quota,
one P100 or 2×T4 (16GB / 32GB combined), sessions capped at roughly 9-12 hours
(confirm the current limit in your notebook's session panel before planning a
long run — it has changed before). 20GB persistent output storage.

**Cost baseline (from the existing run):** 300 steps ≈ 13.3 min (baseline) /
16 min (QGFD) ≈ 0.5 GPU-hr per seed-pair at this scale. Cheap — most of the
budget below goes to phases that buy real evidence (variance, longer horizon),
not to repeating the same 300-step config.

---

## Phase 0 — Fix known bugs before spending any more GPU time

### 1. LoRA `target_modules` is wrong for this architecture

```python
# WRONG — "out_proj" is the standalone MultiHeadQGFDLayer's naming,
# not LlamaQGFDAttention's. patch_llama_with_qgfd preserves the
# original Llama attribute name, which is o_proj.
target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],

# FIX
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
```
As written, PEFT either errors on an unmatched target module or silently
leaves the output projection un-adapted in both arms. Confirm this is fixed
by checking `model.print_trainable_parameters()` shows LoRA attached to all
four projections before trusting any comparison.

### 2. `run_experiment`'s `warmup_steps` argument is dead code

```python
def run_experiment(args, seed, tag, enable_qgfd, target_alpha, warmup_steps):
    ...
    model = patch_llama_with_qgfd(
        model,
        diffusion_steps=args.diffusion_steps,
        target_alpha=target_alpha,
        warmup_steps=warmup_steps,      # was: args.warmup_steps (ignored the param)
        early_stop_eps=0.0,
        enable_qgfd=enable_qgfd,
        verbose=args.verbose,
    )
```
The function signature takes `warmup_steps`, but the body always used
`args.warmup_steps` (=30) instead. The comment says alpha "ramps 0→0.05 over
the run" (300 steps); what actually happened is alpha hit its target by step
30 and sat flat for the remaining 270. Decide which schedule you want and
make the code match it.

### 3. No memory cleanup between sequential runs

```python
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
```
`main()` currently runs both arms in the same process without releasing the
first model. Harmless at 1.3B/4-bit maybe, but it will bite the moment you
loop over seeds/alphas below, and it can quietly bias `wall_s` on later runs.

### 4. Full seeding

Swap `torch.manual_seed(seed)` for `transformers.set_seed(seed)` (covers
`random` / `numpy` too) — cheap insurance for reproducibility across seeds.

**Before any phase below:** run a 20-step smoke test of both arms with these
fixes applied. Confirm no crashes, loss decreasing, LoRA attached correctly,
and — temporarily set `debug=True` on the kernel or assert `alpha_eff != 0`
after a few steps — confirm the QGFD path is actually taking the diffusion
branch. Rule out "QGFD never activates" as a trivial explanation for a null
result before spending real budget.

---

## Phase 1 — Smoke test

**Budget:** ~20 min · **Config:** 1 seed, 20 steps, both arms, `save_strategy="no"`

Purpose: verify Phase 0 fixes work end-to-end. Check:
- No crashes post-fix
- `model.print_trainable_parameters()` shows LoRA on `q_proj`, `k_proj`,
  `v_proj`, `o_proj` in both arms
- QGFD kernel actually diffuses (non-zero alpha mid-run)
- Throughput numbers look sane (no wildly different `it/s` from data loading
  stalls, etc.)

---

## Phase 2 — Establish variance at the current config

**Budget:** ~2.5 GPU-hr · **Config:** 5 seeds (42, 43, 44, 45, 46) ×
{baseline, QGFD α=0.10, diffusion_steps=3} × 300 steps

The existing result is n=1. Run 5 seeds for both arms. Log full-precision
`eval_loss` (4+ decimals) per run to a CSV in `/kaggle/working/`.

For each seed, compute the paired difference:

```
Δ_seed = eval_loss_baseline(seed) - eval_loss_qgfd(seed)
```

(Paired, since each seed also reshuffles the train/test split identically
for both arms.) Report:
- mean(Δ), std(Δ)
- paired 95% CI (t-distribution is fine at n=5; a Wilcoxon signed-rank test
  is a reasonable nonparametric cross-check if you're worried about
  normality)
- mean ± std of the `it/s` ratio across the same 5 pairs — check whether
  1.19x is stable or itself noisy

**Decision rule:** if the CI excludes zero, you have a real (if small)
quality effect in one direction. If it straddles zero, the result is
genuinely indistinguishable from noise — a legitimate, publishable finding,
not a failure of the experiment.

---

## Phase 3 — Alpha / diffusion_steps ablation

**Budget:** ~2 GPU-hr (sweep) + ~2.5 GPU-hr (replication of the winner)

**Sweep pass** (single seed=42, 300 steps):
- `target_alpha ∈ {0.02, 0.05, 0.10, 0.20}` at fixed `diffusion_steps=3` (4 runs)
- `diffusion_steps ∈ {1, 2, 4}` at whichever alpha looked most promising (3 runs)

Rationale: the diffusion correction term scales roughly with alpha, so it's
plausible α=0.10 is too weak or too strong at this model/task scale to show
anything. This is a cheap first pass to check whether the Phase 2 null
result is representative of the mechanism or just a bad point on the curve.

**Replication:** whichever config wins the sweep, rerun Phase 2's 5-seed
protocol on it before trusting the result.

---

## Phase 4 — Does the null hold at a longer horizon?

**Budget:** ~5 GPU-hr, one session · **Config:** 1 seed, 3,000 steps
(≈2.2 hr baseline / 2.7 hr QGFD)

300 steps is a small fraction of even one epoch on Alpagasus (~9K examples).
Changes needed:
- `save_strategy="no"` → `save_strategy="steps"`, `save_steps=500` (so a
  session timeout doesn't lose the run — resume via `resume_from_checkpoint`)
- Add `eval_strategy="steps"`, `eval_steps=200` to get an eval-loss
  **trajectory**, not just an endpoint

Plot both curves. This is the only way to see whether QGFD separates from
baseline later in training rather than assuming 300 steps was sufficient to
judge the mechanism.

---

## Phase 5 — Profile the 19% overhead

**Budget:** ~30 min, cheap (forward/backward passes only, no full training)

Wrap `build_transition_from_keys` and the diffusion loop in
`torch.profiler` or manual `time.perf_counter()` calls around a few forward
passes on the real model. Break the overhead into:
- similarity matmul (`K_norm @ K_norm^T`)
- diffusion loop iterations
- Python / dispatch overhead

Tells you whether 1.19x has room to shrink (fusing the diffusion loop,
caching P, etc.) or is close to the mechanism's floor. Relevant regardless
of what Phases 2-4 show — even a quality win isn't worth much at a bad cost
floor.

---

## Phase 6 — Test axes other than generic eval loss

**Budget:** ~1.5 GPU-hr

Perplexity may simply be the wrong lens for what this mechanism changes. On
the Phase 2/4 checkpoints, cheaply check:

- **Attention entropy / sink concentration** — forward-hook a handful of
  eval examples through both models, compare attention entropy and how much
  mass sits on position 0. Near-zero cost, directly tests what QGFD's
  mechanism is supposed to change.
- **Generation-quality metric** on held-out Alpagasus prompts — ROUGE-L
  against reference outputs. (An LLM-judge would need external API access —
  confirm your Kaggle notebook has internet enabled if you want that route
  instead.)

If quality and efficiency both stay flat through Phases 2-4, this is where
you'd look for a reason to keep the mechanism alive — or a reason to write
the honest negative result and stop.

---

## Budget summary

| Phase | GPU-hr | Purpose |
|---|---|---|
| 0-1. Bug fixes + smoke test | ~0.3 | Confirm fixes work, QGFD path activates |
| 2. 5-seed variance | ~2.5 | Turn n=1 into a real Δ with a CI |
| 3. Ablation + replication | ~4.5 | Rule out "wrong α/steps" before concluding null |
| 4. Long-horizon run | ~5 | Rule out "null result is just early-training" |
| 5. Profiling | ~0.5 | Ground the 1.19x claim, find slack if any |
| 6. Alternative axes | ~1.5 | Test the mechanism where it's more likely to matter |
| **Total** | **~14.5** | Fits inside one week's 30-hr quota with margin |

Leave slack in-week against the quota — don't schedule phases back-to-back
with zero margin for a crashed session or a bug found mid-phase.

## What each phase lets you claim

- **Phase 2 alone**: whether the original -0.0008 result is signal or noise.
- **Phase 2 + 3**: whether the null (if it holds) is specific to α=0.10,
  diffusion_steps=3, or general across the mechanism's hyperparameter range.
- **Phase 2 + 3 + 4**: whether the null holds at a training horizon long
  enough to matter, not just at 300 steps.
- **Phase 5**: whether the ~19% compute overhead has room to shrink or is
  near the mechanism's floor — needed context for any cost/benefit claim.
- **Phase 6**: whether QGFD's effect (if any) shows up on properties the
  mechanism more plausibly changes (attention concentration, robustness)
  even if generic LM eval loss stays flat.

If quality and efficiency both stay flat through Phases 2-4, the defensible
write-up is: *"Under a matched training-step budget, QGFD attention does not
improve validation perplexity relative to standard softmax attention (Δ not
distinguishable from zero across N seeds and an α/diffusion_steps sweep),
while incurring ~15-20% additional compute per step."* That is a legitimate
negative result — report the compute cost plainly rather than treating it as
a tradeoff.
