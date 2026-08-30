# Gated QGFD — experimental plan

**One page:** fixed-α QGFD does not work, we know exactly why, and the fix is to make
the mixing coefficient *learned, zero-initialised and per-head* so diffusion only
happens where a head earns it. This document is the plan for testing that on a single
free Kaggle T4, with a 40-minute screen in front of the expensive part.

- **Code:** `scripts/gated_experiments.py`
- **Notebook:** `QGFD_Gated_Experiments.ipynb`
- **Tests:** `tests/test_gated_experiments.py` (32 tests, CPU, no downloads)
- **Budget:** under 8 GPU-hours for all four stages; 70 minutes if the screen says stop

---

## 1. Where we are

QGFD replaces the attention probabilities with a one-step random walk over a
key-similarity graph:

```
p⁰ = softmax(QKᵀ/√d)                    ← standard attention
p  = (1−α)·p⁰ + α·(p⁰P),  P = softmax(KKᵀ/√d)   ← QGFD
```

At α = 0 this is bit-for-bit softmax (verified: max |Δlogit| = 0.000e+00), so it is a
safe drop-in. The claim under test was that a fixed α > 0 buys robustness to input
noise.

**It does not.** Across 3 models × 3 seeds × 4 noise rates, QGFD's *absolute*
perplexity under noise was lower in **0 of 36 measurements**. Clean perplexity was
worse on 3 of 3 models and prefill was 1.3–1.8× slower.

The published-looking gap came from the metric. `robustness_gap_pct` is a difference
of *relative* degradations, `100·(noisy − clean)/clean`, computed per arm against its
own clean run. QGFD's clean perplexity is higher, so its denominator is larger, and
its Δ% shrinks even at identical accuracy. Decomposing it:

| Model | Reported gap | Denominator artefact | Real residual | Absolute noisy PPL (softmax − QGFD) |
|---|---|---|---|---|
| SmolLM2-135M | +12.60 ± 3.26 | **+30.76 ± 5.04** (244%) | **−18.16 ± 3.32** * | −5.17 ± 1.06 |
| Qwen2.5-0.5B | +8.18 ± 3.31 | **+15.87 ± 1.18** (194%) | **−7.70 ± 2.14** * | −1.79 ± 0.59 |
| TinyLlama-1.1B | −1.16 ± 0.83 | +7.57 ± 1.58 | **−8.73 ± 0.76** * | −1.20 ± 0.09 |

The artefact is larger than the whole reported gap. The residual — the only part that
means anything — is negative and significant everywhere.

The same arithmetic shows up independently in the fine-tuning run, with no QGFD
involved in the comparison: LoRA improved SmolLM2's clean perplexity 27.87 → 18.81
*and* its absolute noisy perplexity 451.9 → 397.0, while the reported "robustness"
number got **428 pp worse**. A strictly better model scored much worse on the
statistic. That is a property of the statistic, not of the model.

## 2. Why it failed, and what that implies about the fix

At α = 0, QGFD *is* softmax. So a fixed α > 0 is a constant perturbation of an already
converged optimum. The gradient at a converged point is ≈ 0, so the leading term of
the loss change is quadratic:

```
ΔL ≈ ½ δᵀHδ = O(α²) ≥ 0
```

for *every* direction δ. A perturbation applied to every head, at every position, on
every input cannot pay for itself — it can only be worth its cost where it is applied
selectively. Two consequences drive the whole redesign:

1. **The coefficient has to be learned and zero-initialised.** Then the model starts at
   the optimum it already found and pays only where it profits.
2. **The coefficient has to be per-head and per-position.** A precision-retrieval head
   should stay ordinary softmax; a head reading a corrupted span might not.

Revised hypothesis, and it is falsifiable as stated:

> *Some attention heads benefit from learned, selective diffusion when the input or
> context is unreliable, while precision-retrieval heads should remain ordinary
> softmax.*

## 3. The design

```
p  = (1 − g_{h,t})·p⁰ + g_{h,t}·(p⁰P)
g_{h,t} = g_max · tanh( a_h·Ĥ(p⁰)_t + b_h·margin(p⁰)_t + r_h ),   a = b = r = 0 at init
```

Four decisions worth defending, because each of them is a change from the obvious
version:

**tanh, not sigmoid.** A one-sided gate `g = g_max·σ(z) ∈ [0, g_max]` cannot be
initialised at exactly 0: `σ(0) = ½`, and `g = 0` is reachable only at a boundary
where the gradient is 0 and training never starts. `g_max·tanh(z)` is exactly 0 at
z = 0 with `tanh'(0) = 1`. It also makes **g < 0** reachable — i.e. sharpening — which
matters because the evidence so far is consistent with heads wanting to sharpen, not
smooth. Both properties are pinned by tests.

**Two features, not four.** Entropy and the top-2 probability margin both fall out of
the `p⁰` that is already materialised, so they cost O(n²) against the diffusion's
O(n³). Entropy is normalised by the causal prefix length — position *t* of a causal row
holds at most log(t+1) nats, so raw entropy is not comparable across positions.
Cross-layer disagreement needs plumbing across module boundaries and is deferred until
the free features earn it.

**ρ is dropped.** Damped diffusion `p̃ = (1−ρ)p⁰ + ρ(p⁰P)` composed with the gate gives
`(1 − gρ)p⁰ + gρ(p⁰P)`. ρ is *exactly* reparameterisable into g — a redundant
hyper-parameter and a redundant stage.

**No teacher, no KL term.** At g = 0 the student **is** the frozen softmax model,
exactly. A teacher would be a third forward pass and a second copy of the weights in
16 GB to enforce a constraint that the clean-CE term plus ε-constrained checkpoint
selection already enforce. Reinstate a cached top-k logit KL only if Stage 1 shows
clean drift that the L1 penalty cannot hold.

## 4. Measurement rules

Learned the hard way, and non-negotiable:

- **Primary metric: paired per-window cross-entropy in nats**, on byte-identical
  corrupted text. Never a relative delta of perplexities — PPL = exp(CE), so any ratio
  statistic re-imports exactly the denominator sensitivity that produced the phantom
  +12.6 pp.
- **Pair at the window level** (100–300 windows), not the seed level (n = 3). Seeds
  measure training variance; windows measure the effect. Same tokeniser, same bytes,
  same window index in both arms: the difference has no corpus-sampling variance in it.
- **Success = constrained Pareto improvement.** Clean CE regression ≤ ε *and* noisy CE
  strictly lower, with the paired CI excluding zero. ε = 0.003 nats ≈ 0.3% relative
  perplexity. A zero-initialised gate starts at ε = 0 by construction, so the question
  is only how well the L1 penalty holds it.
- **The falsifiers stay in.** Every trained gate is re-evaluated with `uniform` P and
  `shuffled` P (row values permuted within the causal prefix, holding each row's
  entropy, max and sparsity fixed and changing only *which* key gets the mass). If the
  gate still wins with a structure-free P, it learned "spread mass when uncertain" —
  an adaptive temperature schedule, which is free and folds into `W_Q`. That gets
  reported as such, not as graph routing.

## 5. Corruption families

Character noise re-tokenises words into subword fragments, so a result there is
confounded with fragmentation repair and may not even instantiate the key-perturbation
hypothesis. The families are therefore split, and the split is reported:

| Family | Tokenisation | What it tests |
|---|---|---|
| `word_drop` | preserved | information removed, surviving token ids intact |
| `word_swap` | preserved | order destroyed, token multiset intact |
| `token_replace` | preserved | misleading context, vocabulary unchanged |
| `repeat_token` | preserved | redundant keys — can attention discount them? |
| `char` | **broken** | the original headline family; confounded |
| `ocr` | **broken** | realistic confusables, still re-tokenises |

Training and evaluation default to `word_drop`. Curriculum: 2% → 8%.

## 6. The four stages

### Stage 0 — does the headroom exist? (~40 min, no training)

The point of Stage 0 is that a zero-initialised gate can only exploit heads where
diffusion *already* reduces noisy loss at the margin. That is directly measurable
without writing a gate at all.

| Probe | Method | Cost |
|---|---|---|
| **0a** per-head α-gradient | `learnable_alpha=True` gives a per-head `alpha_param`; one forward+backward at α ≈ 0 yields `∂CE_noisy/∂α_h` for **every head in every layer simultaneously** | minutes |
| **0b** per-layer α oracle | α = 0.05 on exactly one layer, 0 elsewhere; absolute noisy CE. Confirms 0a at the real α and catches second-order rescues | ~5 min |
| **0c** corruption screen | fixed α, absolute paired CE, one row per family | ~15 min |

Reading 0a:

- `∂CE_noisy/∂α_h < 0` → diffusion on head *h* lowers noisy loss at the margin. Real
  headroom; a gate can find it.
- `∂CE_noisy/∂α_h > 0` → the optimiser wants α **negative** on that head: it wants to
  *sharpen*. That inverts the mechanism claim, and sharpening is a temperature change
  that folds into `W_Q` — so the E3 temperature control would be the winner, not the
  null. Cheap to find, and publishable either way.

**The stop rule, fixed before the numbers are seen** (a stop rule that gets overridden
is decoration):

| Outcome | Decision |
|---|---|
| no head with a negative noisy gradient, no layer helps, no family wins | **stop** — publish the negative result plus the metric critique. Nothing downstream can rescue this: a zero-init gate cannot reach states a fixed α did not already probe. |
| gradients uniformly positive | **pivot** to the sharpening/temperature story |
| some heads or some family show headroom | **proceed**, with the Stage 1 gate restricted to the layers 0b flagged |

### Stage 1 — gate only (~1.5 h)

Base model entirely frozen; the only trainable parameters are 3 per head (≈1k total).
Smallest model, 3 seeds.

```
L = CE_clean + λ_n(t)·CE_noisy + γ·mean|g|
```

λ_n ramps from 0 over the first 30% of steps and the corruption rate follows the
curriculum, so the gate first learns not to break clean text and only then learns when
to fire. The L1 term on |g| is the "stay at softmax unless you earn it" pressure, and
it is symmetric, so it penalises sharpening and smoothing equally.

There is nothing here that can overfit in an interesting way — which is the point. If
the gate cannot win with the base weights untouched, then a win after LoRA is a win by
the adapters, not by diffusion.

The softmax reference is the *same weights with the gate switched off*, so the
comparison is exactly paired: one model, one set of windows, one difference.

### Stage 2 — trust-aware sparse graph + Q/K LoRA (~3 h)

Only if Stage 1 clears ε **and** beats the shuffled-P control.

```
P_ij = softmax_j( k̂_i·k̂_j/τ + s·[i=j] + λ·(−z_j) ),   restricted to top-k in the causal prefix
```

- **top-k (k=8)** is not optional — it takes `p⁰P` from O(n³) to O(n²k), which is what
  makes this stage affordable at all.
- **Causal mask and top-k are applied to the logits, before normalisation**, so every
  row is a proper distribution over exactly the surviving edges.
- **Reliability `log r_j` is a free signal, not a learned sub-model.** `z_j` is the
  standardised key norm; unusually long keys dominate similarity scores for reasons
  unrelated to content, so `−z_j` is a reasonable prior. λ is one scalar per layer,
  initialised to 0, so the term starts inert.
- **`agreement_ij` is deferred** — it needs the previous layer's attention inside this
  layer's operator, the most expensive feature in the design and the least justified
  before the cheap ones show something.
- **LoRA on q_proj/k_proj only.** Not q/k/v/o: if the hypothesis is about the key
  graph, V and O adapters let the model win for unrelated reasons and the attribution
  is gone.

Once LoRA moves the weights, "gate off" is no longer the right control, so Stage 2
trains two arms — LoRA-only vs gate+trust+LoRA — on identical seed, data, steps and
LR, and pairs them at the window level.

### Stage 3 — recipe transfer (~2 h)

Qwen2.5-0.5B and TinyLlama-1.1B, one seed, hyper-parameters transferred as-is.
Deliberately un-tuned: if the effect is a mechanism it survives a transfer; if it needs
refitting per checkpoint it is a fit, and it gets reported as one.

## 7. Why this fits on a free T4

Four levers, in order of payoff:

| Lever | Why | Gain |
|---|---|---|
| **fp16, not bf16** | the run log printed `bf16 tensor cores (sm>=80): False → EMULATED bf16 on pre-Ampere hardware`. Turing has fp16 tensor cores and no bf16 ones. `p⁰`, `P` and the loss stay fp32, which is where precision actually matters | ~2× |
| **both T4s** | Kaggle gives 2 devices; the run resolved everything to `cuda:0`. Arms are independent by construction — one per device | 2× |
| **seq_len 128** for method development | eager attention is O(n²) materialised and the diffusion product O(n³); we pay both, because the operator must see the probability matrix, so no FlashAttention | ~2–2.5× |
| **top-k sparse P** | needed for Stage 2 anyway | large |

Together these turn the measured 300-step SmolLM2 run from ~33 min into ~4–6 min. The
corollary is worth stating in the paper: the earlier 9-hour burn was seq-256 dense eager
attention in emulated bf16, not the diffusion — QGFD itself added only ~15% per step
(6.0 → 6.9 s).

**Session plan:**

| Block | Time | Gate to proceed |
|---|---|---|
| setup, dtype check, verify α=0 exactness, verify patch | 30 min | — |
| Stage 0a + 0b + 0c | 40 min | headroom exists / pivot / stop |
| Stage 1, 3 seeds + shuffled-P control | 1.5 h | ε cleared **and** beats shuffled P |
| Stage 2, 2 arms | 3 h | absolute noisy CE win, paired CI excludes 0 |
| Stage 3, 2 models × 1 seed | 2 h | — |

Under 8 h with margin, against ~30–40 GPU-hours for the six-variant × three-seed ×
teacher-forward version of the same idea. If Stage 0 says stop, the cost was 70 minutes.

## 8. The proposal's six-variant ablation, collapsed

| Original variant | What actually happens |
|---|---|
| A softmax baseline | already measured — do not rerun |
| B fixed global α | already measured (this is the 0/36) — do not rerun |
| C learned per-head α_h | **Stage 0a**, and it is free rather than a training run: `alpha_param` already exists and is already per-head |
| D learned gate | **Stage 1** — the main run |
| E + clean KL | cut; replaced by g = 0 init + ε-constrained selection |
| F + trust-aware sparse graph | **Stage 2** — not a separate variant, since the sparsity is a speed requirement |

Six training variants become two, and the two cheapest rows become eval-only.

`α_max ∈ {0.02, 0.05, 0.10}`, `diffusion_steps = 1`, `top-k ∈ {4, 8, 16}`. No T > 1
until one-step adaptive diffusion beats the baseline — at α = 0.05 the T = 4 term
carries 6e-6 of the mass, so a flat depth curve is the expected outcome, not a finding.

## 9. Diffusion-utilisation table

Mostly free — it all comes out of the same forward pass:

- mean signed gate by layer and head
- gate on clean vs on corrupted input
- fraction of heads with |g| ≈ 0 (the "stayed softmax" count — the interesting number
  if the hypothesis is right)
- **gate AUC for discriminating clean from corrupted input.** If this is high, what was
  built is a noise detector plus a smoother, and it should be described that way rather
  than as graph routing.
- clean and noisy paired ΔCE, latency, peak memory

## 10. If it still fails

Then the honest conclusion is the one the data already supports: key-similarity
diffusion is not the mechanism for this noise family, and the paper is the negative
result plus the metric critique. That critique is the most transferable thing in the
project — it applies to any robustness paper that reports relative degradation, and the
LoRA run demonstrates it without QGFD in the comparison at all.

Contributions 1 (α = 0 exactness) and 3 (the single-GPU LoRA recipe) stand either way.

## 11. Running it

```bash
# CPU rehearsal of every code path, a few minutes, no GPU (run this first)
python -m scripts.gated_experiments --stages 0,1,2 --quick --out_dir /tmp/gq

# the real screen
python -m scripts.gated_experiments --stages 0 --model_id HuggingFaceTB/SmolLM2-135M

# stages, obeying Stage 0's stop rule (add --force to override, and say so if you do)
python -m scripts.gated_experiments --stages 0,1,2,3 --seeds 0,1,2
```

Or open `QGFD_Gated_Experiments.ipynb` on Kaggle, set the accelerator to GPU T4 ×2, and
run top to bottom. Each stage writes its own JSON to `out_dir`, so the run can be
stopped after any stage and still be coherent.

Related: [interpreting-results.md](interpreting-results.md) for what a number licenses,
[paper-experiments.md](paper-experiments.md) for the original fixed-α tracks.
