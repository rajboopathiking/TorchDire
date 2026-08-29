# How QGFD is wired into a model

Two independent integration paths exist. They have **different defaults and
different failure modes**, and mixing up which one you are using is the most common
source of confusion in this repo.

---

## The math

```
P    = softmax(KKᵀ/√d)              row-stochastic key-similarity transition matrix
p⁰   = softmax(QKᵀ/√d)              the ordinary attention distribution
pᵗ⁺¹ = (1−α)·p⁰ + α·(pᵗ P)          one diffusion step, repeated `diffusion_steps` times
```

`p⁰` is re-injected every step, so this is a lazy random walk anchored to the
original distribution rather than a free-running one. At `α = 0` the update is the
identity: QGFD **is** softmax, bit for bit (verified, `max |Δlogit| = 0.000e+00`).

---

## Path A — the operator path (use this)

```python
from torchdire import QGFDOperator, wrap_model_with_qgfd_operator

op = QGFDOperator(diffusion_steps=1, target_alpha=0.05,
                  mode="full", detach_P=True, is_causal=True)
model = wrap_model_with_qgfd_operator(model, op, verbose=False)
```

`wrap_model_with_qgfd_operator` replaces each attention module with an
`AttentionOperatorAdapter` subclass — `LlamaAttentionAdapter`, or its `Qwen2` /
`Mistral` subclasses. The adapter reimplements attention with the probability step
factored out into a swappable operator object, so both arms of every experiment run
through **the same adapter code**; only the operator differs. That is what makes the
A/B a clean comparison rather than two separate implementations.

Helpers: `collect_qgfd_operators(model)` for introspection,
`register_qgfd_operator_step_callback(trainer, model)` for α warmup.

## Path B — the legacy kernel path

```python
from torchdire.nn.llama_qgfd import patch_llama_with_qgfd
model = patch_llama_with_qgfd(model, diffusion_steps=1, target_alpha=0.05)
```

Patches `QGFDKernel` into Llama attention in place. Helpers:
`collect_qgfd_kernels`, `register_qgfd_step_callback`, `unfreeze_qgfd_alpha`
(needed to re-enable a learnable α after PEFT freezes parameters),
`dump_learned_alphas`.

`scripts/finetune_qgfd.py --backend {operator,kernel}` selects between them.

### The defaults differ — do not copy one path's config into the other

| Parameter | `QGFDOperator` (path A) | `QGFDKernel` (path B) |
| --- | --- | --- |
| `diffusion_steps` | 1 | **4** |
| `target_alpha` | 0.05 | **0.02** |
| `warmup_steps` | 0 | **20000** |
| `detach_P` | **True** | False |
| `is_causal` | False | auto-detected |

Two of these are traps:

* **`QGFDKernel(warmup_steps=20000)`** with a 300-step fine-tune means α ramps to
  `300/20000 = 1.5%` of target — QGFD is effectively off for the entire run, and
  nothing errors. `report_alpha()` in `scripts/finetune_qgfd.py` exists to catch this.
* **`QGFDOperator(is_causal=False)`** on a causal LM is a correctness bug, below.

`scripts/review_experiments.py:build_operator()` forces `is_causal=True` and threads
`detach_P` from config, so anything driven through the harness is safe. Hand-written
code is not.

---

## `is_causal=True` is mandatory on a causal LM

With `is_causal=False`, `P` is unmasked, so `p⁰ @ P` diffuses probability mass onto
**future** keys. Measured consequence: editing a future token changed logits at
*earlier* positions by ~4e-3. That is a causality violation, and it deflates
teacher-forced perplexity — QGFD would look better than it is, for the worst
possible reason.

With `is_causal=True` the leakage is exactly `0.0`, matching the base model.

---

## `detach_P`: is the graph structure learned?

`detach_P=True` (the operator default) stops gradient through `P`, treating the
key-similarity graph as a **fixed routing structure** that the walk moves over.
`detach_P=False` lets the model learn the graph itself.

This was a config field only recently: `ExperimentConfig` had no `detach_P` and
`build_operator` hard-coded `True`, so an ablation that varied it would have produced
eight rows in which four pairs were identical **by construction** — and nothing in
the output would have revealed it. The field now exists and is threaded through
(`scripts/review_experiments.py:158`).

---

## The dead-LoRA bug

**Any operator-path LoRA fine-tune written before this fix trained nothing.** If you
have results from one, discard them.

`AttentionOperatorAdapter.__init__` aliases the original module's projections onto
itself, so `self.q_proj` and `self.original_attention.q_proj` were the *same*
`nn.Linear` reachable under two names. `nn.Module.named_modules()` **de-duplicates
shared submodules**, yielding only the first name it encounters — which is the
`original_attention.*` one, registered first in the base `__init__`.

PEFT builds its target list from `named_modules()` and rebinds by `setattr` on the
parent. So LoRA landed on `original_attention.q_proj`: a module the adapter's
`forward()` never calls. The live projection stayed a frozen `nn.Linear`, the
adapters were dead weight, and with nothing else trainable, backward raised
`element 0 of tensors does not require grad`.

**The fix** (`_detach_original_from_tree`, `torchdire/nn/attention_adapters.py:264`)
unregisters `original_attention` as a submodule once its projections are aliased.
The object stays intact and reachable as a plain attribute for attribute delegation,
so callers holding a reference are unaffected, and the stock checkpoint key layout
(`self_attn.q_proj.weight`) is restored as a side benefit.

It is deliberately conservative: if the architecture has extra parametrised
submodules that are *not* aliased onto the adapter (e.g. `q_norm` / `k_norm`),
dropping the original from the tree would silently exclude them from `state_dict()`
and `.to()`. In that case it keeps the original registered and **warns that this
architecture must not be fine-tuned through the operator path** — a loud
non-guarantee instead of a quiet wrong answer.

**The guard.** `verify_lora_live()` (`scripts/finetune_qgfd.py:237`) runs a probe
forward+backward and refuses to report a result unless a `lora_B` tensor actually
receives gradient. It prints:

```
LoRA live: N trainable tensors, M with non-zero grad on the probe
```

`M ≈ N/2` at initialisation is **correct**, not a warning sign — `lora_B` is
initialised to zero, so `lora_A` receives no gradient on the first probe.
`M = 0` raises. Regression coverage: `tests/test_operator_lora_liveness.py`.

---

## α warmup must be driven from outside `forward()`

α ramps linearly `0 → target_alpha` over `warmup_steps`. The step counter must be
advanced by a `TrainerCallback` calling `set_step(global_step)`:

```python
from torchdire.nn.attention_operators import register_qgfd_operator_step_callback
register_qgfd_operator_step_callback(trainer, model)   # operator path
```

Incrementing a counter inside `forward()` instead **diverges under gradient
checkpointing**, because recompute calls `forward()` a second time for the same
optimizer step — the counter runs ahead, the forward and recomputed backward see
different α values, and the run fails with `CheckpointError`.

`get_alpha()` behaves differently by mode: in **eval** mode it ignores warmup and
returns the full `target_alpha`; in **train** mode it scales by
`min(1, step_count / warmup_steps)`. So a model that never warmed up during training
will still evaluate at full strength — which is exactly why `report_alpha()` checks
`step_count ≥ warmup_steps` rather than trusting the eval-time value.

---

## Supported architectures, and why unsupported ones fail loudly

Genuinely wired: **Llama, Qwen2, Mistral**. GPT-2 and OPT adapters are `pass` stubs
whose patched forward raises `TypeError`; GPT-Neo patches **zero layers**, which
would mean QGFD silently has no effect at all.

`verify_patch()` (`scripts/review_experiments.py:175`) raises if zero layers were
patched, if any layer fell back to `GenericAttentionAdapter`, or if the operator is
never invoked during a probe forward pass. A silent no-op therefore cannot be
reported as a QGFD result. A green run prints, per arm:

```
[qgfd] patch verified: 22 x LlamaAttentionAdapter, operator invoked 22x
```

> The README has previously claimed drop-in support for OPT and GPT-2. That claim is
> not currently true — either implement those adapters or drop it.

---

## Why QGFD cannot use FlashAttention

The diffusion step needs `p` as an explicit matrix. Fused attention kernels never
materialise it — that is precisely how they save memory. So QGFD is architecturally
incompatible with SDPA/FlashAttention, and the honest baseline for any QGFD timing is
**eager materialised softmax**. See
[interpreting-results.md](interpreting-results.md#latency-is-measured-against-eager-softmax-not-flashattention).

---

## See also

* [../QGFD_CONFIG_REFERENCE.md](../QGFD_CONFIG_REFERENCE.md) — per-parameter ranges
* [paper-experiments.md](paper-experiments.md) — running the tracks
* [../CAUSAL_MASKING_FIX.md](../CAUSAL_MASKING_FIX.md) — the causal-masking fix in detail



