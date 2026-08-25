# QGFD Config Reference (min / max / recommended)

All parameters go through `wrap_model_with_qgfd(model, **config)` (or `patch_llama_with_qgfd`).
Validated default config used by the sanity harness:

```python
DEFAULT_QGFD_CONFIG = {
    "diffusion_steps": 1,        # proven best; 2 is a fine test, 3+ degrades loss
    "target_alpha": 0.05,        # validated by QGFD_Sanity_Checks
    "warmup_steps": 0,           # full strength from step 1 (set >0 only with a trainer callback)
    "early_stop_eps": 0.0,       # no early stop
    "max_full_seq_len": 512,     # QGFD stays active at long context via conv fallback
    "full_fallback_mode": "conv",# linear-cost local diffusion instead of disabling
}
```

| Parameter | Type | Min | Max (practical) | Recommended | Notes |
|---|---|---|---|---|---|
| `diffusion_steps` | int | 1 | 8 (cost grows linearly) | 1–2 | Diffusion rounds. Sanity data: 1 = best loss; 2 testable; 3+ worsens eval loss vs softmax. |
| `target_alpha` | float | >0 | <1 (practical 0.20) | 0.05 | Blend weight p = (1−α)·p0 + α·(p0·P). 0.01 = barely different from softmax; 0.10+ = strong smoothing. |
| `warmup_steps` | int | 0 | ≤ 30% of total optimizer steps | 0 (or ~20–30) | α ramps linearly 0→target over these steps. 0 = immediate. **Requires `register_qgfd_step_callback` if >0 during training**; too-large values silently disable QGFD for the run. |
| `early_stop_eps` | float | 0.0 | 1e-3 | 0.0 | Early-stop threshold on diffusion convergence. 0 = always run all steps. |
| `kernel_size` | int (odd) | 1 | 21 (odd only) | 5 | Conv smoothing window (mode="conv" / fallback). Asserted odd ≥ 1. |
| `max_full_seq_len` | int | 1 | context length | 512 | Above this, full diffusion switches to fallback mode. |
| `full_fallback_mode` | str | — | — | `"conv"` | `"conv"` = linear-cost local diffusion; `"disable"` = plain softmax beyond limit. |
| `mode` | str | — | — | `"full"` (kernel default) | `"full"` or `"conv"` (asserted). Replacer default passes `"conv"` only via fallback. |
| `max_alpha` | float | >0 | <1 | 0.10 | Cap for `learnable_alpha`. |
| `learnable_alpha` | bool | — | — | False | Per-head learned α; requires `num_heads`; adds an optimizer parameter. |
| `num_heads` | int | 1 | model heads | auto | Needed only when `learnable_alpha=True`. |
| `temp` | float | >0 (≤0 → forced 1.0) | 5.0 | 1.0 | Diffusion temperature. |
| `mask_threshold` | float | — | — | -1e4 | Scores below this are treated as masked in the valid mask. |
| `detach_P` | bool | — | — | False | Stop gradient through the transition matrix P. |
| `enable_qgfd` | bool | — | — | True | False = kernel returns plain softmax (handy as a no-op ablation). |
| `is_causal` | bool | — | — | auto-detected | Causal self-attention flag for the kernel. |

## Safe / sensible combos

- **Production default (validated):** steps=1, α=0.05, warmup=0, fallback="conv"
- **Stronger smoothing test:** steps=2, α=0.05–0.10, warmup=0 (or 20–30 with callback)
- **Training warmup recipe:** warmup ≤ 30% of max_steps + `register_qgfd_step_callback(trainer, model)`
- **Long context:** keep `max_full_seq_len=512` + `full_fallback_mode="conv"` (never "disable" — it silently reverts to softmax)