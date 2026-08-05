# 🔬 QGFD: Diffusion-Regularized Attention Replacement

*A universal, model-agnostic way to inject Query Graph Flow Diffusion into any PyTorch / HuggingFace transformer.*

---

## 📘 Overview

This repository provides a **drop-in, training-free robustness and regularization mechanism** for transformer attention using:

- **`MultiHeadQGFDLayer`** → a diffusion-regularized multi-head attention implementation  
- **`SafeWrappedAttention`** → a universal wrapper that replaces existing attention modules while preserving behavior  
- **`wrap_model_with_qgfd(...)`** → recursively rewrites a full model in-place, wrapping all attention layers  

QGFD can be used as:

- A **research tool** for experimenting with diffusion-regularized attention, and
- A **production-ish inference upgrade** to improve robustness to noisy inputs (OCR/ASR errors, typos, etc.) *without retraining or modifying weights*.

---

## ✨ Key Ideas

### What QGFD Does (Intuitively)

Standard attention:

1. Computes `scores = QKᵀ / √d`
2. Applies softmax: `p₀ = softmax(scores)`
3. Uses `p₀` to aggregate values: `output = p₀ V`

QGFD adds **diffusion** over the attention distribution:

```
p_{t+1} = (1 - α) p_0 + α (p_t P)
```

- `p₀` – baseline attention
- `P` – transition matrix over keys (either global key-similarity or local conv)
- `α` – diffusion strength
- `t` – number of diffusion steps

This makes attention less brittle: mass can flow to nearby/similar keys, which improves robustness when inputs are noisy or slightly corrupted.

---

## 🧠 `MultiHeadQGFDLayer`: Core API

Defined in `qgfd_attention.py` as:

```python
class MultiHeadQGFDLayer(nn.Module):
    ...
```

### Modes

**`mode="full"`**

Builds a full key-similarity transition matrix P ∈ ℝ^{B×H×Lk×Lk} and performs global diffusion.

- **Pros:** expressive, strong robustness gains on some models
- **Cons:** O(L²) memory in the key dimension

**`mode="conv"`**

Uses a local CausalConv1D over keys to approximate diffusion:

- **Pros:** O(L·K), much cheaper, more production-friendly
- **Cons:** less expressive, more model-dependent

### Optional Safety / Production Knobs

QGFD can be:

- Disabled entirely (`enable_qgfd=False`) → behaves like standard attention
- Limited to short sequences in full mode via `max_full_seq_len` + `full_fallback_mode`
- Clamped in strength via `max_alpha` to avoid overly aggressive diffusion

### Constructor Arguments

```python
MultiHeadQGFDLayer(
    embed_dim: int,
    num_heads: int = 8,
    proj_dim: int | None = None,
    diffusion_steps: int = 4,
    target_alpha: float = 0.02,
    warmup_steps: int = 20000,
    use_bias: bool = True,
    early_stop_eps: float = 1e-5,
    detach_P: bool = False,
    temp: float = 1.0,
    mode: str = "full",
    kernel_size: int = 5,
    enable_qgfd: bool = True,
    max_alpha: float = 0.10,
    max_full_seq_len: int = 512,
    full_fallback_mode: str = "disable",
    mask_threshold: float = -1e4,
    debug: bool = False,
    **kwargs,
)
```

#### Core attention / projection args

- **`embed_dim`** – input embedding dimension
- **`num_heads`** – number of attention heads
- **`proj_dim`** – projection dimension for Q/K/V (defaults to embed_dim)
- **`use_bias`** – whether to use bias in projection layers

#### Diffusion configuration

- **`diffusion_steps`** – number of diffusion iterations (≥ 0)
- **`target_alpha`** – desired diffusion mixing strength α
- **`warmup_steps`** – linearly ramp α from 0 → target_alpha over this many steps

During forward, the effective alpha is:

```python
alpha_eff = get_alpha()  # warmup + clamp
```

#### Mode & structure

- **`mode`**: `"full"` or `"conv"`
  - `"full"` → builds P using cosine similarity of keys
  - `"conv"` → uses 1D conv kernel to diffuse over keys
- **`kernel_size`** (conv mode) – odd kernel size (e.g., 3, 5, 7)

#### Safety & production controls

- **`enable_qgfd`** – `False` → skips diffusion completely (acts like vanilla attention)
- **`max_alpha`** – Clamp `alpha_eff` to `[-max_alpha, max_alpha]` after warmup
- **`max_full_seq_len`** – Maximum allowed key length Lk for `mode="full"`; if exceeded, behavior depends on `full_fallback_mode`
- **`full_fallback_mode`**:
  - `"disable"` → if Lk > max_full_seq_len, QGFD is skipped, p = p0
  - `"conv"` → if Lk > max_full_seq_len, fall back to conv diffusion
- **`mask_threshold`** – For additive masks: values ≤ mask_threshold are treated as "masked out" when building key masks in conv mode
- **`debug`** – If True, prints lightweight diagnostic messages (e.g., when falling back from full → conv)

#### Numerical stability / masking

- **`early_stop_eps`** – stop iterating when max change in p is below this threshold
- **`detach_P`** – whether to detach the transition matrix P from autograd (for full mode)
- **`temp`** – scaling temperature before softmax when building P in full mode
- **`_eps`** – dtype-aware epsilon (1e-6 for fp32, 1e-3 for fp16/bf16)

---

## 🔄 Forward Signature

```python
def forward(
    self,
    hidden_states: torch.Tensor,       # (B, Lq, D)
    kv: torch.Tensor | None = None,    # (B, Lk, D) or None for self-attn
    attention_mask: torch.Tensor | None = None,
    head_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
    **kwargs,
):
    ...
```

- **`hidden_states`** – query input (B, Lq, D)
- **`kv`** – optional key/value input (B, Lk, D); if None, self-attention
- **`attention_mask`** – HF-style mask (bool or additive; supports shapes (B,Lk) or (B,1,1,Lk))
- **`head_mask`** – per-head scaling mask
- **`output_attentions`** – if True, returns `(attn_output, p)` where p is attention probs

---

## 🔁 How Diffusion Works Internally

1. **Compute Q/K/V and base attention distribution:**

```python
scores = Q Kᵀ / √(head_dim)
scores = apply_attention_mask(scores, attention_mask)
p0 = softmax(scores, dim=-1)
```

2. **Decide if QGFD is active:**

```python
alpha_eff = get_alpha()
qgfd_active = enable_qgfd and diffusion_steps > 0 and abs(alpha_eff) > 0
```

- If not active → p = p0

3. **If active:**
   - If `mode == "full"` and sequence length Lk ≤ `max_full_seq_len`:
     - Build transition matrix P from keys
     - Do diffusion using `p @ P`
   - If `mode == "full"` and Lk > `max_full_seq_len`:
     - Fall back according to `full_fallback_mode`
   - If `mode == "conv"`:
     - Use conv-based local diffusion with causal padding and mask-aware renormalization

4. **Compute final output:**

```python
attn_output = (p V) projected back to embed_dim
```

---

## 🧱 Universal Wrapping: SafeWrappedAttention & wrap_model_with_qgfd

`universal_qgfd_replacer.py` provides:

- **`SafeWrappedAttention`** – wraps an existing attention module:
  - stores original as `._orig`
  - creates a `MultiHeadQGFDLayer` as `.qgfd`
  - copies attributes and transfers weights
  - preserves caching & HF-compatible forward signature

- **`wrap_model_with_qgfd(model, MultiHeadQGFDLayer, ...)`** – walks the module tree, finds leaf attentions, and replaces them with `SafeWrappedAttention` instances.

### Example:

```python
from qgfd_attention import MultiHeadQGFDLayer
from universal_qgfd_replacer import wrap_model_with_qgfd
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

model = wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    diffusion_steps=1,
    target_alpha=0.01,
    warmup_steps=0,
    kernel_size=5,
    early_stop_eps=0.0,
    mode="full",                # or "conv"
    enable_qgfd=True,
    max_alpha=0.05,
    max_full_seq_len=512,
    full_fallback_mode="conv",  # fallback to conv for long seq
    debug=False,
    verbose=True,
)
```

After wrapping, you use model exactly like before.

---

## 🚀 Usage Patterns

### 🔬 Research Mode (maximal flexibility)

Use full diffusion, no hard limits, enable gradient flow through P, use larger `diffusion_steps`, higher alpha:

```python
model = wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    diffusion_steps=2,
    target_alpha=0.05,
    warmup_steps=0,
    mode="full",
    enable_qgfd=True,
    max_alpha=0.10,
    max_full_seq_len=4096,      # effectively no full-mode cutoff
    full_fallback_mode="disable",
    detach_P=False,             # propagate gradients through P
    debug=True,
)
```

### 🏢 Production-ish Mode (safe & bounded)

Use small α, fewer steps, clamp α, limit full mode to short sequences, fall back to conv or no diffusion for long sequences:

```python
model = wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    diffusion_steps=1,
    target_alpha=0.01,
    warmup_steps=0,
    mode="full",                # use full when seq is short
    enable_qgfd=True,
    max_alpha=0.02,
    max_full_seq_len=512,
    full_fallback_mode="conv",  # use conv when seq > 512
    kernel_size=5,
    detach_P=True,              # safer for inference-only
    debug=False,
)
```

### 📴 Turn QGFD Off (for A/B testing)

```python
model = wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    enable_qgfd=False,      # qgfd layer present but inactive
)
```

---

## 📊 Empirical Benchmarks & Correctness Verification

Run the full correctness, numerical stability, autograd, microbenchmark, and sequence scaling suite via:

```bash
python run_benchmarks_and_validation.py
```

See [DOCUMENTATION.md](DOCUMENTATION.md#9-empirical-validation--microbenchmark-suite-results) and [EXPERIMENTATION_DOCS.md](EXPERIMENTATION_DOCS.md#10-empirical-microbenchmarks--scaling-suite-results) for full detailed theoretical proofs, empirical tables, and memory/latency scaling curves.

---

## 📦 Installation

1. **Clone:**

```bash
git clone https://github.com/rajboopathiking/TorchDire.git
cd TorchDire
```

2. **Install dependencies:**

```bash
pip install torch transformers
```

(Transformers only required for HuggingFace-based tests.)

---

## 📜 License

MIT License — free for commercial and research use.

---

## 🤝 Contributing

PRs and issues are welcome for:

- Optimized kernels (Triton/CUDA) for diffusion
- Support for more HF architectures
- Additional sanity tests and benchmarks
- New diffusion schemes (e.g., learned kernels, adaptive per-head α)
