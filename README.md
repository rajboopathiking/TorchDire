# 🔬 QGFD: Diffusion-Regularized Attention Replacement

This repository provides a **universal, model-agnostic wrapper** that injects a
**QGFD (Quasi-Gaussian Feature Diffusion)** attention mechanism into *any*
PyTorch / HuggingFace transformer model — without modifying model internals.

The system includes:

- **`MultiHeadQGFDLayer`** — a drop-in replacement for multi-head attention  
- **`SafeWrappedAttention`** — a universal wrapper that replaces existing attention modules  
- **`wrap_model_with_qgfd(...)`** — recursively rewrites a full model in-place  
- **`QGFD_Sanity_Checks.py`** — automated shape, gradient, and wrapping tests

This library is intended for research experiments in stabilizing attention,
regularizing attention distributions via diffusion, and replacing dot-product
attention with a more structured update rule.

---

## ✨ Features

### ✔ Universal Attention Replacement
The wrapper detects any leaf module whose class name contains `"Attention"` and that
has parameters typical of Q/K/V projections. It then replaces it with
`SafeWrappedAttention`, preserving:

- original module behavior  
- caching (`present_key_value`)  
- attention masks  
- `output_attentions=True` compatibility  

### ✔ QGFD Diffusion Layer
`MultiHeadQGFDLayer` performs:

1. standard Q/K/V projection  
2. baseline softmax attention  
3. **repeated diffusion steps**:  
   \[
     p_{t+1} = (1 - \alpha)p_0 + \alpha(p_t P)
   \]  
   where `P` is a normalized *key-similarity transition matrix*

4. final value projection

Includes:
- cosine-sim transition matrix  
- temperature scaling  
- warmup schedule for α  
- early stopping  
- gradient-enabled diffusion (`detach_P=False` by default)  

### ✔ Safety and Stability
The wrapper includes:
- robust attribute copying  
- weight transfer from original attention  
- verification step ensuring proper `.qgfd` and `._orig` attachment  
- fallback heuristics for ambiguous architectures  

### ✔ Fully Tested
Run:

```bash
python QGFD_Sanity_Checks.py
```

📦 Installation
Clone:

```bash
git clone https://github.com/rajboopathiking/TorchDire.git
cd TorchDire
```
Install dependencies:

```bash
pip install torch transformers
```
(Transformers optional unless running HF smoke test)

🚀 Usage
1. Import components
```python
from qgfd_attention import MultiHeadQGFDLayer
from universal_qgfd_replacer import wrap_model_with_qgfd
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
# Wrap all its attention layers
model = wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    diffusion_steps=4,
    target_alpha=0.02,
    warmup_steps=20000,
    detach_P=False,
    temp=1.0,
    verbose=True
)
# Run inference as usual
outputs = model(input_ids)

```
