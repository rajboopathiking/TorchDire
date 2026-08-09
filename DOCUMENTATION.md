# 📚 QGFD & TorchDire: Complete R&D Technical Documentation

> **Query–Graph Flow Diffusion (QGFD): A Diffusion-Regularized Attention Mechanism**  
> **Repository:** [TorchDire](https://github.com/rajboopathiking/TorchDire.git)  
> **Version:** 1.0.0  
> **Author:** Raj Boopathi

---

## 📋 Table of Contents
1. [Overview & Architectural Vision](#1-overview--architectural-vision)
2. [Mathematical Foundations & Theorem Proofs](#2-mathematical-foundations--theorem-proofs)
3. [Step-by-Step Numerical Walkthrough](#3-step-by-step-numerical-walkthrough)
4. [TorchDire Module Reference](#4-torchdire-module-reference)
5. [Universal Model Wrapping & Integration](#5-universal-model-wrapping--integration)
6. [Benchmarking, Profiling & Ablation Engine](#6-benchmarking-profiling--ablation-engine)
7. [Google Colab & Cloud Setup Guide](#7-google-colab--cloud-setup-guide)
8. [IEEE Publication & Benchmark Guidelines](#8-ieee-publication--benchmark-guidelines)
9. [Empirical Validation & Microbenchmark Suite Results](#9-empirical-validation--microbenchmark-suite-results)

---

## 1. Overview & Architectural Vision

Standard Transformer attention evaluates pairwise query-key interactions:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### Limitations of Vanilla Attention:
- **Single-Hop Bottleneck:** Cannot aggregate context across multi-hop semantic key paths within a single layer.
- **Brittleness to Noise:** OCR/ASR transcription errors or minor typos distort attention scores without neighbor smoothing.

### How QGFD Solves This:
QGFD treats attention refinement as an iterative graph diffusion process over a key-similarity transition matrix $P$:
$$P = \text{softmax}\left(\frac{KK^\top}{\sqrt{d_k} \cdot \tau}\right)$$
$$p^{(t+1)} = (1 - \alpha) p^{(0)} + \alpha (p^{(t)} P)$$
$$h^{(T)} = p^{(T)} V$$

- $p^{(0)}$: Baseline softmax attention logits.
- $P$: Row-stochastic key transition matrix over valid keys in the KV cache.
- $\alpha$: Diffusion mixing strength ($\alpha \in [0.01, 0.05]$).
- $T$: Number of diffusion steps ($T \in [1, 4]$).

### Autoregressive KV-Cache & Key Transition Matrix Design ($P$):
In causal language model decoding (incremental $q\_len = 1$ with KV-cache of length $L_k$), all keys $\{0, 1, \dots, L_k - 1\}$ currently present in the cache represent valid past tokens for the current query step $q = L_k - 1$.

- **Pitfall of Lower-Triangular Masking on $P$:** Applying a lower-triangular causal mask to $P$ ($P_{n, m} = 0$ for $m > n$) causes extreme column sum asymmetry:
  $$\sum_{n=0}^{L_k-1} P_{n, 0} \approx \ln(L_k) \gg 1, \quad P_{L_k-1, L_k-1} \approx \frac{1}{L_k} \ll 1$$
  This creates an artificial **attention probability sink** at Position 0, which constantly drains attention weight away from recent tokens and funnels it into Position 0 during incremental decoding (causing output collapse into repeating tokens like `이이이...`).
- **Balanced Stochastic Formulation:** Since all cached keys $m \le q$ belong to the causal past of query $q$, key transitions among cached keys $\{0, \dots, L_k-1\}$ are unconstrained by future tokens. Constructing $P = \text{softmax}\left(\frac{KK^\top}{\sqrt{d_k} \tau}\right)$ over all cached keys yields a balanced row-stochastic matrix with column sums $\approx 1.0$, completely eliminating probability sinks while preserving $O(1)$ per-step incremental decoding stability.

---

## 2. Mathematical Foundations & Theorem Proofs

```
+-----------------------------------------------------------------------------------------+
|                                    QGFD THEOREMS                                        |
+-----------------------------------------------------------------------------------------+
| Theorem 1: Equivalence to Softmax Attention  ==>  alpha=0 => p^(1) = p^(0)              |
| Theorem 2: Geometric Convergence          ==>  ||p^(T) - p^(inf)|| <= alpha^T ||err_0||   |
| Theorem 3: Multi-Hop Expansion             ==>  p^(T) = (1-a) sum a^k p^(0) P^k + a^T ...|
| Theorem 4: Dense Attention Approximation   ==>  ||p^(T) - p_full|| <= C |lambda_2(P)|^T   |
| Theorem 5: Oversmoothing Bounds            ==>  ||p^(T) - pi|| <= C * gamma^T             |
+-----------------------------------------------------------------------------------------+
```

### Theorem 1: Equivalence to Softmax Attention
When $\alpha = 0$, QGFD reduces exactly to standard dot-product attention:
$$p^{(1)} = (1 - 0) p^{(0)} + 0 \cdot (p^{(0)} P) = p^{(0)} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

### Theorem 2: Geometric Convergence to Fixed Point
For $0 \le \alpha < 1$, the infinite-step diffusion converges to a unique fixed point:
$$p^{(\infty)} = (1 - \alpha) p^{(0)} (I - \alpha P)^{-1}$$
The approximation error decays geometrically:
$$\|p^{(T)} - p^{(\infty)}\| \le \alpha^T \|p^{(0)} - p^{(\infty)}\|$$

### Theorem 3: Multi-Hop Expansion
After $T$ steps, QGFD can be expanded into explicit path traversals:
$$p^{(T)} = (1 - \alpha) \sum_{k=0}^{T-1} \alpha^k p^{(0)} P^k + \alpha^T p^{(0)} P^T$$

### Theorem 4: Dense Attention Approximation from Sparse Graphs
For sparse initial graph $A$ and key transition $P$, diffusion recovers full attention expressivity with error controlled by the second-largest eigenvalue $|\lambda_2(P)|$:
$$\|p^{(T)} - \text{softmax}(QK^\top / \sqrt{d_k})\| \le C |\lambda_2(P)|^T$$

### Theorem 5: Oversmoothing Bounds
As $T \to \infty$ or $\alpha \to 1$, $p^{(T)}$ converges to the unique stationary distribution $\pi$ of $P$:
$$\|p^{(T)} - \pi\| \le C \cdot \gamma^T, \quad \text{where } \gamma = \max(\alpha, |\lambda_2(P)|) < 1$$
Query outputs collapse: $\lim_{T \to \infty} \|h_i^{(T)} - h_j^{(T)}\| = 0$.

---

## 3. Step-by-Step Numerical Walkthrough

Consider $N=3$ tokens with $d_k=2, \sqrt{d_k} = \sqrt{2} \approx 1.414$, and inputs:
$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}$$

### Step 1: Raw Similarities
$$QK^\top = KK^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$
Scaled by $\sqrt{2}$:
$$\frac{QK^\top}{\sqrt{d_k}} \approx \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}$$

### Step 2: Initial Attention $p^{(0)}$ & Transition Matrix $P$
Applying row-wise softmax:
$$p^{(0)} = P \approx \begin{bmatrix} 0.40 & 0.20 & 0.40 \\ 0.20 & 0.40 & 0.40 \\ 0.25 & 0.25 & 0.50 \end{bmatrix}$$
Standard Attention Output $h_{\text{std}} = p^{(0)} V$:
$$h_{\text{std}} = \begin{bmatrix} 0.40(2) + 0.20(3) + 0.40(4) \\ 0.20(2) + 0.40(3) + 0.40(4) \\ 0.25(2) + 0.25(3) + 0.50(4) \end{bmatrix} = \begin{bmatrix} 3.00 \\ 3.20 \\ 3.25 \end{bmatrix}$$

### Step 3: Diffusion Iteration ($T=2, \alpha=0.5$)
#### Iteration $T=1$:
$$p^{(0)} P \approx \begin{bmatrix} 0.30 & 0.26 & 0.44 \\ 0.26 & 0.30 & 0.44 \\ 0.2875 & 0.2625 & 0.45 \end{bmatrix}$$
$$p^{(1)} = 0.5 p^{(0)} + 0.5 (p^{(0)} P) \approx \begin{bmatrix} 0.35 & 0.23 & 0.42 \\ 0.23 & 0.35 & 0.42 \\ 0.2688 & 0.2563 & 0.475 \end{bmatrix}$$

#### Iteration $T=2$:
$$p^{(2)} = 0.5 p^{(0)} + 0.5 (p^{(1)} P) \approx \begin{bmatrix} 0.3455 & 0.2335 & 0.4210 \\ 0.2455 & 0.3335 & 0.4210 \\ 0.2769 & 0.2575 & 0.4750 \end{bmatrix}$$

### Step 4: Final QGFD Output $h^{(2)} = p^{(2)} V$
$$h^{(2)} = \begin{bmatrix} 3.075 \\ 3.175 \\ 3.226 \end{bmatrix}$$

---

## 4. TorchDire Module Reference

### `MultiHeadQGFDLayer`
Defined in `torchdire.nn.qgfd`:
```python
class MultiHeadQGFDLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        proj_dim: int | None = None,
        diffusion_steps: int = 4,
        target_alpha: float = 0.02,
        warmup_steps: int = 20000,
        detach_P: bool = False,
        mode: str = "full",  # "full" or "conv"
        max_full_seq_len: int = 512,
        full_fallback_mode: str = "conv",
    ): ...
```

---

## 5. Universal Model Wrapping & Integration

Inject QGFD into any PyTorch / HuggingFace model in one line:
```python
from transformers import AutoModelForCausalLM
from torchdire import wrap_model_with_qgfd

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = wrap_model_with_qgfd(
    model,
    diffusion_steps=2,
    target_alpha=0.02,
    mode="full",
    detach_P=True,
    max_full_seq_len=512,
    full_fallback_mode="conv",
    verbose=True,
)
```

---

## 6. Benchmarking, Profiling & Ablation Engine

### Verification of Theorems:
```python
from torchdire import verify_qgfd_theorems
results = verify_qgfd_theorems(verbose=True)
```

### Profiling Latency & VRAM:
```python
from torchdire import profile_qgfd_efficiency
profile = profile_qgfd_efficiency(batch_size=4, seq_len=512, embed_dim=768)
print(profile)
```

### Running Automated Ablations:
```python
from torchdire import run_ablation_study
results = run_ablation_study(save_csv_path="ablation_results.csv")
```

---

## 7. Google Colab & Cloud Setup Guide

Execute directly on Google Colab GPU:
```python
!git clone https://github.com/rajboopathiking/TorchDire.git
%cd TorchDire
!pip install -e .
!python -c "from torchdire import verify_qgfd_theorems; verify_qgfd_theorems()"
```

---

## 8. IEEE Publication & Benchmark Guidelines

When preparing paper experiments for IEEE journals:
1. **Base Models:** SmolLM2-135M or Qwen2.5-0.5B.
2. **Settings:** Diffusion steps $T=2, \alpha=0.02, \text{detach\_P}=\text{True}, \text{warmup}=2000$.
3. **Metrics:** ROUGE-L, BLEU, BERTScore F1, Perplexity, Latency (ms), and VRAM (MB).

---

## 9. Empirical Validation & Microbenchmark Suite Results

The full validation suite and benchmarks can be executed via `run_benchmarks_and_validation.py`.

### 9.1 Numerical Stability & Correctness Verification

| Test Suite | Conditions Tested | Status | Max Error / Metric |
| :--- | :--- | :---: | :--- |
| **Baseline Equivalence** | $\alpha=0$ or `enable_qgfd=False` vs standard attention | **PASSED** | Output diff $= 0.00$, Attn diff $= 0.00$ |
| **Probability Distribution Validity** | Row stochasticity ($\sum_k p_k = 1.0$) & non-negativity ($p \ge 0$) | **PASSED** | Max sum err $= 2.38 \times 10^{-7}$, Range: $[0.006, 0.137]$, No NaNs/Infs |
| **Dtype Precision Stability** | FP32, FP16, BF16 across `full` and `conv` modes | **PASSED** | Stable outputs for all precisions |
| **Autograd & Gradient Flow** | Gradient computation for input $X$, projections ($W_q, W_k, W_v, W_o$), & `conv_kernel` | **PASSED** | Non-zero gradients, no NaNs/Infs |
| **Extreme Value & Edge Case Stress** | Zeros, large inputs ($10^4$), large negative inputs ($-10^4$), $L=1$, 90% padding mask | **PASSED** | Stable execution across all edge cases |
| **Sequence Length Fallbacks** | $L > \text{max\_full\_seq\_len}$ ($L=32, \text{max}=16$) with `conv` and `disable` fallback | **PASSED** | Graceful transition to fallback mode |

### 9.2 Microbenchmarks (Latency, Memory, Throughput)
*Configuration: Batch Size = 4, Sequence Length = 256, Embedding Dim = 512, Heads = 8*

| Layer Configuration | Forward Latency (ms) | Backward Latency (ms) | Total Step Latency (ms) | Throughput (tokens/sec) | Peak Tensor Memory (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Attention** | **115.18** | **512.34** | **627.52** | **8890.6** | **16.00** |
| **QGFD Full** ($T=1, \alpha=0.02$) | 411.67 | 988.57 | 1400.24 | 2487.4 | 32.00 |
| **QGFD Full** ($T=2, \alpha=0.02$) | 516.95 | 1983.14 | 2500.09 | 1980.9 | 40.00 |
| **QGFD Full** ($T=4, \alpha=0.02$) | 683.98 | 1271.66 | 1955.65 | 1497.1 | 40.00 |
| **QGFD Conv** ($T=1, \alpha=0.02$) | 504.99 | 1963.55 | 2468.54 | 2027.7 | **16.16** |
| **QGFD Conv** ($T=2, \alpha=0.02$) | 674.04 | 1969.15 | 2643.19 | 1519.2 | **16.16** |
| **QGFD Conv** ($T=4, \alpha=0.02$) | 904.53 | 4189.77 | 5094.30 | 1132.1 | **16.16** |

### 9.3 Sequence Length Scaling Suite
*Sweep across sequence lengths $L \in [64, 128, 256, 512, 1024, 2048, 4096]$*

| Sequence Length ($L$) | Baseline Attention (Fwd ms / Mem MB) | QGFD Full ($T=2$) (Fwd ms / Mem MB) | QGFD Conv ($T=2, K=5$) (Fwd ms / Mem MB) | QGFD Full w/ Conv Fallback ($\text{max\_L}=512$) |
| :---: | :---: | :---: | :---: | :---: |
| **64** | 55.24 ms / 0.63 MB | 74.60 ms / 1.00 MB | 61.35 ms / 0.64 MB | 47.59 ms / 1.00 MB |
| **128** | 32.06 ms / 1.50 MB | 163.06 ms / 3.00 MB | 127.29 ms / 1.52 MB | 285.85 ms / 3.00 MB |
| **256** | 121.18 ms / 4.00 MB | 306.84 ms / 10.00 MB | 380.04 ms / 4.04 MB | 413.75 ms / 10.00 MB |
| **512** | 247.77 ms / 12.00 MB | 800.95 ms / 36.00 MB | 616.31 ms / 12.08 MB | 684.69 ms / 36.00 MB |
| **1024** | 312.94 ms / 40.00 MB | 1773.61 ms / 136.00 MB | 2613.29 ms / 40.16 MB | 1741.78 ms / 136.00 MB |
| **2048** | 1076.42 ms / 144.00 MB | 10224.12 ms / 528.00 MB | 7431.67 ms / 144.31 MB | 4946.77 ms / 528.00 MB |
| **4096** | 2954.45 ms / 544.00 MB | 69555.46 ms / 2080.00 MB | 20111.86 ms / 544.63 MB | 24535.69 ms / 2080.00 MB |

