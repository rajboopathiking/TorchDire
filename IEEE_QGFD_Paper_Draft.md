# Query–Graph Flow Diffusion: A Diffusion-Regularized Attention Mechanism

**Author:** Raj Boopathi  
**Target Venue:** IEEE Transactions on Neural Networks and Learning Systems / IEEE Conference  
**Codebase:** [TorchDire Framework](https://github.com/rajboopathiking/TorchDire.git)

> **Status: theory sections are current; empirical sections are pending re-measurement.**
> An earlier revision of this draft reported ROUGE-L / BLEU / BERTScore figures taken
> from `torchdire/experiments/ablation.py`. Those figures were **hard-coded constants**,
> not measurements — the ablator computed `trainer.evaluate(...)`, discarded the result,
> and returned arithmetic on literals instead. They have been removed rather than
> restated, and the ablator has been fixed to report what it measures. Its real output
> is still not publishable: proxy ROUGE-L of `0.0219`, identical across $T$, on a
> randomly-initialised toy model. Section IV now documents the protocol that will
> replace the removed table; run `QGFD_Paper_Experiments.ipynb` to populate
> `paper/REPORT.md`, then transcribe from there.
> See [docs/interpreting-results.md](docs/interpreting-results.md).


---

## Abstract

Self-attention mechanisms in modern Transformer models rely on single-step, direct pairwise inner products between Queries and Keys. Consequently, standard attention cannot propagate information across higher-order key neighborhoods within a single layer, rendering models vulnerable to input noise and limiting multi-hop reasoning. In this paper, we propose **Query–Graph Flow Diffusion (QGFD)**, a diffusion-regularized attention mechanism that frames attention refinement as an iterative Markovian random walk over a key-similarity graph. QGFD starts from a baseline attention distribution $p^{(0)}$ and updates it over $T$ steps using a row-stochastic transition matrix $P = \text{softmax}(KK^\top / \sqrt{d_k})$. We present five formal theorems establishing: (1) exact mathematical equivalence to standard softmax attention when the diffusion strength parameter $\alpha = 0$; (2) geometric convergence to a unique fixed point $p^{(\infty)} = (1 - \alpha) p^{(0)} (I - \alpha P)^{-1}$; (3) explicit multi-hop expansion capturing path traversal; (4) dense attention approximation from sparse initial graphs; and (5) upper bounds on representation oversmoothing when $T \to \infty$. Theorem 1 is verified in code to bit-exactness (maximum absolute logit difference $0.0$ between $\text{QGFD}(\alpha{=}0)$ and softmax on a pretrained checkpoint), establishing QGFD as a safe drop-in. Our empirical claim is **robustness-led**: on clean text QGFD is expected to be perplexity-neutral to slightly worse, and the effect we test for is a smaller *relative degradation* under character-level input noise, evaluated on three Small Language Models (SmolLM2-135M, Qwen2.5-0.5B, TinyLlama-1.1B) with paired within-seed statistics and t-based 95% confidence intervals over three seeds. Multi-hop routing is assessed separately via controlled induction and passkey-retrieval probes.

---

## I. Introduction

Transformer architectures have become the foundational backbone of natural language processing, computer vision, and multimodal learning. The core operation driving Transformers is scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Despite its empirical success, standard attention exhibits fundamental architectural constraints:
1. **Single-Hop Routing:** Attention routing occurs purely through direct $Q_i K_j^\top$ interactions. If token $i$ is semantically connected to token $k$ via an intermediate key token $j$ ($i \to j \to k$), standard attention cannot capture this path within a single layer.
2. **Brittle Probability Mass Allocation:** In noisy domains (OCR/ASR transcriptions, typos, ungrammatical text), slight key embedding shifts cause softmax probabilities to collapse onto incorrect tokens.

To resolve these challenges without increasing model parameter counts, we introduce **Query–Graph Flow Diffusion (QGFD)**. QGFD constructs a key-similarity transition matrix $P$ and diffuses initial attention weights $p^{(0)}$ across $P$:
$$p^{(t+1)} = (1 - \alpha) p^{(0)} + \alpha (p^{(t)} P)$$

---

## II. Theoretical Foundations

We establish five core theorems governing QGFD behavior:

### Theorem 1 (Equivalence to Softmax Attention)
*Statement:* For $\alpha = 0$ or $T = 1$ with $\alpha = 0$, QGFD collapses strictly to standard scaled dot-product attention.  
*Proof:* Setting $\alpha = 0$ into $p^{(t+1)} = (1 - \alpha) p^{(0)} + \alpha p^{(t)} P$ yields $p^{(1)} = p^{(0)} = \text{softmax}(QK^\top / \sqrt{d_k})$. $\blacksquare$

### Theorem 2 (Geometric Convergence to Fixed Point)
*Statement:* For $0 \le \alpha < 1$, QGFD diffusion converges geometrically to $p^{(\infty)} = (1 - \alpha) p^{(0)} (I - \alpha P)^{-1}$ with error bound $\|p^{(T)} - p^{(\infty)}\| \le \alpha^T \|p^{(0)} - p^{(\infty)}\|$.  
*Proof:* The linear recurrence operator $T(p) = (1-\alpha)p^{(0)} + \alpha p P$ has spectral radius $\rho(\alpha P) \le \alpha < 1$. By the Banach Fixed-Point Theorem, the Neumann series $(I - \alpha P)^{-1} = \sum_{k=0}^{\infty} \alpha^k P^k$ exists and converges geometrically. $\blacksquare$

### Theorem 3 (Multi-Hop Expansion)
*Statement:* After $T$ diffusion steps, QGFD expands to:
$$p^{(T)} = (1 - \alpha) \sum_{k=0}^{T-1} \alpha^k p^{(0)} P^k + \alpha^T p^{(0)} P^T$$
*Proof:* Proof by mathematical induction on $T$. Base case $T=1$ holds. Substituting $p^{(T)}$ into $p^{(T+1)}$ yields the summation. $\blacksquare$

### Theorem 4 (Approximation of Full Attention via Diffusion)
*Statement:* Starting from a sparse initial graph $A$, diffusion propagates probability mass across connected key neighborhoods, approximating full dense attention exponentially fast with spectral bound $\|p^{(T)} - \text{softmax}(QK^\top / \sqrt{d_k})\| \le C |\lambda_2(P)|^T$.

### Theorem 5 (Oversmoothing Bounds)
*Statement:* As $T \to \infty$ or $\alpha \to 1$, $p^{(T)}$ converges to the unique stationary distribution $\pi$ of $P$:
$$\|p^{(T)} - \pi\| \le C \gamma^T, \quad \text{where } \gamma = \max(\alpha, |\lambda_2(P)|) < 1$$
Consequently, query output representations collapse: $\lim_{T \to \infty} \|h_i^{(T)} - h_j^{(T)}\| = 0$.

### Causal Key Transition Matrix Design ($P$)

> **Correction.** An earlier revision of this section argued that $P$ should be left
> **unmasked** across cached keys, on the grounds that all cached keys $\{K_0,\dots,K_q\}$
> already belong to the causal past, and that masking $P$ induces a column-sum asymmetry
> ($\sum_n P_{n,0} \approx \ln(L_k) \gg 1$) creating an artificial sink at position 0.
> That argument holds only for **incremental decoding**, where the KV cache contains
> nothing but past keys. It does not hold for **teacher-forced training or evaluation**,
> where the full sequence is present in one forward pass. Measurements below.

Two regimes must be distinguished.

**Teacher-forced training and evaluation ($L_q = L_k$).** The full sequence is resident,
so an unmasked $P$ lets $p^{(0)} P$ move probability mass onto keys at positions *later*
than the query. This is a causality violation, and it is measurable: with
`is_causal=False`, perturbing a token at position $j$ changed logits at positions $i < j$
by $\approx 4\times10^{-3}$, whereas the base model's leakage is exactly $0$. Because the
leaked information is the very token being predicted, the effect *deflates* teacher-forced
perplexity — QGFD scores better for a reason that has nothing to do with attention
quality. Setting `is_causal=True` restores exact zero leakage.

**Incremental decoding ($L_q = 1$).** Here every cached key is genuinely in the past and
the distinction is vacuous; the sink-asymmetry argument applies, and balanced column sums
($\approx 1.0$) with a near-uniform stationary distribution are what keep generation from
collapsing at $O(1)$ per-step cost.

**Consequence for the experiments.** Every reported number in this paper is teacher-forced,
so `is_causal=True` is mandatory and is enforced rather than left to the caller:
`build_operator()` in `scripts/review_experiments.py` hard-codes it. Any QGFD perplexity
measured with an unmasked $P$ should be treated as invalid.


---

## III. System Implementation: TorchDire Framework

The `TorchDire` library implements QGFD via a dual-mode engine:
1. **Full Matrix Diffusion (`mode="full"`):** Computes global key similarity matrix $P \in \mathbb{R}^{B \times H \times L_k \times L_k}$. Ideal for sequence lengths $L_k \le 512$.
2. **Convolutional Local Diffusion (`mode="conv"`):** Approximates local key diffusion using a CausalConv1D kernel over sequence dimension with $O(L \cdot K)$ memory complexity.
3. **Universal Model Injection:** Rewrites HuggingFace model module trees in-place via `wrap_model_with_qgfd_operator`, preserving state-dict keys and KV-caching behaviour. Support is genuine for the Llama-family adapters (`LlamaAttentionAdapter` and its Qwen2 / Mistral subclasses); GPT-2, OPT and GPT-Neo are **not** supported — their adapters are stubs or patch zero layers.
4. **Verification guards.** `verify_patch()` raises if zero layers were patched, if any layer fell back to a generic adapter, or if the operator is never invoked during a probe forward pass — so an unsupported architecture cannot silently report plain-softmax numbers under a QGFD label. `verify_lora_live()` refuses to report a fine-tuning result unless a LoRA `lora_B` tensor actually receives gradient, and `report_alpha()` asserts that the $\alpha$ warmup schedule genuinely advanced. These exist because each corresponding failure mode was observed silently in practice.

---

## IV. Experimental Protocol & Results

> **Results pending.** The table previously printed here was not measured. See the status
> note at the top of this document. This section specifies the protocol; populate it from
> `paper/REPORT.md` after running `QGFD_Paper_Experiments.ipynb`.

### A. Arms

Four arms, all sharing one adapter implementation so that only the probability operator
differs: (A) eager softmax baseline, (B) QGFD zero-shot, (C) LoRA-only fine-tuned,
(D) QGFD + LoRA fine-tuned. Arms C and D receive identical seed, data order, learning
rate, schedule, step count, LoRA rank and target modules ($q/k/v/o$), so any difference is
attributable to the operator rather than to adapter capacity.

### B. Metrics

| Metric | Corpus | Role |
| --- | --- | --- |
| Clean perplexity | WikiText-2 | Cost of the drop-in; expected neutral-to-slightly-worse |
| Perplexity at noise $\in \{0,5,10,15\}\%$ | WikiText-2 | **Primary claim** — relative degradation |
| Attention entropy, sink mass at position 0 | WikiText-2 | Mechanism evidence, reported as deltas only |
| Prefill latency, peak VRAM at $L \in \{128,256,512\}$ | — | Cost accounting |
| Induction / passkey exact-match accuracy | Synthetic | Secondary: multi-hop routing |

### C. Statistics

Three seeds; every figure carries a two-sided t-based 95% confidence interval. The
statistic that carries the headline claim is the **paired**, within-seed difference
$\Delta\%_{\text{softmax}} - \Delta\%_{\text{QGFD}}$: both arms see identical text
subsamples and the identical noise realisation within a seed, so pairing removes
between-seed corpus variance, which is substantially larger than the effect under test.
Per-arm means and their spreads are reported as context, not as the result.

### D. Threats to validity, stated up front

1. **The latency baseline is eager materialised softmax, not FlashAttention.** QGFD
   requires the explicit probability matrix, so it is architecturally incompatible with
   fused attention kernels. Measured overhead was $1.58\times$ at $L=128$ and
   $1.91\times$ at $L=256$ against eager softmax; against a fused baseline the true gap is
   larger.
2. **Free-tier hardware runs fp16, not bf16.** bf16 requires compute capability $\ge 8.0$;
   a T4 is `sm_75` and a P100 is `sm_60`. Both arms share the dtype, so it is not a
   confound, but the executed dtype is recorded per run.
3. **Exact-match accuracy is coarse.** At $\alpha=0.05$ the arms disagree on $\approx 1.4\%$
   of argmax predictions, so identical synthetic scores at small $n$ are expected and do
   not indicate an inactive operator. Operator provenance is recorded alongside every score.
4. **The induction probe uses a single-token vocabulary,** which makes target alignment
   exact but the probe more artificial than natural-text induction.
5. **$n=3$ is small.** At $n=2$ the t-critical value is $12.7$ and essentially nothing
   reaches significance; the report builder flags any under-powered aggregate automatically.

### E. Ablation

$T \in \{1,2\} \times \alpha \in \{0.02, 0.05\} \times$ `detach_P` $\in \{$True, False$\}$
on the smallest model, one seed per cell — sufficient for **direction only**. Configurations
of interest must be re-run at $n \ge 3$ before any is claimed better. All ablation figures
come from the same harness as the main results; the legacy `QGFDAblator` is not used.


---

## V. Conclusion

Query–Graph Flow Diffusion (QGFD) reframes attention refinement as a short Markovian
random walk over a key-similarity graph, and does so as a genuine drop-in: Theorem 1's
$\alpha=0$ equivalence is verified to bit-exactness in code, so the mechanism can be
introduced into a pretrained model without changing its behaviour until $\alpha$ is raised.
The theoretical contributions — geometric convergence, the explicit multi-hop expansion,
and the oversmoothing bound — characterise what the walk does and where it stops being
useful.

Whether that mechanism yields a *measurable* robustness or multi-hop advantage on real
Small Language Models is the open question this paper is designed to answer, and it is not
yet answered here. The protocol in Section IV is falsifiable and pre-registered: a
robustness gap whose paired 95% confidence interval includes zero is a null result, and
will be reported as one. The `TorchDire` framework provides the operator implementation,
the verification guards that make a silent no-op impossible to report as a result, and the
single-GPU harness needed to settle the question on free-tier hardware.

