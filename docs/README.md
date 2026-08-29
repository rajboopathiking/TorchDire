# TorchDire / QGFD documentation

QGFD (**Query–Graph Flow Diffusion**) replaces the attention probability vector

```
p = softmax(QKᵀ/√d)          # standard attention
p⁽ᵗ⁺¹⁾ = (1−α)·p⁽⁰⁾ + α·(p⁽ᵗ⁾ P),   P = softmax(KKᵀ/√d)   # QGFD
```

with a short Markovian random walk over a key-similarity graph. At `α = 0` it is
bit-for-bit softmax, so it is a drop-in.

## Which document do you want

| I want to… | Read |
| --- | --- |
| Run the paper experiments and get `paper/REPORT.md` | [paper-experiments.md](paper-experiments.md) |
| Understand what a number in the report actually licenses | [interpreting-results.md](interpreting-results.md) |
| Consume the result JSON from my own code | [results-schema.md](results-schema.md) |
| Know how QGFD is wired into a model, and why LoRA once died | [architecture.md](architecture.md) |
| Look up a config parameter's range | [../QGFD_CONFIG_REFERENCE.md](../QGFD_CONFIG_REFERENCE.md) |
| Reproduce the Milestone-1 zero-shot review only | [../REVIEW_RUN_GUIDE.md](../REVIEW_RUN_GUIDE.md) |

## The four experiment tracks at a glance

| Track | Driver | Question it answers |
| --- | --- | --- |
| 1 · Zero-shot | `scripts/review_experiments.py` | Does QGFD cost clean perplexity, and does it degrade *less* under input noise? |
| 2 · LoRA A/B | `scripts/finetune_qgfd.py` | Does letting the model adapt to the diffused distribution help? |
| 3 · Synthetic | `scripts/eval_synthetic.py` | Does QGFD improve two-hop routing (induction) and retrieval (passkey)? |
| 4 · Ablation | `QGFD_Paper_Experiments.ipynb` | Which `(T, α, detach_P)` direction is worth pursuing? |

`scripts/build_report.py` ingests whatever the tracks produced and writes
`paper/REPORT.md`. Tracks you skipped render as `_Not yet run._`, never as blanks.

## The one-sentence honest framing

Zero-shot QGFD is expected to be **perplexity-neutral to slightly worse** on clean
text. The claim is the *robustness gap* — QGFD's perplexity should degrade less as
the input is corrupted — and the statistic that carries it is the **paired**,
within-seed difference with a t-based 95% CI. Everything in
[interpreting-results.md](interpreting-results.md) exists to keep that claim honest.
