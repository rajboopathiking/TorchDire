"""Builds QGFD_Review_Experiments.ipynb from the verified review_experiments.py.
Run once, then deleted. Guarantees the notebook harness == the tested module."""
import json, os

REPO = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(REPO, "scripts", "review_experiments.py")

with open(MODULE) as f:
    src = f.read()
# Strip the CLI __main__ block so the cell only defines the harness on run.
marker = 'if __name__ == "__main__":'
harness_src = src.split(marker)[0].rstrip() + "\n\nprint('QGFD review harness loaded.')\n"


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


cells = []

cells.append(md("""# QGFD Milestone-1 Review — Experiment Suite

**Query–Graph Flow Diffusion vs standard softmax attention.**
Zero-shot evaluation (no training) — a full sweep runs in a few minutes on one GPU.

Produces everything needed for the review report/video:
1. **Perplexity** on WikiText-2 (softmax vs QGFD)
2. **Noise robustness** — perplexity degradation vs input corruption
3. **Attention entropy & sink concentration** — the mechanism's effect
4. **Compute overhead** — prefill latency / tokens-per-sec
5. **Qualitative generation** samples

Outputs: `qgfd_review_results/results.json`, `robustness_curve.png`, `attention_stats.png`.

> **Correctness note:** QGFD is constructed with `is_causal=True`. This is required
> for causal LMs — with `is_causal=False` the key-graph diffusion leaks future tokens
> into earlier positions and artificially deflates perplexity. Verified on CPU."""))

cells.append(md("## 1 · Install dependencies"))
cells.append(code(
    "!pip -q install --upgrade git+https://github.com/rajboopathiking/TorchDire.git\n"
    "!pip -q install transformers datasets matplotlib accelerate\n"))

cells.append(md("## 2 · No Hugging Face token needed\n"
                "The default model is **ungated**, so you can ignore the `HF_TOKEN` warning Colab prints.\n\n"
                "QGFD only genuinely patches **Llama / Mistral / Qwen2** architectures. GPT-2, OPT and\n"
                "GPT-Neo either crash or patch zero layers (silently no-op), so stick to this list:\n\n"
                "| model | notes |\n|---|---|\n"
                "| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | llama, 1.1B, 32 heads / 4 KV — real GQA, **default** |\n"
                "| `HuggingFaceTB/SmolLM2-135M` | llama, 135M, fast |\n"
                "| `JackFram/llama-160m` | llama, 160M, fastest smoke test |\n"
                "| `Qwen/Qwen2.5-0.5B` | qwen2, 0.5B |\n\n"
                "`meta-llama/Llama-3.2-1B` also works but is **gated** — it needs "
                "`login(token='hf_...')` and an approved licence."))
cells.append(code(
    "# Optional: only if you insist on the gated meta-llama checkpoint.\n"
    "# from huggingface_hub import login\n"
    "# login(token='hf_xxx')\n"
    "print('No token required for the default ungated model.')\n"))

cells.append(md("## 3 · Load the experiment harness\n"
                "This cell is the verified `review_experiments.py` module, inlined so the "
                "notebook is self-contained."))
cells.append(code(harness_src))

cells.append(md("""## 4 · Configure & run

Defaults are ungated and tuned for a fast, defensible run — no HF token required.
Swap `model_id` for any entry in the table above. Reduce `ppl_num_texts` /
`robustness_num_texts` if you are tight on time.

`make_model()` verifies the patch actually took effect and raises if zero layers were
patched, so a silent no-op can never be reported as a QGFD result."""))
cells.append(code(
    "cfg = ExperimentConfig(\n"
    "    model_id=\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",   # ungated, real GQA\n"
    "    device=\"auto\",\n"
    "    dtype=\"bfloat16\",\n"
    "    target_alpha=0.05,\n"
    "    diffusion_steps=1,\n"
    "    ppl_num_texts=200,\n"
    "    robustness_num_texts=60,\n"
    "    out_dir=\"./qgfd_review_results\",\n"
    ")\n"
    "results = run_all(cfg)\n"))

cells.append(md("## 5 · Show plots inline (for report / slides)"))
cells.append(code(
    "from IPython.display import Image, display\n"
    "import os\n"
    "for name in ('robustness_curve.png', 'attention_stats.png'):\n"
    "    p = os.path.join(cfg.out_dir, name)\n"
    "    if os.path.exists(p):\n"
    "        display(Image(filename=p))\n"))

cells.append(md("## 6 · Generation samples (side-by-side)"))
cells.append(code(
    "for arm in ('softmax', 'qgfd'):\n"
    "    print(f\"\\n===== {arm.upper()} =====\")\n"
    "    for g in results['arms'][arm]['generation']:\n"
    "        print('PROMPT :', g['prompt'])\n"
    "        print('OUTPUT :', g['completion'])\n"
    "        print('-' * 60)\n"))

cells.append(md("""## 7 · (Optional, heavy) Fine-tuning A/B

The zero-shot suite above is enough for Milestone-1. If you have spare GPU time and
want a trained comparison, `scripts/compare_softmax_vs_qgfd.py` runs an equal-budget
LoRA SFT A/B on Alpagasus (see `qgfd_experimentation_plan.md` for the multi-seed
protocol). That is Phase-2 work and is **not** required for this review."""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(REPO, "QGFD_Review_Experiments.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "cells:", len(cells))
