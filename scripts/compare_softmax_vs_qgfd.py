# ============================================================
# Equal-budget A/B: softmax baseline vs QGFD diffusion
# on the Alpagasus instruction-following task.
#
# Goal: does QGFD (alpha>0) beat softmax (alpha=0) on a simple
# downstream task, at identical compute budget?
#
#   Baseline: QGFD patch with enable_qgfd=False -> the kernel is a
#             pure softmax passthrough (zero diffusion cost).
#   QGFD:     enable_qgfd=True, target_alpha=0.05,
#             warmup_steps=args.steps (alpha ramps 0->0.05 over the run).
#
# Metric: held-out EVAL loss + perplexity (train loss saturates at ~0
# via memorization and carries no signal).
# Cost:   wall time and it/s for each run.
#
# Usage:  python compare_softmax_vs_qgfd.py --steps 100
# ============================================================
import os

# Kaggle hosts expose 2 T4s, so TRL/transformers wraps the model in
# nn.DataParallel regardless of device_map={"": 0}. bitsandbytes 4-bit
# linears are NOT thread-safe under DataParallel and corrupt memory, which
# CUDA reports asynchronously as "CUDA error: an illegal memory access" at
# the next kernel. Expose a single GPU so the Trainer never DataParallel
# wraps (override with QGFD_CUDA_VISIBLE_DEVICES to opt back into 2 GPUs).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import inspect
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from torchdire import patch_llama_with_qgfd, register_qgfd_step_callback

MODEL_ID = "42dot/42dot_LLM-SFT-1.3B"
OUT_ROOT = "./42dot-ab"


def make_sft_config_kwargs(args, seed):
    kwargs = dict(
        output_dir=os.path.join(OUT_ROOT, f"run-{seed}-{args.tag}"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        num_train_epochs=3,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field="text",
        packing=False,
        loss_type="nll",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        data_seed=seed,
    )
    # TRL renamed max_seq_length -> max_length in 1.x; accept whichever exists
    try:
        fields = SFTConfig.model_fields
    except AttributeError:
        fields = set(inspect.signature(SFTConfig).parameters)
    if "max_length" in fields:
        kwargs["max_length"] = args.max_length
    elif "max_seq_length" in fields:
        kwargs["max_seq_length"] = args.max_length
    return kwargs


def format_example(example):
    if example.get("input"):
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        )
    return {"text": prompt}


def run_experiment(args, seed, tag, enable_qgfd, target_alpha, warmup_steps):
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        ),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    model = patch_llama_with_qgfd(
        model,
        diffusion_steps=args.diffusion_steps,
        target_alpha=target_alpha,
        warmup_steps=warmup_steps,
        early_stop_eps=0.0,
        enable_qgfd=enable_qgfd,
        verbose=args.verbose,
    )

    ds = load_dataset("arbml/alpagasus_cleaned")["train"].map(format_example)
    ds = ds.train_test_split(test_size=0.2, seed=seed)
    train_ds, test_ds = ds["train"], ds["test"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=SFTConfig(**make_sft_config_kwargs(args, seed)),
        peft_config=lora_config,
    )
    register_qgfd_step_callback(trainer, model)

    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0
    eval_metrics = trainer.evaluate()

    steps = args.steps
    loss = eval_metrics.get("eval_loss", float("nan"))
    return {
        "tag": tag,
        "alpha": target_alpha if enable_qgfd else 0.0,
        "eval_loss": loss,
        "eval_ppl": float(torch.exp(torch.tensor(loss))),
        "wall_s": round(wall_time, 1),
        "it_per_s": round(steps / wall_time, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--diffusion_steps", type=int, default=2)
    ap.add_argument("--target_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = []
    results.append(run_experiment(
        args, args.seed, "softmax-baseline",
        enable_qgfd=False, target_alpha=0.0, warmup_steps=0,
    ))
    results.append(run_experiment(
        args, args.seed, "qgfd",
        enable_qgfd=True, target_alpha=args.target_alpha, warmup_steps=args.steps,
    ))

    print("\n=== RESULT (equal budget: %d steps, seed %d) ===" % (args.steps, args.seed))
    print(f"{'run':<18}{'alpha':<8}{'eval_loss':<12}{'eval_ppl':<12}{'wall_s':<10}{'it/s':<8}")
    for r in results:
        print(
            f"{r['tag']:<18}{r['alpha']:<8.4f}{r['eval_loss']:<12.4f}"
            f"{r['eval_ppl']:<12.2f}{r['wall_s']:<10}{r['it_per_s']:<8.3f}"
        )
    gain = results[0]["eval_loss"] - results[1]["eval_loss"]
    print(f"\nDelta eval_loss (softmax - QGFD): {gain:+.4f}  "
          f"(negative => QGFD worse, positive => QGFD better)")
    print(f"QGFD compute cost ratio: {results[1]['wall_s'] / results[0]['wall_s']:.2f}x baseline")


if __name__ == "__main__":
    main()