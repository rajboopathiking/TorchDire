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
# Usage:
#   single GPU (default): python compare_softmax_vs_qgfd.py --steps 100
#   multi GPU (DDP, 2x T4): accelerate launch --multi_gpu --num_processes 2 \
#       compare_softmax_vs_qgfd.py --steps 100
#
# Quick iteration: edit the CONFIG dataclass below (main()); any flag passed
# on the command line overrides the corresponding CONFIG field.
# ============================================================
import argparse
import dataclasses
import inspect
import os
import time

# Kaggle hosts expose 2 T4s. In a plain `python` run, TRL/transformers wraps
# the model in nn.DataParallel regardless of device_map={"": 0}, and
# bitsandbytes 4-bit linears are NOT safe under DataParallel (async "CUDA
# error: an illegal memory access" at the next kernel). Pin a single GPU so
# the Trainer never DataParallel-wraps. For real multi-GPU use accelerate
# launch (DDP): accelerate sets LOCAL_RANK/RANK before this module imports,
# so the pin is skipped and each process gets its own GPU.
if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import torch
from dataclasses import dataclass
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from torchdire import (
    dump_learned_alphas,
    patch_llama_with_qgfd,
    register_qgfd_step_callback,
)

MODEL_ID = "42dot/42dot_LLM-SFT-1.3B"
OUT_ROOT = "./42dot-ab"


@dataclass
class Config:
    # === EDIT HERE for quick iteration ===
    seed: int = 42
    steps: int = 300
    target_alpha: float = 0.1
    warmup_steps: int = 30  # steps over which alpha ramps 0 -> target_alpha
    diffusion_steps: int = 3
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 2e-4
    max_length: int = 512
    verbose: bool = True
    tag: str = "run"
    learnable_alpha: bool = False  # per-head alpha learned by the model
    alpha_sweep: str = ""          # e.g. "0,0.005,0.01,0.02,0.05" (0 = softmax)
    max_train_samples: int = 0     # low-data regularizer experiment, 0 = all
    max_eval_samples: int = 0      # 0 = full test split (lower variance)
    dump_alphas: str = ""          # JSON path for learned per-head alphas


def make_sft_config_kwargs(cfg, seed):
    kwargs = dict(
        output_dir=os.path.join(OUT_ROOT, f"run-{seed}-{cfg.tag}"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_steps=cfg.steps,
        num_train_epochs=3,
        learning_rate=cfg.lr,
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
        kwargs["max_length"] = cfg.max_length
    elif "max_seq_length" in fields:
        kwargs["max_seq_length"] = cfg.max_length
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


def _main_process() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def run_experiment(cfg, seed, tag, enable_qgfd, target_alpha, warmup_steps, learnable_alpha=False):
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **load_kwargs,
    )
    model.config.use_cache = False

    model = patch_llama_with_qgfd(
        model,
        diffusion_steps=cfg.diffusion_steps,
        target_alpha=target_alpha,
        warmup_steps=warmup_steps,
        early_stop_eps=0.0,
        enable_qgfd=enable_qgfd,
        learnable_alpha=learnable_alpha if enable_qgfd else False,
        verbose=cfg.verbose,
    )

    ds = load_dataset("arbml/alpagasus_cleaned")["train"].map(format_example)
    ds = ds.train_test_split(test_size=0.2, seed=seed)
    train_ds, test_ds = ds["train"], ds["test"]
    if cfg.max_train_samples > 0:
        train_ds = train_ds.select(range(min(len(train_ds), cfg.max_train_samples)))
    if cfg.max_eval_samples > 0:
        test_ds = test_ds.select(range(min(len(test_ds), cfg.max_eval_samples)))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=SFTConfig(**make_sft_config_kwargs(cfg, seed)),
        peft_config=lora_config,
    )
    if cfg.verbose and _main_process():
        trainer.model.print_trainable_parameters()

    register_qgfd_step_callback(trainer, model)
    if learnable_alpha:
        from torchdire import unfreeze_qgfd_alpha

        n = unfreeze_qgfd_alpha(model)  # after PEFT froze the base model
        if cfg.verbose:
            print(f"[learnable_alpha] unfroze alpha_param on {n} kernels")

    t0 = time.time()
    trainer.train()
    wall_time = time.time() - t0
    eval_metrics = trainer.evaluate()

    if learnable_alpha and cfg.dump_alphas and _main_process():
        out = dump_learned_alphas(model, cfg.dump_alphas)
        if cfg.verbose:
            print(f"[learnable_alpha] dumped {len(out)} layers to {cfg.dump_alphas}")

    steps = cfg.steps
    loss = eval_metrics.get("eval_loss", float("nan"))
    train_losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    train_loss = train_losses[-1] if train_losses else float("nan")

    res = {
        "tag": tag,
        "alpha": target_alpha if enable_qgfd else 0.0,
        "train_loss": train_loss,
        "eval_loss": loss,
        "eval_ppl": float(torch.exp(torch.tensor(loss))),
        "wall_s": round(wall_time, 1),
        "it_per_s": round(steps / wall_time, 3),
    }

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def main():
    cfg = Config()

    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if f.name == "verbose":
            ap.add_argument("--verbose", action="store_true", default=None)
        elif f.type is bool:
            ap.add_argument(f"--{f.name}", action="store_true", default=None)
        else:
            ap.add_argument(f"--{f.name}", type=f.type, default=None)
    args = ap.parse_args()
    for f in dataclasses.fields(Config):
        val = getattr(args, f.name)
        if val is not None:
            setattr(cfg, f.name, val)

    results = []

    if cfg.alpha_sweep:
        for a in cfg.alpha_sweep.split(","):
            a = float(a.strip())
            results.append(run_experiment(
                cfg, cfg.seed, f"alpha-{a:g}",
                enable_qgfd=a > 0, target_alpha=a,
                warmup_steps=cfg.warmup_steps if a > 0 else 0,
                learnable_alpha=cfg.learnable_alpha,
            ))
    else:
        results.append(run_experiment(
            cfg, cfg.seed, "softmax-baseline",
            enable_qgfd=False, target_alpha=0.0, warmup_steps=0,
        ))
        results.append(run_experiment(
            cfg, cfg.seed,
            "qgfd-learn" if cfg.learnable_alpha else "qgfd",
            enable_qgfd=True, target_alpha=cfg.target_alpha,
            warmup_steps=cfg.warmup_steps,
            learnable_alpha=cfg.learnable_alpha,
        ))

    if not _main_process():
        return

    print("\n=== RESULT (equal budget: %d steps, seed %d%s) ===" % (
        cfg.steps, cfg.seed,
        ", train samples: %d" % cfg.max_train_samples if cfg.max_train_samples else "",
    ))
    print(f"{'run':<18}{'alpha':<8}{'train_loss':<12}{'eval_loss':<12}{'gap':<10}{'eval_ppl':<12}{'wall_s':<10}{'it/s':<8}")
    for r in results:
        gap = r["train_loss"] - r["eval_loss"]
        print(
            f"{r['tag']:<18}{r['alpha']:<8.4f}{r['train_loss']:<12.4f}"
            f"{r['eval_loss']:<12.4f}{gap:<10.4f}{r['eval_ppl']:<12.2f}"
            f"{r['wall_s']:<10}{r['it_per_s']:<8.3f}"
        )
    best = min(results, key=lambda r: r["eval_loss"])
    base = results[0]
    print(f"\nBest eval_loss: {best['tag']} ({best['eval_loss']:.4f})")
    if len(results) > 1:
        print(f"Delta vs baseline: {base['eval_loss'] - best['eval_loss']:+.4f} "
              f"(positive => improvement over baseline)")
    print(f"QGFD compute cost ratio: {results[1]['wall_s'] / results[0]['wall_s']:.2f}x baseline")


if __name__ == "__main__":
    main()