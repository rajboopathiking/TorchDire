"""
QGFD LoRA Fine-Tuning A/B  (paper track 2)
==========================================
Equal-budget comparison of two fine-tuning arms on a single free-tier GPU:

    arm "softmax"  ->  LoRA only          (SoftmaxOperator; numerically stock attention)
    arm "qgfd"     ->  LoRA + QGFD        (QGFDOperator, alpha warmed up over training)

Identical seed, data, LR, schedule and step count in both arms — the ONLY
difference is the attention probability operator. After training, each arm is
evaluated with the same clean-perplexity and noise-robustness code the zero-shot
harness uses, so the fine-tuned numbers are directly comparable to Table 1.

Design notes
------------
* Plain `transformers.Trainer`, not TRL. This is ordinary causal-LM fine-tuning;
  Trainer's API is far more stable across versions than SFTConfig, which has
  renamed `tokenizer`/`max_seq_length`/`dataset_text_field` repeatedly.
* The OPERATOR backend is the default so training and evaluation go through the
  exact same QGFD code path. (The legacy `patch_llama_with_qgfd` kernel backend
  is available via --backend kernel.)
* alpha warmup is driven from OUTSIDE forward() by a Trainer callback calling
  set_step(global_step). Mutating step inside forward diverges on gradient-
  checkpoint recompute and raises CheckpointError.
* LoRA targets q/k/v/o. Both arms get identical adapters, so the comparison
  isolates the operator rather than adapter capacity.

Usage
-----
    # CPU smoke (~2 min, no GPU):
    python scripts/finetune_qgfd.py --quick --model_id JackFram/llama-160m

    # Paper run on a T4:
    python scripts/finetune_qgfd.py --model_id HuggingFaceTB/SmolLM2-135M --seeds 0,1,2
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, replace, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.review_experiments import (  # noqa: E402
    ExperimentConfig,
    UNGATED_MODELS,
    _plot_aggregate,
    _print_aggregate,
    aggregate_runs,
    compute_perplexity,
    load_wikitext,
    resolve_device,
    robustness_sweep,
)

# __CHUNK2__

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class FinetuneConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    dtype: str = "bfloat16"
    device: str = "auto"
    backend: str = "operator"          # "operator" (recommended) | "kernel" (legacy)

    # --- QGFD (arm "qgfd" only) --------------------------------------------
    diffusion_steps: int = 1
    target_alpha: float = 0.05
    warmup_steps: int = 100            # alpha ramps 0 -> target over this many steps
    detach_P: bool = True
    mode: str = "full"
    max_full_seq_len: int = 512
    full_fallback_mode: str = "conv"
    learnable_alpha: bool = False

    # --- LoRA (both arms, identical) ---------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # --- Data ---------------------------------------------------------------
    train_num_texts: int = 1500        # WikiText-2 *train* paragraphs
    block_size: int = 256              # <= max_full_seq_len keeps mode="full"

    # --- Optimisation (T4-friendly) ----------------------------------------
    max_steps: int = 300
    batch_size: int = 2
    grad_accum: int = 8                # effective batch 16
    learning_rate: float = 2e-4
    lr_warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    seed: int = 0

    # --- Post-training evaluation (mirrors the zero-shot harness) ----------
    eval_ppl_num_texts: int = 60
    eval_robustness_num_texts: int = 40
    eval_max_length: int = 512
    noise_rates: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)

    out_dir: str = "./qgfd_finetune_results"

    def eval_config(self) -> ExperimentConfig:
        """An ExperimentConfig so the shared eval helpers can be reused verbatim."""
        return ExperimentConfig(
            model_id=self.model_id, dtype=self.dtype, device=self.device,
            diffusion_steps=self.diffusion_steps, target_alpha=self.target_alpha,
            ppl_max_length=self.eval_max_length, ppl_stride=self.eval_max_length,
            noise_rates=tuple(self.noise_rates), robustness_seed=self.seed,
            seed=self.seed,
        )
# __CHUNK3__


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def resolve_dtype(cfg: FinetuneConfig, device: str) -> torch.dtype:
    dtype = _DTYPES[cfg.dtype]
    if device == "cpu" and dtype is not torch.float32:
        dtype = torch.float32          # bf16/fp16 matmuls are slow/unstable on CPU
    return dtype


def build_arm_model(arm: str, cfg: FinetuneConfig):
    """
    Load a fresh pretrained model and install the arm's attention operator.

    arm="softmax": SoftmaxOperator — bit-identical to stock attention, but routed
    through the same adapter as the QGFD arm so the two arms differ ONLY in the
    probability computation (same eager materialisation, same code path).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg, device)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    model.config.use_cache = False     # required with gradient checkpointing

    if cfg.backend == "kernel":
        from torchdire import patch_llama_with_qgfd
        model = patch_llama_with_qgfd(
            model,
            diffusion_steps=cfg.diffusion_steps,
            target_alpha=cfg.target_alpha,
            warmup_steps=cfg.warmup_steps,
            enable_qgfd=(arm == "qgfd"),
            detach_P=cfg.detach_P,
            mode=cfg.mode,
            max_full_seq_len=cfg.max_full_seq_len,
            full_fallback_mode=cfg.full_fallback_mode,
            learnable_alpha=cfg.learnable_alpha,
            is_causal=True,            # P must not diffuse mass onto future keys
            auto_eval=False,
            verbose=False,
        )
    elif cfg.backend == "operator":
        from torchdire import (
            QGFDOperator, SoftmaxOperator, wrap_model_with_qgfd_operator,
        )
        if arm == "softmax":
            operator = SoftmaxOperator()
        else:
            n_heads = getattr(model.config, "num_attention_heads", None)
            operator = QGFDOperator(
                diffusion_steps=cfg.diffusion_steps,
                target_alpha=cfg.target_alpha,
                warmup_steps=cfg.warmup_steps,
                detach_P=cfg.detach_P,
                mode=cfg.mode,
                max_full_seq_len=cfg.max_full_seq_len,
                full_fallback_mode=cfg.full_fallback_mode,
                learnable_alpha=cfg.learnable_alpha,
                num_heads=n_heads if cfg.learnable_alpha else None,
                is_causal=True,        # P must not diffuse mass onto future keys
            )
        model = wrap_model_with_qgfd_operator(model, operator, verbose=False)
        model.config.use_cache = False
    else:
        raise ValueError(f"backend must be 'operator' or 'kernel', got {cfg.backend!r}")

    verify_arm(model, arm, cfg)
    model.to(device)
    return tok, model, device
# __CHUNK4__


def verify_arm(model, arm: str, cfg: FinetuneConfig) -> None:
    """Fail loudly on a silently-no-op patch (unsupported architecture)."""
    if cfg.backend == "operator":
        from torchdire.nn.attention_adapters import (
            AttentionOperatorAdapter, GenericAttentionAdapter,
        )
        adapters = [m for m in model.modules() if isinstance(m, AttentionOperatorAdapter)]
        if not adapters:
            raise RuntimeError(
                f"[{arm}] no attention module patched for "
                f"'{model.config.model_type}'. Use a Llama/Mistral/Qwen2 model — "
                f"e.g. {', '.join(list(UNGATED_MODELS)[:2])}.")
        n_generic = sum(1 for m in adapters if isinstance(m, GenericAttentionAdapter))
        if n_generic:
            raise RuntimeError(
                f"[{arm}] {n_generic}/{len(adapters)} layers fell back to "
                f"GenericAttentionAdapter, which does not intercept softmax.")
        n_ops = len({id(m.prob_operator) for m in adapters})
        print(f"  [{arm}] {len(adapters)} x {type(adapters[0]).__name__}, "
              f"{n_ops} operator instance(s)")
    else:
        from torchdire import collect_qgfd_kernels
        kernels = collect_qgfd_kernels(model)
        if not kernels:
            raise RuntimeError(f"[{arm}] patch_llama_with_qgfd installed 0 kernels.")
        active = sum(1 for k in kernels if k.enable_qgfd)
        expected = len(kernels) if arm == "qgfd" else 0
        if active != expected:
            raise RuntimeError(
                f"[{arm}] {active}/{len(kernels)} kernels have enable_qgfd=True, "
                f"expected {expected}.")
        print(f"  [{arm}] {len(kernels)} QGFD kernels ({active} active)")


def verify_lora_live(model, tok, device: str) -> None:
    """
    Assert LoRA sits on the projections the forward pass actually uses.

    Regression guard: the operator adapter used to alias q/k/v/o while also
    registering the original attention, and named_modules() de-duplicates shared
    submodules — so PEFT injected LoRA into a module the adapter never called and
    training silently did nothing (see tests/test_operator_lora_liveness.py).
    """
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters after attaching LoRA.")
    ids = tok("liveness probe sentence for the adapters", return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    live = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    model.zero_grad(set_to_none=True)
    if not any("lora_B" in n for n in live):
        raise RuntimeError(
            "LoRA adapters received no gradient — they are not on the live "
            "attention path. Refusing to report a fine-tuning result.")
    print(f"  LoRA live: {len(trainable)} trainable tensors, "
          f"{len(live)} with non-zero grad on the probe")
# __CHUNK5__


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_lm_dataset(tok, texts: List[str], block_size: int):
    """Concatenate, tokenize once, and chunk into fixed-size causal-LM blocks."""
    from datasets import Dataset

    ids = tok("\n\n".join(texts))["input_ids"]
    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        raise RuntimeError(
            f"Corpus too small: {len(ids)} tokens < block_size={block_size}. "
            f"Raise train_num_texts or lower block_size.")
    blocks = [ids[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    print(f"  train data: {n_blocks} blocks x {block_size} tokens "
          f"= {n_blocks * block_size} tokens")
    return Dataset.from_dict({
        "input_ids": blocks,
        "attention_mask": [[1] * block_size for _ in blocks],
        "labels": [list(b) for b in blocks],
    })


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_arm(arm: str, cfg: FinetuneConfig, train_texts: List[str],
              eval_texts: List[str], return_model: bool = False) -> Dict:
    """
    Fine-tune one arm and evaluate it. Returns a run_all()-shaped arm dict.

    With return_model=True the trained model is kept alive and returned as
    ``(arm_result, tok, model, device)`` — used by the synthetic track to probe a
    post-LoRA checkpoint without duplicating this Trainer setup. The caller then
    owns the VRAM.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Trainer, TrainingArguments, default_data_collator, set_seed,
    )

    print(f"\n--- arm: {arm} (backend={cfg.backend}) ---", flush=True)
    set_seed(cfg.seed)
    tok, model, device = build_arm_model(arm, cfg)

    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_targets),
    ))
    if cfg.learnable_alpha:
        # PEFT froze everything that is not an adapter; alpha must be re-enabled
        # AFTER get_peft_model or it never trains.
        from torchdire import unfreeze_qgfd_alpha
        n = unfreeze_qgfd_alpha(model)
        print(f"  re-enabled {n} learnable alpha parameter(s)")
    if cfg.gradient_checkpointing:
        # With every base weight frozen the checkpointed segment has no input
        # requiring grad, so autograd would drop the whole block.
        model.enable_input_require_grads()
    verify_lora_live(model, tok, device)

    dataset = build_lm_dataset(tok, train_texts, cfg.block_size)
# __CHUNK6__

    dtype = resolve_dtype(cfg, device)
    ta_kwargs = dict(
        output_dir=os.path.join(cfg.out_dir, f"_trainer_{arm}"),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.lr_warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        # Clamped so a short run still logs at least twice: with the default
        # logging_steps=10 a 4-step smoke run logs nothing and the loss-history
        # check below fires spuriously.
        logging_steps=max(1, min(cfg.logging_steps, cfg.max_steps // 2 or 1)),
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=(device == "cuda" and dtype is torch.bfloat16),
        fp16=(device == "cuda" and dtype is torch.float16),
    )
    # Drop anything the installed transformers does not know about.
    import inspect
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in ta_kwargs if k not in allowed]
    for k in dropped:
        ta_kwargs.pop(k)
    if dropped:
        print(f"  (TrainingArguments: dropped unsupported {dropped})")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # alpha warmup must be driven from outside forward(): mutating step_count
    # inside forward diverges on gradient-checkpoint recompute (CheckpointError).
    if arm == "qgfd":
        if cfg.backend == "operator":
            from torchdire import register_qgfd_operator_step_callback as reg
        else:
            from torchdire import register_qgfd_step_callback as reg
        if reg(trainer, model) is None:
            raise RuntimeError(f"[{arm}] alpha warmup callback found no QGFD modules.")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if device == "cuda" else None)

    losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    if not losses:
        raise RuntimeError(f"[{arm}] no training loss was logged.")
    print(f"  trained {cfg.max_steps} steps in {train_seconds:.1f}s | "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
# __CHUNK7__

    alpha_report = report_alpha(model, cfg, arm)

    # ---- post-training evaluation (same code as the zero-shot harness) ----
    model.eval()
    ecfg = cfg.eval_config()
    print(f"  [{arm}] clean perplexity ...", flush=True)
    with torch.no_grad():
        clean_ppl = compute_perplexity(
            model, tok, eval_texts[:cfg.eval_ppl_num_texts], device,
            max_length=cfg.eval_max_length, stride=cfg.eval_max_length)
        print(f"  [{arm}] robustness sweep ...", flush=True)
        robustness = robustness_sweep(
            model, tok, eval_texts[:cfg.eval_robustness_num_texts], device, ecfg)
    print(f"  [{arm}] clean ppl = {clean_ppl:.4f}")

    arm_result = {
        "kind": arm,
        "clean_ppl": clean_ppl,
        "robustness": {float(k): v for k, v in robustness.items()},
        "train": {
            "losses": losses,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "steps": cfg.max_steps,
            "seconds": train_seconds,
            "peak_vram_mb": peak_vram_mb,
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        },
        "alpha": alpha_report,
    }

    if return_model:
        del trainer
        return arm_result, tok, model, device

    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arm_result


def report_alpha(model, cfg: FinetuneConfig, arm: str) -> Dict:
    """Confirm the warmup schedule actually advanced, and record final alpha."""
    if arm != "qgfd":
        return {"active": False}
    if cfg.backend == "operator":
        from torchdire import collect_qgfd_operators as collect
    else:
        from torchdire import collect_qgfd_kernels as collect
    mods = collect(model)
    if not mods:
        return {"active": True, "warning": "no QGFD modules found post-training"}
    step = int(mods[0].step_count.item())
    was_training = mods[0].training
    mods[0].train()
    a_train = mods[0].get_alpha()
    mods[0].eval()
    a_eval = mods[0].get_alpha()
    mods[0].train(was_training)

    def _f(a):
        return float(a.mean().item()) if isinstance(a, torch.Tensor) else float(a)

    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        print(f"  WARNING: step_count={step} < warmup_steps={cfg.warmup_steps} — "
              f"alpha never reached target_alpha during training.")
    return {"active": True, "step_count": step, "n_modules": len(mods),
            "alpha_train_mode": _f(a_train), "alpha_eval_mode": _f(a_eval),
            "target_alpha": cfg.target_alpha}
# __CHUNK8__


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_seed(cfg: FinetuneConfig, arms: Sequence[str] = ("softmax", "qgfd")) -> Dict:
    """Train + evaluate every arm at cfg.seed. Returns a run_all()-shaped dict."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading WikiText-2 train split ({cfg.train_num_texts} paragraphs) ...",
          flush=True)
    train_texts = load_wikitext(cfg.train_num_texts, split="train")
    n_eval = max(cfg.eval_ppl_num_texts, cfg.eval_robustness_num_texts)
    print(f"Loading WikiText-2 test split ({n_eval} paragraphs) ...", flush=True)
    eval_texts = load_wikitext(n_eval, split="test")

    result = {
        "config": asdict(cfg),
        "device": resolve_device(cfg.device),
        "arms": {},
    }
    for arm in arms:
        result["arms"][arm] = train_arm(arm, cfg, train_texts, eval_texts)

    path = os.path.join(cfg.out_dir, "finetune_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {path}")
    _print_seed_summary(result)
    return result


def _print_seed_summary(result: Dict) -> None:
    arms = result["arms"]
    print("\n" + "-" * 66)
    print(f"{'arm':<12}{'final loss':>12}{'clean ppl':>12}{'ppl @15% noise':>18}")
    print("-" * 66)
    for name, a in arms.items():
        noisy = a["robustness"].get(0.15) or list(a["robustness"].values())[-1]
        print(f"{name:<12}{a['train']['final_loss']:>12.4f}"
              f"{a['clean_ppl']:>12.4f}{noisy:>18.4f}")
    print("-" * 66)


def run_all_seeds(cfg: FinetuneConfig, seeds: Sequence[int] = (0, 1, 2),
                  arms: Sequence[str] = ("softmax", "qgfd")) -> Dict:
    """
    Repeat the A/B across seeds and aggregate with the shared statistics code.

    Both arms share a seed's data order and noise realisation, so the paired
    per-seed differences reported by aggregate_runs() are the meaningful numbers.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    runs, seeds = [], list(seeds)
    for i, s in enumerate(seeds):
        print(f"\n{'#' * 74}\n# FINETUNE SEED {s}  ({i + 1}/{len(seeds)})\n{'#' * 74}",
              flush=True)
        runs.append(run_seed(
            replace(cfg, seed=s, out_dir=os.path.join(cfg.out_dir, f"seed{s}")),
            arms=arms))

    if len(arms) < 2:
        print("(single arm — skipping paired aggregation)")
        return {"runs": runs}

    agg = aggregate_runs(runs, seeds)
    agg["meta"]["track"] = "finetune"
    agg["meta"]["backend"] = cfg.backend
    agg["meta"]["max_steps"] = cfg.max_steps
    agg["meta"]["lora"] = {"r": cfg.lora_r, "alpha": cfg.lora_alpha,
                           "targets": list(cfg.lora_targets)}
    agg["train"] = {
        arm: {
            "final_loss": _stat_from([r["arms"][arm]["train"]["final_loss"] for r in runs]),
            "seconds": _stat_from([r["arms"][arm]["train"]["seconds"] for r in runs]),
        } for arm in arms
    }
    path = os.path.join(cfg.out_dir, "finetune_aggregated.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved aggregated fine-tuning results -> {path}")
    _print_aggregate(agg)
    _plot_aggregate(agg, cfg.out_dir)
    return agg


def _stat_from(values):
    from scripts.review_experiments import _stat
    return _stat(values)
# __CHUNK9__


def apply_quick(cfg: FinetuneConfig) -> FinetuneConfig:
    """Tiny CPU-runnable budget — proves the plumbing, not the science."""
    return replace(
        cfg,
        max_steps=6, warmup_steps=3, logging_steps=1,
        batch_size=1, grad_accum=1, block_size=64,
        train_num_texts=40, lora_r=4, lora_alpha=8,
        gradient_checkpointing=False,
        eval_ppl_num_texts=4, eval_robustness_num_texts=4,
        eval_max_length=128, noise_rates=(0.0, 0.15),
    )


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="QGFD LoRA fine-tuning A/B")
    ap.add_argument("--model_id", default=FinetuneConfig.model_id,
                    help=f"one of: {', '.join(UNGATED_MODELS)}")
    ap.add_argument("--backend", default=FinetuneConfig.backend,
                    choices=("operator", "kernel"))
    ap.add_argument("--dtype", default=FinetuneConfig.dtype)
    ap.add_argument("--device", default=FinetuneConfig.device)
    ap.add_argument("--arms", default="softmax,qgfd",
                    help="comma-separated subset of softmax,qgfd")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds, e.g. '0,1,2' (default: single seed 0)")
    ap.add_argument("--max_steps", type=int, default=FinetuneConfig.max_steps)
    ap.add_argument("--batch_size", type=int, default=FinetuneConfig.batch_size)
    ap.add_argument("--grad_accum", type=int, default=FinetuneConfig.grad_accum)
    ap.add_argument("--block_size", type=int, default=FinetuneConfig.block_size)
    ap.add_argument("--learning_rate", type=float, default=FinetuneConfig.learning_rate)
    ap.add_argument("--diffusion_steps", type=int, default=FinetuneConfig.diffusion_steps)
    ap.add_argument("--target_alpha", type=float, default=FinetuneConfig.target_alpha)
    ap.add_argument("--warmup_steps", type=int, default=FinetuneConfig.warmup_steps)
    ap.add_argument("--learnable_alpha", action="store_true")
    ap.add_argument("--no_grad_checkpoint", action="store_true")
    ap.add_argument("--out_dir", default=FinetuneConfig.out_dir)
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke budget")
    a = ap.parse_args(argv)

    cfg = FinetuneConfig(
        model_id=a.model_id, backend=a.backend, dtype=a.dtype, device=a.device,
        max_steps=a.max_steps, batch_size=a.batch_size, grad_accum=a.grad_accum,
        block_size=a.block_size, learning_rate=a.learning_rate,
        diffusion_steps=a.diffusion_steps, target_alpha=a.target_alpha,
        warmup_steps=a.warmup_steps, learnable_alpha=a.learnable_alpha,
        gradient_checkpointing=not a.no_grad_checkpoint, out_dir=a.out_dir,
    )
    if a.quick:
        cfg = apply_quick(cfg)

    arms = tuple(s.strip() for s in a.arms.split(",") if s.strip())
    unknown = [x for x in arms if x not in ("softmax", "qgfd")]
    if unknown:
        ap.error(f"unknown arm(s): {unknown}")

    if a.seeds:
        run_all_seeds(cfg, seeds=[int(s) for s in a.seeds.split(",") if s.strip()],
                      arms=arms)
    else:
        run_seed(cfg, arms=arms)


if __name__ == "__main__":
    main()
