import gc
import inspect
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from torchdire import (
    patch_llama_with_qgfd,
    register_qgfd_step_callback,
    collect_qgfd_kernels,
)


def build_synthetic_llama_model(vocab_size=50257):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.config.use_cache = False
    return model


def build_synthetic_dataset():
    data = {"text": [f"### Instruction:\nSolve step {i}\n\n### Response:\nAnswer {i}" for i in range(50)]}
    return Dataset.from_dict(data)


def test_phase1_smoke_both_arms():
    """
    Phase 1 Smoke Test:
    - Verifies Phase 0 fixes work end-to-end.
    - 20 steps for baseline (enable_qgfd=False) and QGFD arm (enable_qgfd=True).
    - Checks LoRA target_modules adapts q_proj, k_proj, v_proj, o_proj.
    - Checks QGFD kernel alpha warmup schedule updates and diffuses.
    - Checks memory cleanup between sequential runs.
    """
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_synthetic_dataset()

    # -------------------------------------------------------------
    # Arm 1: Softmax Baseline (enable_qgfd=False)
    # -------------------------------------------------------------
    model_base = build_synthetic_llama_model(vocab_size=len(tokenizer))
    model_base = patch_llama_with_qgfd(
        model_base,
        diffusion_steps=3,
        target_alpha=0.10,
        warmup_steps=10,
        enable_qgfd=False,
        verbose=False,
    )

    lora_config_base = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config_base = SFTConfig(
        output_dir="./tmp_smoke_base",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=20,
        learning_rate=1e-3,
        logging_steps=5,
        save_strategy="no",
        dataset_text_field="text",
        packing=False,
        seed=42,
    )

    trainer_kwargs_base = dict(
        model=model_base,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=sft_config_base,
        peft_config=lora_config_base,
    )
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs_base["processing_class"] = tokenizer
    else:
        trainer_kwargs_base["tokenizer"] = tokenizer

    trainer_base = SFTTrainer(**trainer_kwargs_base)

    # Verify LoRA targets q_proj, k_proj, v_proj, o_proj
    trainable_names = [n for n, p in trainer_base.model.named_parameters() if p.requires_grad]
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        assert any(proj in n for n in trainable_names), f"LoRA missing target module: {proj}"

    trainer_base.train()
    base_train_losses = [l["loss"] for l in trainer_base.state.log_history if "loss" in l]
    assert len(base_train_losses) > 0, "No training losses logged for baseline arm"
    assert base_train_losses[-1] < base_train_losses[0], "Baseline loss did not decrease"

    del model_base, trainer_base
    gc.collect()

    # -------------------------------------------------------------
    # Arm 2: QGFD Arm (enable_qgfd=True)
    # -------------------------------------------------------------
    set_seed(42)
    model_qgfd = build_synthetic_llama_model(vocab_size=len(tokenizer))
    model_qgfd = patch_llama_with_qgfd(
        model_qgfd,
        diffusion_steps=3,
        target_alpha=0.10,
        warmup_steps=10,
        enable_qgfd=True,
        verbose=False,
    )

    lora_config_qgfd = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config_qgfd = SFTConfig(
        output_dir="./tmp_smoke_qgfd",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=20,
        learning_rate=1e-3,
        logging_steps=5,
        save_strategy="no",
        dataset_text_field="text",
        packing=False,
        seed=42,
    )

    trainer_kwargs_qgfd = dict(
        model=model_qgfd,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=sft_config_qgfd,
        peft_config=lora_config_qgfd,
    )
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs_qgfd["processing_class"] = tokenizer
    else:
        trainer_kwargs_qgfd["tokenizer"] = tokenizer

    trainer_qgfd = SFTTrainer(**trainer_kwargs_qgfd)

    register_qgfd_step_callback(trainer_qgfd, model_qgfd)

    kernels = collect_qgfd_kernels(model_qgfd)
    assert len(kernels) > 0, "No QGFDKernels found in patched model!"

    # Before training starts, step_count is 0 -> alpha_eff is 0
    assert float(kernels[0].step_count.item()) == 0.0

    trainer_qgfd.train()

    qgfd_train_losses = [l["loss"] for l in trainer_qgfd.state.log_history if "loss" in l]
    assert len(qgfd_train_losses) > 0, "No training losses logged for QGFD arm"
    assert qgfd_train_losses[-1] < qgfd_train_losses[0], "QGFD loss did not decrease"

    # After 20 steps with warmup_steps=10, step_count has advanced, and alpha_eff should be target_alpha (0.10)
    step_val = kernels[0].step_count.item()
    alpha_val = kernels[0].get_alpha()
    assert step_val >= 10, f"Expected step_count >= 10, got {step_val}"
    assert abs(alpha_val - 0.10) < 1e-5, f"Expected alpha_eff == 0.10, got {alpha_val}"

    del model_qgfd, trainer_qgfd
    gc.collect()


if __name__ == "__main__":
    test_phase1_smoke_both_arms()
    print("Phase 1 Smoke Test passed successfully!")
