# ============================================================
# Fixed SFT (LoRA) training: 42dot 1.3B on Alpagasus
# Fixes over the Colab version:
#   1. trainer.train() now runs BEFORE trainer.evaluate().
#      The old cell called evaluate() first (train() was commented out),
#      so the table showed "Training Loss: No log" at step 0 and
#      eval_num_tokens=0.0 (TRL's num_tokens counter is only
#      incremented during training, sft_trainer.py:1817).
#   2. EvalAwareSFTTrainer reports the TRUE eval token count in
#      eval_num_tokens instead of TRL's cumulative train-token counter.
#   3. Memory-safe config (smaller batch, seq len 512, grad
#      checkpointing) to stop the CUDA OOM retries on 2x16GB.
# ============================================================
import inspect

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "42dot/42dot_LLM-SFT-1.3B"
OUTPUT_DIR = "./42dot-alpagasus-lora"

# --- 1. Model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- 1b. Optional: QGFD attention patch (research pipeline) ---
USE_QGFD = True
if USE_QGFD:
    from torchdire.nn.llama_qgfd import patch_llama_with_qgfd

    model = patch_llama_with_qgfd(
        model,
        diffusion_steps=2,
        target_alpha=0.02,
        warmup_steps=0,
        verbose=True,
    )
    # patch_llama_with_qgfd switches to eval mode (auto_eval=True); SFT needs
    # train mode so dropout and QGFD training behavior are active.
    model.train()
model.config.use_cache = False

# --- 2. Dataset ---
ds = load_dataset("arbml/alpagasus_cleaned")


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


ds = ds["train"].map(format_example)
ds = ds.train_test_split(test_size=0.2)
train_ds, test_ds = ds["train"], ds["test"]

# --- 3. LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 4. Training config (memory-safe) ---
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,          # was 4 -> OOM on 16GB
    per_device_eval_batch_size=4,           # was 8 -> OOM during eval
    gradient_accumulation_steps=8,          # effective batch stays ~32
    max_steps=100,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=50,
    max_seq_length=512,                     # was 1024 -> biggest OOM driver
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    dataset_text_field="text",
    packing=False,
    loss_type="nll",
)


# --- 5. Trainer subclass: fix eval_num_tokens ---
class EvalAwareSFTTrainer(SFTTrainer):
    """SFTTrainer that reports the true eval token count in eval_num_tokens.

    TRL fills `num_tokens` with `_total_train_tokens`, a counter that is only
    incremented during training (sft_trainer.py:1806-1817). Evaluating before
    any training step therefore reports 0 tokens, and after training it reports
    cumulative *train* tokens rather than the eval tokens actually scored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_eval_num_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training:
            labels = inputs.get("shift_labels", inputs.get("labels"))
            if labels is not None:
                num_tokens = (labels != -100).sum()
                self._total_eval_num_tokens += (
                    self.accelerator.gather_for_metrics(num_tokens).sum().item()
                )
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    def log(self, logs, start_time=None):
        if not self.model.training and hasattr(self, "_metrics"):
            self._metrics["eval"]["num_tokens"] = [self._total_eval_num_tokens]
        super().log(logs, start_time)
        self._total_eval_num_tokens = 0


# --- 6. Trainer (processing_class vs tokenizer kwarg across TRL versions) ---
trainer_kwargs = dict(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=training_args,
    peft_config=lora_config,
)
if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer
trainer = EvalAwareSFTTrainer(**trainer_kwargs)

# --- 6b. QGFD warmup step must be driven OUTSIDE the model forward ---
# step_count mutated inside forward() makes gradient-checkpoint recomputation
# diverge (alpha flips 0 -> 1e-6, diffusion branch flips) and crashes with
# torch.utils.checkpoint.CheckpointError. The callback sets the step once per
# optimizer step instead, so forward and recompute see the same alpha.
if USE_QGFD:
    from torchdire.nn.qgfd_kernel import register_qgfd_step_callback

    register_qgfd_step_callback(trainer, model)

# --- 7. Train FIRST, then evaluate ---
trainer.train()
trainer.evaluate()
trainer.save_model(OUTPUT_DIR)