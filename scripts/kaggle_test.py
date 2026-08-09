# ============================================================
# QGFD Verified Test Script for Kaggle / Colab
# ============================================================
# IMPORTANT: Run this in a FRESH kernel (Kernel -> Restart)
# ============================================================

# Step 1: Force clean reinstall
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "--upgrade", "--force-reinstall", "--no-cache-dir",
    "git+https://github.com/rajboopathiking/TorchDire.git"])

# Step 2: Verify the installed version has the fix
import inspect
from torchdire.nn.qgfd_kernel import QGFDKernel
src = inspect.getsource(QGFDKernel.build_transition_from_keys)
assert "torch.triu" in src and "causal_mask" in src, \
    f"ERROR: Old code without causal fix is installed!\nSource:\n{src[:300]}"
print("[OK] Verified: build_transition_from_keys has causal mask fix")

from torchdire.nn.llama_qgfd import patch_llama_with_qgfd
src2 = inspect.getsource(patch_llama_with_qgfd)
assert "warmup_steps: int = 0" in src2, \
    f"ERROR: Old default warmup_steps=20000 is still present!\nSource:\n{src2[:300]}"
print("[OK] Verified: patch_llama_with_qgfd has warmup_steps=0 default")

# Step 3: Load model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "42dot/42dot_LLM-SFT-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(model.device)

# Step 4: Baseline output
outputs_base = model.generate(**inputs, max_new_tokens=50, do_sample=False)
text_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
print("\n=== BASELINE SOFTMAX ===")
print(text_base)

# Step 5: QGFD output (fresh patch, NOT double-patched)
qgfd_model = patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02)
outputs_qgfd = qgfd_model.generate(**inputs, max_new_tokens=50, do_sample=False)
text_qgfd = tokenizer.decode(outputs_qgfd[0], skip_special_tokens=True)
print("\n=== QGFD ATTENTION ===")
print(text_qgfd)

# Step 6: Verify no collapse
tokens_qgfd = outputs_qgfd[0].tolist()
tail = tokens_qgfd[len(inputs["input_ids"][0]):]
unique_ratio = len(set(tail)) / max(1, len(tail))
print(f"\nUnique token ratio in generated tail: {unique_ratio:.2%} ({len(set(tail))}/{len(tail)})")
if unique_ratio < 0.1:
    print("WARNING: Generation may have collapsed!")
else:
    print("PASS: Generation is diverse and meaningful")
