import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchdire import patch_llama_with_qgfd

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("42dot/42dot_LLM-SFT-1.3B")
model = AutoModelForCausalLM.from_pretrained(
    "42dot/42dot_LLM-SFT-1.3B",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto"
)

inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(model.device)

print("Patching model...")
qgfd_model = patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.0)

print("Generating without cache...")
outputs = qgfd_model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=False)
print("QGFD (no cache):", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Generating with cache...")
outputs_cached = qgfd_model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=True)
print("QGFD (cache):", tokenizer.decode(outputs_cached[0], skip_special_tokens=True))
