import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchdire.nn.llama_qgfd import patch_llama_with_qgfd

model_id = '42dot/42dot_LLM-SFT-1.3B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(model.device)
qgfd_model = patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)

out_cache = qgfd_model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=True)
text_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
print("QGFD (cache):", text_cache)
