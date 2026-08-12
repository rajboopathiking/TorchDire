import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchdire import patch_llama_with_qgfd

model = AutoModelForCausalLM.from_pretrained(
    "42dot/42dot_LLM-SFT-1.3B",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    device_map="cpu"
)

def hook(module, args, kwargs, output):
    print("attention_mask type:", type(kwargs.get("attention_mask")))
    if kwargs.get("attention_mask") is not None:
        print("attention_mask shape:", kwargs["attention_mask"].shape)
    else:
        print("attention_mask is None!")

model.model.layers[0].self_attn.register_forward_hook(hook, with_kwargs=True)

inputs = torch.randint(0, 100, (1, 8))
print("Running original SDPA model...")
model(inputs)

# Now patch
patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.0)

# If we change config _attn_implementation to eager
if hasattr(model, "config"):
    print("Old implementation:", getattr(model.config, "_attn_implementation", None))
    model.config._attn_implementation = "eager"
    
print("Running patched model...")
# Hook needs to be added again because layer was replaced
model.model.layers[0].self_attn.register_forward_hook(hook, with_kwargs=True)
model(inputs)
