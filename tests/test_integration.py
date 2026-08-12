import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchdire.nn.llama_qgfd import patch_llama_with_qgfd
from torchdire.utils.replacer import wrap_model_with_qgfd

@pytest.mark.parametrize("use_cache", [True, False])
def test_gpt2_generation_with_qgfd(use_cache):
    """Test that GPT-2 generation works flawlessly with SafeWrappedAttention (wrap_model_with_qgfd)"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Patch the model using the generic wrapper
    wrapped_model = wrap_model_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, auto_eval=True)
    
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    
    with torch.no_grad():
        out = wrapped_model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=use_cache)
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    assert len(text) > 0

def test_double_patch_llama_tiny():
    """Test that patch_llama_with_qgfd doesn't break if applied twice."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Patch once
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, auto_eval=True)
    # Patch twice
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, auto_eval=True)
    
    inputs = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
    
    assert out.shape[1] > inputs["input_ids"].shape[1]

def test_gpt_neo_generation_parity():
    """Test that GPT-Neo generation output exactly matches baseline when QGFD alpha=0 (due to warmup_steps)."""
    model_id = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Baseline Model
    baseline_model = AutoModelForCausalLM.from_pretrained(model_id)
    baseline_model.eval()
    
    # QGFD Model
    qgfd_model = AutoModelForCausalLM.from_pretrained(model_id)
    # The user's snippet called patch_llama_with_qgfd, which will fallback to wrap_model_with_qgfd for non-llama
    patch_llama_with_qgfd(qgfd_model, diffusion_steps=4, target_alpha=0.02, warmup_steps=20000, auto_eval=True)
    qgfd_model.eval()
    
    inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
    
    with torch.no_grad():
        baseline_out = baseline_model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        qgfd_out = qgfd_model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    qgfd_text = tokenizer.decode(qgfd_out[0], skip_special_tokens=True)
    
    assert baseline_out[0].tolist() == qgfd_out[0].tolist(), "Token outputs do not strictly match!"
    assert baseline_text == qgfd_text, "Decoded text outputs do not match!"

def test_real_slm_generation_parity_gpt_neo_125m():
    """Test that a REAL SLM (gpt-neo-125m) produces identical generation under QGFD with alpha=0."""
    model_id = "EleutherAI/gpt-neo-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Baseline Model
    baseline_model = AutoModelForCausalLM.from_pretrained(model_id)
    baseline_model.eval()
    
    # QGFD Model
    qgfd_model = AutoModelForCausalLM.from_pretrained(model_id)
    patch_llama_with_qgfd(qgfd_model, diffusion_steps=4, target_alpha=0.02, warmup_steps=20000, auto_eval=True)
    qgfd_model.eval()
    
    inputs = tokenizer("Hello, I am a language model", return_tensors="pt")
    
    with torch.no_grad():
        baseline_out = baseline_model.generate(**inputs, max_new_tokens=15, do_sample=False, use_cache=True)
        qgfd_out = qgfd_model.generate(**inputs, max_new_tokens=15, do_sample=False, use_cache=True)
        
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    qgfd_text = tokenizer.decode(qgfd_out[0], skip_special_tokens=True)
    
    assert baseline_out[0].tolist() == qgfd_out[0].tolist(), "Token outputs do not strictly match for real SLM!"
    assert baseline_text == qgfd_text, "Decoded text outputs do not match for real SLM!"
