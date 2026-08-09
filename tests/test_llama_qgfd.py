import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM, LlamaAttention
from torchdire.nn.qgfd_kernel import QGFDKernel
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd


def test_qgfd_kernel_standalone():
    kernel = QGFDKernel(diffusion_steps=3, target_alpha=0.03, warmup_steps=0)
    scores = torch.randn(2, 4, 16, 16)
    keys = torch.randn(2, 4, 16, 16)
    
    probs = kernel(scores, keys)
    assert probs.shape == (2, 4, 16, 16)
    assert not torch.isnan(probs).any()
    assert (probs >= -1e-6).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


def test_qgfd_kernel_gqa():
    # 4 query heads, 2 key heads (GQA)
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
    scores = torch.randn(2, 4, 16, 16)
    keys = torch.randn(2, 2, 16, 16)
    
    probs = kernel(scores, keys)
    assert probs.shape == (2, 4, 16, 16)
    assert not torch.isnan(probs).any()


def test_llama_qgfd_attention_forward():
    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    attn = LlamaQGFDAttention(config, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
    x = torch.randn(2, 16, 64)
    
    output, attn_weights, past_kv = attn(x)
    assert output.shape == (2, 16, 64)
    assert not torch.isnan(output).any()


def test_patch_llama_model():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    
    # Store reference to layer 0 q_proj before patching
    orig_q_proj = model.model.layers[0].self_attn.q_proj
    
    # Patch model
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)
    
    # Verify layer type updated
    assert isinstance(model.model.layers[0].self_attn, LlamaQGFDAttention)
    # Verify q_proj object identity preserved (crucial for LoRA / QLoRA!)
    assert model.model.layers[0].self_attn.q_proj is orig_q_proj
    
    # Verify model forward pass and loss backward
    input_ids = torch.randint(0, 100, (2, 16))
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    assert loss is not None and not torch.isnan(loss)
    
    loss.backward()
    assert orig_q_proj.weight.grad is not None
    assert not torch.isnan(orig_q_proj.weight.grad).any()


def test_llama_qgfd_attribute_preservation():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)
    
    attn_layer = model.model.layers[0].self_attn
    assert isinstance(attn_layer, LlamaQGFDAttention)
    assert hasattr(attn_layer, "num_heads") and attn_layer.num_heads == 4
    assert hasattr(attn_layer, "num_key_value_heads") and attn_layer.num_key_value_heads == 2
    assert hasattr(attn_layer, "head_dim") and attn_layer.head_dim == 16
    assert hasattr(attn_layer, "num_key_value_groups") and attn_layer.num_key_value_groups == 2
    assert hasattr(attn_layer, "vocab_size") and attn_layer.vocab_size == 100  # From config fallback __getattr__


def test_llama_qgfd_rotary_fallback_and_double_patch():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    # Remove rotary_emb from attention layer to simulate newer transformers / custom architecture
    if hasattr(model.model.layers[0].self_attn, "rotary_emb"):
        delattr(model.model.layers[0].self_attn, "rotary_emb")

    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)
    # Patch second time (double-patching safety test)
    patch_llama_with_qgfd(model, diffusion_steps=3, target_alpha=0.03, warmup_steps=0, verbose=False)

    x = torch.randn(2, 16, 64)
    out, _, _ = model.model.layers[0].self_attn(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_llama_qgfd_generation_kv_cache():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)

    input_ids = torch.randint(0, 100, (1, 8))

    # Test generation with KV cache (use_cache=True)
    out_cache = model.generate(input_ids, max_new_tokens=20, min_new_tokens=20, do_sample=False, use_cache=True)
    assert out_cache.shape == (1, 28)
    assert not torch.isnan(out_cache.float()).any()

    # Test generation without KV cache (use_cache=False)
    out_nocache = model.generate(input_ids, max_new_tokens=20, min_new_tokens=20, do_sample=False, use_cache=False)
    assert out_nocache.shape == (1, 28)
    assert not torch.isnan(out_nocache.float()).any()

    # Verify diversity (no single-token collapse attractor loop)
    generated_tokens = out_cache[0, 8:].tolist()
    unique_tokens = set(generated_tokens)
    assert len(unique_tokens) > 1, f"Generation collapsed into a single repeating token: {generated_tokens}"


def test_llama_softmax_equivalence():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    baseline = LlamaForCausalLM(config)
    qgfd = LlamaForCausalLM(config)
    qgfd.load_state_dict(baseline.state_dict())

    patch_llama_with_qgfd(qgfd, diffusion_steps=4, target_alpha=0.0, warmup_steps=0, verbose=False)

    input_ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        base_logits = baseline(input_ids).logits
        qgfd_logits = qgfd(input_ids).logits

    diff = (base_logits - qgfd_logits).abs()
    assert diff.max().item() < 1e-5, f"Softmax equivalence failed! Max diff: {diff.max().item()}"
    assert diff.mean().item() < 1e-6, f"Softmax equivalence failed! Mean diff: {diff.mean().item()}"


if __name__ == "__main__":
    test_qgfd_kernel_standalone()
    test_qgfd_kernel_gqa()
    test_llama_qgfd_attention_forward()
    test_patch_llama_model()
    test_llama_qgfd_attribute_preservation()
    test_llama_qgfd_rotary_fallback_and_double_patch()
    test_llama_qgfd_generation_kv_cache()
    test_llama_softmax_equivalence()
    print("All Llama QGFD tests passed successfully!")


