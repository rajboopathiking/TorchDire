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


if __name__ == "__main__":
    test_qgfd_kernel_standalone()
    test_qgfd_kernel_gqa()
    test_llama_qgfd_attention_forward()
    test_patch_llama_model()
    test_llama_qgfd_attribute_preservation()
    print("All Llama QGFD tests passed successfully!")
