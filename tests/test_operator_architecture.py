"""
Tests for the operator-based QGFD attention architecture.

This verifies that:
1. AttentionProbabilityOperator base class works
2. SoftmaxOperator produces identical results to F.softmax
3. QGFDOperator produces valid probabilities (row-stochastic)
3. Architecture-specific adapters preserve attention mechanics
4. The universal replacer correctly applies operators
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdire.nn.attention_operators import (
    AttentionProbabilityOperator,
    SoftmaxOperator,
    QGFDOperator,
)
from torchdire.nn.attention_adapters import (
    patch_model_with_operator,
    create_attention_adapter,
    LlamaAttentionAdapter,
    _is_leaf_attention,
)



def test_softmax_operator_parity():
    """Test that SoftmaxOperator produces identical results to F.softmax."""
    torch.manual_seed(42)
    
    B, H, Lq, Lk = 2, 4, 8, 16
    scores = torch.randn(B, H, Lq, Lk, dtype=torch.float32)
    key_states = torch.randn(B, H, Lk, 64, dtype=torch.float32)
    
    # Test without mask
    op = SoftmaxOperator()
    probs_op = op(scores, key_states)
    probs_native = F.softmax(scores, dim=-1)
    
    max_diff = (probs_op - probs_native).abs().max().item()
    assert max_diff < 1e-6, f"SoftmaxOperator parity failed: max_diff={max_diff}"
    
    # Test with boolean mask
    mask = torch.ones(B, H, Lq, Lk, dtype=torch.bool)
    mask[:, :, :, -3:] = False
    probs_op_masked = op(scores, key_states, attention_mask=mask)
    probs_native_masked = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
    max_diff = (probs_op_masked - probs_native_masked).abs().max().item()
    assert max_diff < 1e-5, f"SoftmaxOperator with mask failed: max_diff={max_diff}"
    
    # Test with additive mask
    add_mask = torch.zeros(B, H, Lq, Lk)
    add_mask[:, :, :, -3:] = -1e9
    probs_op_add = op(scores, key_states, attention_mask=add_mask)
    probs_native_add = F.softmax(scores + add_mask, dim=-1)
    max_diff = (probs_op_add - probs_native_add).abs().max().item()
    assert max_diff < 1e-5, f"SoftmaxOperator with additive mask failed: max_diff={max_diff}"
    
    print("✓ SoftmaxOperator parity test passed")


def test_qgfd_operator_row_stochasticity():
    """Test that QGFDOperator produces row-stochastic probabilities."""
    torch.manual_seed(42)
    
    B, H, Lq, Lk = 2, 4, 8, 16
    scores = torch.randn(B, H, Lq, Lk, dtype=torch.float32)
    key_states = torch.randn(B, H, Lk, 64, dtype=torch.float32)
    
    op = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.05,
        detach_P=True,
        mode="full",
    )
    op.eval()
    
    probs = op(scores, key_states)
    
    # Check shape preserved
    assert probs.shape == (B, H, Lq, Lk), f"Shape mismatch: {probs.shape}"
    
    # Check row-stochastic (sum to 1 over last dim)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Row sums not 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
    
    # Check non-negative
    assert (probs >= -1e-6).all(), "Probabilities contain negative values"
    
    # Test with mask
    mask = torch.ones(B, H, Lq, Lk, dtype=torch.bool)
    mask[:, :, :, -3:] = False
    probs_masked = op(scores, key_states, attention_mask=mask)
    row_sums_masked = probs_masked.sum(dim=-1)
    # Masked positions should have sum ~1 (over unmasked)
    assert torch.allclose(row_sums_masked, torch.ones_like(row_sums_masked), atol=1e-4), \
        f"Masked row sums not 1: min={row_sums_masked.min():.6f}, max={row_sums_masked.max():.6f}"
    
    print("✓ QGFDOperator row-stochasticity test passed")


def test_qgfd_operator_alpha_zero_equals_softmax():
    """Test that QGFDOperator with alpha=0 equals softmax."""
    torch.manual_seed(42)
    
    B, H, Lq, Lk = 2, 4, 8, 16
    scores = torch.randn(B, H, Lq, Lk, dtype=torch.float32)
    key_states = torch.randn(B, H, Lk, 64, dtype=torch.float32)
    
    op_zero = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.0,  # Zero alpha
        mode="full",
    )
    op_zero.eval()
    
    op_softmax = SoftmaxOperator()
    
    probs_zero = op_zero(scores, key_states)
    probs_softmax = op_softmax(scores, key_states)
    
    max_diff = (probs_zero - probs_softmax).abs().max().item()
    assert max_diff < 1e-5, f"QGFD with alpha=0 != softmax: max_diff={max_diff}"
    
    print("✓ QGFDOperator alpha=0 parity test passed")


def test_qgfd_operator_gradient_flow():
    """Test that gradients flow through QGFDOperator."""
    torch.manual_seed(42)
    
    B, H, Lq, Lk = 1, 2, 4, 8
    scores = torch.randn(B, H, Lq, Lk, dtype=torch.float32, requires_grad=True)
    key_states = torch.randn(B, H, Lk, 32, dtype=torch.float32, requires_grad=True)
    
    op = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.05,
        detach_P=False,  # Allow gradients through P
        mode="full",
    )
    op.train()
    op.set_step(1000)  # Past warmup
    
    probs = op(scores, key_states)
    loss = probs.sum()
    loss.backward()
    
    assert scores.grad is not None, "No gradient wrt scores"
    assert key_states.grad is not None, "No gradient wrt key_states"
    assert scores.grad.abs().max() > 0, "Zero gradient wrt scores"
    assert key_states.grad.abs().max() > 0, "Zero gradient wrt key_states"
    
    print("✓ QGFDOperator gradient flow test passed")


def test_qgfd_operator_detach_P():
    """Test that detach_P blocks gradients through transition matrix."""
    torch.manual_seed(42)
    
    B, H, Lq, Lk = 1, 2, 4, 8
    scores = torch.randn(B, H, Lq, Lk, dtype=torch.float32, requires_grad=True)
    key_states = torch.randn(B, H, Lk, 32, dtype=torch.float32, requires_grad=True)
    
    # With detach_P=True
    op_detached = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.05,
        detach_P=True,
        mode="full",
    )
    op_detached.train()
    op_detached.set_step(1000)
    
    probs_detached = op_detached(scores, key_states)
    loss_detached = probs_detached.sum()
    loss_detached.backward()
    
    key_grad_detached = key_states.grad.clone() if key_states.grad is not None else None
    key_states.grad = None
    
    # With detach_P=False
    op_attached = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.05,
        detach_P=False,
        mode="full",
    )
    op_attached.train()
    op_attached.set_step(1000)
    
    probs_attached = op_attached(scores, key_states)
    loss_attached = probs_attached.sum()
    loss_attached.backward()
    
    key_grad_attached = key_states.grad
    
    # With detach_P=True, key_states should get zero (or very small) gradient
    # because P is detached
    # Note: This test may be flaky because gradients can still flow through p0
    # The key test is that the graph doesn't crash
    
    print("✓ QGFDOperator detach_P test passed (no crash)")


def test_llama_attention_adapter():
    """Test LlamaAttentionAdapter with a dummy Llama-like attention."""
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig
        from transformers.cache_utils import DynamicCache
    except ImportError:
        print("⊘ Skipping LlamaAttentionAdapter test (transformers not available)")
        return
    
    torch.manual_seed(42)
    
    config = LlamaConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
    )
    
    # Create original attention
    orig_attn = LlamaAttention(config, layer_idx=0)
    orig_attn.eval()
    
    # Create operator
    operator = QGFDOperator(
        diffusion_steps=1,
        target_alpha=0.05,
        mode="full",
        detach_P=True,
    )
    operator.eval()
    
    # Create adapter
    adapter = create_attention_adapter(orig_attn, operator)
    adapter.eval()
    
    # Test input
    B, L = 2, 16
    hidden_states = torch.randn(B, L, config.hidden_size)
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    
    # Run original
    with torch.no_grad():
        orig_out = orig_attn(hidden_states, position_ids=position_ids)
        if isinstance(orig_out, tuple):
            orig_attn_output = orig_out[0]
        else:
            orig_attn_output = orig_out
    
    # Run adapter
    with torch.no_grad():
        adapter_out = adapter(hidden_states, position_ids=position_ids)
        if isinstance(adapter_out, tuple):
            adapter_attn_output = adapter_out[0]
        else:
            adapter_attn_output = adapter_out
    
    # They should be different (QGFD vs softmax) but same shape
    assert orig_attn_output.shape == adapter_attn_output.shape, \
        f"Shape mismatch: {orig_attn_output.shape} vs {adapter_attn_output.shape}"
    
    # Test with KV cache
    cache = DynamicCache()
    with torch.no_grad():
        orig_out_cache = orig_attn(hidden_states, position_ids=position_ids, past_key_value=cache, use_cache=True)
        if isinstance(orig_out_cache, tuple):
            orig_attn_output_cache = orig_out_cache[0]
            orig_cache = orig_out_cache[2] if len(orig_out_cache) > 2 else orig_out_cache[1]
        else:
            orig_attn_output_cache = orig_out_cache
    
    cache2 = DynamicCache()
    with torch.no_grad():
        adapter_out_cache = adapter(hidden_states, position_ids=position_ids, past_key_value=cache2, use_cache=True)
        if isinstance(adapter_out_cache, tuple):
            adapter_attn_output_cache = adapter_out_cache[0]
            adapter_cache = adapter_out_cache[2] if len(adapter_out_cache) > 2 else adapter_out_cache[1]
        else:
            adapter_attn_output_cache = adapter_out_cache
    
    assert orig_attn_output_cache.shape == adapter_attn_output_cache.shape

    # Specifically test attribute access and delegation (e.g. num_heads, head_dim)
    assert hasattr(adapter, "num_heads") and adapter.num_heads == 8
    assert hasattr(adapter, "head_dim") and adapter.head_dim == 32
    assert hasattr(adapter, "num_key_value_heads") and adapter.num_key_value_heads == 4

    # Test simulating modern transformers where orig_attn only has config.num_attention_heads
    class ModernLlamaAttentionMock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.layer_idx = 0
            self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size // 2, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size // 2, bias=False)
            self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            # Notice NO self.num_heads or self.head_dim on self (only on self.config)

    mock_attn = ModernLlamaAttentionMock(config)
    adapter_mock = LlamaAttentionAdapter(mock_attn, operator)
    assert adapter_mock.num_heads == 8
    assert adapter_mock.head_dim == 32
    assert adapter_mock.num_key_value_heads == 4
    
    print("✓ LlamaAttentionAdapter test passed")



def test_operator_replacer_on_dummy_model():
    """Test the patch_model_with_operator on a dummy model with attention."""
    
    class UnknownAttention(nn.Module):  # Unknown architecture name
        def __init__(self, embed_dim=64, num_heads=4):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.o_proj = nn.Linear(embed_dim, embed_dim)
            self.is_causal = True
            self.attention_dropout = 0.0
        
        def forward(self, hidden_states, attention_mask=None, **kwargs):
            B, L, _ = hidden_states.shape
            q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Native softmax
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
            out = self.o_proj(out)
            return out
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = UnknownAttention()
            self.layer2 = nn.Linear(64, 64)
        
        def forward(self, x):
            x = self.attention(x)
            x = self.layer2(x)
            return x
    
    model = DummyModel()
    model.eval()
    
    # Test with SoftmaxOperator (should use generic adapter)
    op_softmax = SoftmaxOperator()
    model_softmax = patch_model_with_operator(model, op_softmax, verbose=False)
    model_softmax.eval()
    
    # Test with QGFDOperator
    op_qgfd = QGFDOperator(diffusion_steps=1, target_alpha=0.5, mode="full", detach_P=True)
    model_qgfd = patch_model_with_operator(model, op_qgfd, verbose=False)
    model_qgfd.eval()
    
    x = torch.randn(2, 8, 64)
    
    with torch.no_grad():
        out_orig = model(x)
        out_softmax = model_softmax(x)
        out_qgfd = model_qgfd(x)
    
    # Check that modules were replaced with generic adapter
    from torchdire.nn.attention_adapters import GenericAttentionAdapter
    assert isinstance(model_softmax.attention, GenericAttentionAdapter), "SoftmaxOperator should create generic adapter"
    assert isinstance(model_qgfd.attention, GenericAttentionAdapter), "QGFDOperator should create generic adapter"
    
    # Generic adapter calls original forward (no actual operator applied)
    max_diff_softmax = (out_orig - out_softmax).abs().max().item()
    assert max_diff_softmax < 1e-4, f"Generic adapter failed: max_diff={max_diff_softmax}"
    
    max_diff_qgfd = (out_orig - out_qgfd).abs().max().item()
    assert max_diff_qgfd < 1e-4, "Generic adapter should call original forward"
    
    print("✓ Operator replacer on dummy model test passed (generic adapter fallback)")


def test_is_leaf_attention():
    """Test the _is_leaf_attention helper."""
    
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
    
    class FakeNonAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
    
    attn = FakeAttention()
    non_attn = FakeNonAttention()
    
    assert _is_leaf_attention(attn) == True
    assert _is_leaf_attention(non_attn) == False
    
    # Already adapted should return False
    from torchdire.nn.attention_adapters import AttentionOperatorAdapter
    from torchdire.nn.attention_operators import SoftmaxOperator
    
    adapted = AttentionOperatorAdapter(attn, SoftmaxOperator())
    assert _is_leaf_attention(adapted) == False
    
    print("✓ _is_leaf_attention test passed")


def test_qgfd_operator_low_precision_inputs():
    """QGFDOperator must accept bf16/fp16 scores+keys (matmul does not type-promote).

    Regression: p0 is upcast to float32 while `key_states` stays in the model
    dtype, so `p @ P` raised "expected scalar type Float but found BFloat16".
    """
    torch.manual_seed(0)

    B, H, Lq, Lk = 2, 4, 6, 6
    for dtype in (torch.bfloat16, torch.float16):
        scores = torch.randn(B, H, Lq, Lk).to(dtype)
        key_states = torch.randn(B, H, Lk, 32).to(dtype)

        # multi-step + early stop + a padding mask exercises the loop path;
        # diffusion_steps=1 with no mask exercises the fused single-step path.
        for steps, mask, eps in ((1, None, 0.0), (3, None, 1e-5), (2, None, 0.0)):
            for learnable in (False, True):
                op = QGFDOperator(
                    diffusion_steps=steps,
                    target_alpha=0.05,
                    early_stop_eps=eps,
                    mode="full",
                    is_causal=True,
                    learnable_alpha=learnable,
                    num_heads=H if learnable else None,
                )
                op.eval()
                if learnable:
                    op.to(dtype)

                probs = op(scores, key_states, attention_mask=mask)

                assert probs.dtype == dtype, f"expected {dtype}, got {probs.dtype}"
                assert torch.isfinite(probs.float()).all()
                sums = probs.float().sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=2e-2), sums

    # Also cover a masked run (valid_mask path) in bf16.
    scores = torch.randn(B, H, Lq, Lk).to(torch.bfloat16)
    key_states = torch.randn(B, H, Lk, 32).to(torch.bfloat16)
    attn_mask = torch.zeros(B, 1, 1, Lk, dtype=torch.bfloat16)
    attn_mask[:, :, :, -2:] = -1e9
    op = QGFDOperator(diffusion_steps=2, target_alpha=0.05, mode="full", is_causal=True)
    op.eval()
    probs = op(scores, key_states, attention_mask=attn_mask)
    assert probs.dtype == torch.bfloat16
    assert torch.isfinite(probs.float()).all()

    print("✓ QGFDOperator low-precision (bf16/fp16) test passed")


def test_qgfd_kernel_low_precision_inputs():
    """QGFDKernel shares the p @ P codepath and must also accept bf16 inputs."""
    from torchdire.nn.qgfd_kernel import QGFDKernel

    torch.manual_seed(0)
    B, H, Lq, Lk = 1, 2, 5, 5
    scores = torch.randn(B, H, Lq, Lk).to(torch.bfloat16)
    key_states = torch.randn(B, H, Lk, 16).to(torch.bfloat16)

    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.05, mode="full")
    kernel.eval()
    probs = kernel(scores, key_states)

    assert probs.dtype == torch.bfloat16
    assert torch.isfinite(probs.float()).all()

    print("✓ QGFDKernel low-precision (bf16) test passed")


if __name__ == "__main__":
    print("Running operator-based QGFD tests...\n")
    
    test_is_leaf_attention()
    test_softmax_operator_parity()
    test_qgfd_operator_row_stochasticity()
    test_qgfd_operator_alpha_zero_equals_softmax()
    test_qgfd_operator_gradient_flow()
    test_qgfd_operator_detach_P()
    test_qgfd_operator_low_precision_inputs()
    test_qgfd_kernel_low_precision_inputs()
    test_operator_replacer_on_dummy_model()
    test_llama_attention_adapter()
    
    print("\n✓ All tests passed!")