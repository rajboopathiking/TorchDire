import torch
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention


def test_qgfd_full_mode():
    layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode="full", diffusion_steps=2, target_alpha=0.02)
    x = torch.randn(2, 16, 64)
    out = layer(x)[0]
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_qgfd_conv_mode():
    layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode="conv", diffusion_steps=2, target_alpha=0.02)
    x = torch.randn(2, 16, 64)
    out = layer(x)[0]
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_gating_attention():
    layer = QGFDMultiHeadAttention(embed_dim=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    out = layer(x)[0]
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_baseline_equivalence():
    layer_std = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, enable_qgfd=False)
    layer_zero = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, enable_qgfd=True, target_alpha=0.0, warmup_steps=0)
    layer_zero.load_state_dict(layer_std.state_dict(), strict=False)

    x = torch.randn(2, 16, 64)
    out_std, p_std = layer_std(x, output_attentions=True)
    out_zero, p_zero = layer_zero(x, output_attentions=True)

    assert torch.allclose(out_std, out_zero, atol=1e-6)
    assert torch.allclose(p_std, p_zero, atol=1e-6)


def test_probability_distribution_properties():
    for mode in ["full", "conv"]:
        layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=3, target_alpha=0.04, warmup_steps=0)
        x = torch.randn(2, 16, 64)
        _, p = layer(x, output_attentions=True)

        row_sums = p.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        assert (p >= -1e-7).all()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()


def test_dtype_precision_stability():
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        for mode in ["full", "conv"]:
            layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=2, target_alpha=0.02, warmup_steps=0).to(dtype=dt)
            x = torch.randn(2, 16, 64, dtype=dt)
            out, p = layer(x, output_attentions=True)
            assert out.shape == (2, 16, 64)
            assert not torch.isnan(out).any()
            assert not torch.isnan(p).any()


def test_autograd_gradient_flow():
    for mode in ["full", "conv"]:
        layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = layer(x)[0]
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


def test_edge_case_inputs():
    edge_cases = [
        torch.zeros(2, 16, 64),
        torch.randn(2, 16, 64) * 1e4,
        torch.randn(2, 16, 64) * -1e4,
        torch.randn(2, 1, 64),  # single token
    ]
    for x in edge_cases:
        for mode in ["full", "conv"]:
            layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
            out, p = layer(x, output_attentions=True)
            assert not torch.isnan(out).any()
            assert not torch.isnan(p).any()


def test_sequence_length_fallbacks():
    # Fallback to conv
    layer_conv = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode="full", max_full_seq_len=16, full_fallback_mode="conv", diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
    x = torch.randn(2, 32, 64)
    out_conv = layer_conv(x)[0]
    assert out_conv.shape == (2, 32, 64)
    assert not torch.isnan(out_conv).any()

    # Fallback to disable
    layer_dis = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode="full", max_full_seq_len=16, full_fallback_mode="disable", diffusion_steps=2, target_alpha=0.02, warmup_steps=0)
    out_dis = layer_dis(x)[0]
    assert out_dis.shape == (2, 32, 64)
    assert not torch.isnan(out_dis).any()


if __name__ == "__main__":
    test_qgfd_full_mode()
    test_qgfd_conv_mode()
    test_gating_attention()
    test_baseline_equivalence()
    test_probability_distribution_properties()
    test_dtype_precision_stability()
    test_autograd_gradient_flow()
    test_edge_case_inputs()
    test_sequence_length_fallbacks()
    print("All QGFD layer correctness and stability unit tests passed successfully!")

