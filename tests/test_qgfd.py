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


if __name__ == "__main__":
    test_qgfd_full_mode()
    test_qgfd_conv_mode()
    test_gating_attention()
    print("All QGFD layer unit tests passed successfully!")
