import torch
import torch.nn.functional as F
from torchdire.kernels.fused_qgfd import fused_conv_diffusion, TRITON_AVAILABLE
from torchdire.nn.qgfd_kernel import QGFDKernel


def test_fused_conv_diffusion_correctness():
    B, H, Lq, Lk = 2, 4, 16, 16
    scores = torch.randn(B, H, Lq, Lk)
    p0 = F.softmax(scores, dim=-1)
    kernel_tensor = torch.tensor([0.2, 0.6, 0.2]).view(1, 1, 3)
    alpha = 0.05
    steps = 2

    # Execute fused conv diffusion
    p_fused = fused_conv_diffusion(p0, kernel_tensor, alpha=alpha, steps=steps)

    assert p_fused.shape == (B, H, Lq, Lk)
    assert not torch.isnan(p_fused).any()
    assert (p_fused >= 0.0).all()
    assert torch.allclose(p_fused.sum(dim=-1), torch.ones_like(p_fused.sum(dim=-1)), atol=1e-5)


def test_fused_conv_diffusion_dtypes():
    B, H, Lq, Lk = 1, 2, 8, 8
    scores = torch.randn(B, H, Lq, Lk)
    kernel_tensor = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        p0 = F.softmax(scores.to(dtype=dtype), dim=-1)
        p_out = fused_conv_diffusion(p0, kernel_tensor.to(dtype=torch.float32), alpha=0.02, steps=1)
        assert p_out.dtype == dtype
        assert not torch.isnan(p_out).any()


if __name__ == "__main__":
    test_fused_conv_diffusion_correctness()
    test_fused_conv_diffusion_dtypes()
    print("All kernel tests passed successfully!")
