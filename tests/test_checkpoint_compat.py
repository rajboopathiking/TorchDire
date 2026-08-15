import torch
from torchdire.nn.qgfd_kernel import QGFDKernel
from torchdire.nn.qgfd import MultiHeadQGFDLayer

# Regression tests for the gradient-checkpointing crash seen during LoRA SFT:
# torch.utils.checkpoint.CheckpointError - "Recomputed values ... have
# different metadata than during the forward pass".
#
# Root cause: step_count was incremented inside QGFDKernel.forward(), so a
# checkpoint recomputation ran with step_count+1, alpha flipped from 0 to
# ~1e-6, the diffusion branch flipped, and forward/recompute built different
# computational graphs. The fix makes the step externally controlled
# (kernel.set_step() / QGFDStepCallback) and enforces a strict [B, H, Q, K]
# shape invariant, so forward and recompute replay identical ops AND values.


def _checkpoint_step(kernel, scores, K):
    def run():
        return kernel(scores=scores, key_states=K, attention_mask=None)

    out = torch.utils.checkpoint.checkpoint(run, use_reentrant=False)
    out.sum().backward()
    return out


def test_qgfd_kernel_checkpoint_compat():
    torch.manual_seed(0)
    B, H, L, d = 4, 32, 32, 64  # B*H = 128, head_dim 64 (SFT/LoRA workload shape)
    for warmup in (0, 20000):
        for mode in ("full", "conv"):
            kernel = QGFDKernel(
                diffusion_steps=3,
                target_alpha=0.02,
                warmup_steps=warmup,
                mode=mode,
                is_causal=True,
            )
            if warmup > 0:
                kernel.set_step(20000)  # trainer drives the step outside forward
            kernel.train()
            scores = torch.randn(B, H, L, L, requires_grad=True)
            K = torch.randn(B, H, L, d, requires_grad=True)

            out = _checkpoint_step(kernel, scores, K)
            assert out.shape == (B, H, L, L)
            assert scores.grad is not None and torch.isfinite(scores.grad).all()
            # "full" mode uses K for the transition matrix; "conv" mode never touches K.
            if mode == "full":
                assert K.grad is not None and torch.isfinite(K.grad).all()


def test_warmup_without_set_step_skips_safely():
    # warmup>0 with no external step: alpha stays 0, diffusion is skipped in
    # BOTH forward and recompute, so checkpointing must still work and the
    # output must be exactly the plain softmax path (p0).
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 16, 32
    kernel = QGFDKernel(diffusion_steps=3, target_alpha=0.02, warmup_steps=20000, is_causal=True)
    kernel.train()
    scores = torch.randn(B, H, L, L, requires_grad=True)
    K = torch.randn(B, H, L, d, requires_grad=True)

    out = _checkpoint_step(kernel, scores, K)
    p0 = torch.softmax(scores.detach(), dim=-1)
    assert torch.allclose(out, p0, atol=1e-6)
    assert K.grad is None  # K was never used


def test_recompute_exactness_with_external_step():
    # The externally-controlled step must make checkpoint recomputation
    # EXACT: identical alpha in forward and recompute -> bitwise-identical
    # values and gradients, for both warmup=0 and an active warmup schedule.
    torch.manual_seed(0)
    B, H, L, d = 2, 8, 16, 32
    for warmup in (0, 20000):
        kernel = QGFDKernel(
            diffusion_steps=2, target_alpha=0.02, warmup_steps=warmup, is_causal=True
        )
        if warmup > 0:
            kernel.set_step(10000)
        kernel.train()

        scores = torch.randn(B, H, L, L, requires_grad=True)
        K = torch.randn(B, H, L, d, requires_grad=True)

        out_ref = kernel(scores=scores, key_states=K, attention_mask=None)
        out_ref.sum().backward()

        scores2 = scores.detach().clone().requires_grad_(True)
        K2 = K.detach().clone().requires_grad_(True)
        out_ckpt = _checkpoint_step(kernel, scores2, K2)

        assert torch.allclose(out_ckpt, out_ref, atol=1e-6)
        assert torch.allclose(scores2.grad, scores.grad, atol=1e-6)
        assert torch.allclose(K2.grad, K.grad, atol=1e-6)


def test_set_step_controls_alpha():
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.02, warmup_steps=20000)
    assert kernel.get_alpha() == 0.0
    kernel.set_step(10000)
    assert abs(kernel.get_alpha() - 0.01) < 1e-9  # 0.02 * 10000/20000
    kernel.set_step(20000)
    assert abs(kernel.get_alpha() - 0.02) < 1e-9
    kernel.set_step(10**9)
    assert abs(kernel.get_alpha() - 0.02) < 1e-9  # capped at target_alpha


def test_multibead_qgfd_checkpoint_compat():
    torch.manual_seed(0)
    layer = MultiHeadQGFDLayer(
        embed_dim=64, num_heads=4, mode="full", diffusion_steps=2,
        target_alpha=0.02, warmup_steps=0,
    )
    layer.train()
    x = torch.randn(2, 16, 64, requires_grad=True)

    def run():
        return layer(x)[0]

    out = torch.utils.checkpoint.checkpoint(run, use_reentrant=False)
    out.sum().backward()
    assert out.shape == (2, 16, 64)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_qgfd_kernel_grad_stability_with_detach_P():
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 12, 16
    for detach_P in (True, False):
        kernel = QGFDKernel(
            diffusion_steps=2,
            target_alpha=0.02,
            warmup_steps=0,
            detach_P=detach_P,
            is_causal=True,
        )
        kernel.train()
        scores = torch.randn(B, H, L, L, requires_grad=True)
        K = torch.randn(B, H, L, d, requires_grad=True)
        out = _checkpoint_step(kernel, scores, K)
        assert torch.isfinite(out).all()