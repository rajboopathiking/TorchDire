import torch
import pytest
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


def test_eval_uses_full_alpha_regardless_of_step():
    # Regression: at inference step_count is 0 (the callback only runs during
    # training), so the old logic computed alpha=0 and the diffusion was
    # silently DISABLED during generation of a QGFD-trained model. Eval mode
    # must use target_alpha regardless of step_count.
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 16, 32
    for mode in ("full", "conv"):
        kernel = QGFDKernel(
            diffusion_steps=2,
            target_alpha=0.05,
            warmup_steps=20000,
            mode=mode,
            kernel_size=5,
            is_causal=True,
        )
        kernel.eval()  # step_count untouched, alpha must still be full
        assert kernel.get_alpha() == 0.05
        assert kernel.get_alpha() == 0.05  # stable across calls

        scores = torch.randn(B, H, L, L)
        K = torch.randn(B, H, L, d)
        out = kernel(scores=scores, key_states=K, attention_mask=None)
        assert out.shape == (B, H, L, L)
        # diffusion is active (alpha=0.05), so P != plain softmax
        assert not torch.allclose(out, torch.softmax(scores, dim=-1), atol=1e-6)

    # train mode keeps the warmup schedule
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.05, warmup_steps=20000)
    kernel.train()
    assert kernel.get_alpha() == 0.0


def test_callback_fires_on_both_step_hooks():
    # TRL>=1.0 skips on_step_begin in its training loop; the callback must also
    # advance the step from on_optimizer_step, otherwise the warmup schedule
    # silently stays at alpha=0 during training.
    from torchdire import QGFDStepCallback

    class Args:
        pass

    class Control:
        pass

    class State:
        global_step = 0

    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.02, warmup_steps=20000)
    cb = QGFDStepCallback([kernel])
    state = State()

    cb.on_optimizer_step(Args(), state, Control())
    state.global_step += 1
    cb.on_step_begin(Args(), state, Control())
    state.global_step += 1
    cb.on_optimizer_step(Args(), state, Control())

    # step advanced through both hooks without the warmup footgun warning
    assert kernel.get_alpha() > 0.0
    assert not hasattr(kernel, "_warned_step_control")


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


def test_warning_training_mode_only_fires_without_grad():
    # The training-mode warning exists to catch inference without model.eval().
    # Real training always runs with grad enabled and must stay quiet.
    import warnings

    torch.manual_seed(0)
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.05, warmup_steps=0, is_causal=True)
    kernel.train()
    scores = torch.randn(2, 4, 8, 8, requires_grad=True)
    K = torch.randn(2, 4, 8, 8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kernel(scores=scores, key_states=K, attention_mask=None).sum().backward()
        assert not any("training=True" in str(x.message) for x in w), \
            "must not warn during real training (grad enabled)"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.no_grad():
            kernel(scores=scores.detach(), key_states=K, attention_mask=None)
        assert any("training=True" in str(x.message) for x in w), \
            "must warn for inference without model.eval()"


def test_warning_step_control_requires_no_callback():
    # The warmup step-control warning must fire only when NO QGFDStepCallback
    # is registered (register_qgfd_step_callback sets the class flag); a
    # callback-driven training loop must never see it.
    import warnings
    from torchdire import QGFDKernel as KClass

    torch.manual_seed(0)
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.05, warmup_steps=100, is_causal=True)
    kernel.train()
    scores = torch.randn(2, 4, 8, 8, requires_grad=True)
    K = torch.randn(2, 4, 8, 8)

    try:
        KClass._qgfd_callback_registered = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel(scores=scores, key_states=K, attention_mask=None)
            assert any("no QGFDStepCallback" in str(x.message) for x in w), \
                "must warn when no callback drives the warmup"

        KClass._qgfd_callback_registered = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kernel(scores=scores, key_states=K, attention_mask=None)
            assert not any("no QGFDStepCallback" in str(x.message) for x in w), \
                "must stay quiet when a callback is registered"
    finally:
        KClass._qgfd_callback_registered = False


def test_dump_learned_alphas_json():
    import json
    import tempfile

    from torchdire import dump_learned_alphas, patch_llama_with_qgfd
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    patch_llama_with_qgfd(
        model, diffusion_steps=2, target_alpha=0.05, warmup_steps=0,
        learnable_alpha=True, verbose=False,
    )
    # heads 0..2 of layer 1 stay, head 3 collapses toward softmax
    layer1 = model.model.layers[1].self_attn.qgfd
    with torch.no_grad():
        layer1.alpha_param.copy_(torch.tensor([0.05, 0.04, 0.03, 0.0]))

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/alphas.json"
        out = dump_learned_alphas(model, path)
        assert set(out.keys()) == {"0", "1"}
        assert out["1"]["mean"] == pytest.approx(0.03, abs=1e-6)
        assert out["1"]["nonzero"] == 3
        assert out["1"]["head_alphas"] == pytest.approx([0.05, 0.04, 0.03, 0.0], abs=1e-6)
        with open(path) as f:
            assert json.load(f) == out


def test_mixed_dtype_scores_fp32_keys_bf16():
    # Regression: real bf16 training (42dot 1.3B + bf16 autocast) upcasts
    # scores to fp32 but passes bf16 key_states; the transition matrix then
    # mismatched p0's dtype and torch.matmul raised
    # "expected scalar type Float but found BFloat16" in the diffusion loop.
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 16, 32
    for mode in ("full", "conv"):
        kernel = QGFDKernel(
            diffusion_steps=2,
            target_alpha=0.05,
            warmup_steps=0,
            mode=mode,
            kernel_size=5,
            is_causal=True,
        )
        kernel.train()
        scores = torch.randn(B, H, L, L, requires_grad=True)  # fp32 (attention wrapper upcast)
        K = torch.randn(B, H, L, d, dtype=torch.bfloat16, requires_grad=True)  # model dtype
        out = _checkpoint_step(kernel, scores, K)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    # reverse mix: bf16 scores with fp32 keys must also work
    kernel = QGFDKernel(diffusion_steps=2, target_alpha=0.05, warmup_steps=0, is_causal=True)
    kernel.train()
    scores = torch.randn(B, H, L, L, dtype=torch.bfloat16, requires_grad=True)
    K = torch.randn(B, H, L, d, requires_grad=True)
    out = _checkpoint_step(kernel, scores, K)
    assert torch.isfinite(out).all()


def test_learnable_alpha_trains_per_head():
    # Per-head selectivity hypothesis: heads must be able to LEARN their own
    # alpha (self-select diffusion), so alpha_param must receive gradients
    # through the checkpointed diffusion and move under an optimizer.
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 16, 32
    kernel = QGFDKernel(
        diffusion_steps=2,
        target_alpha=0.05,
        warmup_steps=0,
        learnable_alpha=True,
        num_heads=H,
        is_causal=True,
    )
    kernel.train()
    assert kernel.alpha_param.shape == (H,)

    W = torch.rand(B, H, L, L)  # fixed weights: out.sum() is alpha-invariant (row mass is constant)

    def step():
        scores = torch.randn(B, H, L, L)
        K = torch.randn(B, H, L, d)
        out = torch.utils.checkpoint.checkpoint(
            lambda: kernel(scores=scores, key_states=K, attention_mask=None),
            use_reentrant=False,
        )
        (out * W).sum().backward()
        return out

    for _ in range(3):
        out = step()
        assert kernel.alpha_param.grad is not None and torch.isfinite(kernel.alpha_param.grad).all()
        with torch.no_grad():
            kernel.alpha_param.sub_(0.05 * kernel.alpha_param.grad)
        kernel.alpha_param.grad = None
        assert torch.isfinite(kernel.alpha_param).all()

    # per-head alpha is used as [1, H, 1, 1] and at least one head diverged
    a = kernel.get_alpha()
    assert a.shape == (1, H, 1, 1)
    assert (a != a[0, 0].item()).any() or (a != 0.05).any()


def test_learnable_alpha_eval_and_unfreeze():
    from torchdire import unfreeze_qgfd_alpha

    torch.manual_seed(0)
    H = 4
    kernel = QGFDKernel(
        diffusion_steps=2,
        target_alpha=0.05,
        warmup_steps=20000,
        learnable_alpha=True,
        num_heads=H,
        is_causal=True,
    )
    kernel.train()
    assert kernel.get_alpha().shape == (1, H, 1, 1)

    # freeze (simulating prepare_model_for_kbit_training / PEFT) then unfreeze
    kernel.alpha_param.requires_grad_(False)
    assert kernel.alpha_param.requires_grad is False
    n = unfreeze_qgfd_alpha(kernel)
    assert n == 1
    assert kernel.alpha_param.requires_grad is True

    # eval mode: full learned alpha, step-independent
    kernel.eval()
    a = kernel.get_alpha()
    assert a.shape == (1, H, 1, 1)
    assert torch.allclose(a[0, :, 0, 0], kernel.alpha_param.clamp(-kernel.max_alpha, kernel.max_alpha))