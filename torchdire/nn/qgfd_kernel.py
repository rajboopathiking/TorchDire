import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class QGFDKernel(nn.Module):
    """
    Query-Graph Flow Diffusion (QGFD) Kernel.

    Operates on raw attention scores (logits QK^T / sqrt(d)) and Key states.
    It replaces standard softmax(scores) with diffusion-regularized attention probabilities.

    Inputs:
        scores: Attention scores of shape (B, H, Lq, Lk)
        key_states: Key projections of shape (B, H_k, Lk, head_dim)
        attention_mask: Optional attention mask (additive or boolean)
        head_mask: Optional head mask of shape (H,) or (1, H, 1, 1)

    Outputs:
        attention_probs: Diffusion-regularized probabilities of shape (B, H, Lq, Lk)
    """

    # Set True by register_qgfd_step_callback once any trainer drives the
    # warmup schedule; suppresses the "no callback" training warning.
    _qgfd_callback_registered = False

    def __init__(
        self,
        diffusion_steps: int = 4,
        target_alpha: float = 0.02,
        warmup_steps: int = 20000,
        early_stop_eps: float = 1e-5,
        detach_P: bool = False,
        temp: float = 1.0,
        mode: str = "full",
        kernel_size: int = 5,
        enable_qgfd: bool = True,
        max_alpha: float = 0.10,
        max_full_seq_len: int = 512,
        full_fallback_mode: str = "disable",
        mask_threshold: float = -1e4,
        debug: bool = False,
        learnable_alpha: bool = False,
        num_heads: Optional[int] = None,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)
        self.detach_P = bool(detach_P)
        self.temp = float(temp) if temp > 0.0 else 1.0

        assert mode in ("full", "conv"), f"mode must be 'full' or 'conv', got {mode}"
        self.mode = mode

        self.enable_qgfd = bool(enable_qgfd)
        self.max_alpha = float(max_alpha)
        self.max_full_seq_len = int(max_full_seq_len)
        assert full_fallback_mode in ("disable", "conv"), f"full_fallback_mode must be 'disable' or 'conv'"
        self.full_fallback_mode = full_fallback_mode
        self.mask_threshold = float(mask_threshold)
        self.debug = bool(debug)
        self.learnable_alpha = bool(learnable_alpha)
        self.num_heads = num_heads
        self.is_causal = bool(is_causal)

        if self.learnable_alpha:
            assert num_heads is not None, "num_heads must be provided when learnable_alpha=True"
            self.alpha_param = nn.Parameter(torch.full((num_heads,), fill_value=float(target_alpha)))

        # Step count buffer for warmup schedule.
        # NOTE: this must NEVER be mutated inside forward(). The trainer drives
        # it via set_step() (see QGFDStepCallback) so that a gradient-checkpoint
        # recomputation replays forward() with the exact same alpha schedule.
        # Incrementing it in forward() makes the warmup branch diverge between
        # the forward pass and its recompute (alpha flips from 0 to ~1e-6,
        # switching the diffusion branch) and raises
        # torch.utils.checkpoint.CheckpointError: "Recomputed values ... have
        # different metadata than during the forward pass".
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # Conv mode parameter setup
        if self.mode == "conv" or self.full_fallback_mode == "conv":
            assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd >= 1"
            self.kernel_size = kernel_size
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size)),
            )

    def set_step(self, step: int) -> None:
        """Externally set the global training step driving the alpha warmup.

        The trainer must call this once per optimizer step, outside the model
        forward. Keeping the step external is what makes the QGFD kernel
        gradient-checkpoint safe: forward and recompute then see the same
        alpha, so they build the identical computational graph.
        """
        self.step_count.fill_(int(step))

    def get_alpha(self) -> Union[float, torch.Tensor]:
        """Calculate effective alpha based on step_count and max_alpha bound."""
        if not self.training:
            # Warmup is a training-schedule concept: at inference the
            # diffusion runs at full strength (target_alpha / learned alpha)
            # regardless of step_count, matching the behavior the model was
            # trained toward. Otherwise generation with a fresh model would
            # see step_count=0 -> alpha=0 -> diffusion silently disabled.
            if self.learnable_alpha:
                return torch.clamp(self.alpha_param, -self.max_alpha, self.max_alpha).view(1, -1, 1, 1)
            return float(max(-self.max_alpha, min(self.max_alpha, self.target_alpha)))

        if self.warmup_steps <= 0:
            factor = 1.0
        else:
            factor = min(1.0, float(self.step_count.item()) / float(self.warmup_steps))

        if self.learnable_alpha:
            alpha = self.alpha_param * factor
            return torch.clamp(alpha, -self.max_alpha, self.max_alpha).view(1, -1, 1, 1)

        alpha = self.target_alpha * factor
        return float(max(-self.max_alpha, min(self.max_alpha, alpha)))

    @staticmethod
    def _eps(x: torch.Tensor) -> float:
        return 1e-3 if x.dtype in (torch.float16, torch.bfloat16) else 1e-6

    def build_transition_from_keys(self, K: torch.Tensor, target_heads: Optional[int] = None, is_causal: Optional[bool] = None) -> torch.Tensor:
        """
        Build key-based row-stochastic transition matrix P from key projections.
        Args:
            K: (B, H_k, Lk, head_dim)
            target_heads: Number of query heads H (repeats K heads if GQA/MQA).
            is_causal: If True, applies lower-triangular causal masking to P.
                       Defaults to self.is_causal (False) so key transitions remain unmasked across
                       already-cached past keys, preventing probability sinks at position 0.
        Returns:
            P: (B, H, Lk, Lk) transition matrix
        """
        if is_causal is None:
            is_causal = getattr(self, "is_causal", False)
        B, H_k, Lk, head_dim = K.shape
        if target_heads is not None and H_k != target_heads:
            repeat_factor = target_heads // H_k
            K = torch.repeat_interleave(K, repeat_factor, dim=1)

        K_norm = F.normalize(K, p=2, dim=-1, eps=self._eps(K))
        # torch.matmul (not torch.einsum): einsum's internal batch-matmul path
        # flattens batch dims to 3D views whose saved-tensor metadata can differ
        # between a forward pass and its gradient-checkpoint recompute
        # (use_reentrant=False), raising "Recomputed values ... different metadata".
        # matmul on 4D operands always saves the native 4D tensors.
        sim = torch.matmul(K_norm, K_norm.transpose(-1, -2))
        
        if is_causal:
            causal_mask = torch.tril(torch.ones((Lk, Lk), device=K.device, dtype=torch.bool))
            sim = sim.masked_fill(~causal_mask[None, None, :, :], torch.finfo(sim.dtype).min)
            
        P = F.softmax(sim, dim=-1)

        # 2. Isolate Position 0 (BOS / Attention Sink): Position 0 only transitions to itself
        # This prevents the LLM's attention sink weight at Pos 0 (often 80%+) from polluting semantic key diffusion
        if Lk > 1:
            P_row0 = torch.zeros_like(P[:, :, :1, :])
            P_row0[:, :, 0, 0] = 1.0
            P = torch.cat([P_row0, P[:, :, 1:, :]], dim=2)

        jitter = self._eps(P)
        P = P * (1.0 - jitter) + (jitter / P.size(-1))

        if self.detach_P:
            P = P.detach()

        return P

    def diffuse_via_conv(
        self,
        p0: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        alpha_eff: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Perform 1D local conv-based diffusion over the key sequence dimension."""
        B, H, Lq, Lk = p0.shape
        p = p0
        prev_p = None

        if not hasattr(self, "conv_kernel") or self.conv_kernel is None:
            kernel_size = getattr(self, "kernel_size", 5)
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size).to(device=p0.device, dtype=p0.dtype)),
            )

        kernel = self.conv_kernel.to(device=p0.device, dtype=p0.dtype)
        K_size = kernel.shape[-1]

        for _ in range(self.diffusion_steps):
            x = p.view(B * H * Lq, 1, Lk)
            x_padded = F.pad(x, (K_size - 1, 0))
            x_conv = F.conv1d(x_padded, kernel, groups=1)
            p_conv = x_conv.view(B, H, Lq, Lk)

            p_conv = p_conv.clamp(min=self._eps(p_conv))
            if valid_mask is not None:
                p_conv = p_conv * valid_mask.to(p_conv.dtype)

            Z = p_conv.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_conv))
            p_conv = p_conv / Z

            p_next = (1.0 - alpha_eff) * p0 + alpha_eff * p_conv
            if valid_mask is not None:
                p_next = p_next * valid_mask.to(p_next.dtype)
            Z = p_next.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_next))
            p_next = p_next / Z

            if prev_p is not None and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                p = p_next
                break

            prev_p = p
            p = p_next

        return p

    def apply_attention_mask(self, scores: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply boolean or additive attention mask to raw score logits."""
        if attention_mask is None:
            return scores

        if attention_mask.dtype == torch.bool:
            additive = (~attention_mask).to(scores.dtype) * -1e9
        else:
            additive = attention_mask.to(scores.dtype)

        if additive.dim() == 2:
            additive = additive[:, None, None, :]

        return scores + additive

    def _build_valid_mask(self, scores: torch.Tensor, p0: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            else:
                valid = attention_mask > -1e4
            if valid.all():
                return None
            return valid
            
        mask = (scores > self.mask_threshold) & (p0 > 1e-12)
        if mask.all():
            return None
        return mask

    def forward(
        self,
        scores: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion-regularized attention probabilities.

        Args:
            scores: (B, H, Lq, Lk) raw attention scores
            key_states: (B, H_k, Lk, head_dim) key tensors
            attention_mask: Optional mask tensor
            head_mask: Optional head mask

        Returns:
            attention_probs: (B, H, Lq, Lk)
        """
        if self.training and not torch.is_grad_enabled() and not hasattr(self, "_warned_training_mode"):
            # Real training always runs with grad enabled, so this only fires
            # for inference/generation paths that forgot model.eval() — the
            # alpha warmup schedule and dropout are training-mode-dependent.
            import warnings
            warnings.warn(
                "QGFDKernel.forward called with module.training=True. If you're doing "
                "inference/generation, call model.eval() first — dropout and the alpha "
                "warmup schedule are both training-mode-dependent.",
                stacklevel=2,
            )
            self._warned_training_mode = True

        if self.debug:
            print(
                "[QGFD]",
                "training=", self.training,
                "grad=", torch.is_grad_enabled(),
                "scores=", tuple(scores.shape),
                "dtype=", scores.dtype,
                "device=", scores.device,
            )

        # Shape invariant: scores and the returned probabilities must ALWAYS
        # be [B, H, Lq, Lk]. Any internal flattening (e.g. [B*H, Lq, Lk] for
        # the conv kernel) must be restored before returning; a shape change
        # between forward and checkpoint recomputation is rejected by
        # torch.utils.checkpoint with "Recomputed values ... different
        # metadata".
        assert scores.dim() == 4, f"QGFD expected scores [B,H,Q,K], got {scores.shape}"
        B, H, Lq, Lk = scores.shape

        # key_states arrives in the model dtype (e.g. bfloat16) while scores are
        # upcast to fp32 by the attention wrapper. Mixing dtypes in the diffusion
        # loop crashes with "expected scalar type Float but found BFloat16" in
        # torch.matmul(p, P). Align K with scores so the transition matrix
        # matches p0's dtype (also gives fp32 numerics for the diffusion).
        key_states = key_states.to(scores.dtype)

        if (
            self.training
            and torch.is_grad_enabled()
            and self.warmup_steps > 0
            and float(self.step_count.item()) == 0.0
            and not QGFDKernel._qgfd_callback_registered
            and not hasattr(self, "_warned_step_control")
        ):
            # Training always starts at step 0, so step==0 alone is not a bug;
            # the warning only means "no QGFDStepCallback is driving the warmup"
            # (register_qgfd_step_callback sets the class flag). Eval-mode
            # forwards are excluded: they use full alpha regardless of step.
            import warnings
            warnings.warn(
                "QGFD warmup_steps>0 but no QGFDStepCallback is registered on the trainer: "
                "the warmup schedule is externally controlled — call kernel.set_step(step) "
                "(or register the QGFDStepCallback) once per optimizer step, otherwise alpha "
                "stays 0 and diffusion never activates during training.",
                stacklevel=2,
            )
            self._warned_step_control = True

        if attention_mask is not None:
            scores = self.apply_attention_mask(scores, attention_mask)

        # Base softmax
        p0 = F.softmax(scores, dim=-1)

        alpha_eff = self.get_alpha()
        alpha_nonzero = (alpha_eff != 0.0).any().item() if isinstance(alpha_eff, torch.Tensor) else abs(alpha_eff) > 0.0
        # `alpha` derives from step_count, which is only changed by the trainer
        # via set_step() OUTSIDE the forward — so forward and checkpoint
        # recomputation always see the same alpha and take the same branch.
        qgfd_active = self.enable_qgfd and self.diffusion_steps > 0 and alpha_nonzero

        if not qgfd_active:
            p = p0
        else:
            valid_mask = self._build_valid_mask(scores, p0, attention_mask)
            mode = self.mode
            if mode == "full" and Lk > self.max_full_seq_len:
                if self.full_fallback_mode == "conv":
                    mode = "conv"
                else:
                    p = p0
                    mode = None

            if mode is None:
                p = p0
            elif mode == "full":
                P = self.build_transition_from_keys(key_states, target_heads=H, is_causal=self.is_causal)
                p = p0
                prev_p = None

                for _ in range(self.diffusion_steps):
                    p_next = (1.0 - alpha_eff) * p0 + alpha_eff * torch.matmul(p, P)

                    if valid_mask is not None:
                        p_next = p_next * valid_mask.to(p_next.dtype)
                    Z = p_next.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_next))
                    p_next = p_next / Z

                    # The early-stop break is data-dependent, but it is safe
                    # under checkpointing because alpha is externally fixed:
                    # recomputation reproduces bitwise-identical p_next values
                    # and therefore takes the same branch.
                    if prev_p is not None and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                        p = p_next
                        break

                    prev_p = p
                    p = p_next
            else:
                p = self.diffuse_via_conv(p0, valid_mask, alpha_eff)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            p = p * head_mask

        assert p.dim() == 4 and p.shape == (B, H, Lq, Lk), (
            f"QGFD changed attention shape: expected {(B, H, Lq, Lk)}, got {p.shape}"
        )

        if self.debug:
            print("[QGFD OUT]", "P=", tuple(p.shape), "alpha=", alpha_eff)

        return p


try:
    from transformers import TrainerCallback

    _HAS_TRAINER_CALLBACK = True
except ImportError:
    TrainerCallback = None
    _HAS_TRAINER_CALLBACK = False


if _HAS_TRAINER_CALLBACK:

    class QGFDStepCallback(TrainerCallback):
        """Advance every QGFDKernel's step_count once per optimizer step.

        The step must be set OUTSIDE the model forward: a counter mutated
        inside forward() changes alpha between the forward pass and its
        gradient-checkpoint recomputation, which can flip the diffusion
        branch and crash with
        torch.utils.checkpoint.CheckpointError ("Recomputed values ... have
        different metadata"). Register via `register_qgfd_step_callback`.
        """

        def __init__(self, kernels):
            self.kernels = list(kernels)

        def on_step_begin(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0))
            for kernel in self.kernels:
                kernel.set_step(step)

        def on_optimizer_step(self, args, state, control, **kwargs):
            # TRL>=1.0 no longer invokes on_step_begin in its training loop, so
            # the warmup step would never advance there. on_optimizer_step is
            # called right after each optimizer step (global_step already
            # incremented) and keeps forward/recompute on the same alpha for
            # the NEXT forward. Harmless if both fire (same global_step).
            step = int(getattr(state, "global_step", 0))
            for kernel in self.kernels:
                kernel.set_step(step)


def collect_qgfd_kernels(model) -> list:
    """Return every QGFDKernel instance found in `model` (incl. PEFT wrappers)."""
    kernels = []
    for module in model.modules():
        if isinstance(module, QGFDKernel):
            kernels.append(module)
    return kernels


def dump_learned_alphas(model, path) -> dict:
    """Save each QGFDKernel's learned per-head alpha values to JSON.

    For the per-head selectivity hypothesis this shows which heads/layers kept
    diffusion (|alpha| > 0) after training vs collapsed toward softmax (alpha
    ~ 0). Keys are the layer index parsed from the module path (e.g.
    model.layers.4.self_attn.qgfd -> "4"); any kernel without a layer index in
    its name uses the full module path as key.
    """
    import json
    import re

    out = {}
    for name, module in model.named_modules():
        if isinstance(module, QGFDKernel) and module.learnable_alpha:
            m = re.search(r"layers\.(\d+)", name)
            key = m.group(1) if m else name
            alphas = [float(a) for a in module.alpha_param.detach().cpu()]
            out[key] = {
                "mean": sum(alphas) / len(alphas),
                "nonzero": sum(1.0 for a in alphas if abs(a) > 1e-4),
                "head_alphas": alphas,
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def unfreeze_qgfd_alpha(model) -> int:
    """Make learnable per-head alpha params trainable (requires_grad=True).

    prepare_model_for_kbit_training / PEFT freeze every base-model parameter,
    which would freeze QGFDKernel.alpha_param and silently disable the
    per-head selectivity experiment (hypothesis: heads self-select whether
    diffusion helps). Call AFTER the trainer is created (so PEFT's freeze
    pass has already run and the optimizer will pick up alpha_param).
    Returns the number of unfrozen kernels.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, QGFDKernel) and module.learnable_alpha:
            module.alpha_param.requires_grad_(True)
            n += 1
    return n


def register_qgfd_step_callback(trainer, model):
    """Register a QGFDStepCallback on `trainer` for all QGFD kernels in `model`.

    Required for the alpha warmup schedule to progress during training with
    gradient checkpointing. Returns the callback, or None if the model has no
    QGFD kernels.
    """
    if not _HAS_TRAINER_CALLBACK:
        raise ImportError("register_qgfd_step_callback requires `transformers` (TrainerCallback).")
    kernels = collect_qgfd_kernels(model)
    if not kernels:
        return None
    QGFDKernel._qgfd_callback_registered = True
    callback = QGFDStepCallback(kernels)
    trainer.add_callback(callback)
    return callback
