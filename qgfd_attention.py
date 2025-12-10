import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadQGFDLayer(nn.Module):
    """
    Multi-head QGFD attention layer.

    Modes:
      - mode="full": use full key-based transition matrix P (O(L^2) memory)
      - mode="conv": use local CausalConv1D smoothing over attention weights (O(L·K))

    QGFD can be optionally:
      - disabled entirely (enable_qgfd=False → plain attention)
      - limited to short sequences in "full" mode (via max_full_seq_len + full_fallback_mode)

    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        proj_dim: Dimensionality of projected Q/K/V vectors (defaults to embed_dim).
        diffusion_steps: Number of diffusion steps to apply (>= 0).
        target_alpha: Target diffusion mixing factor α.
        warmup_steps: Steps to linearly ramp up α (for stable training).
        use_bias: Whether to use bias in linear projections.
        early_stop_eps: Threshold to stop diffusion early if updates are small.
        detach_P: Whether to detach transition matrix P from autograd (for full mode).
        temp: Temperature scaling for the key-based transition softmax (for full mode).
        mode: "full" (global key-based P) or "conv" (local conv-based diffusion).
        kernel_size: kernel size for conv mode (must be odd).
        enable_qgfd: if False, skip diffusion and behave like standard attention.
        max_alpha: upper bound for |alpha_eff| (for safety).
        max_full_seq_len: maximum sequence length allowed in mode="full".
        full_fallback_mode:
            - "disable": if Lk > max_full_seq_len, skip QGFD (use p0).
            - "conv":    if Lk > max_full_seq_len, fall back to conv-based diffusion.
        mask_threshold: threshold for interpreting additive masks (<= threshold = masked).
        debug: if True, logs some simple runtime diagnostics (no heavy logging).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        proj_dim: int | None = None,
        diffusion_steps: int = 4,
        target_alpha: float = 0.02,
        warmup_steps: int = 20000,
        use_bias: bool = True,
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
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        assert self.proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.head_dim = self.proj_dim // num_heads

        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)
        self.detach_P = bool(detach_P)
        self.temp = float(temp) if temp > 0.0 else 1.0

        assert mode in ("full", "conv"), "mode must be 'full' or 'conv'"
        self.mode = mode

        self.enable_qgfd = bool(enable_qgfd)
        self.max_alpha = float(max_alpha)
        self.max_full_seq_len = int(max_full_seq_len)
        assert full_fallback_mode in ("disable", "conv"), "full_fallback_mode must be 'disable' or 'conv'"
        self.full_fallback_mode = full_fallback_mode
        self.mask_threshold = float(mask_threshold)
        self.debug = bool(debug)

        # used for alpha warmup (and eval-time control)
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

        # --- conv mode: learnable local kernel (Toeplitz band for P) ---
        if self.mode == "conv" or self.full_fallback_mode == "conv":
            assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd >= 1"
            self.kernel_size = kernel_size
            # Simple symmetric kernel, normalized
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size))  # (out_ch=1, in_ch=1, K)
            )

    # ------------ alpha schedule ------------
    def get_alpha(self) -> float:
        """
        Linear warmup from 0 to target_alpha over warmup_steps,
        then clamp to [-max_alpha, max_alpha] for safety.
        """
        if self.warmup_steps <= 0:
            alpha = self.target_alpha
        else:
            factor = min(1.0, float(self.step_count.item()) / float(self.warmup_steps))
            alpha = self.target_alpha * factor

        # safety clamp
        alpha = max(-self.max_alpha, min(self.max_alpha, alpha))
        return float(alpha)

    # ------------ key-based transition matrix (full mode) ------------
    def build_transition_from_keys(self, K: torch.Tensor) -> torch.Tensor:
        """
        Build key-based transition matrix P from keys.

        Args:
            K: (B, H, Lk, head_dim)

        Returns:
            P: (B, H, Lk, Lk) transition matrix based on cosine similarities.
        """
        # normalize last dim
        K_norm = F.normalize(K, p=2, dim=-1, eps=self._eps(K))  # (B,H,Lk,head_dim)

        # cosine similarity: (B,H,Lk,Lk)
        sim = torch.einsum("bhid,bhjd->bhij", K_norm, K_norm)

        # scale and temperature
        sim = sim / max(1.0, math.sqrt(self.head_dim))
        sim = sim / self.temp

        # softmax to get transitions
        P = F.softmax(sim, dim=-1)

        # tiny jitter for numerical stability
        jitter = self._eps(P)
        P = P * (1.0 - jitter) + (jitter / P.size(-1))

        if self.detach_P:
            P = P.detach()

        return P

    # ------------ conv mode diffusion over p ------------
    def diffuse_via_conv(
        self,
        p0: torch.Tensor,
        attention_mask: torch.Tensor | None,
        alpha_eff: float,
    ) -> torch.Tensor:
        """
        Diffusion using a local CausalConv1D over the key dimension.

        p0: (B,H,Lq,Lk) baseline attention
        returns p: (B,H,Lq,Lk)
        """
        B, H, Lq, Lk = p0.shape
        p = p0
        prev_p = None

        # Prepare key mask for renormalization (True = keep, False = masked)
        key_mask = self._build_key_mask(attention_mask, B, Lk)  # (B,Lk) or None

        # conv1d expects (N, C_in, L)
        kernel = self.conv_kernel  # (1,1,K)
        K = kernel.shape[-1]

        for _ in range(self.diffusion_steps):
            # reshape p to (N = B*H*Lq, C=1, L=Lk)
            x = p.view(B * H * Lq, 1, Lk)

            # causal padding: pad on the left by K-1
            x_padded = F.pad(x, (K - 1, 0))  # (N,1,Lk+K-1)

            x_conv = F.conv1d(x_padded, kernel, groups=1)  # (N,1,Lk)

            # reshape back
            p_conv = x_conv.view(B, H, Lq, Lk)

            # zero-out masked keys if we have a key_mask
            if key_mask is not None:
                km = key_mask.view(B, 1, 1, Lk)  # (B,1,1,Lk)
                p_conv = p_conv * km.to(p_conv.dtype)

            # renormalize per query so distribution sums to 1 over keys
            p_conv = p_conv.clamp(min=self._eps(p_conv))
            Z = p_conv.sum(dim=-1, keepdim=True)  # (B,H,Lq,1)
            # Avoid division by zero, just in case all keys masked
            Z = Z.clamp(min=self._eps(Z))
            p_conv = p_conv / Z

            # mix with original p0
            p_next = (1.0 - alpha_eff) * p0 + alpha_eff * p_conv

            # early stop on small change
            if (
                prev_p is not None
                and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps
            ):
                p = p_next
                break

            prev_p = p
            p = p_next

        return p

    # ------------ mask handling ------------
    def apply_attention_mask(self, scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """
        attention_mask: can be
            - (B, Lk)        bool or additive
            - (B, 1, 1, Lk)  additive (HF style)
            - bool mask      (True = keep, False = mask)
            - additive mask  already in logits space

        Returns:
            scores with additive mask applied.
        """
        if attention_mask is None:
            return scores

        if attention_mask.dtype == torch.bool:
            # bool mask: True = keep, False = mask
            additive = (~attention_mask).to(scores.dtype) * -1e9
        else:
            # assume HF-style additive mask: 0 for keep, large negative for masked
            additive = attention_mask.to(scores.dtype)

        if additive.dim() == 2:
            additive = additive[:, None, None, :]  # (B,1,1,Lk)

        return scores + additive

    def _build_key_mask(self, attention_mask: torch.Tensor | None, B: int, Lk: int) -> torch.Tensor | None:
        """
        Attempt to build a boolean key mask (B,Lk) from attention_mask.

        True = valid key, False = masked-out key.

        This is used in conv mode for zeroing and renormalization. If
        we can't infer a clean key mask, returns None and we trust p0.
        """
        if attention_mask is None:
            return None

        if attention_mask.dtype == torch.bool:
            am = attention_mask
            # am may be (B,Lk) or (B,1,1,Lk)
            if am.dim() == 4:
                am = am[:, 0, 0, :]
            return am

        # additive mask: 0 for keep, large negative for masked.
        am = attention_mask
        if am.dim() == 4:
            am = am[:, 0, 0, :]  # (B,Lk) or (B,seq_len) depending on caller

        if am.dim() != 2 or am.shape[0] != B or am.shape[1] != Lk:
            # shape mismatch; bail out
            return None

        # positions <= mask_threshold are considered masked
        key_mask = ~(am <= self.mask_threshold)
        return key_mask

    # ------------ dtype-aware epsilon ------------
    @staticmethod
    def _eps(x: torch.Tensor) -> float:
        """
        Return a small epsilon appropriate to dtype.
        """
        if x.dtype in (torch.float16, torch.bfloat16):
            return 1e-3
        else:
            return 1e-6

    # ------------ forward ------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states: (B, Lq, D)
            kv: optional key/value input (B, Lk, D). If None, self-attn on hidden_states.
            attention_mask: HF-style attention mask.
            head_mask: optional per-head scaling.
            output_attentions: whether to return attention probabilities.

        Returns:
            If output_attentions:
                (attn_output, p)
            else:
                (attn_output,)
        """
        B, Lq, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        Lk = kv.shape[1]

        # project and reshape to (B, H, L, head_dim)
        Q = (
            self.q_proj(hidden_states)
            .view(B, Lq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lq,hd)
        K = (
            self.k_proj(kv)
            .view(B, Lk, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lk,hd)
        V = (
            self.v_proj(kv)
            .view(B, Lk, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lk,hd)

        # scaled dot-product scores (before softmax)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)

        # apply attention mask
        scores = self.apply_attention_mask(scores, attention_mask)

        # baseline softmax attention distribution
        p0 = F.softmax(scores, dim=-1)  # (B,H,Lq,Lk)

        # decide if QGFD is active
        alpha_eff = self.get_alpha()
        qgfd_active = (
            self.enable_qgfd
            and self.diffusion_steps > 0
            and abs(alpha_eff) > 0.0
        )

        if not qgfd_active:
            p = p0
        else:
            # optionally guard full mode on long sequences
            mode = self.mode
            if mode == "full" and Lk > self.max_full_seq_len:
                if self.full_fallback_mode == "conv":
                    mode = "conv"
                    if self.debug:
                        print(
                            f"[QGFD] full mode disabled for Lk={Lk} > {self.max_full_seq_len}, "
                            f"falling back to conv mode."
                        )
                else:  # "disable"
                    if self.debug:
                        print(
                            f"[QGFD] full mode disabled for Lk={Lk} > {self.max_full_seq_len}, "
                            f"falling back to plain attention."
                        )
                    p = p0
                    mode = None  # skip diffusion

            if mode is None:
                p = p0
            elif mode == "full":
                # key-based global transition
                P = self.build_transition_from_keys(K)  # (B,H,Lk,Lk)
                p = p0
                prev_p = None

                for _ in range(self.diffusion_steps):
                    # p_{t+1} = (1 - alpha) * p0 + alpha * (p_t @ P)
                    p_next = (1.0 - alpha_eff) * p0 + alpha_eff * torch.einsum(
                        "bhqn,bhnm->bhqm", p, P
                    )

                    # early stop on small change
                    if (
                        prev_p is not None
                        and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps
                    ):
                        p = p_next
                        break

                    prev_p = p
                    p = p_next
            else:
                # conv-based local diffusion (CausalConv1D)
                p = self.diffuse_via_conv(p0, attention_mask, alpha_eff)

        # optional per-head scaling
        if head_mask is not None:
            p = p * head_mask.view(1, -1, 1, 1)

        # attention output
        attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, V)  # (B,H,Lq,hd)
        attn_output_raw = (
            attn_output_raw.transpose(1, 2)
            .contiguous()
            .view(B, Lq, self.proj_dim)
        )
        attn_output = self.out_proj(attn_output_raw)  # final projection back to embed_dim

        # increment step counter with overflow protection
        self.step_count += 1
        if self.step_count.item() > 10**12:
            self.step_count.zero_()

        if output_attentions:
            return (attn_output, p)
        return (attn_output,)
