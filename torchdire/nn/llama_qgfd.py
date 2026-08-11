import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaConfig,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    from transformers.cache_utils import Cache
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False
    LlamaAttention = object
    LlamaConfig = object

from torchdire.nn.qgfd_kernel import QGFDKernel


_LLAMA_ATTN_RETURNS_2_TUPLE = False
if HAS_LLAMA:
    try:
        import transformers
        v_parts = [int(p) for p in transformers.__version__.split(".")[:2] if p.isdigit()]
        if len(v_parts) >= 2 and tuple(v_parts) >= (4, 43):
            _LLAMA_ATTN_RETURNS_2_TUPLE = True
        else:
            import inspect
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            decoder_src = inspect.getsource(LlamaDecoderLayer.forward)
            if "hidden_states, _ =" in decoder_src or "hidden_states, self_attn_weights =" in decoder_src:
                _LLAMA_ATTN_RETURNS_2_TUPLE = True
            else:
                attn_src = inspect.getsource(LlamaAttention.forward)
                if "return attn_output, attn_weights\n" in attn_src or "return attn_output, attn_weights\r\n" in attn_src:
                    _LLAMA_ATTN_RETURNS_2_TUPLE = True
    except Exception:
        pass


if HAS_LLAMA:
    class LlamaQGFDAttention(LlamaAttention):
        """
        QGFD Attention module for Hugging Face LLaMA models.

        Subclasses LlamaAttention directly, preserving original q_proj, k_proj, v_proj, o_proj,
        rotary_emb, KV cache, and GQA/MQA handling. Replaces only softmax(attn_scores) with QGFDKernel.
        """

        def __init__(
            self,
            config: LlamaConfig,
            layer_idx: Optional[int] = None,
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
            **kwargs,
        ):
            super().__init__(config, layer_idx=layer_idx)

            # Ensure essential LLaMA attention attributes exist regardless of transformers version or custom config
            num_heads = (
                getattr(config, "num_attention_heads", None)
                or getattr(config, "num_heads", None)
                or getattr(config, "n_head", None)
                or getattr(self, "num_heads", None)
            )
            num_key_value_heads = (
                getattr(config, "num_key_value_heads", None)
                or getattr(config, "num_kv_heads", None)
                or getattr(self, "num_key_value_heads", None)
                or num_heads
            )
            hidden_size = getattr(config, "hidden_size", getattr(self, "hidden_size", None))
            head_dim = (
                getattr(config, "head_dim", None)
                or getattr(self, "head_dim", None)
                or (hidden_size // num_heads if hidden_size and num_heads else None)
            )
            num_key_value_groups = (
                getattr(self, "num_key_value_groups", None)
                or (num_heads // num_key_value_heads if num_heads and num_key_value_heads else 1)
            )

            self.num_heads = num_heads
            self.num_key_value_heads = num_key_value_heads
            self.hidden_size = hidden_size
            self.head_dim = head_dim
            self.num_key_value_groups = num_key_value_groups

            self.qgfd = QGFDKernel(
                diffusion_steps=diffusion_steps,
                target_alpha=target_alpha,
                warmup_steps=warmup_steps,
                early_stop_eps=early_stop_eps,
                detach_P=detach_P,
                temp=temp,
                mode=mode,
                kernel_size=kernel_size,
                enable_qgfd=enable_qgfd,
                max_alpha=max_alpha,
                max_full_seq_len=max_full_seq_len,
                full_fallback_mode=full_fallback_mode,
                mask_threshold=mask_threshold,
                debug=debug,
                learnable_alpha=learnable_alpha,
                num_heads=self.num_heads,
                **kwargs,
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len, _ = hidden_states.size()
            if position_ids is None:
                if cache_position is not None:
                    position_ids = cache_position.unsqueeze(0)
                else:
                    past_len = 0
                    if past_key_value is not None:
                        if hasattr(past_key_value, "get_seq_length"):
                            try:
                                past_len = past_key_value.get_seq_length(self.layer_idx)
                            except TypeError:
                                past_len = past_key_value.get_seq_length()
                        elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
                            past_len = past_key_value[0].shape[-2]
                    position_ids = torch.arange(past_len, past_len + q_len, device=hidden_states.device).unsqueeze(0)

            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                if not hasattr(self, "_fallback_rotary_emb"):
                    head_dim = getattr(self, "head_dim", value_states.shape[-1])
                    max_pos = getattr(self, "max_position_embeddings", 2048)
                    base = getattr(self, "rope_theta", 10000.0)
                    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                    self._fallback_rotary_emb = LlamaRotaryEmbedding(
                        head_dim,
                        max_position_embeddings=max_pos,
                        base=base,
                    ).to(device=value_states.device)
                cos, sin = self._fallback_rotary_emb(value_states, position_ids)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                if isinstance(past_key_value, Cache):
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                    past_key, past_value = past_key_value[0], past_key_value[1]
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)
                    past_key_value = (key_states, value_states)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    if (attention_mask < 0).any():
                        causal_mask = attention_mask[:, None, None, :]
                    else:
                        min_dtype = torch.finfo(query_states.dtype).min
                        causal_mask = (1.0 - attention_mask[:, None, None, :].to(query_states.dtype)) * min_dtype
                else:
                    causal_mask = attention_mask

                Lk = key_states.shape[-2]
                if causal_mask.shape[-1] < Lk:
                    pad_len = Lk - causal_mask.shape[-1]
                    pad = torch.zeros((*causal_mask.shape[:-1], pad_len), device=causal_mask.device, dtype=causal_mask.dtype)
                    causal_mask = torch.cat([causal_mask, pad], dim=-1)
                elif causal_mask.shape[-1] > Lk:
                    causal_mask = causal_mask[..., :Lk]

                scores = attn_weights + causal_mask
            elif q_len > 1:
                min_dtype = torch.finfo(query_states.dtype).min
                causal_mask = torch.full(
                    (q_len, key_states.shape[-2]), fill_value=min_dtype, device=query_states.device, dtype=query_states.dtype
                )
                causal_mask = torch.triu(causal_mask, diagonal=1 + key_states.shape[-2] - q_len).view(1, 1, q_len, key_states.shape[-2])
                scores = attn_weights + causal_mask
            else:
                scores = attn_weights

            # --- QGFD Diffusion kernel replaces standard softmax ---
            attn_weights = self.qgfd(
                scores=scores,
                key_states=key_states,
                attention_mask=None,
            ).to(query_states.dtype)

            if self.training and self.attention_dropout > 0.0:
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=True)
                denom = attn_weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(attn_weights.dtype).eps)
                attn_weights = attn_weights / denom
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            if _LLAMA_ATTN_RETURNS_2_TUPLE:
                return attn_output, attn_weights
            else:
                return attn_output, attn_weights, past_key_value

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                if "config" in self.__dict__ and hasattr(self.__dict__["config"], name):
                    return getattr(self.__dict__["config"], name)
                raise

else:
    LlamaQGFDAttention = None


def patch_llama_with_qgfd(
    model: nn.Module,
    diffusion_steps: int = 4,
    target_alpha: float = 0.02,
    warmup_steps: int = 0,
    verbose: bool = True,
    auto_eval: bool = True,
    **qgfd_kwargs,
) -> nn.Module:
    """
    In-place replace all LlamaAttention layers in a model with LlamaQGFDAttention.
    Reuses existing q_proj, k_proj, v_proj, o_proj, and rotary_emb submodules so that
    LoRA/QLoRA adapters and original projection weights remain fully intact.
    """
    if not HAS_LLAMA:
        raise RuntimeError("Hugging Face transformers (LlamaAttention) is not installed.")

    if auto_eval and model.training:
        model.eval()
        if verbose:
            print("[QGFD Patch] Switched model to evaluation mode (model.eval()).")

    replaced_count = 0
    updated_count = 0
    model_rotary = getattr(model, "rotary_emb", None)
    if model_rotary is None and hasattr(model, "model"):
        model_rotary = getattr(model.model, "rotary_emb", None)

    for name, module in list(model.named_modules()):
        if isinstance(module, LlamaAttention):
            if isinstance(module, LlamaQGFDAttention):
                # Update QGFD parameters if layer is already patched
                for k, v in qgfd_kwargs.items():
                    if hasattr(module.qgfd, k):
                        setattr(module.qgfd, k, v)
                module.qgfd.diffusion_steps = diffusion_steps
                module.qgfd.target_alpha = target_alpha
                module.qgfd.warmup_steps = warmup_steps
                updated_count += 1
                continue

            layer_idx = getattr(module, "layer_idx", None)
            param_sample = next(module.parameters(), None)
            device = param_sample.device if param_sample is not None else torch.device("cpu")
            dtype = param_sample.dtype if param_sample is not None else torch.float32

            new_attn = LlamaQGFDAttention(
                config=module.config,
                layer_idx=layer_idx,
                diffusion_steps=diffusion_steps,
                target_alpha=target_alpha,
                warmup_steps=warmup_steps,
                **qgfd_kwargs,
            ).to(device=device, dtype=dtype)
            new_attn.train(module.training)

            # Copy all original layer attributes to preserve model-specific config or transformers overrides
            for attr in [
                "num_heads",
                "num_key_value_heads",
                "head_dim",
                "num_key_value_groups",
                "hidden_size",
                "max_position_embeddings",
                "rope_theta",
                "attention_dropout",
                "is_causal",
                "layer_idx",
            ]:
                if hasattr(module, attr):
                    setattr(new_attn, attr, getattr(module, attr))

            # Reuse original modules (preserves LoRA / QLoRA adapters & weight references!)
            new_attn.q_proj = module.q_proj
            new_attn.k_proj = module.k_proj
            new_attn.v_proj = module.v_proj
            new_attn.o_proj = module.o_proj
            if hasattr(module, "rotary_emb"):
                new_attn.rotary_emb = module.rotary_emb
            elif model_rotary is not None:
                new_attn.rotary_emb = model_rotary

            parent_name, _, child_name = name.rpartition(".")
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, new_attn)
            replaced_count += 1

    if verbose:
        if replaced_count > 0:
            print(f"[QGFD Patch] Successfully replaced {replaced_count} LlamaAttention layers with LlamaQGFDAttention.")
        elif updated_count > 0:
            print(f"[QGFD Patch] Model was already patched; updated parameters for {updated_count} LlamaQGFDAttention layers.")
    return model
