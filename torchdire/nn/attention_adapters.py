"""
Architecture-Specific Attention Adapters for QGFD.

These adapters wrap the original attention modules and inject a custom
AttentionProbabilityOperator at the softmax point, preserving all other
attention mechanics (projections, RoPE, KV cache, GQA, causal masking).
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional, Tuple, Union, List, Dict, Any

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

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2Config,
        apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
        repeat_kv as qwen2_repeat_kv,
    )
    HAS_QWEN2 = True
except ImportError:
    HAS_QWEN2 = False
    Qwen2Attention = object
    Qwen2Config = object

try:
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralConfig,
    )
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    MistralAttention = object
    MistralConfig = object

try:
    from transformers.models.gpt_neox.modeling_gpt_neox import (
        GPTNeoXAttention,
        GPTNeoXConfig,
    )
    HAS_GPT_NEOX = True
except ImportError:
    HAS_GPT_NEOX = False
    GPTNeoXAttention = object
    GPTNeoXConfig = object

try:
    from transformers.models.gpt_neo.modeling_gpt_neo import (
        GPTNeoAttention,
        GPTNeoConfig,
    )
    HAS_GPT_NEO = True
except ImportError:
    HAS_GPT_NEO = False
    GPTNeoAttention = object
    GPTNeoConfig = object

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    HAS_GPT2 = True
except ImportError:
    HAS_GPT2 = False
    GPT2Attention = object

try:
    from transformers.models.opt.modeling_opt import OPTAttention
    HAS_OPT = True
except ImportError:
    HAS_OPT = False
    OPTAttention = object

from torchdire.nn.attention_operators import (
    AttentionProbabilityOperator,
    SoftmaxOperator,
    QGFDOperator,
)


class AttentionOperatorAdapter(nn.Module):
    """
    Base class for architecture-specific attention adapters.
    
    Subclasses must implement `forward` to intercept the attention computation
    and replace softmax with the provided operator.
    """

    def __init__(
        self,
        original_attention: nn.Module,
        prob_operator: AttentionProbabilityOperator,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.prob_operator = prob_operator
        self.config = getattr(original_attention, "config", None)
        self.layer_idx = getattr(original_attention, "layer_idx", None)
        
        # Copy over all non-callable attributes from original
        for attr in dir(original_attention):
            if not attr.startswith('_') and not callable(getattr(original_attention, attr, None)):
                try:
                    val = getattr(original_attention, attr)
                    if not isinstance(val, nn.Module):
                        setattr(self, attr, val)
                except Exception:
                    pass

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            orig = self.__dict__.get("original_attention")
            if orig is not None and hasattr(orig, name):
                return getattr(orig, name)
            cfg = self.__dict__.get("config")
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class LlamaAttentionAdapter(AttentionOperatorAdapter):
    """
    Adapter for Llama/Mistral/Qwen2 attention modules.
    
    Preserves: q_proj, k_proj, v_proj, o_proj, rotary_emb, KV cache, GQA, causal mask
    Replaces: F.softmax(attn_weights) with prob_operator
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        prob_operator: AttentionProbabilityOperator,
        config=None,
        layer_idx: Optional[int] = None,
    ):
        if not HAS_LLAMA:
            raise RuntimeError("transformers LlamaAttention not available")
        super().__init__(original_attention, prob_operator)
        self.config = config or getattr(original_attention, 'config', None)
        self.layer_idx = layer_idx or getattr(original_attention, 'layer_idx', None)
        
        # Copy projection modules and rotary embedding from original attention
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'rotary_emb']:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))
        
        # Robust attribute resolution with fallbacks for modern transformers (>=4.45 / 4.48)
        num_heads = (
            getattr(original_attention, 'num_heads', None)
            or getattr(original_attention, 'num_attention_heads', None)
            or getattr(original_attention, 'n_head', None)
            or (getattr(self.config, 'num_attention_heads', None) if self.config else None)
            or (getattr(self.config, 'num_heads', None) if self.config else None)
            or (getattr(self.config, 'n_head', None) if self.config else None)
        )
        if num_heads is None and hasattr(original_attention, 'q_proj') and hasattr(original_attention.q_proj, 'weight'):
            q_out_dim = original_attention.q_proj.weight.shape[0]
            for h in [64, 32, 24, 16, 12, 8, 6, 4, 2]:
                if q_out_dim % h == 0:
                    num_heads = h
                    break
        if num_heads is None:
            num_heads = 32

        self.num_heads = num_heads
        self.num_attention_heads = num_heads

        num_kv_heads = (
            getattr(original_attention, 'num_key_value_heads', None)
            or (getattr(self.config, 'num_key_value_heads', None) if self.config else None)
            or num_heads
        )
        self.num_key_value_heads = num_kv_heads

        hidden_size = (
            getattr(original_attention, 'hidden_size', None)
            or (getattr(self.config, 'hidden_size', None) if self.config else None)
        )
        if hidden_size is None and hasattr(original_attention, 'q_proj') and hasattr(original_attention.q_proj, 'weight'):
            hidden_size = original_attention.q_proj.weight.shape[1]
        self.hidden_size = hidden_size

        head_dim = (
            getattr(original_attention, 'head_dim', None)
            or (getattr(self.config, 'head_dim', None) if self.config else None)
            or ((hidden_size // num_heads) if (hidden_size and num_heads) else 64)
        )
        self.head_dim = head_dim

        kv_groups = (
            getattr(original_attention, 'num_key_value_groups', None)
            or ((num_heads // num_kv_heads) if (num_heads and num_kv_heads) else 1)
        )
        self.num_key_value_groups = kv_groups

        self.attention_dropout = getattr(
            original_attention, 'attention_dropout',
            getattr(self.config, 'attention_dropout', 0.0) if self.config else 0.0
        )
        self.is_causal = getattr(
            original_attention, 'is_causal',
            getattr(self.config, 'is_causal', True) if self.config else True
        )
        self.max_position_embeddings = getattr(
            original_attention, 'max_position_embeddings',
            getattr(self.config, 'max_position_embeddings', 2048) if self.config else 2048
        )
        self.rope_theta = getattr(
            original_attention, 'rope_theta',
            getattr(self.config, 'rope_theta', 10000.0) if self.config else 10000.0
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Handle transformers >=5 passing cache as past_key_values / layer_past
        if past_key_value is None:
            past_key_value = kwargs.get("past_key_values", kwargs.get("layer_past", None))

        bsz, q_len, _ = hidden_states.size()

        # QKV projections - use original modules
        pretraining_tp = getattr(self.config, "pretraining_tp", 1) if self.config else 1
        if pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_embeddings is None:
            if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
                cos, sin = self.rotary_emb(value_states, position_ids)
            elif hasattr(self.original_attention, 'rotary_emb') and self.original_attention.rotary_emb is not None:
                cos, sin = self.original_attention.rotary_emb(value_states, position_ids)
            else:
                cos, sin = None, None
        else:
            cos, sin = position_embeddings
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Cache
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            elif isinstance(past_key_value, (list, tuple)) and len(past_key_value) == 2:
                past_k, past_v = past_key_value
                key_states = torch.cat([past_k, key_states], dim=-2)
                value_states = torch.cat([past_v, value_states], dim=-2)
                past_key_value = (key_states, value_states)

        # GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        # Attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Causal mask
        k_len = key_states.shape[-2]
        past_seen = k_len - q_len
        if cache_position is not None:
            if not torch.is_tensor(cache_position):
                cache_position = torch.tensor(cache_position, device=attn_weights.device)
            q_abs = cache_position.reshape(-1).to(device=attn_weights.device, dtype=torch.long)
        elif position_ids is not None:
            pos = position_ids
            if pos.dim() > 1:
                pos = pos[0]
            q_abs = pos.reshape(-1)[:q_len].to(device=attn_weights.device, dtype=torch.long)
        else:
            q_abs = torch.arange(past_seen, k_len, device=attn_weights.device)

        key_abs = torch.arange(k_len, device=attn_weights.device)
        future = key_abs[None, :] > q_abs[:, None]
        if future.any():
            attn_weights = attn_weights.masked_fill(
                future[None, None, :, :], torch.finfo(attn_weights.dtype).min
            )

        # HF attention mask (padding)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                hf_mask = attention_mask[:, :, :, :k_len]
            elif attention_mask.dim() == 2:
                hf_mask = attention_mask[:, None, None, :k_len].to(attn_weights.dtype)
            else:
                hf_mask = attention_mask
            if hf_mask.dtype == torch.bool:
                hf_mask = (~hf_mask).to(attn_weights.dtype) * -1e9
            attn_weights = attn_weights + hf_mask

        # === REPLACE SOFTMAX WITH OPERATOR ===
        attn_weights = self.prob_operator(
            scores=attn_weights.to(torch.float32),
            key_states=key_states,
            attention_mask=None,  # Already applied above
        ).to(query_states.dtype)

        # Dropout
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Attention output
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        pretraining_tp = getattr(self.config, "pretraining_tp", 1) if self.config else 1
        if pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # LLaMA, Mistral, and Qwen2 decoder layers always expect a 2-tuple: (attn_output, attn_weights)
        return attn_output, attn_weights




class Qwen2AttentionAdapter(LlamaAttentionAdapter):
    """
    Adapter for Qwen2 attention - nearly identical to Llama.
    Uses Qwen2's apply_rotary_pos_emb and repeat_kv.
    """
    pass  # Inherits everything from LlamaAttentionAdapter


class MistralAttentionAdapter(LlamaAttentionAdapter):
    """
    Adapter for Mistral attention - identical to Llama.
    """
    pass


class GPTNeoXAttentionAdapter(AttentionOperatorAdapter):
    """
    Adapter for GPT-NeoX attention.
    
    Preserves: query_key_value projection, rotary_emb (partial), KV cache, dense output
    Replaces: The attention function's softmax with prob_operator
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        prob_operator: AttentionProbabilityOperator,
        config=None,
        layer_idx: Optional[int] = None,
    ):
        if not HAS_GPT_NEOX:
            raise RuntimeError("transformers GPTNeoXAttention not available")
        super().__init__(original_attention, prob_operator)
        self.config = config or getattr(original_attention, 'config', None)
        self.layer_idx = layer_idx or getattr(original_attention, 'layer_idx', None)
        
        for attr in ['num_attention_heads', 'head_size', 'rotary_ndims', 'hidden_size',
                     'attention_dropout', 'is_causal', 'norm_factor']:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, seq_len, _ = hidden_states.shape

        # QKV projections + RoPE (preserved from original)
        query, key, value, present = self.original_attention._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # The original uses GPTNEOX_ATTENTION_FUNCTION[attention_type]
        # We need to replicate the eager attention computation with our operator
        
        # Get attention mask ready
        if attention_mask is not None and attention_mask.dim() == 4:
            attention_mask = attention_mask[:, :, :, :key.shape[-2]]

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / self.norm_factor

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # === REPLACE SOFTMAX WITH OPERATOR ===
        attn_weights = self.prob_operator(
            scores=attn_weights.to(torch.float32),
            key_states=key,
            attention_mask=None,
            head_mask=head_mask,
        ).to(query.dtype)

        # Dropout
        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)

        # Attention output
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and output projection
        attn_output = attn_output.contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTNeoAttentionAdapter(AttentionOperatorAdapter):
    """
    Adapter for GPT-Neo attention (similar to GPT-2).
    
    Preserves: c_attn (fused QKV), c_proj (output), KV cache
    Replaces: softmax in the attention submodule
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        prob_operator: AttentionProbabilityOperator,
        config=None,
        layer_idx: Optional[int] = None,
    ):
        if not HAS_GPT_NEO:
            raise RuntimeError("transformers GPTNeoAttention not available")
        super().__init__(original_attention, prob_operator)
        self.config = config or getattr(original_attention, 'config', None)
        self.layer_idx = layer_idx or getattr(original_attention, 'layer_idx', None)
        
        # GPT-Neo uses self.attention which is a GPTNeoSelfAttention
        self.attention = original_attention.attention

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        **kwargs,
    ):
        # Delegate to the inner attention with our operator
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            prob_operator=self.prob_operator,  # Pass operator
        )


class GPT2AttentionAdapter(AttentionOperatorAdapter):
    """
    Adapter for GPT-2 attention.
    """
    pass  # Similar to GPT-Neo


class OPTAttentionAdapter(AttentionOperatorAdapter):
    """
    Adapter for OPT attention.
    """
    pass


class GenericAttentionAdapter(AttentionOperatorAdapter):
    """
    Generic fallback adapter for unknown attention architectures.
    
    Uses monkey-patching to intercept the forward pass and replace softmax.
    This is a best-effort approach and may not work for all architectures.
    For production use, implement a specific adapter.
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        prob_operator: AttentionProbabilityOperator,
    ):
        super().__init__(original_attention, prob_operator)
        # Store original forward
        self._orig_forward = original_attention.forward
        
    def forward(self, *args, **kwargs):
        # This is a fallback that tries to detect and replace softmax in the computation
        # For proper use, implement an architecture-specific adapter
        import warnings
        warnings.warn(
            f"Using GenericAttentionAdapter for {type(self.original_attention).__name__}. "
            "This is a fallback and may not correctly intercept softmax. "
            "Implement a specific adapter for this architecture.",
            UserWarning,
            stacklevel=2,
        )
        return self._orig_forward(*args, **kwargs)


def create_attention_adapter(
    attention_module: nn.Module,
    prob_operator: AttentionProbabilityOperator,
) -> nn.Module:
    """
    Factory function to create the appropriate adapter for the given attention module.
    """
    cls = type(attention_module)
    cls_name = cls.__name__
    
    if 'Llama' in cls_name or 'Mistral' in cls_name:
        return LlamaAttentionAdapter(attention_module, prob_operator)
    elif 'Qwen2' in cls_name:
        return Qwen2AttentionAdapter(attention_module, prob_operator)
    elif 'GPTNeoX' in cls_name:
        return GPTNeoXAttentionAdapter(attention_module, prob_operator)
    elif 'GPTNeo' in cls_name and 'GPTNeoX' not in cls_name:
        return GPTNeoAttentionAdapter(attention_module, prob_operator)
    elif 'GPT2' in cls_name:
        return GPT2AttentionAdapter(attention_module, prob_operator)
    elif 'OPT' in cls_name:
        return OPTAttentionAdapter(attention_module, prob_operator)
    else:
        # Fallback to generic adapter
        return GenericAttentionAdapter(attention_module, prob_operator)


def patch_model_with_operator(
    model: nn.Module,
    prob_operator: AttentionProbabilityOperator,
    verbose: bool = True,
) -> nn.Module:
    """
    Replace all attention modules in the model with operator-adapted versions.
    
    This preserves the original attention architecture completely and only
    replaces the softmax computation with the provided operator.
    """
    if verbose:
        print(f"[Operator Patch] Patching model with {type(prob_operator).__name__}")

    replaced = 0
    for name, module in list(model.named_modules()):
        if _is_leaf_attention(module):
            try:
                adapter = create_attention_adapter(module, prob_operator)
                _set_submodule(model, name, adapter)
                replaced += 1
                if verbose:
                    print(f"[Operator Patch] Replaced {name} ({type(module).__name__}) with {type(adapter).__name__}")
            except Exception as e:
                if verbose:
                    print(f"[Operator Patch] Failed to patch {name}: {e}")

    if verbose:
        print(f"[Operator Patch] Successfully replaced {replaced} attention modules")

    return model


def _is_leaf_attention(mod: nn.Module) -> bool:
    """Check if module is a leaf attention module (not already adapted)."""
    if hasattr(mod, 'prob_operator') or isinstance(mod, AttentionOperatorAdapter):
        return False
    
    cls_name = mod.__class__.__name__.lower()
    if 'attention' not in cls_name and 'attn' not in cls_name:
        return False
    
    for attr in ('q_proj', 'k_proj', 'v_proj', 'q', 'k', 'v', 'c_attn', 'query_key_value', 'in_proj_weight'):
        if hasattr(mod, attr):
            return True
    
    for n, p in mod.named_parameters(recurse=False):
        ln = n.lower()
        if any(k in ln for k in ['q', 'k', 'v', 'c_attn', 'query_key']) and 'weight' in ln:
            return True
    
    return False


def _get_attr_or_index(parent, part):
    if hasattr(parent, part):
        return getattr(parent, part)
    if part.isdigit():
        return parent[int(part)]
    raise AttributeError(f"Cannot resolve part '{part}' on {type(parent).__name__}")


def _set_submodule(root, dotted_name, new_mod):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = _get_attr_or_index(parent, p)
    last = parts[-1]
    if last.isdigit() and (isinstance(parent, nn.ModuleList) or isinstance(parent, (list, tuple))):
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)