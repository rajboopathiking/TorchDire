import gc
import inspect
import math
import traceback
import torch
import torch.nn as nn
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.llama_qgfd import HAS_LLAMA, LlamaAttention, patch_llama_with_qgfd

try:  # GPT-2 uses transformers' Conv1D (weights stored [in, out], applied as x @ W^T)
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    _HFConv1D = None


def _is_conv1d(mod: nn.Module) -> bool:
    return _HFConv1D is not None and isinstance(mod, _HFConv1D) or type(mod).__name__ == "Conv1D"

# Plain config-like attributes copied from the original attention module onto
# the wrapper so HF parent layers keep reading the values they expect.
# NEVER copy callables or nn.Module/nn.Parameter values: a copied `forward`
# bound method in the instance __dict__ shadows SafeWrappedAttention.forward
# and silently disables QGFD (the original softmax path runs instead), and
# copied submodules are unregistered plain attributes (not moved by .to(),
# not part of state_dict).
_COPYABLE_ATTRS = (
    "n_head",
    "num_heads",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "embed_dim",
    "d_model",
    "is_causal",
    "causal",
    "is_decoder",
    "attention_dropout",
    "attn_dropout",
    "max_position_embeddings",
    "layer_idx",
    "scale_attn_weights",
    "use_cache",
    "config",
)


class SafeWrappedAttention(nn.Module):
    """
    Universal Wrapper for HuggingFace / PyTorch Attention Modules.

    Preserves state dict keys, original layer attributes, and HuggingFace forward signature:
    (attn_output, present, attn_weights) or (attn_output, present).
    """

    def __init__(
        self,
        orig_mod: nn.Module,
        QGFDLayerClass=MultiHeadQGFDLayer,
        diffusion_steps: int = 1,
        target_alpha: float = 0.05,
        warmup_steps: int = 0,
        kernel_size: int = 5,
        early_stop_eps: float = 0.0,
        max_full_seq_len: int = 512,
        full_fallback_mode: str = "conv",
        **qgfd_kwargs,
    ):
        super().__init__()
        object.__setattr__(self, "_orig", orig_mod)

        # Infer embed_dim
        embed_dim = (
            getattr(orig_mod, "d_model", None)
            or getattr(orig_mod, "embed_dim", None)
            or getattr(orig_mod, "hidden_size", None)
        )
        if embed_dim is None:
            for n, p in orig_mod.named_parameters():
                ln = n.lower()
                if p.ndim == 2 and any(tag in ln for tag in ["q_proj", "q", "in_proj"]):
                    embed_dim = p.shape[1]
                    break
        if embed_dim is None:
            raise RuntimeError(
                f"Cannot infer embed_dim for SafeWrappedAttention on {orig_mod.__class__.__name__}"
            )

        # Infer num_heads. Prefer explicit module attributes (GPT-2 uses n_head,
        # OPT uses num_heads), then config, then the q-projection divisor heuristic.
        num_heads = (
            getattr(orig_mod, "n_head", None)
            or getattr(orig_mod, "num_heads", None)
            or getattr(orig_mod, "num_attention_heads", None)
        )
        if num_heads is None and hasattr(orig_mod, "config"):
            num_heads = (
                getattr(orig_mod.config, "num_attention_heads", None)
                or getattr(orig_mod.config, "num_heads", None)
                or getattr(orig_mod.config, "n_head", None)
            )
        q_proj = getattr(orig_mod, "q_proj", None) or getattr(orig_mod, "q", None)
        if num_heads is None and q_proj is not None and hasattr(q_proj, "weight"):
            q_out_dim = q_proj.weight.shape[0]
            for h in [32, 24, 16, 12, 8, 6, 4, 2]:
                if q_out_dim % h == 0:
                    num_heads = h
                    break
        if num_heads is None:
            num_heads = 8

        # Infer proj_dim from the actual Q projection output width so the
        # head reshape matches the pretrained weights (avoids head scrambling).
        proj_dim = embed_dim
        if q_proj is not None and hasattr(q_proj, "weight"):
            q_out_dim = q_proj.weight.shape[0]
            if q_out_dim % num_heads == 0:
                proj_dim = q_out_dim

        qgfd = QGFDLayerClass(
            embed_dim=embed_dim,
            num_heads=num_heads,
            proj_dim=proj_dim,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            kernel_size=kernel_size,
            early_stop_eps=early_stop_eps,
            max_full_seq_len=max_full_seq_len,
            full_fallback_mode=full_fallback_mode,
            **qgfd_kwargs,
        )
        # Register the QGFD layer as a proper submodule (NOT object.__setattr__:
        # a plain attribute is invisible to .eval()/.to()/state_dict() — the
        # kernel would stay in training mode during inference, weights would
        # never move to GPU, and checkpoints would lose every QGFD parameter).
        self.add_module("qgfd", qgfd)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "head_dim", proj_dim // num_heads)

        # Copy only whitelisted config-like attributes (see _COPYABLE_ATTRS).
        # Copying everything from dir(orig_mod) put `forward` (and q_proj /
        # k_proj / v_proj / out_proj) into this instance's __dict__, so
        # wrapper.forward resolved to the ORIGINAL module's bound method and
        # QGFD never ran. The projection modules are transferred into
        # self.qgfd below instead.
        for attr in _COPYABLE_ATTRS:
            if not hasattr(orig_mod, attr):
                continue
            try:
                val = getattr(orig_mod, attr)
                if not callable(val):
                    object.__setattr__(self, attr, val)
            except Exception:
                pass

        # Transfer projection weights
        with torch.no_grad():
            if hasattr(orig_mod, "c_attn") and hasattr(orig_mod.c_attn, "weight"):
                w = orig_mod.c_attn.weight.data
                b = orig_mod.c_attn.bias.data if getattr(orig_mod.c_attn, "bias", None) is not None else None
                if w.shape[1] == 3 * embed_dim:
                    q_w, k_w, v_w = w.chunk(3, dim=1)
                    self.qgfd.q_proj.weight.copy_(q_w.T)
                    self.qgfd.k_proj.weight.copy_(k_w.T)
                    self.qgfd.v_proj.weight.copy_(v_w.T)
                    if b is not None:
                        q_b, k_b, v_b = b.chunk(3, dim=0)
                        self.qgfd.q_proj.bias.copy_(q_b)
                        self.qgfd.k_proj.bias.copy_(k_b)
                        self.qgfd.v_proj.bias.copy_(v_b)
                elif w.shape[0] == 3 * embed_dim:
                    q_w, k_w, v_w = w.chunk(3, dim=0)
                    self.qgfd.q_proj.weight.copy_(q_w)
                    self.qgfd.k_proj.weight.copy_(k_w)
                    self.qgfd.v_proj.weight.copy_(v_w)
                    if b is not None:
                        q_b, k_b, v_b = b.chunk(3, dim=0)
                        self.qgfd.q_proj.bias.copy_(q_b)
                        self.qgfd.k_proj.bias.copy_(k_b)
                        self.qgfd.v_proj.bias.copy_(v_b)

            if hasattr(orig_mod, "c_proj") and hasattr(orig_mod.c_proj, "weight"):
                w = orig_mod.c_proj.weight.data
                b = orig_mod.c_proj.bias.data if getattr(orig_mod.c_proj, "bias", None) is not None else None
                out_w = self.qgfd.out_proj.weight
                # GPT-2's Conv1D stores weights as [in, out] and computes
                # x @ W^T; nn.Linear stores [out, in] and computes x @ W.
                # Square weights are shape-ambiguous, so use the module class
                # to pick the orientation — a blind shape check silently
                # copies the transposed linear map and corrupts every layer.
                if _is_conv1d(orig_mod.c_proj):
                    out_w.copy_(w.T)
                elif w.shape == out_w.shape:
                    out_w.copy_(w)
                elif w.T.shape == out_w.shape:
                    out_w.copy_(w.T)
                if b is not None and getattr(self.qgfd.out_proj, "bias", None) is not None:
                    self.qgfd.out_proj.bias.copy_(b)

            for src_name, dst_name in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("o", "out_proj"),
                ("out", "out_proj"),
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("out_proj", "out_proj"),
            ]:
                src = getattr(orig_mod, src_name, None)
                if src is not None and hasattr(src, "weight"):
                    dst = getattr(self.qgfd, dst_name, None)
                    if dst is not None and hasattr(dst, "weight"):
                        try:
                            dst.weight.copy_(src.weight.data)
                            if hasattr(src, "bias") and getattr(src, "bias", None) is not None and getattr(dst, "bias", None) is not None:
                                dst.bias.copy_(src.bias.data)
                        except Exception:
                            pass

        self.__class__.__name__ = orig_mod.__class__.__name__

        # Return-tuple layout differs across HF attention classes and the
        # callers unpack positionally, so the wrapper must mirror the
        # original module's convention:
        #   "w3":     always 3, order (output, attn_weights, present)
        #            (OPT, Bloom, GPT-NeoX, T5, GPT-J — attn_weights=None
        #            when not requested)
        #   "cond":   (output, present) or (output, present, attn_weights)
        #            depending on output_attentions (GPT-2, Llama, ...)
        #   "always2":(output, attn_weights) always (DummyAttention, ...)
        _cls_name = type(orig_mod).__name__
        if any(t in _cls_name for t in ("OPT", "Bloom", "NeoX", "T5", "GPTJ")):
            _layout = "w3"
        else:
            try:
                _src = inspect.getsource(orig_mod.forward)
                if "output_attentions" not in _src:
                    _layout = "always2"
                elif "attention_interface" in _src or "cache_position" in _src:
                    # merged modern attention (transformers >= 4.52): the
                    # attention interface always returns (output, attn_weights)
                    _layout = "always2"
                else:
                    _layout = "cond"
            except Exception:
                _layout = "cond"
        object.__setattr__(self, "_return_layout", _layout)

        # Register the original projection names as aliases of the transferred
        # QGFD projections (SHARED weights, not duplicated), so pretrained
        # state dicts keep their original keys (q_proj/k_proj/v_proj/out_proj)
        # and load unchanged into the wrapped model.
        for src_name, dst_name in [
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("out_proj", "out_proj"),
        ]:
            if hasattr(orig_mod, src_name):
                self.add_module(src_name, getattr(self.qgfd, dst_name))

        # All weights have been transferred into self.qgfd; drop the original
        # module so its weights are freed instead of doubling memory.
        object.__delattr__(self, "_orig")

    def _is_causal_self_attn(self) -> bool:
        causal = getattr(self, "is_causal", None)
        if causal is None:
            causal = getattr(self, "causal", None)
        if causal is None:
            cfg = getattr(self, "config", None)
            if cfg is not None:
                causal = bool(getattr(cfg, "is_decoder", False))
        return bool(causal)

    @staticmethod
    def _causal_mask_4d(Lq: int, Lk: int, past_len: int, device, dtype) -> torch.Tensor | None:
        """Additive [1, 1, Lq, Lk] causal mask (finfo.min on future keys).

        past_len is the number of cached keys already present; the absolute
        position of query i is past_len + i, which may attend keys 0..past+i.
        """
        if Lk <= past_len + 1 and Lq == 1:
            return None
        q_abs = torch.arange(past_len, past_len + Lq, device=device)
        k_abs = torch.arange(Lk, device=device)
        future = k_abs[None, :] > q_abs[:, None]  # [Lq, Lk]
        if not future.any():
            return None
        mask = torch.zeros(Lq, Lk, device=device, dtype=dtype)
        mask.masked_fill_(future, torch.finfo(dtype).min)
        return mask[None, None, :, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
        layer_past: tuple | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        p_kv = past_key_value if past_key_value is not None else layer_past
        if p_kv is None:
            p_kv = kwargs.get("past_key_values")  # newer transformers kwarg name

        # transformers >= 4.48 / v5 pass a Cache object (DynamicCache etc.):
        # the attention layer must update it IN PLACE via .update() and
        # receives the full concatenated key/value states back. Legacy
        # transformers pass a per-layer (k, v) 2-tuple. Anything else is
        # treated as "no cache" — silently ignoring a modern Cache object
        # makes every decode step attend only to the current token and the
        # model degenerates into repetition ("foxesessss...").
        modern_cache = False
        if p_kv is not None and not isinstance(p_kv, (list, tuple)):
            if hasattr(p_kv, "self_attention_cache"):  # EncoderDecoderCache
                p_kv = p_kv.self_attention_cache
            if hasattr(p_kv, "update") and getattr(self, "layer_idx", None) is not None:
                modern_cache = True
            else:
                p_kv = None
        legacy_cache = p_kv is not None and isinstance(p_kv, (list, tuple)) and len(p_kv) == 2

        # SDPA-style attention layers receive attention_mask=None from the
        # model because they build their causal mask internally; the QGFD
        # kernel does not, so without this the softmax (and the diffusion)
        # would see future keys and the model collapses to near-uniform
        # logits. Only self-attention over a causal model gets the mask,
        # and it is ANDed with any caller-supplied mask, mirroring the
        # SDPA semantics of is_causal combined with the attention mask.
        eff_mask = attention_mask
        if key_value_states is None and self._is_causal_self_attn():
            Lq = hidden_states.shape[1]
            if legacy_cache:
                Lk = p_kv[0].shape[2] + Lq
                past_len = Lk - Lq
            elif modern_cache:
                past_len = p_kv.get_seq_length()  # before this step's update
                Lk = past_len + Lq
            else:
                Lk = Lq
                past_len = 0
            if eff_mask is None:
                eff_mask = self._causal_mask_4d(Lq, Lk, past_len, hidden_states.device, hidden_states.dtype)
            elif eff_mask.dtype == torch.bool:
                causal_bool = None
                cm = self._causal_mask_4d(Lq, Lk, past_len, hidden_states.device, torch.bool)
                if cm is not None:
                    causal_bool = cm.bool()
                if causal_bool is not None:
                    eff_mask = eff_mask & causal_bool
            else:
                causal_add = self._causal_mask_4d(Lq, Lk, past_len, hidden_states.device, eff_mask.dtype)
                if causal_add is not None:
                    eff_mask = eff_mask + causal_add

        if legacy_cache or modern_cache:
            if legacy_cache:
                prev_k, prev_v = p_kv
            k_out = self.qgfd.k_proj.weight.shape[0]
            kv_heads = k_out // self.head_dim
            k_new = self.qgfd.k_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], kv_heads, self.head_dim).transpose(1, 2)
            v_new = self.qgfd.v_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], kv_heads, self.head_dim).transpose(1, 2)

            if modern_cache:
                full_k, full_v = p_kv.update(k_new, v_new, self.layer_idx)
                present = None  # cache was mutated in place; model ignores layer returns
            else:
                full_k = torch.cat([prev_k, k_new], dim=2)
                full_v = torch.cat([prev_v, v_new], dim=2)
                present = (full_k, full_v)

            Q = self.qgfd.q_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.einsum("bhqd,bhkd->bhqk", Q, full_k) / math.sqrt(self.head_dim)

            p = self.qgfd.kernel(scores=scores, key_states=full_k, attention_mask=eff_mask)

            attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, full_v)
            attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.embed_dim)
            attn_output = self.qgfd.out_proj(attn_output_raw)

            # Return in the original module's tuple layout (see __init__).
            # "cond" mirrors GPT-2 exactly: 2 values unless attentions were
            # requested — newer transformers blocks unpack `attn_output, _ =`
            # positionally and crash on a 3-tuple.
            layout = getattr(self, "_return_layout", "cond")
            if layout == "w3":
                return attn_output, (p if output_attentions else None), present
            if layout == "always2":
                return attn_output, (p if output_attentions else None)
            if output_attentions:
                return attn_output, present, p
            return attn_output, present

        kv_input = key_value_states if key_value_states is not None else hidden_states

        attn_out_tuple = self.qgfd(
            hidden_states,
            kv=kv_input,
            attention_mask=eff_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_out_tuple[0]
        attn_probs = attn_out_tuple[1] if output_attentions and len(attn_out_tuple) > 1 else None

        # The original attention returns a KV cache whenever use_cache=True,
        # including the first (prefill) pass; returning None here poisons the
        # HF cache tuple (next step: past_key_values[0][0] -> NoneType error).
        # OPT goes further: OPTAttention has no use_cache flag and
        # unconditionally returns (key_states, value_states) as the cache
        # when self.is_decoder — even for prefill and even when the caller
        # did not request caching.
        present = None
        if use_cache or getattr(self, "is_decoder", False):
            k_out = self.qgfd.k_proj.weight.shape[0]
            kv_heads = k_out // self.head_dim
            k_new = self.qgfd.k_proj(hidden_states).view(
                hidden_states.shape[0], hidden_states.shape[1], kv_heads, self.head_dim
            ).transpose(1, 2)
            v_new = self.qgfd.v_proj(hidden_states).view(
                hidden_states.shape[0], hidden_states.shape[1], kv_heads, self.head_dim
            ).transpose(1, 2)
            present = (k_new, v_new)

        if output_attentions:
            attn_weights = attn_probs.mean(dim=1) if (attn_probs is not None and attn_probs.dim() == 4) else attn_probs
        else:
            attn_weights = None
        # Return in the original module's tuple layout (see __init__).
        layout = getattr(self, "_return_layout", "cond")
        if layout == "w3":
            return attn_output, attn_weights, present
        if layout == "always2":
            return attn_output, attn_weights
        if output_attentions:
            return attn_output, present, attn_weights
        return attn_output, present


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


def is_leaf_attention(mod: nn.Module) -> bool:
    if hasattr(mod, "qgfd"):
        return False

    clsname = mod.__class__.__name__.lower()
    if "attention" not in clsname and "attn" not in clsname:
        return False

    for a in ("q", "k", "v", "q_proj", "k_proj", "v_proj", "c_attn", "in_proj_weight"):
        if hasattr(mod, a):
            return True
    for n, p in mod.named_parameters(recurse=False):
        ln = n.lower()
        if any(k in ln for k in ["q", "k", "v", "c_attn"]) and "weight" in ln:
            return True
    return False


def wrap_model_with_qgfd(
    model: nn.Module,
    QGFDLayerClass=MultiHeadQGFDLayer,
    diffusion_steps: int = 1,
    target_alpha: float = 0.05,
    warmup_steps: int = 0,
    kernel_size: int = 5,
    early_stop_eps: float = 0.0,
    max_full_seq_len: int = 512,
    full_fallback_mode: str = "conv",
    verbose: bool = True,
    auto_eval: bool = True,
    **qgfd_kwargs,
) -> nn.Module:
    """
    Recursively replaces attention modules in model with QGFD.
    For Llama models, uses zero-overhead subclassing (patch_llama_with_qgfd).
    For other architectures, uses SafeWrappedAttention.

    Defaults match the configuration validated by QGFD_Sanity_Checks
    (diffusion_steps=1, target_alpha=0.05, no warmup): stronger diffusion
    (alpha=0.1, steps=3) degrades eval loss vs softmax while only adding cost.
    max_full_seq_len=512 + full_fallback_mode="conv" keeps QGFD active at long
    context with linear-cost local diffusion instead of silently disabling it.
    """
    gc.collect()

    if hasattr(model, "config"):
        model.config.use_cache = True

    if auto_eval and model.training:
        model.eval()
        if verbose:
            print("[QGFD Replacer] Switched model to evaluation mode (model.eval()).")

    # Check if model contains LlamaAttention
    has_llama_layers = False
    if HAS_LLAMA:
        for mod in model.modules():
            if isinstance(mod, LlamaAttention):
                has_llama_layers = True
                break

    if has_llama_layers:
        if verbose:
            print("[QGFD Replacer] Detected Llama model architecture. Applying direct LlamaQGFDAttention subclassing.")
        return patch_llama_with_qgfd(
            model,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            kernel_size=kernel_size,
            early_stop_eps=early_stop_eps,
            max_full_seq_len=max_full_seq_len,
            full_fallback_mode=full_fallback_mode,
            verbose=verbose,
            auto_eval=auto_eval,
            **qgfd_kwargs,
        )

    candidates = []
    for name, mod in model.named_modules():
        if name and is_leaf_attention(mod):
            candidates.append((name, mod.__class__.__name__))

    if verbose:
        print(f"[QGFD Replacer] Found {len(candidates)} leaf attention layer candidates.")

    instantiated = []
    for name, cls in candidates:
        parts = name.split(".")
        parent = model
        try:
            for p in parts[:-1]:
                parent = _get_attr_or_index(parent, p)
            orig = _get_attr_or_index(parent, parts[-1])
        except Exception:
            continue

        if hasattr(orig, "qgfd"):
            continue

        try:
            wrapper = SafeWrappedAttention(
                orig,
                QGFDLayerClass=QGFDLayerClass,
                diffusion_steps=diffusion_steps,
                target_alpha=target_alpha,
                warmup_steps=warmup_steps,
                kernel_size=kernel_size,
                early_stop_eps=early_stop_eps,
                max_full_seq_len=max_full_seq_len,
                full_fallback_mode=full_fallback_mode,
                **qgfd_kwargs,
            )
            instantiated.append((name, wrapper))
        except Exception as e:
            if verbose:
                print(f"[QGFD Replacer] Could not wrap {name}: {e}")

    replaced = 0
    for name, wrapper in instantiated:
        try:
            _set_submodule(model, name, wrapper)
            replaced += 1
        except Exception as e:
            if verbose:
                print(f"[QGFD Replacer] Failed setting submodule {name}: {e}")

    if verbose:
        print(f"[QGFD Replacer] Successfully replaced {replaced} attention modules with QGFD.")

    return model
