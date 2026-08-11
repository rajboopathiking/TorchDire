import gc
import traceback
import torch
import torch.nn as nn
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.llama_qgfd import HAS_LLAMA, LlamaAttention, patch_llama_with_qgfd


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
        diffusion_steps: int = 2,
        target_alpha: float = 0.02,
        warmup_steps: int = 0,
        kernel_size: int = 5,
        early_stop_eps: float = 0.0,
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

        # Infer num_heads
        num_heads = getattr(orig_mod, "n_heads", None) or getattr(orig_mod, "num_heads", None)
        if num_heads is None:
            q_proj = getattr(orig_mod, "q_proj", None) or getattr(orig_mod, "q", None)
            if q_proj is not None and hasattr(q_proj, "weight"):
                q_out_dim = q_proj.weight.shape[0]
                for h in [32, 24, 16, 12, 8, 4, 2]:
                    if q_out_dim % h == 0:
                        num_heads = h
                        break
        if num_heads is None:
            num_heads = 8

        proj_dim = embed_dim

        qgfd = QGFDLayerClass(
            embed_dim=embed_dim,
            num_heads=num_heads,
            proj_dim=proj_dim,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            kernel_size=kernel_size,
            early_stop_eps=early_stop_eps,
            **qgfd_kwargs,
        )
        object.__setattr__(self, "qgfd", qgfd)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "head_dim", proj_dim // num_heads)

        # Copy original attributes
        for attr in dir(orig_mod):
            if attr.startswith("_") or attr in ("qgfd", "_orig", "num_heads", "embed_dim", "head_dim"):
                continue
            try:
                val = getattr(orig_mod, attr)
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
                if w.shape == out_w.shape:
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

        if p_kv is not None and isinstance(p_kv, (list, tuple)) and len(p_kv) == 2:
            prev_k, prev_v = p_kv
            k_new = self.qgfd.k_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            v_new = self.qgfd.v_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

            full_k = torch.cat([prev_k, k_new], dim=2)
            full_v = torch.cat([prev_v, v_new], dim=2)
            present = (full_k, full_v)

            Q = self.qgfd.q_proj(hidden_states).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.einsum("bhqd,bhkd->bhqk", Q, full_k) / math.sqrt(self.head_dim)

            p = self.qgfd.kernel(scores=scores, key_states=full_k, attention_mask=attention_mask)

            attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, full_v)
            attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.embed_dim)
            attn_output = self.qgfd.out_proj(attn_output_raw)

            if output_attentions:
                return attn_output, present, p
            return attn_output, present

        kv_input = key_value_states if key_value_states is not None else hidden_states

        attn_out_tuple = self.qgfd(
            hidden_states,
            kv=kv_input,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_out_tuple[0]
        attn_probs = attn_out_tuple[1] if output_attentions and len(attn_out_tuple) > 1 else None

        if output_attentions:
            attn_weights = attn_probs.mean(dim=1) if (attn_probs is not None and attn_probs.dim() == 4) else attn_probs
            return attn_output, None, attn_weights
        return attn_output, None


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
    if hasattr(mod, "qgfd") and hasattr(mod, "_orig"):
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
    diffusion_steps: int = 2,
    target_alpha: float = 0.02,
    warmup_steps: int = 0,
    kernel_size: int = 5,
    early_stop_eps: float = 0.0,
    verbose: bool = True,
    auto_eval: bool = True,
    **qgfd_kwargs,
) -> nn.Module:
    """
    Recursively replaces attention modules in model with QGFD.
    For Llama models, uses zero-overhead subclassing (patch_llama_with_qgfd).
    For other architectures, uses SafeWrappedAttention.
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

        if hasattr(orig, "qgfd") and hasattr(orig, "_orig"):
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
