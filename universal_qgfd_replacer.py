import gc
import traceback
import torch
import torch.nn as nn


class SafeWrappedAttention(nn.Module):
    """
    Wrap an attention module by:
    - storing the original module in ._orig
    - creating .qgfd (MultiHeadQGFDLayer)
    - copying public attributes from the original into this wrapper
    - implementing a compatible forward that returns:
        (attn_output, present) or
        (attn_output, present, attn_weights)

    Stealth mode: class name is spoofed to match the original attention class.

    QGFD-specific behavior (mode, enable_qgfd, max_alpha, etc.) is controlled
    via kwargs passed to MultiHeadQGFDLayer from wrap_model_with_qgfd.
    """

    def __init__(
        self,
        orig_mod: nn.Module,
        MultiHeadQGFDLayer,
        diffusion_steps: int = 1,
        target_alpha: float = 0.02,
        warmup_steps: int = 0,
        kernel_size: int = 5,
        early_stop_eps: float = 0.0,
        **qgfd_kwargs,
    ):
        """
        Args:
            orig_mod: original attention module to be wrapped.
            MultiHeadQGFDLayer: class implementing QGFD attention.
            diffusion_steps, target_alpha, warmup_steps, kernel_size, early_stop_eps:
                core hyperparameters for QGFD.
            **qgfd_kwargs:
                Any additional keyword arguments forwarded to MultiHeadQGFDLayer,
                e.g. mode="full"/"conv", enable_qgfd, max_alpha, max_full_seq_len,
                full_fallback_mode, mask_threshold, debug, detach_P, temp, etc.
        """
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
            num_heads = 8  # fallback

        proj_dim = embed_dim

        # Instantiate QGFD (all extra behavior via **qgfd_kwargs)
        qgfd = MultiHeadQGFDLayer(
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

        # Copy public attributes from original module
        for attr in dir(orig_mod):
            if attr.startswith("_"):
                continue
            if attr in ("qgfd", "_orig", "num_heads", "embed_dim", "head_dim"):
                continue
            try:
                val = getattr(orig_mod, attr)
            except Exception:
                continue
            try:
                object.__setattr__(self, attr, val)
            except Exception:
                pass

        # Try to copy projection weights into QGFD
        with torch.no_grad():
            for src_name, dst_name in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("o", "out_proj"),
                ("out", "out_proj"),
            ]:
                src = getattr(orig_mod, src_name, None)
                if src is not None and hasattr(src, "weight"):
                    dst = getattr(self.qgfd, dst_name, None)
                    if dst is not None and hasattr(dst, "weight"):
                        try:
                            dst.weight.copy_(src.weight.data)
                            if hasattr(src, "bias") and getattr(dst, "bias", None) is not None:
                                dst.bias.copy_(src.bias.data)
                        except Exception:
                            pass

            # Fallback by name substring
            for n, p in orig_mod.named_parameters():
                ln = n.lower()
                try:
                    if "q" in ln and "weight" in ln:
                        self.qgfd.q_proj.weight.copy_(p.data)
                    if "k" in ln and "weight" in ln:
                        self.qgfd.k_proj.weight.copy_(p.data)
                    if "v" in ln and "weight" in ln:
                        self.qgfd.v_proj.weight.copy_(p.data)
                    if ("o" in ln or "out" in ln) and "weight" in ln:
                        self.qgfd.out_proj.weight.copy_(p.data)
                except Exception:
                    pass

        # Stealth: spoof class name to look like original attention
        self.__class__.__name__ = orig_mod.__class__.__name__

    def __repr__(self):
        return self._orig.__repr__()

    def _shape_proj_for_present(self, proj_layer, x):
        proj = proj_layer(x)
        if proj.dim() != 3:
            return proj
        b, t, d = proj.shape
        if d % self.num_heads != 0:
            return proj.unsqueeze(1)
        hd = d // self.num_heads
        return proj.view(b, t, self.num_heads, hd).transpose(1, 2)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        position_bias=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        """
        HF-compatible attention forward:
        returns (attn_output, present) or
        (attn_output, present, attn_weights) if output_attentions=True
        """
        kv_input = key_value_states if key_value_states is not None else hidden_states

        attn_output, attn_probs = self.qgfd(
            hidden_states,
            kv=kv_input,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # KV cache
        present = None
        try:
            k_proj = self._shape_proj_for_present(self.qgfd.k_proj, kv_input)
            v_proj = self._shape_proj_for_present(self.qgfd.v_proj, kv_input)
            if (
                past_key_value is not None
                and isinstance(past_key_value, (list, tuple))
                and len(past_key_value) == 2
            ):
                prev_k, prev_v = past_key_value
                try:
                    k = torch.cat([prev_k, k_proj], dim=2)
                    v = torch.cat([prev_v, v_proj], dim=2)
                except Exception:
                    k, v = k_proj, v_proj
            else:
                k, v = k_proj, v_proj
            present = (k, v)
        except Exception:
            present = None

        attn_weights = None
        if output_attentions and attn_probs is not None:
            try:
                attn_weights = attn_probs.mean(dim=1)
            except Exception:
                attn_weights = attn_probs

        if output_attentions:
            return attn_output, present, attn_weights
        else:
            return attn_output, present


# -------- helper traversal --------
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
    if last.isdigit():
        idx = int(last)
        if isinstance(parent, nn.ModuleList):
            parent[idx] = new_mod
            return
        if isinstance(parent, (list, tuple)):
            parent[idx] = new_mod
            return
        setattr(parent, last, new_mod)
    else:
        setattr(parent, last, new_mod)


def is_leaf_attention(mod: nn.Module) -> bool:
    if hasattr(mod, "qgfd") and hasattr(mod, "_orig"):
        return False

    clsname = mod.__class__.__name__.lower()
    if "attention" not in clsname:
        return False

    for a in ("q", "k", "v", "q_proj", "k_proj", "v_proj"):
        if hasattr(mod, a):
            return True
    for n, p in mod.named_parameters(recurse=False):
        ln = n.lower()
        if ("q" in ln and "weight" in ln) or ("k" in ln and "weight" in ln) or ("v" in ln and "weight" in ln):
            return True
    return False


def wrap_model_with_qgfd(
    model: nn.Module,
    MultiHeadQGFDLayer,
    diffusion_steps: int = 1,
    target_alpha: float = 0.02,
    warmup_steps: int = 0,
    kernel_size: int = 5,
    early_stop_eps: float = 0.0,
    verbose: bool = True,
    **qgfd_kwargs,
):
    """
    In-place replacement of attention modules with SafeWrappedAttention+QGFD.
    Returns the same model object (for convenience).

    Args:
        model: nn.Module with attention layers (e.g. GPT-2, OPT, etc.)
        MultiHeadQGFDLayer: QGFD attention implementation class.
        diffusion_steps, target_alpha, warmup_steps, kernel_size, early_stop_eps:
            core QGFD hyperparameters passed to every wrapped attention.
        verbose: if True, prints replacement diagnostics.
        **qgfd_kwargs:
            Extra keyword args forwarded to MultiHeadQGFDLayer via SafeWrappedAttention.
            Examples (depending on your MultiHeadQGFDLayer signature):
                - mode="full" or "conv"
                - enable_qgfd=True/False
                - max_alpha=0.05
                - max_full_seq_len=512
                - full_fallback_mode="conv" or "disable"
                - mask_threshold=-1e4
                - debug=True
                - detach_P=True
                - temp=1.0
            Any unknown kwargs will be passed through; if MultiHeadQGFDLayer
            does not accept them, Python will raise a TypeError at construction,
            making misconfigurations easy to catch.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidates = []
    for name, mod in model.named_modules():
        if not name:
            continue
        try:
            if is_leaf_attention(mod):
                candidates.append((name, mod.__class__.__name__))
        except Exception:
            pass

    if verbose:
        print(f"Leaf attention candidates found: {len(candidates)}")
        for i, (n, cls) in enumerate(candidates[:200]):
            print(f"[{i}] {n} -> {cls}")

    instantiated = []
    already_wrapped = []
    for name, cls in candidates:
        parts = name.split(".")
        parent = model
        try:
            for p in parts[:-1]:
                parent = _get_attr_or_index(parent, p)
            orig = _get_attr_or_index(parent, parts[-1])
        except Exception as e:
            if verbose:
                print("Skip candidate traversal fail:", name, e)
            continue

        if hasattr(orig, "qgfd") and hasattr(orig, "_orig"):
            already_wrapped.append(name)
            if verbose:
                print("Already wrapped, skipping:", name)
            continue

        try:
            wrapper = SafeWrappedAttention(
                orig,
                MultiHeadQGFDLayer=MultiHeadQGFDLayer,
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
                print("Instantiate wrapper failed for", name, e)
                traceback.print_exc()

    if verbose:
        print("Instantiated wrappers count:", len(instantiated))
        print("Already wrapped modules count:", len(already_wrapped))

    replaced = 0
    for name, wrapper in instantiated:
        try:
            _set_submodule(model, name, wrapper)
            replaced += 1
            if verbose:
                print("Replaced:", name)
        except Exception as e:
            if verbose:
                print("Failed to set wrapper at", name, e)
                traceback.print_exc()

    if verbose:
        print("Replacement pass done. Replaced:", replaced)

        print("\n=== REPLACED ATTENTION LAYERS (SUMMARY) ===")
        if instantiated:
            for name, wrapper in instantiated:
                print(f"{name} -> SafeWrappedAttention (Masked as {wrapper.__class__.__name__})")
        else:
            print("(No new layers wrapped this call.)")
        print("=== END OF REPLACED LAYERS ===\n")

    # Verify wrappers
    if instantiated:
        malformed = []
        for name, _ in instantiated:
            parts = name.split(".")
            parent = model
            try:
                for p in parts[:-1]:
                    parent = _get_attr_or_index(parent, p)
                live = _get_attr_or_index(parent, parts[-1])
            except Exception as e:
                malformed.append((name, f"traverse_fail:{e}"))
                continue

            ok_orig = hasattr(live, "_orig")
            ok_qgfd = hasattr(live, "qgfd")
            if not (ok_orig and ok_qgfd):
                malformed.append(
                    (
                        name,
                        {
                            "ok_orig": ok_orig,
                            "ok_qgfd": ok_qgfd,
                            "type": type(live).__name__,
                        },
                    )
                )

        if malformed:
            if verbose:
                print("Malformed wrappers (showing up to 200):")
                for m in malformed[:200]:
                    print(m)
            raise RuntimeError(
                f"Malformed wrappers found: {len(malformed)}. Aborting."
            )
        elif verbose:
            print("All newly created wrappers verified OK.")

    else:
        if verbose:
            if already_wrapped:
                print("No new attention layers were wrapped; model appears already wrapped.")
            else:
                print("No attention layers were wrapped and no existing wrappers were found.")

    return model
