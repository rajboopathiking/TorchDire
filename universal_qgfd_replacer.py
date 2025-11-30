# ---------- UNIVERSAL QGFD REPLACER (library-style) ----------
import gc
import traceback
import torch
import torch.nn as nn
# -------------------------------------------------------------
# SafeWrappedAttention
# -------------------------------------------------------------
class SafeWrappedAttention(nn.Module):
    """
    Wrap an attention module by:
    - storing the original module in ._orig
    - creating .qgfd (MultiHeadQGFDLayer)
    - copying public attributes from the original into this wrapper
    - implementing a compatible forward that returns:
        (attn_output, present) or
        (attn_output, present, attn_weights)
    """
    def __init__(
        self,
        orig_mod,
        MultiHeadQGFDLayer,
        diffusion_steps=4,
        target_alpha=0.02,
        warmup_steps=20000,
        detach_P=False,
        temp=1.0,
    ):
        super().__init__()
        # store original module and create qgfd
        object.__setattr__(self, "_orig", orig_mod)

        # -------- infer embed_dim --------
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

        # -------- infer num_heads --------
        num_heads = getattr(orig_mod, "n_heads", None) or getattr(orig_mod, "num_heads", None)

        if num_heads is None:
            # Try to infer from q_proj weight shape
            q_proj = getattr(orig_mod, "q_proj", None) or getattr(orig_mod, "q", None)
            if q_proj is not None and hasattr(q_proj, "weight"):
                q_out_dim = q_proj.weight.shape[0]
                # Try common head counts
                for h in [16, 12, 8, 4, 2]:
                    if q_out_dim % h == 0:
                        num_heads = h
                        break

        if num_heads is None:
            # last resort – this may be wrong for exotic models
            num_heads = 8

        proj_dim = embed_dim

        # instantiate qgfd layer
        object.__setattr__(
            self,
            "qgfd",
            MultiHeadQGFDLayer(
                embed_dim,
                num_heads,
                proj_dim,
                diffusion_steps=diffusion_steps,
                target_alpha=target_alpha,
                warmup_steps=warmup_steps,
                detach_P=detach_P,
                temp=temp,
            ),
        )
        # store metadata
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "head_dim", proj_dim // num_heads)

        # Copy public attributes and callables from orig into self
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
                if isinstance(val, (nn.Module, torch.nn.parameter.Parameter)) or not callable(val):
                    object.__setattr__(self, attr, val)
                else:
                    object.__setattr__(self, attr, val)
            except Exception:
                pass

        # Try to copy projection weights into qgfd's q_proj/k_proj/v_proj/out_proj (best-effort)
        with torch.no_grad():
            # direct attribute mapping
            for src_name, dst_name in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("o", "out_proj"),
                ("out", "out_proj"),
            ]:
                src = getattr(orig_mod, src_name, None)
                if src is not None and hasattr(src, "weight"):
                    try:
                        getattr(self.qgfd, dst_name).weight.copy_(src.weight.data)
                        if hasattr(src, "bias") and getattr(
                            getattr(self.qgfd, dst_name), "bias", None
                        ) is not None:
                            getattr(self.qgfd, dst_name).bias.copy_(src.bias.data)
                    except Exception:
                        pass

            # fallback by parameter name substring
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

    def _shape_proj_for_present(self, proj_layer, x):
        proj = proj_layer(x)
        if proj.dim() != 3:
            # best-effort; don't crash
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
        Return signature compatible with HF attention:
        - (attn_output, present) or
        - (attn_output, present, attn_weights) if output_attentions=True
        """
        kv_input = key_value_states if key_value_states is not None else hidden_states
        attn_output, attn_probs = self.qgfd(
            hidden_states, kv=kv_input, attention_mask=attention_mask
        )

        # present for caching
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


# -------------------------------------------------------------
# Helper traversal functions
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Detect leaf attention modules
# -------------------------------------------------------------
def is_leaf_attention(mod: nn.Module) -> bool:
    clsname = mod.__class__.__name__.lower()
    if "attention" not in clsname:
        return False
    for a in ("q", "k", "v", "q_proj", "k_proj", "v_proj"):
        if hasattr(mod, a):
            return True
    for n, p in mod.named_parameters(recurse=False):
        ln = n.lower()
        if ("q" in ln and "weight" in ln) or ("k" in ln and "weight" in ln) or (
            "v" in ln and "weight" in ln
        ):
            return True
    return False


# -------------------------------------------------------------
# Main public helper: wrap_model_with_qgfd
# -------------------------------------------------------------
def wrap_model_with_qgfd(
    model,
    MultiHeadQGFDLayer,
    diffusion_steps=4,
    target_alpha=0.02,
    warmup_steps=20000,
    detach_P=False,
    temp=1.0,
    verbose=True,
):
    """
    In-place replacement of attention modules with SafeWrappedAttention+QGFD.
    Returns the same model object (for convenience).

    Args:
        model: HF model instance (e.g. AutoModelForSeq2SeqLM.from_pretrained(...))
        MultiHeadQGFDLayer: class implementing QGFD layer
        diffusion_steps, target_alpha, warmup_steps, detach_P, temp: passed to SafeWrappedAttention / QGFD
        verbose: whether to print diagnostics

    Returns:
        model (with attention layers wrapped)
    """
    # clean some garbage in current process (optional)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- snapshot candidates ----------
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

    # ---------- instantiate wrappers (no mutation yet) ----------
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

        # skip if already our wrapper
        if isinstance(orig, SafeWrappedAttention) or orig.__class__.__name__ == SafeWrappedAttention.__name__:
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
                detach_P=detach_P,
                temp=temp,
            )
            instantiated.append((name, wrapper))
        except Exception as e:
            if verbose:
                print("Instantiate wrapper failed for", name, e)
                traceback.print_exc()

    if verbose:
        print("Instantiated wrappers count:", len(instantiated))
        print("Already wrapped modules count:", len(already_wrapped))

    # ---------- set wrappers ----------
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

    # ---------- REPLACED LAYERS SUMMARY ----------
    if verbose:
        print("\n=== REPLACED ATTENTION LAYERS (SUMMARY) ===")
        if instantiated:
            for name, wrapper in instantiated:
                print(f"{name} -> SafeWrappedAttention")
        else:
            print("(No new layers wrapped this call.)")
        print("=== END OF REPLACED LAYERS ===\n")

    # ---------- robust verify (only for newly instantiated wrappers) ----------
    if instantiated:
        malformed = []
        verified = []

        for name, wrapper in instantiated:
            parts = name.split(".")
            parent = model
            try:
                for p in parts[:-1]:
                    parent = _get_attr_or_index(parent, p)
                live = _get_attr_or_index(parent, parts[-1])
            except Exception as e:
                malformed.append((name, f"traverse_fail:{e}"))
                continue

            ok_orig = True
            ok_qgfd = True
            try:
                _orig = getattr(live, "_orig")
            except Exception:
                ok_orig = False
                _orig = None
            try:
                _qgfd = getattr(live, "qgfd")
            except Exception:
                ok_qgfd = False
                _qgfd = None

            if not (ok_orig and ok_qgfd):
                malformed.append(
                    (
                        name,
                        {
                            "ok_orig": ok_orig,
                            "ok_qgfd": ok_qgfd,
                            "type": type(live).__name__,
                            "repr": repr(live)[:400],
                        },
                    )
                )
            else:
                verified.append(
                    (
                        name,
                        {
                            "wrapper_type": type(live).__name__,
                            "orig_type": type(_orig).__name__
                            if _orig is not None
                            else None,
                        },
                    )
                )

        if malformed:
            if verbose:
                print("Malformed wrappers (showing up to 200):")
                for m in malformed[:200]:
                    print(m)
            raise RuntimeError(
                f"Malformed wrappers found: {len(malformed)} (see printed details). Aborting."
            )

        if verbose:
            print("All newly created wrappers verified OK (._orig and .qgfd reachable).")
            if verified:
                print("Verified wrapped attention modules (showing up to 200):")
                for name, info in verified[:200]:
                    print(
                        f"{name} -> wrapper={info['wrapper_type']}  orig={info['orig_type']}"
                    )
            else:
                print("No verified newly wrapped modules were found (this is unexpected).")

    else:
        # No new wrappers; model may already be fully wrapped
        if verbose:
            if already_wrapped:
                print("No new attention layers were wrapped; model appears to be already fully wrapped.")
                print("Existing SafeWrappedAttention modules:")
                for name, mod in model.named_modules():
                    if isinstance(mod, SafeWrappedAttention):
                        print(f"  {name} -> SafeWrappedAttention")
            else:
                print("No attention layers were wrapped and no existing wrappers were found.")

    return model
