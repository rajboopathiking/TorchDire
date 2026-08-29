"""
Regression test: LoRA must land on the projections the operator adapter ACTUALLY uses.

`LlamaAttentionAdapter` aliases the original q/k/v/o projections onto itself while
also holding the original attention module. Both names pointed at the same Linear,
and nn.Module.named_modules() de-duplicates shared submodules — yielding only the
first name, `original_attention.q_proj`. PEFT builds its target list from
named_modules() and rebinds by setattr, so LoRA was injected into a module the
adapter's forward() never calls: the live projection stayed a frozen nn.Linear,
the adapters were dead weight, and backward raised
"element 0 of tensors does not require grad and does not have a grad_fn".

These tests pin the fix from both ends: the module-tree shape and, more
importantly, an end-to-end check that perturbing lora_B changes the loss.
"""
import torch
import torch.nn as nn
import pytest

from torchdire import QGFDOperator, SoftmaxOperator, wrap_model_with_qgfd_operator
from torchdire.nn.attention_adapters import LlamaAttentionAdapter

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model             # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _tiny_llama():
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    ))


def _qgfd_op():
    return QGFDOperator(diffusion_steps=1, target_alpha=0.05, mode="full",
                        detach_P=True, is_causal=True)


def _wrapped(operator):
    return wrap_model_with_qgfd_operator(_tiny_llama(), operator, verbose=False)


@pytest.mark.parametrize("operator_factory", [_qgfd_op, SoftmaxOperator],
                         ids=["qgfd", "softmax"])
def test_original_attention_is_not_in_the_module_tree(operator_factory):
    model = _wrapped(operator_factory())
    names = [n for n, _ in model.named_modules()]
    assert not any("original_attention" in n for n in names), \
        "original_attention must be unregistered so PEFT targets the live projections"
    # ...but it is still reachable as a plain attribute for delegation.
    attn = model.model.layers[0].self_attn
    assert isinstance(attn, LlamaAttentionAdapter)
    assert attn.original_attention is not None


def test_state_dict_keeps_stock_checkpoint_keys():
    model = _wrapped(_qgfd_op())
    keys = set(model.state_dict())
    for proj in TARGETS:
        assert f"model.layers.0.self_attn.{proj}.weight" in keys, \
            f"{proj} lost its stock checkpoint key"
    assert not any("original_attention" in k for k in keys)


def test_projection_parameters_are_not_duplicated():
    """Aliasing must not make the adapter report the same weight twice."""
    model = _wrapped(_qgfd_op())
    attn = model.model.layers[0].self_attn
    ids = [id(p) for p in attn.parameters()]
    assert len(ids) == len(set(ids)), "duplicate parameter registrations in adapter"


def test_lora_targets_the_live_projections():
    model = _wrapped(_qgfd_op())
    pm = get_peft_model(model, LoraConfig(r=4, target_modules=TARGETS,
                                          task_type="CAUSAL_LM"))
    attn = pm.base_model.model.model.layers[0].self_attn
    for proj in TARGETS:
        mod = getattr(attn, proj)
        assert hasattr(mod, "lora_A"), \
            f"{proj} used by forward() is not a LoRA layer — adapters are dead"


def test_lora_actually_changes_the_loss():
    """
    End-to-end liveness: backward must work, and moving lora_B off zero must
    change the loss. lora_B starts at zeros, so a LoRA layer that is wired in but
    never called is indistinguishable from a live one until you perturb it.
    """
    torch.manual_seed(0)
    model = _wrapped(_qgfd_op())
    pm = get_peft_model(model, LoraConfig(r=4, target_modules=TARGETS,
                                          task_type="CAUSAL_LM"))
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters at all"

    ids = torch.randint(0, 256, (2, 16))
    out = pm(input_ids=ids, labels=ids)
    out.loss.backward()                       # regression: used to raise RuntimeError
    before = out.loss.item()

    for n, p in pm.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0.0, 0.5)
    after = pm(input_ids=ids, labels=ids).loss.item()

    assert abs(after - before) > 1e-6, \
        "perturbing lora_B did not change the loss — LoRA is not on the live path"


def test_adapted_model_still_runs_and_original_module_survives():
    """Unregistering must not break the original module a caller may still hold."""
    torch.manual_seed(0)
    base = _tiny_llama().eval()
    orig_attn = base.model.layers[0].self_attn
    model = wrap_model_with_qgfd_operator(base, _qgfd_op(), verbose=False).eval()

    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=ids).logits).all()
        # The original attention object is intact and independently callable.
        hs = torch.randn(1, 8, 64)
        pos = torch.arange(8).unsqueeze(0)
        out = orig_attn(hs, position_ids=pos)
        assert torch.isfinite((out[0] if isinstance(out, tuple) else out)).all()


def test_unaliased_parametrised_children_keep_the_original_registered():
    """
    An architecture with extra parametrised submodules (q_norm/k_norm style) must
    NOT be silently stripped — those params would vanish from state_dict()/.to().
    The adapter keeps the original registered and warns instead.
    """
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    from transformers.models.llama.modeling_llama import LlamaAttention

    attn = LlamaAttention(cfg, layer_idx=0)
    attn.q_norm = nn.LayerNorm(16)             # unaliased, parametrised

    with pytest.warns(RuntimeWarning, match="q_norm"):
        adapter = LlamaAttentionAdapter(attn, _qgfd_op())

    names = [n for n, _ in adapter.named_modules()]
    assert any("original_attention.q_norm" in n for n in names), \
        "q_norm must stay reachable in the module tree"
