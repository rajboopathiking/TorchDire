import torch
import torch.nn as nn
from torchdire.utils.replacer import wrap_model_with_qgfd, is_leaf_attention, SafeWrappedAttention


class DummyAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 4
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        return self.out_proj(x), None


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.ModuleDict({"self_attn": DummyAttention()})
        self.layer1 = nn.ModuleDict({"self_attn": DummyAttention()})

    def forward(self, x):
        x, _ = self.layer0["self_attn"](x)
        x, _ = self.layer1["self_attn"](x)
        return x


def test_model_replacer_basic():
    model = DummyModel()
    orig_keys = set(model.state_dict().keys())

    wrapped_model = wrap_model_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, verbose=False)

    x = torch.randn(2, 16, 64)
    out = wrapped_model(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()
    assert hasattr(wrapped_model.layer0["self_attn"], "qgfd")
    assert hasattr(wrapped_model.layer1["self_attn"], "qgfd")


def test_state_dict_preservation():
    model = DummyModel()
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    wrapped_model = wrap_model_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, verbose=False)
    wrapped_state_keys = set(wrapped_model.state_dict().keys())

    # Verify that projection weight names in state dict are preserved
    for orig_key in orig_state.keys():
        assert any(orig_key in wk for wk in wrapped_state_keys), f"Key {orig_key} missing from wrapped state dict!"


def test_gradient_flow_through_wrapper():
    model = DummyModel()
    wrapped_model = wrap_model_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, verbose=False)

    x = torch.randn(2, 16, 64, requires_grad=True)
    out = wrapped_model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    # Check that QGFD projection parameters received gradients
    for name, param in wrapped_model.named_parameters():
        if "qgfd" in name and param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any()


def test_leaf_attention_detection():
    attn = DummyAttention()
    assert is_leaf_attention(attn)
    linear = nn.Linear(64, 64)
    assert not is_leaf_attention(linear)


if __name__ == "__main__":
    test_model_replacer_basic()
    test_state_dict_preservation()
    test_gradient_flow_through_wrapper()
    test_leaf_attention_detection()
    print("All model replacer tests passed successfully!")
