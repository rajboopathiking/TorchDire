import torch
import torch.nn as nn
from torchdire.utils.replacer import wrap_model_with_qgfd


class DummyAttention(nn.Module):

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 4
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        return x, None


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn1 = DummyAttention()
        self.attn2 = DummyAttention()

    def forward(self, x):
        x, _ = self.attn1(x)
        x, _ = self.attn2(x)
        return x


def test_model_replacer():
    model = DummyModel()
    wrapped_model = wrap_model_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, verbose=True)

    x = torch.randn(2, 16, 64)
    out = wrapped_model(x)
    assert out.shape == (2, 16, 64)
    assert hasattr(wrapped_model.attn1, "qgfd")
    assert hasattr(wrapped_model.attn2, "qgfd")


if __name__ == "__main__":
    test_model_replacer()
    print("All model replacer tests passed successfully!")
