import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from torchdire.benchmarks.tradeoff import compare_qgfd_vs_softmax, run_single_benchmark


class MockTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = [hash(w) % (self.vocab_size - 4) + 4 for w in text.split()]
        if not tokens:
            tokens = [1, 2, 3]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": input_ids}
        return {"input_ids": tokens}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(f"tok_{t}" for t in token_ids)


def test_compare_qgfd_vs_softmax_tradeoff():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    tokenizer = MockTokenizer(vocab_size=100)

    summary = compare_qgfd_vs_softmax(
        model=model,
        tokenizer=tokenizer,
        diffusion_steps=2,
        target_alpha=0.02,
        max_new_tokens=10,
        verbose=False,
    )

    assert "baseline" in summary
    assert "qgfd" in summary
    assert "tradeoff" in summary

    assert "ppl_clean" in summary["baseline"]
    assert "ppl_noisy" in summary["baseline"]
    assert "tokens_per_sec" in summary["baseline"]

    assert "robustness_improvement_percent" in summary["tradeoff"]
    assert "latency_overhead_percent" in summary["tradeoff"]
    assert "tps_speed_ratio" in summary["tradeoff"]


if __name__ == "__main__":
    test_compare_qgfd_vs_softmax_tradeoff()
    print("Tradeoff endpoint unit test passed successfully!")
