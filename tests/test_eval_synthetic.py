"""
Unit tests for scripts/eval_synthetic.py.

The prompt construction is where this track can silently become meaningless — a
misaligned target, an ill-posed corrupted position, or a tokenizer that splits
the "one word = one token" assumption would all still produce a number. These
tests pin the construction against a real tokenizer and check the two probes on a
synthetic model, without needing a large download.
"""
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("transformers")

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

from scripts.eval_synthetic import (  # noqa: E402
    SyntheticConfig,
    aggregate_synthetic,
    apply_quick,
    eval_induction,
    eval_passkey,
    greedy_generate,
    make_induction_examples,
    make_passkey_prompt,
    single_token_vocab,
)


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    ))
    return m.eval()


@pytest.fixture
def cfg():
    return apply_quick(SyntheticConfig(device="cpu", dtype="float32"))


def test_single_token_vocab_words_really_cost_one_token(tok):
    base_ids, ids = single_token_vocab(tok)
    assert len(ids) >= 40, f"only {len(ids)} single-token words found"
    assert len(set(ids)) == len(ids), "duplicate ids returned"
    # Each id must round-trip: base + decoded(id) retokenizes to base_ids + [id].
    for tid in ids[:10]:
        text = tok.decode(base_ids) + tok.decode([tid])
        assert tok(text, add_special_tokens=False)["input_ids"] == base_ids + [tid]


def test_induction_sequence_is_a_doubled_random_sequence(tok, cfg):
    ids, tgt, valid, pos, ctrl = make_induction_examples(tok, cfg, random.Random(0))
    k = cfg.induction_seq_len
    assert ids.shape[0] == cfg.induction_num_examples
    row = ids[0].tolist()
    first, second = row[-2 * k:-k], row[-k:]
    assert first == second, "clean example must repeat S verbatim"
    assert len(set(first)) == k, "S must have no repeated word (match must be unique)"


def test_induction_targets_are_the_successors_in_the_first_copy(tok, cfg):
    ids, tgt, valid, pos, ctrl = make_induction_examples(tok, cfg, random.Random(0))
    row = ids[0].tolist()
    # pos[j] indexes the second copy; its target must be the token that followed
    # the SAME word in the first copy.
    for j, p in enumerate(pos.tolist()):
        query = row[p]
        first_hit = row.index(query)
        assert row[first_hit + 1] == tgt[0, j].item()
        assert ctrl[j].item() + cfg.induction_seq_len == p


def test_induction_noise_corrupts_and_masks_those_positions(tok, cfg):
    ids, tgt, valid, pos, _ = make_induction_examples(
        tok, cfg, random.Random(0), noise_rate=0.5)
    k = cfg.induction_seq_len
    row = ids[0].tolist()
    first, second = row[-2 * k:-k], row[-k:]
    assert first != second, "noise_rate>0 must actually change the second copy"
    assert not bool(valid.all()), "corrupted positions must be masked out"
    # Every scored position that is still valid must have an intact query token.
    for j, p in enumerate(pos.tolist()):
        if valid[0, j]:
            assert row[p] == first[p - (len(row) - k)]


def test_induction_zero_noise_scores_every_position(tok, cfg):
    _, _, valid, _, _ = make_induction_examples(tok, cfg, random.Random(0))
    assert bool(valid.all())


def test_eval_induction_reports_every_noise_rate(tok, model, cfg):
    res = eval_induction(model, tok, "cpu", cfg, micro_batch=2)
    assert res["noise_rates"] == ["0.00", "0.25"]
    assert set(res["by_noise"]) == {"0.00", "0.25"}
    assert res["acc"] == res["by_noise"]["0.00"]["acc"]
    for d in res["by_noise"].values():
        assert 0.0 <= d["acc"] <= 1.0
        assert d["n_predictions"] > 0
    # A random-init model cannot do induction; both scores should be near zero,
    # which is exactly what makes control_acc a usable floor on a real model.
    assert res["acc"] < 0.5


def test_eval_induction_rejects_total_corruption(tok, model, cfg):
    import dataclasses
    bad = dataclasses.replace(cfg, induction_noise_rates=(1.0,))
    with pytest.raises(RuntimeError, match="corrupted every scored position"):
        eval_induction(model, tok, "cpu", bad, micro_batch=2)


def test_passkey_prompt_contains_key_and_ends_with_the_question(tok, cfg):
    for depth in (0.1, 0.5, 0.9):
        prompt, key = make_passkey_prompt(tok, cfg, random.Random(1), depth)
        assert key in prompt
        assert prompt.endswith("The pass key is")
        assert len(key) == cfg.passkey_digits and not key.startswith("0")
        assert prompt.count(key) == 2, "key is stated twice by construction"


def test_passkey_depth_moves_the_key(tok, cfg):
    shallow, k1 = make_passkey_prompt(tok, cfg, random.Random(1), 0.1)
    deep, k2 = make_passkey_prompt(tok, cfg, random.Random(1), 0.9)
    assert k1 == k2, "same rng seed must give the same key"
    assert shallow.index(k1) < deep.index(k2)
    assert abs(len(shallow) - len(deep)) < len(k1) + 80, "total length must be ~fixed"


def test_greedy_generate_is_bounded_and_decodes(tok, model, cfg):
    out = greedy_generate(model, tok, "The pass key is", "cpu", max_new_tokens=5)
    assert isinstance(out, str)
    assert len(tok(out, add_special_tokens=False)["input_ids"]) <= 5


def test_eval_passkey_shape(tok, model, cfg):
    res = eval_passkey(model, tok, "cpu", cfg)
    assert set(res["by_depth"]) == {"0.10", "0.90"}
    assert res["n"] == cfg.passkey_num_examples * len(cfg.passkey_depths)
    assert res["context_tokens"] > 0
    assert 0.0 <= res["acc"] <= 1.0
    for d in res["by_depth"].values():
        assert d["acc"] <= d["contains"], "strict must be <= lenient"


def _fake_run(seed, sm, qg, cfg_dict):
    def arm(kind, ind, ind_noisy, pk):
        return {
            "kind": kind, "post_lora": False,
            "induction": {"acc": ind, "control_acc": 0.01, "n_predictions": 100,
                          "n_examples": 4, "seq_len_tokens": 40,
                          "noise_rates": ["0.00", "0.25"],
                          "by_noise": {"0.00": {"acc": ind},
                                       "0.25": {"acc": ind_noisy}}},
            "passkey": {"acc": pk, "n": 4, "context_tokens": 142,
                        "by_depth": {"0.10": {"acc": pk, "contains": pk, "n": 2},
                                     "0.90": {"acc": pk, "contains": pk, "n": 2}}},
            "operator": {"qgfd_active": kind == "qgfd"},
        }
    return {"config": dict(cfg_dict, seed=seed), "post_lora": False, "device": "cpu",
            "arms": {"softmax": arm("softmax", *sm), "qgfd": arm("qgfd", *qg)}}


def test_aggregate_synthetic_paired_gaps(cfg):
    import dataclasses
    cd = dataclasses.asdict(cfg)
    runs = [_fake_run(0, (0.90, 0.70, 1.0), (0.92, 0.76, 1.0), cd),
            _fake_run(1, (0.80, 0.60, 1.0), (0.82, 0.66, 1.0), cd)]
    agg = aggregate_synthetic(runs, [0, 1])

    assert agg["meta"]["track"] == "synthetic"
    assert agg["meta"]["n_seeds"] == 2
    assert agg["meta"]["operator"]["qgfd"]["qgfd_active"] is True
    assert agg["arms"]["softmax"]["induction_acc"]["mean"] == pytest.approx(0.85)
    # Between-seed spread is 0.05, but the paired gap is a constant 0.02 with zero
    # variance — the whole point of pairing.
    assert agg["arms"]["softmax"]["induction_acc"]["std"] > 0.05
    assert agg["paired"]["induction_gap"]["mean"] == pytest.approx(0.02)
    assert agg["paired"]["induction_gap"]["std"] == pytest.approx(0.0, abs=1e-12)
    assert agg["paired"]["induction_gap_by_noise"]["0.25"]["mean"] == pytest.approx(0.06)
    assert agg["paired"]["passkey_gap"]["mean"] == pytest.approx(0.0)


def test_aggregate_synthetic_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_synthetic([], [])
