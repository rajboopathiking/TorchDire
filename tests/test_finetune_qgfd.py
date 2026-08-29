"""
Unit tests for scripts/finetune_qgfd.py on a synthetic Llama.

Kept CPU-only and tiny (a 2-layer/64-dim Llama saved to a temp dir, so no large
download): these pin the plumbing the paper's fine-tuning track depends on —
LoRA reaching the live projections, the alpha warmup advancing via the Trainer
callback, and the two arms actually differing.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

from scripts.finetune_qgfd import (  # noqa: E402
    FinetuneConfig,
    apply_quick,
    build_arm_model,
    build_lm_dataset,
    train_arm,
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A synthetic Llama + gpt2 tokenizer saved to disk, usable as a model_id."""
    d = tmp_path_factory.mktemp("tiny_llama")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256,
    ))
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def _cfg(model_dir, out_dir, **over):
    cfg = apply_quick(FinetuneConfig(
        model_id=model_dir, device="cpu", dtype="float32",
        out_dir=str(out_dir), warmup_steps=2, max_steps=4,
    ))
    return cfg if not over else __import__("dataclasses").replace(cfg, **over)


@pytest.fixture(scope="module")
def texts():
    return [f"Paragraph {i}: the quick brown fox jumps over the lazy dog "
            f"while diffusion spreads attention mass across the key graph. " * 4
            for i in range(40)]


def test_build_arm_model_installs_expected_operator(tiny_model_dir, tmp_path):
    from torchdire import SoftmaxOperator, QGFDOperator

    cfg = _cfg(tiny_model_dir, tmp_path)
    for arm, expected in (("softmax", SoftmaxOperator), ("qgfd", QGFDOperator)):
        _, model, device = build_arm_model(arm, cfg)
        ops = {type(m.prob_operator) for m in model.modules()
               if hasattr(m, "prob_operator")}
        assert ops == {expected}, f"arm {arm} got {ops}"
        assert device == "cpu"
        assert model.config.use_cache is False, "use_cache must be off for training"


def test_build_arm_model_rejects_unknown_backend(tiny_model_dir, tmp_path):
    import dataclasses
    cfg = dataclasses.replace(_cfg(tiny_model_dir, tmp_path), backend="nope")
    with pytest.raises(ValueError, match="backend"):
        build_arm_model("qgfd", cfg)


def test_qgfd_operator_is_causal_and_detached(tiny_model_dir, tmp_path):
    """P must be causally masked, or diffusion leaks future keys into the past."""
    cfg = _cfg(tiny_model_dir, tmp_path)
    _, model, _ = build_arm_model("qgfd", cfg)
    from torchdire import collect_qgfd_operators
    ops = collect_qgfd_operators(model)
    assert ops, "no QGFDOperator installed"
    for op in ops:
        assert op.is_causal is True
        assert op.detach_P is True


def test_build_lm_dataset_blocks_and_labels(tiny_model_dir, texts):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ds = build_lm_dataset(tok, texts, block_size=32)
    assert len(ds) > 0
    row = ds[0]
    assert len(row["input_ids"]) == 32
    assert row["labels"] == row["input_ids"], "causal LM labels == inputs"
    assert row["attention_mask"] == [1] * 32


def test_build_lm_dataset_rejects_corpus_smaller_than_block(tiny_model_dir):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    with pytest.raises(RuntimeError, match="too small"):
        build_lm_dataset(tok, ["hi"], block_size=4096)


def test_train_arm_softmax_runs_and_reports(tiny_model_dir, tmp_path, texts):
    cfg = _cfg(tiny_model_dir, tmp_path / "sm")
    res = train_arm("softmax", cfg, texts, texts[:4])

    assert res["kind"] == "softmax"
    assert res["train"]["steps"] == cfg.max_steps
    assert len(res["train"]["losses"]) > 0
    assert res["train"]["trainable_params"] > 0
    assert res["clean_ppl"] > 1.0 and torch.isfinite(torch.tensor(res["clean_ppl"]))
    assert set(res["robustness"]) == set(cfg.noise_rates)
    # Noise must hurt: this is the axis the paper's headline claim lives on.
    assert res["robustness"][0.15] > res["robustness"][0.0]
    assert res["alpha"] == {"active": False}


def test_train_arm_qgfd_warms_alpha_to_target(tiny_model_dir, tmp_path, texts):
    cfg = _cfg(tiny_model_dir, tmp_path / "qg")
    res = train_arm("qgfd", cfg, texts, texts[:4])

    a = res["alpha"]
    assert a["active"] is True
    assert a["n_modules"] >= 1
    assert a["step_count"] >= cfg.warmup_steps, \
        "Trainer callback did not advance step_count — alpha never warmed up"
    assert a["alpha_train_mode"] == pytest.approx(cfg.target_alpha, abs=1e-6)
    assert a["alpha_eval_mode"] == pytest.approx(cfg.target_alpha, abs=1e-6)


def test_arms_actually_differ(tiny_model_dir, tmp_path, texts):
    """
    A silently no-op QGFD patch would make both arms numerically identical.
    Same seed + same data, so any difference must come from the operator.
    """
    cfg_sm = _cfg(tiny_model_dir, tmp_path / "a")
    cfg_qg = _cfg(tiny_model_dir, tmp_path / "b")
    sm = train_arm("softmax", cfg_sm, texts, texts[:4])
    qg = train_arm("qgfd", cfg_qg, texts, texts[:4])
    assert sm["clean_ppl"] != pytest.approx(qg["clean_ppl"], rel=1e-9), \
        "arms are identical — QGFD had no effect"


def test_run_seed_writes_json(tiny_model_dir, tmp_path, monkeypatch, texts):
    """run_seed() end-to-end, with the corpus stubbed so the test stays offline."""
    import scripts.finetune_qgfd as ft
    monkeypatch.setattr(ft, "load_wikitext", lambda n, split="test": texts[:n])

    out = tmp_path / "seedrun"
    res = ft.run_seed(_cfg(tiny_model_dir, out))

    assert set(res["arms"]) == {"softmax", "qgfd"}
    path = out / "finetune_results.json"
    assert path.exists()
    blob = json.loads(path.read_text())
    assert blob["config"]["model_id"] == tiny_model_dir
    assert blob["arms"]["qgfd"]["alpha"]["active"] is True


def test_run_all_seeds_aggregates(tiny_model_dir, tmp_path, monkeypatch, texts):
    import scripts.finetune_qgfd as ft
    monkeypatch.setattr(ft, "load_wikitext", lambda n, split="test": texts[:n])

    out = tmp_path / "multi"
    agg = ft.run_all_seeds(_cfg(tiny_model_dir, out), seeds=(0, 1))

    assert agg["meta"]["n_seeds"] == 2
    assert agg["meta"]["track"] == "finetune"
    assert agg["meta"]["backend"] == "operator"
    assert "clean_ppl" in agg["arms"]["qgfd"]
    # No latency in the fine-tuning track — the aggregator must tolerate that.
    assert "latency" not in agg["arms"]["qgfd"]
    assert "latency_overhead_x" not in agg["paired"]
    assert agg["train"]["qgfd"]["final_loss"]["n"] == 2
    assert (out / "finetune_aggregated.json").exists()
