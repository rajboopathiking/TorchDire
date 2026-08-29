"""
Unit tests for the multi-seed aggregation in scripts/review_experiments.py.

These run on synthetic result dicts (no model, no download) so they are fast and
pin down the statistics the paper's headline claim rests on:
  * mean / std / t-based 95% CI arithmetic,
  * per-seed PAIRED qgfd-minus-softmax differences,
  * the sign convention of `robustness_gap_pct` (positive == QGFD more robust).
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.review_experiments import (  # noqa: E402
    ExperimentConfig,
    _stat,
    _signif,
    _rate_key,
    aggregate_runs,
    fmt_stat,
)


def _fake_run(seed, sm_ppls, qg_ppls, rates=(0.0, 0.05, 0.10)):
    """One run_all()-shaped dict. sm_ppls/qg_ppls are per-rate perplexities."""
    cfg = ExperimentConfig(model_id="fake/model", seed=seed)

    def arm(kind, ppls):
        return {
            "kind": kind,
            "clean_ppl": ppls[0],
            "robustness": {float(r): p for r, p in zip(rates, ppls)},
            "attention": {"mean_attention_entropy_nats": 1.4 + 0.01 * seed,
                          "mean_sink_mass_pos0": 0.62 - 0.001 * seed},
            "latency": {"prefill_ms": 100.0 + seed, "tokens_per_s": 5000.0},
            "generation": [],
        }

    return {
        "config": {**cfg.__dict__},
        "device": "cpu",
        "arms": {"softmax": arm("softmax", sm_ppls), "qgfd": arm("qgfd", qg_ppls)},
    }


def test_stat_basic_arithmetic():
    s = _stat([10.0, 12.0, 14.0])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(12.0)
    assert s["std"] == pytest.approx(2.0)                      # sample (n-1) std
    assert s["sem"] == pytest.approx(2.0 / math.sqrt(3))
    assert s["ci95"] == pytest.approx(4.303 * 2.0 / math.sqrt(3), rel=1e-6)
    assert s["values"] == [10.0, 12.0, 14.0]


def test_stat_single_value_has_no_spread():
    s = _stat([7.5])
    assert s["mean"] == pytest.approx(7.5)
    assert s["std"] == 0.0 and s["sem"] == 0.0 and s["ci95"] == 0.0
    assert _signif(s) == "?", "n=1 must not be reported as significant"


def test_signif_marks_ci_excluding_zero():
    assert _signif(_stat([1.0, 1.1, 0.9])) == "*"      # tight, far from 0
    assert _signif(_stat([1.0, -1.0, 0.1])) == "ns"    # straddles 0


def test_fmt_stat_uses_std_by_default_and_ci_on_request():
    s = _stat([10.0, 12.0, 14.0])
    assert fmt_stat(s, 2) == "12.00 +/- 2.00"
    assert fmt_stat(s, 2, ci=True).startswith("12.00 +/- 4.9")


def test_aggregate_runs_shapes_and_means():
    runs = [
        _fake_run(0, [10.0, 12.0, 14.0], [10.1, 11.8, 13.4]),
        _fake_run(1, [11.0, 13.4, 15.6], [11.1, 13.1, 14.9]),
        _fake_run(2, [12.0, 14.6, 17.0], [12.1, 14.4, 16.4]),
    ]
    agg = aggregate_runs(runs, seeds=[0, 1, 2])

    assert agg["meta"]["n_seeds"] == 3
    assert agg["meta"]["seeds"] == [0, 1, 2]
    assert agg["meta"]["model_id"] == "fake/model"
    assert agg["meta"]["noise_rates"] == ["0.0000", "0.0500", "0.1000"]

    sm = agg["arms"]["softmax"]
    assert sm["clean_ppl"]["mean"] == pytest.approx(11.0)
    assert sm["clean_ppl"]["n"] == 3
    assert sm["robustness"]["0.0500"]["mean"] == pytest.approx((12.0 + 13.4 + 14.6) / 3)

    # Degradation is measured against each seed's OWN clean run.
    d = sm["robustness_delta_pct"]["0.0500"]
    expected = [100 * (12.0 - 10.0) / 10.0,
                100 * (13.4 - 11.0) / 11.0,
                100 * (14.6 - 12.0) / 12.0]
    assert d["mean"] == pytest.approx(sum(expected) / 3)
    assert sm["robustness_delta_pct"]["0.0000"]["mean"] == pytest.approx(0.0)

    for k in ("mean_attention_entropy_nats", "mean_sink_mass_pos0"):
        assert sm["attention"][k]["n"] == 3
    assert sm["latency"]["prefill_ms"]["mean"] == pytest.approx(101.0)


def test_paired_gap_is_positive_when_qgfd_degrades_less():
    # QGFD starts slightly worse on clean text but degrades less under noise.
    runs = [
        _fake_run(0, [10.0, 13.0, 16.0], [10.1, 12.6, 15.0]),
        _fake_run(1, [11.0, 14.3, 17.6], [11.1, 13.9, 16.6]),
        _fake_run(2, [12.0, 15.6, 19.2], [12.1, 15.2, 18.1]),
    ]
    agg = aggregate_runs(runs, seeds=[0, 1, 2])

    for rate in ("0.0500", "0.1000"):
        gap = agg["paired"]["robustness_gap_pct"][rate]
        assert gap["n"] == 3
        assert gap["mean"] > 0, f"expected QGFD more robust at {rate}"
    assert "0.0000" not in agg["paired"]["robustness_gap_pct"], \
        "clean rate carries no gap (it is 0 by construction)"

    # Paired clean-PPL penalty: QGFD is consistently ~0.1 worse.
    cp = agg["paired"]["clean_ppl_qgfd_minus_softmax"]
    assert cp["mean"] == pytest.approx(0.1, abs=1e-6)
    assert cp["values"] == pytest.approx([0.1, 0.1, 0.1], abs=1e-6)

    ov = agg["paired"]["latency_overhead_x"]
    assert ov["mean"] == pytest.approx(1.0)


def test_paired_gap_is_negative_when_qgfd_degrades_more():
    runs = [_fake_run(s, [10.0, 12.0, 14.0], [10.0, 12.5, 15.0]) for s in (0, 1)]
    agg = aggregate_runs(runs, seeds=[0, 1])
    assert agg["paired"]["robustness_gap_pct"]["0.0500"]["mean"] < 0


def test_aggregate_survives_json_roundtrip_string_keys():
    """Robustness dicts read back from results.json have STRING rate keys."""
    runs = [_fake_run(s, [10.0 + s, 12.0 + s, 14.0 + s],
                      [10.1 + s, 11.9 + s, 13.5 + s]) for s in (0, 1, 2)]
    roundtripped = [json.loads(json.dumps(r)) for r in runs]
    agg_raw = aggregate_runs(runs, seeds=[0, 1, 2])
    agg_json = aggregate_runs(roundtripped, seeds=[0, 1, 2])
    assert (agg_json["paired"]["robustness_gap_pct"]["0.0500"]["mean"]
            == pytest.approx(agg_raw["paired"]["robustness_gap_pct"]["0.0500"]["mean"]))


def test_aggregate_runs_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_runs([])


def test_rate_key_is_stable():
    assert _rate_key(0.05) == _rate_key("0.05") == "0.0500"
    assert _rate_key(0) == "0.0000"


def test_aggregate_is_json_serializable():
    runs = [_fake_run(s, [10.0, 12.0, 14.0], [10.0, 11.9, 13.8]) for s in (0, 1, 2)]
    agg = aggregate_runs(runs, seeds=[0, 1, 2])
    blob = json.dumps(agg)          # must not raise
    assert "robustness_gap_pct" in blob
