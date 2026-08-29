"""
Unit tests for scripts/build_report.py.

The report is the artefact a reader actually sees, so the failure mode that
matters most is a *silently wrong* table: a missing track rendered as blank
instead of "not yet run", a significance marker on an under-powered statistic, or
a robustness gap whose sign flips in transcription. These tests build reports from
hand-written aggregates where the right answer is known.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_report import (  # noqa: E402
    _pm,
    _sig,
    build_report,
    discover,
)


def _stat(mean, std, n=3, ci95=None):
    return {"mean": mean, "std": std, "sem": std / max(1, n) ** 0.5,
            "ci95": ci95 if ci95 is not None else 3.182 * std / max(1, n) ** 0.5,
            "n": n, "values": [mean] * n}


def _zeroshot_agg(model="org/tiny", n=3, gap_at_15=40.0, gap_ci=5.0):
    rates = ["0.0000", "0.0500", "0.1500"]
    arm = lambda ppl: {                                        # noqa: E731
        "clean_ppl": _stat(ppl, 0.5, n),
        "robustness": {r: _stat(ppl * (1 + 3 * float(r)), 1.0, n) for r in rates},
        "robustness_delta_pct": {r: _stat(300 * float(r), 2.0, n) for r in rates},
        "attention": {"mean_attention_entropy_nats": _stat(1.9, 0.05, n),
                      "mean_sink_mass_pos0": _stat(0.02, 0.001, n)},
        "latency": {"prefill_ms": _stat(50.0, 2.0, n),
                    "tokens_per_s": _stat(500.0, 10.0, n)},
    }
    return {
        "meta": {"seeds": list(range(n)), "n_seeds": n, "model_id": model,
                 "device": "cuda", "dtype": "bfloat16", "diffusion_steps": 1,
                 "target_alpha": 0.05, "noise_rates": rates,
                 "ci_method": "two-sided t, 95%", "baseline_note": "eager softmax"},
        "arms": {"softmax": arm(20.0), "qgfd": arm(20.3)},
        "paired": {
            "clean_ppl_qgfd_minus_softmax": _stat(0.30, 0.01, n),
            "latency_overhead_x": _stat(1.4, 0.05, n),
            "robustness_gap_pct": {"0.0500": _stat(3.0, 4.0, n, ci95=9.0),
                                   "0.1500": _stat(gap_at_15, 2.0, n, ci95=gap_ci)},
        },
    }


def _finetune_agg(model="org/tiny", n=3):
    agg = _zeroshot_agg(model, n)
    for arm in agg["arms"].values():           # the fine-tuning track has neither
        arm.pop("attention")
        arm.pop("latency")
    agg["paired"].pop("latency_overhead_x")
    agg["meta"].update(track="finetune", backend="operator", max_steps=300)
    agg["train"] = {k: {"final_loss": _stat(3.2, 0.1, n),
                        "first_loss": _stat(3.9, 0.1, n),
                        "seconds": _stat(600.0, 20.0, n)}
                    for k in ("softmax", "qgfd")}
    return agg


def _synthetic_agg(model="org/tiny", n=3, post_lora=False):
    noise = ["0.00", "0.20"]
    arm = lambda acc: {                                        # noqa: E731
        "induction_acc": _stat(acc, 0.01, n),
        "induction_control_acc": _stat(0.005, 0.001, n),
        "induction_by_noise": {r: _stat(acc - float(r), 0.01, n) for r in noise},
        "passkey_acc": _stat(0.75, 0.05, n),
        "passkey_by_depth": {"0.10": _stat(0.8, 0.05, n),
                             "0.90": _stat(0.7, 0.05, n)},
    }
    return {
        "meta": {"track": "synthetic", "seeds": list(range(n)), "n_seeds": n,
                 "model_id": model, "backend": "operator", "device": "cuda",
                 "post_lora": post_lora, "diffusion_steps": 1, "target_alpha": 0.05,
                 "induction_seq_len": 48, "induction_predictions_per_seed": 2944,
                 "passkey_context_tokens": 400, "passkey_n_per_seed": 72,
                 "ci_method": "t-based", "control_note": "first copy is the floor",
                 "coarse_metric_note": "argmax is coarse",
                 "operator": {"softmax": {"qgfd_active": False},
                              "qgfd": {"qgfd_active": True, "alpha_eval_mode": 0.05}}},
        "arms": {"softmax": arm(0.90), "qgfd": arm(0.92)},
        "paired": {"induction_gap": _stat(0.02, 0.004, n, ci95=0.007),
                   "induction_gap_by_noise": {r: _stat(0.02, 0.004, n, ci95=0.007)
                                              for r in noise},
                   "passkey_gap": _stat(0.0, 0.0, n),
                   "passkey_gap_by_depth": {"0.10": _stat(0.0, 0.0, n),
                                            "0.90": _stat(0.0, 0.0, n)}},
    }


@pytest.fixture
def tree(tmp_path):
    """A results tree with all three tracks, plus a decoy JSON that must be skipped."""
    (tmp_path / "zs").mkdir()
    (tmp_path / "ft" / "nested").mkdir(parents=True)
    (tmp_path / "sy").mkdir()
    (tmp_path / "zs" / "results_aggregated.json").write_text(json.dumps(_zeroshot_agg()))
    (tmp_path / "ft" / "nested" / "finetune_aggregated.json").write_text(
        json.dumps(_finetune_agg()))
    (tmp_path / "sy" / "synthetic_aggregated.json").write_text(json.dumps(_synthetic_agg()))
    (tmp_path / "zs" / "results.json").write_text('{"not": "an aggregate"}')
    return tmp_path


def test_pm_and_sig_formatting():
    assert _pm(_stat(1.2345, 0.02), 3) == "1.234 ± 0.020"
    assert _pm(None) == "—"
    assert _sig(_stat(40.0, 2.0, 3, ci95=5.0)) == "**\\***"      # CI excludes zero
    assert _sig(_stat(3.0, 4.0, 3, ci95=9.0)) == "ns"            # CI includes zero
    assert _sig(_stat(40.0, 0.0, 1, ci95=0.0)) == "n/a"          # single seed
    assert _sig(None) == "n/a"


def test_discover_buckets_by_track_and_skips_unknown_files(tree):
    found = discover([str(tree)])
    assert len(found["zeroshot"]) == 1
    assert len(found["finetune"]) == 1
    assert len(found["synthetic"]) == 1
    assert all(not p.endswith("results.json") or "aggregated" in p
               for p, _ in found["zeroshot"])


def test_discover_is_idempotent_across_overlapping_roots(tree):
    found = discover([str(tree), str(tree / "zs"), str(tree / "zs")])
    assert len(found["zeroshot"]) == 1, "the same file must not be ingested twice"


def test_discover_accepts_a_direct_file_path(tree):
    found = discover([str(tree / "sy" / "synthetic_aggregated.json")])
    assert len(found["synthetic"]) == 1
    assert not found["zeroshot"]


def test_report_has_every_section_and_the_real_numbers(tree, tmp_path):
    out = tmp_path / "paper" / "REPORT.md"
    text = build_report(discover([str(tree)]), str(out))

    assert out.exists()
    for heading in ("## Abstract", "## Tools and System Design",
                    "## Experimental Protocol", "### Table 1", "### Table 2",
                    "### Table 3", "### Table 4", "## Threats to Validity",
                    "## Reproduction"):
        assert heading in text, f"missing {heading}"
    assert "20.000 ± 0.500" in text          # clean PPL softmax
    assert "+40.00 ± 2.00" in text           # robustness gap at 15%
    assert "1.40 ± 0.05" in text             # latency overhead
    assert "3.200 ± 0.100" not in text or "3.2000 ± 0.1000" in text
    assert "0.9200 ± 0.0100" in text         # QGFD induction accuracy


def test_report_marks_significance_per_row_not_globally(tree, tmp_path):
    text = build_report(discover([str(tree)]), str(tmp_path / "R.md"))
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "%" in ln]
    at5 = [r for r in rows if "| 5% |" in r]
    at15 = [r for r in rows if "| 15% |" in r]
    assert at5 and at15
    assert at5[0].rstrip().endswith("ns |"), "wide-CI row must be flagged ns"
    assert at15[0].rstrip().endswith("**\\*** |"), "tight-CI row must be starred"


def test_report_flags_underpowered_runs(tmp_path):
    """n=2 cannot support a claim; the report must say so without being asked."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "results_aggregated.json").write_text(json.dumps(_zeroshot_agg(n=2)))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert "Under-powered" in text and "n=2" in text


def test_report_does_not_flag_a_fully_powered_run(tree, tmp_path):
    text = build_report(discover([str(tree)]), str(tmp_path / "R.md"))
    assert "Under-powered" not in text


def test_missing_tracks_become_explicit_notes(tmp_path):
    d = tmp_path / "only_zs"
    d.mkdir()
    (d / "results_aggregated.json").write_text(json.dumps(_zeroshot_agg()))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert text.count("_Not yet run._") == 2, "fine-tune and synthetic must be flagged"
    assert "### Table 1" in text and "20.000 ± 0.500" in text


def test_empty_input_still_writes_an_honest_skeleton(tmp_path):
    out = tmp_path / "R.md"
    text = build_report(discover([str(tmp_path)]), str(out))
    assert out.exists()
    assert "the headline claim is unsupported" in text
    assert "_None found._" in text
    assert text.count("_Not yet run._") == 4


def test_headline_reports_a_loss_as_a_loss(tmp_path):
    """A negative gap must not be silently narrated as a win."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "results_aggregated.json").write_text(
        json.dumps(_zeroshot_agg(gap_at_15=-12.5)))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert "degraded **less**" in text and "0/1 models" in text
    assert "and **more** on tiny" in text


def test_multiple_models_each_get_a_row(tmp_path):
    for i, name in enumerate(("org/aaa", "org/bbb")):
        d = tmp_path / f"m{i}"
        d.mkdir()
        (d / "results_aggregated.json").write_text(json.dumps(_zeroshot_agg(name)))
    text = build_report(discover([str(tmp_path)]), str(tmp_path / "R.md"))
    assert "`aaa`" in text and "`bbb`" in text
    assert "2/2 models" in text


def test_figure_is_embedded_when_the_plot_exists(tmp_path):
    d = tmp_path / "zs"
    d.mkdir()
    (d / "results_aggregated.json").write_text(json.dumps(_zeroshot_agg()))
    (d / "robustness_aggregated.png").write_bytes(b"\x89PNG\r\n")
    out = tmp_path / "paper" / "REPORT.md"
    text = build_report(discover([str(d)]), str(out))
    assert "![robustness — tiny](../zs/robustness_aggregated.png)" in text


def test_synthetic_post_lora_mode_is_labelled(tmp_path):
    d = tmp_path / "sy"
    d.mkdir()
    (d / "synthetic_aggregated.json").write_text(
        json.dumps(_synthetic_agg(post_lora=True)))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert "post-LoRA" in text
