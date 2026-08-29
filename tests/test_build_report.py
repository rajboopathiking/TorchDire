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
    discover_mechanism,
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
    assert text.count("_Not yet run._") == 3, \
        "fine-tune, synthetic and mechanism must be flagged"
    assert "### Table 1" in text and "20.000 ± 0.500" in text


def test_empty_input_still_writes_an_honest_skeleton(tmp_path):
    out = tmp_path / "R.md"
    text = build_report(discover([str(tmp_path)]), str(out))
    assert out.exists()
    assert "the headline claim is unsupported" in text
    assert "_None found._" in text
    assert text.count("_Not yet run._") == 7


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


# --------------------------------------------------------------------------
# Table 1b / E9 — the denominator-artefact decomposition.
#
# The headline statistic is a difference of RELATIVE degradations, and QGFD pays a
# clean-perplexity cost, so its denominator is larger and its Δ% is mechanically
# smaller for the same absolute perplexity under noise. That manufactures a
# positive "robustness gap" out of a *worse* model. These tests pin the
# decomposition and the four readings.
# --------------------------------------------------------------------------
def _ppl_agg(name, c_s, n_s, c_q, n_q, n=3, jitter=0.0, n_params=None,
             q_noisy_spread=None):
    """A zero-shot aggregate with explicit per-seed clean/noisy perplexities.

    `q_noisy_spread` adds per-seed offsets to the QGFD arm's noisy perplexity only,
    which is how a residual with a CI straddling zero is constructed.
    """
    rates = ["0.0000", "0.1500"]

    def arm(clean, noisy, spread=None):
        cv = [clean + jitter * i for i in range(n)]
        nv = [noisy + jitter * i + (spread[i] if spread else 0.0) for i in range(n)]
        a = _zeroshot_agg(name, n)["arms"]["softmax"]
        a["clean_ppl"] = _stat_from(cv)
        a["robustness"] = {"0.0000": _stat_from(cv), "0.1500": _stat_from(nv)}
        a["robustness_delta_pct"] = {
            r: _stat_from([100.0 * (b - c) / c for c, b in zip(cv, nv)]
                          if r != "0.0000" else [0.0] * n) for r in rates}
        return a

    agg = _zeroshot_agg(name, n)
    agg["meta"]["noise_rates"] = rates
    if n_params is not None:
        agg["meta"]["n_params"] = n_params
    agg["arms"] = {"softmax": arm(c_s, n_s),
                   "qgfd": arm(c_q, n_q, q_noisy_spread)}
    sm, qg = agg["arms"]["softmax"], agg["arms"]["qgfd"]
    agg["paired"]["clean_ppl_qgfd_minus_softmax"] = _stat_from(
        [q - s for s, q in zip(sm["robustness"]["0.0000"]["values"],
                               qg["robustness"]["0.0000"]["values"])])
    agg["paired"]["robustness_gap_pct"] = {"0.1500": _stat_from([
        100.0 * (ns - cs) / cs - 100.0 * (nq - cq) / cq
        for cs, ns, cq, nq in zip(sm["robustness"]["0.0000"]["values"],
                                  sm["robustness"]["0.1500"]["values"],
                                  qg["robustness"]["0.0000"]["values"],
                                  qg["robustness"]["0.1500"]["values"])])}
    return agg


def _stat_from(values):
    from scripts.build_report import _stat
    return _stat(values)


def _write(tmp_path, *aggs):
    for i, agg in enumerate(aggs):
        d = tmp_path / f"m{i}"
        d.mkdir(parents=True)
        (d / "results_aggregated.json").write_text(json.dumps(agg))
    return build_report(discover([str(tmp_path)]), str(tmp_path / "R.md"))


def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_denominator_decomposition_is_exact(tmp_path):
    """observed == denominator-only + residual, to the printed precision."""
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 325.0, jitter=0.3))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    obs, art, _, res = (_cells(row)[2], _cells(row)[3], _cells(row)[4], _cells(row)[5])
    to_f = lambda s: float(s.split(" ±")[0])                   # noqa: E731
    assert to_f(obs) == pytest.approx(to_f(art) + to_f(res), abs=0.02)


def test_denominator_artefact_can_exceed_the_whole_reported_gap(tmp_path):
    """
    The case that matters: nearly the whole reported gap comes from the denominator,
    and the residual — the only part that could be robustness — has a CI straddling
    zero. The report must refuse to treat the gap as a robustness result.
    """
    # softmax 20.0 -> 320.0 (+1500%). QGFD's noisy perplexity is 320.0 on average
    # too, so its whole apparent advantage is the 0.5 PPL clean cost in the
    # denominator; the +/-8 per-seed spread is what makes the residual ns.
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 320.0, jitter=0.3,
                                     q_noisy_spread=[-8.0, 0.0, 8.0]))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    cells = _cells(row)
    assert cells[2].startswith("+"), "the reported gap is positive"
    assert cells[3].startswith("+"), "the denominator-only component is positive"
    assert cells[6] == "ns", "the residual must not reach significance"
    assert float(cells[4].rstrip("%")) >= 50.0, "denominator explains most of it"
    assert "does not survive the control" in text
    assert "no evidence QGFD is more accurate under noise in absolute terms" in text


def test_denominator_check_calls_out_a_significantly_backwards_residual(tmp_path):
    """
    The strongest reading: the relative gap is positive while QGFD's absolute
    perplexity under noise is significantly worse. Small between-seed jitter keeps
    the residual's CI away from zero.
    """
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 325.0, jitter=0.02))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    cells = _cells(row)
    assert cells[2].startswith("+"), "the reported relative gap still reads positive"
    assert cells[5].startswith("-") and cells[6] == "**\\***"
    assert cells[7].startswith("-"), "absolute noisy PPL is worse for QGFD"
    assert "the underlying sign is backwards" in text
    assert "measuring QGFD's clean-perplexity cost, not its robustness" in text


def test_denominator_share_is_blank_for_a_non_positive_gap(tmp_path):
    """A 'share of the gap' is meaningless when the gap itself is negative."""
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 340.0, jitter=0.3))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    assert _cells(row)[2].startswith("-"), "this fixture has a negative gap"
    assert _cells(row)[4] == "n/a"
    assert "left blank where the reported gap is not positive" in text


def test_denominator_check_credits_a_real_absolute_win(tmp_path):
    """QGFD genuinely lower under noise, with between-seed spread small enough to
    reach significance — the residual must be starred and the reading positive."""
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.0, 260.0, jitter=0.2))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    cells = _cells(row)
    assert cells[5].startswith("+") and cells[6] == "**\\***"
    assert "The gap survives the control" in text


def test_denominator_check_reports_a_partial_artefact(tmp_path):
    """A real absolute win that the denominator still inflates by half or more."""
    # QGFD is 4.0 PPL better under noise but costs 1.0 PPL clean; at a +1500%
    # baseline the denominator contributes more than the win does.
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 21.0, 316.0, jitter=0.05))
    assert "Partly artefact" in text
    assert "Quote the residual, not the raw gap" in text


def test_denominator_check_needs_two_noise_rates(tmp_path):
    agg = _ppl_agg("org/a", 20.0, 320.0, 20.5, 331.0)
    for arm in agg["arms"].values():
        arm["robustness"].pop("0.1500")
    text = _write(tmp_path, agg)
    assert "### Table 1b" in text
    assert _block(text, "### Table 1b")[1].startswith("_Not yet run._")


def test_denominator_check_survives_aggregates_without_per_seed_values(tmp_path):
    """Hand-written or trimmed aggregates must degrade to a note, not a crash."""
    agg = _ppl_agg("org/a", 20.0, 320.0, 20.5, 331.0)
    for arm in agg["arms"].values():
        for s in arm["robustness"].values():
            s.pop("values")
    text = _write(tmp_path, agg)
    assert "_Not yet run._" in _block(text, "### Table 1b")[1]


def test_denominator_check_orders_by_scale_like_table_1a(tmp_path):
    text = _write(tmp_path,
                  _ppl_agg("org/big", 12.0, 100.0, 12.1, 101.0, n_params=1_100_000_000),
                  _ppl_agg("org/small", 21.0, 340.0, 21.6, 350.0, n_params=135_000_000))
    rows = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")]
    assert "`small`" in rows[0] and "`big`" in rows[1]


def test_denominator_check_reports_no_positive_residual_without_blaming_the_share(
        tmp_path):
    """
    The in-between reading: QGFD costs nothing clean (so there is no denominator
    artefact to blame) and its noisy perplexity is a wash. Nothing is established,
    but the 'does not survive the control' wording would be wrong here — no share
    of the gap is arithmetic.
    """
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.0, 320.0, jitter=0.3,
                                     q_noisy_spread=[-8.0, 0.0, 8.0]))
    row = [ln for ln in _block(text, "### Table 1b") if ln.startswith("| `")][0]
    assert _cells(row)[6] == "ns", "residual must straddle zero"
    assert "No model has a positive residual distinguishable from zero" in text
    assert "does not survive the control" not in text


# --------------------------------------------------------------------------
# The abstract must not survive E9 either. A relative gap that turns out to be
# arithmetic cannot be sold as a contribution two paragraphs above the table that
# refutes it.
# --------------------------------------------------------------------------
def test_abstract_retracts_contribution_two_when_the_residual_is_backwards(tmp_path):
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 325.0, jitter=0.02))
    abstract = text.split("## Abstract")[1].split("##")[0]
    assert "No robustness benefit survives the denominator control" in abstract
    assert "the absolute effect runs the other way" in abstract
    assert "Contribution (2) is falsified" in abstract
    assert "scale-dependent robustness effect" not in abstract
    assert "does not survive E9" in abstract, "the headline sentence must say so too"


def test_abstract_retracts_contribution_two_when_no_residual_is_positive(tmp_path):
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 320.0, jitter=0.3,
                                     q_noisy_spread=[-8.0, 0.0, 8.0]))
    abstract = text.split("## Abstract")[1].split("##")[0]
    assert "No robustness benefit survives the denominator control" in abstract
    assert "not supported by these runs" in abstract
    assert "the absolute effect runs the other way" not in abstract, \
        "nothing here is significantly backwards"
    assert "E9 does not support reading this as robustness" in abstract


def test_abstract_keeps_contribution_two_when_the_residual_is_real(tmp_path):
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.0, 260.0, jitter=0.2))
    abstract = text.split("## Abstract")[1].split("##")[0]
    assert "Training-free robustness" in abstract
    assert "denominator control" not in abstract
    assert "does not survive E9" not in abstract


def test_abstract_flags_a_partial_artefact_without_retracting(tmp_path):
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 21.0, 316.0, jitter=0.05))
    abstract = text.split("## Abstract")[1].split("##")[0]
    assert "denominator artefact" in abstract
    assert "the number to quote" in abstract
    assert "No robustness benefit survives" not in abstract, \
        "a surviving residual must not be retracted"


def test_abstract_is_unchanged_when_the_decomposition_cannot_be_computed(tmp_path):
    """No per-seed values -> no E9 verdict -> contribution 2 falls back to the gap."""
    agg = _ppl_agg("org/a", 20.0, 320.0, 20.5, 325.0, jitter=0.02)
    for arm in agg["arms"].values():
        for s in arm["robustness"].values():
            s.pop("values")
    abstract = _write(tmp_path, agg).split("## Abstract")[1].split("##")[0]
    assert "denominator control" not in abstract
    assert "Training-free robustness" in abstract


def test_caveats_name_the_relative_denominator_sensitivity(tmp_path):
    text = _write(tmp_path, _ppl_agg("org/a", 20.0, 320.0, 20.5, 325.0, jitter=0.3))
    threats = text.split("## Threats to Validity")[1]
    assert "difference of **relative** degradations" in threats
    assert "Table 1b" in threats


def test_synthetic_post_lora_mode_is_labelled(tmp_path):
    d = tmp_path / "sy"
    d.mkdir()
    (d / "synthetic_aggregated.json").write_text(
        json.dumps(_synthetic_agg(post_lora=True)))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert "post-LoRA" in text


# --------------------------------------------------------------------------
# Table 1a (scale trend) and the data-conditional contributions.
#
# These exist because the first generated report hard-asserted "QGFD lowers
# perplexity degradation at negligible clean-perplexity cost" while its own
# Table 1 showed the gap reversing sign on the largest model. An auto-generated
# report making a claim its own tables refute is the one failure mode that
# cannot be caught by reading the numbers, so it is pinned here.
# --------------------------------------------------------------------------
def _block(text, heading):
    """The lines of one '### ...' section, so a filter cannot match another table."""
    out, inside = [], False
    for ln in text.splitlines():
        if ln.startswith("### "):
            inside = ln.startswith(heading)
        elif inside:
            out.append(ln)
    return out


def _write_scale_tree(tmp_path, spec):
    """spec: [(name, n_params_or_None, clean_ppl, gap_at_15)] -> results tree.

    The per-seed `values` are stripped from the robustness stats so E9 cannot be
    computed: these fixtures exercise the *relative*-gap branches of
    `_contributions`, and the E9 gate (tested separately) would otherwise override
    every one of them.
    """
    for i, (name, n_params, clean, gap) in enumerate(spec):
        d = tmp_path / f"m{i}"
        d.mkdir(parents=True)
        agg = _zeroshot_agg(name, gap_at_15=gap, gap_ci=abs(gap) / 2 or 1.0)
        for arm in agg["arms"].values():
            arm["clean_ppl"] = _stat(clean, 0.5)
            for s in arm["robustness"].values():
                s.pop("values")
        if n_params is not None:
            agg["meta"]["n_params"] = n_params
        (d / "results_aggregated.json").write_text(json.dumps(agg))
    return build_report(discover([str(tmp_path)]), str(tmp_path / "R.md"))


def test_scale_table_orders_by_parameter_count_when_present(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/big", 1_100_000_000, 11.0, -1.26),
        ("org/small", 135_000_000, 21.0, 12.61),
    ])
    assert "### Table 1a" in text
    rows = [ln for ln in _block(text, "### Table 1a") if ln.startswith("| `")]
    assert "`small`" in rows[0] and "`big`" in rows[1], \
        "smallest model must come first regardless of discovery order"
    assert "135M" in text and "1100M" in text
    assert "ordered by parameter count" in text


def test_scale_table_falls_back_to_clean_ppl_ordering(tmp_path):
    """Aggregates written before meta.n_params existed must still render."""
    text = _write_scale_tree(tmp_path, [
        ("org/big", None, 11.0, -1.26),
        ("org/small", None, 21.0, 12.61),
    ])
    rows = [ln for ln in _block(text, "### Table 1a") if ln.startswith("| `")]
    assert "`small`" in rows[0] and "`big`" in rows[1]
    assert all("| — |" in r for r in rows), "params column must be an em dash"
    assert "measured capability" in text


def test_scale_table_calls_out_a_monotone_decline_with_a_sign_flip(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, 12.61),
        ("org/b", 494_000_000, 15.0, 8.14),
        ("org/c", 1_100_000_000, 11.0, -1.26),
    ])
    assert "**The effect does not survive scale.**" in text
    assert "+12.61 → +8.14 → -1.26 pp" in text
    assert "small-model artefact" in text


def test_scale_table_distinguishes_shrinking_from_reversing(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, 12.61),
        ("org/b", 1_100_000_000, 11.0, 4.0),
    ])
    assert "**The gap shrinks with scale**" in text
    assert "does not survive scale" not in text


def test_scale_table_reports_no_trend_as_no_trend(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, 5.0),
        ("org/b", 494_000_000, 15.0, 12.0),
        ("org/c", 1_100_000_000, 11.0, 8.0),
    ])
    assert "No monotone trend" in text
    assert "weak evidence of scale-independence, not evidence of it" in text


def test_scale_table_puts_the_gap_in_proportion_to_the_damage(tmp_path):
    """
    _zeroshot_agg's softmax degrades 45% at 15% noise. A +12.61 pp gap is 28% of
    that, so the 'under 5%' warning must NOT fire; a +1.0 pp gap is 2.2% and must.
    """
    big = _write_scale_tree(tmp_path / "a", [("org/a", 135_000_000, 21.0, 12.61)])
    assert "28.02%" in big
    assert "**under 5%**" not in big
    tiny = _write_scale_tree(tmp_path / "b", [("org/a", 135_000_000, 21.0, 1.0)])
    assert "2.22%" in tiny
    assert "**under 5%**" in tiny


def test_contributions_downgrade_when_the_sign_flips(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, 12.61),
        ("org/c", 1_100_000_000, 11.0, -1.26),
    ])
    assert "scale-dependent robustness effect" in text
    assert "must not be written as one" in text
    assert "negligible clean-perplexity cost" not in text, \
        "the abstract must not assert a benefit the tables contradict"


def test_contributions_report_an_all_negative_run_as_falsified(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, -3.0),
        ("org/c", 1_100_000_000, 11.0, -1.26),
    ])
    assert "No robustness benefit" in text and "falsified" in text


def test_contributions_state_the_win_when_every_model_wins(tmp_path):
    text = _write_scale_tree(tmp_path, [
        ("org/a", 135_000_000, 21.0, 40.0),
        ("org/c", 1_100_000_000, 11.0, 38.0),
    ])
    assert "2/2 models" in text
    assert "scale-dependent" not in text


def test_contributions_report_a_losing_synthetic_probe_as_a_negative_result(tmp_path):
    d = tmp_path / "sy"
    d.mkdir()
    agg = _synthetic_agg("org/tiny")
    agg["paired"]["induction_gap"] = _stat(-0.0033, 0.0005, 3, ci95=0.0012)
    (d / "synthetic_aggregated.json").write_text(json.dumps(agg))
    text = build_report(discover([str(d)]), str(tmp_path / "R.md"))
    assert "does **not** win" in text
    assert "discriminates nothing" in text
    assert "Reported as a negative result" in text


# --------------------------------------------------------------------------
# Track 5 (E2-E8) discovery and Table 5.
# --------------------------------------------------------------------------
def _mech_blob(model="org/tiny", quick=False):
    return {
        "config": {"model_id": model, "dtype": "bfloat16", "target_alpha": 0.05},
        "quick": quick,
        "requested": ["E2", "E3", "E4", "E6", "E7", "E8"],
        "results": {
            "E2": {"clean_curvature": {"T1": {"fit": {
                        "exponent_k": 1.94, "r2": 0.998,
                        "consistent_with_quadratic": True}}},
                   "robustness_by_alpha": {"T1": {"by_alpha": {
                        "0.0500": {"robustness_gap_pp": 12.6}}}}},
            "E3": {"paired": {"0.0500": {
                        "alpha": 0.05, "matched_tau": 1.084, "qgfd_gap_pp": 12.6,
                        "temp_gap_pp": 14.9, "qgfd_minus_temp_pp": -2.3,
                        "verdict": "temperature wins"}}},
            "E4": {"verdict": {"real_gap_pp": 12.6, "best_control": "shuffled",
                               "best_control_gap_pp": 11.9, "margin_pp": 0.7,
                               "reading": "graph structure adds almost nothing"}},
            "E6": {"by_corruption": {
                        "char": {"robustness_gap_pp": 12.6,
                                 "changes_tokenisation": True},
                        "word_drop": {"robustness_gap_pp": 0.4,
                                      "changes_tokenisation": False}},
                   "verdict": {"reading": "the win is specific to tokenisation damage"}},
            "E7": {"verdict": {"best_T": 1, "best_induction_acc": 0.31,
                               "softmax_induction_acc": 0.34,
                               "control_acc_floor": 0.01, "shape": "flat",
                               "reading": "no useful reach beyond T=1",
                               "warning": None}},
            "E8": {"comparison": {"qgfd_overhead_x": 1.53,
                                  "winner_under_noise": "larger_model",
                                  "latency_cheaper": "larger_model",
                                  "reading": "no efficiency framing survives"}},
        },
        "errors": {},
    }


@pytest.fixture
def mech_tree(tmp_path):
    d = tmp_path / "mechanism" / "tiny"
    d.mkdir(parents=True)
    (d / "mechanism_results.json").write_text(json.dumps(_mech_blob()))
    (d / "E3.json").write_text('{"per-experiment dump": "must be ignored"}')
    return tmp_path


def test_discover_mechanism_finds_the_combined_file_only(mech_tree):
    mech = discover_mechanism([str(mech_tree)])
    assert len(mech) == 1
    assert mech[0][0].endswith("mechanism_results.json")
    assert mech[0][1]["config"]["model_id"] == "org/tiny"


def test_discover_mechanism_is_kept_out_of_the_seeded_tracks(mech_tree):
    """A config-only blob in the zeroshot bucket would KeyError on meta.n_seeds."""
    found = discover([str(mech_tree)])
    assert not any(found.values()), "mechanism files must not land in any track"


def test_discover_mechanism_is_idempotent_and_takes_a_file_path(mech_tree):
    path = str(mech_tree / "mechanism" / "tiny" / "mechanism_results.json")
    assert len(discover_mechanism([str(mech_tree), path, path])) == 1
    assert len(discover_mechanism([path])) == 1


def test_table_mechanism_renders_every_requested_experiment(mech_tree, tmp_path):
    mech = discover_mechanism([str(mech_tree)])
    text = build_report({}, str(tmp_path / "R.md"), mechanism=mech)
    assert "### Table 5" in text
    assert "6/6 completed" in text
    for key in ("E2", "E3", "E4", "E6", "E7", "E8"):
        assert f"| {key} |" in text, f"missing row for {key}"
    assert "k = 1.94" in text and "R² = 0.998" in text
    assert "τ=1.084" in text and "-2.30 pp" in text
    assert "margin **+0.70 pp**" in text
    assert "word_drop +0.40 pp" in text
    assert "best T = 1" in text
    assert "1.53×" in text


def test_table_mechanism_reports_the_e3_falsification_in_words(mech_tree, tmp_path):
    """E3 negative is the single most consequential outcome; it must be spelled out."""
    mech = discover_mechanism([str(mech_tree)])
    text = build_report({}, str(tmp_path / "R.md"), mechanism=mech)
    assert "a free temperature rescale matches or beats QGFD" in text
    assert "no mechanism contribution survives" in text


def test_table_mechanism_credits_qgfd_when_e3_goes_the_other_way(tmp_path):
    blob = _mech_blob()
    blob["results"]["E3"]["paired"]["0.0500"]["qgfd_minus_temp_pp"] = 3.1
    d = tmp_path / "m"
    d.mkdir()
    (d / "mechanism_results.json").write_text(json.dumps(blob))
    text = build_report({}, str(tmp_path / "R.md"),
                        mechanism=discover_mechanism([str(d)]))
    assert "QGFD beats the free control" in text
    assert "no mechanism contribution survives" not in text


def test_table_mechanism_surfaces_failures_and_missing_fits(tmp_path):
    blob = _mech_blob()
    blob["results"] = {"E2": {"clean_curvature": {
        "T1": {"fit": {"note": "only 2 usable alphas"}}}}}
    blob["errors"] = {"E3": "RuntimeError: CUDA out of memory",
                      "E4": "ValueError: nope", "E6": "x", "E7": "y", "E8": "z"}
    d = tmp_path / "m"
    d.mkdir()
    (d / "mechanism_results.json").write_text(json.dumps(blob))
    text = build_report({}, str(tmp_path / "R.md"),
                        mechanism=discover_mechanism([str(d)]))
    assert "1/6 completed" in text
    assert "only 2 usable alphas" in text and "inconclusive" in text
    assert "**FAILED** — `RuntimeError: CUDA out of memory`" in text


def test_table_mechanism_labels_a_quick_run_as_not_a_result(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "mechanism_results.json").write_text(json.dumps(_mech_blob(quick=True)))
    text = build_report({}, str(tmp_path / "R.md"),
                        mechanism=discover_mechanism([str(d)]))
    assert "plumbing check, not a result" in text


def test_mechanism_counts_toward_the_ingested_total(mech_tree, tmp_path):
    mech = discover_mechanism([str(mech_tree)])
    text = build_report({}, str(tmp_path / "R.md"), mechanism=mech)
    assert "mechanism 1" in text
    assert "| Aggregates ingested | 1 " in text
    assert "— mechanism" in text, "the source-files list must name the file"
    assert "_None found._" not in text
