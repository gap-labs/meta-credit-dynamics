from __future__ import annotations

import csv
import json

import pytest

from capitalmarket.capitalselector.experiments import run_phase_i_evaluation
from capitalmarket.capitalselector.experiments.phase_ii_evaluation import run_phase_ii_evaluation


def test_phase_ii_evaluation_writes_artifacts_and_hypothesis_ci_fields(tmp_path):
    seeds = [0, 1, 2]
    selectors = ["Selector-3", "Selector-4", "Selector-5"]
    worlds = ["baseline_world"]

    result = run_phase_ii_evaluation(
        seeds=seeds,
        selectors=selectors,
        worlds=worlds,
        steps=20,
        channels=5,
        backend="cpu",
        bootstrap_samples=250,
        bootstrap_seed=123,
        output_dir=tmp_path,
        enforce_min_runs=False,
    )

    rows = result["rows"]
    assert len(rows) == len(seeds) * len(selectors) * len(worlds)

    artifacts = result["artifacts"]
    csv_path = tmp_path / "phase_ii_selector_world_results.csv"
    hypotheses_path = tmp_path / "phase_ii_hypotheses.json"
    summary_path = tmp_path / "phase_ii_summary_metrics.json"

    assert artifacts["selector_world_results_csv"] == str(csv_path)
    assert artifacts["hypotheses_json"] == str(hypotheses_path)
    assert artifacts["summary_metrics_json"] == str(summary_path)

    assert csv_path.exists()
    assert hypotheses_path.exists()
    assert summary_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert len(reader) == len(rows)
    assert {
        "seed",
        "world",
        "selector",
        "terminal_wealth",
        "time_to_death",
        "rollover_failure_frequency",
    }.issubset(set(reader[0].keys()))

    hypotheses = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    assert len(hypotheses) == 2

    hypothesis_keys = {
        "hypothesis",
        "description",
        "world",
        "metric",
        "delta_definition",
        "paired_seed_count",
        "ci_low",
        "ci_high",
        "delta_mean",
        "decision_rule",
        "supported",
    }
    for item in hypotheses:
        assert hypothesis_keys.issubset(set(item.keys()))
        assert int(item["paired_seed_count"]) == len(seeds)
        assert float(item["ci_low"]) <= float(item["ci_high"])

        if item["hypothesis"] == "H1":
            assert item["decision_rule"] == "CI_lower(Delta) > 0"
            assert bool(item["supported"]) == (float(item["ci_low"]) > 0.0)
        elif item["hypothesis"] == "H2":
            assert item["decision_rule"] == "CI_upper(Delta) < 0"
            assert bool(item["supported"]) == (float(item["ci_high"]) < 0.0)
        else:
            raise AssertionError(f"Unexpected hypothesis id: {item['hypothesis']}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "config" in summary
    assert "aggregate_by_world_selector" in summary
    assert "hypothesis_results" in summary
    assert float(summary["config"]["confidence_interval"]) == pytest.approx(0.95)
    assert str(summary["config"]["bootstrap_mode"]) == "paired"


def test_phase_ii_evaluation_min_seed_matrix_guard():
    with pytest.raises(ValueError, match="at least 100 seeds"):
        run_phase_ii_evaluation(
            seeds=[0, 1, 2],
            selectors=["Selector-3", "Selector-4", "Selector-5"],
            worlds=["baseline_world"],
            steps=10,
            channels=5,
            backend="cpu",
            bootstrap_samples=64,
            bootstrap_seed=7,
            enforce_min_runs=True,
        )


def test_phase_i_evaluation_still_exposes_invariance_artifacts(tmp_path):
    result = run_phase_i_evaluation(
        seeds=[0, 1],
        selectors=["random", "myopic"],
        worlds=["baseline_world"],
        steps=10,
        lambda_risk=0.50,
        bootstrap_samples=64,
        bootstrap_seed=7,
        output_dir=tmp_path / "phase_i_guard",
    )

    assert "rows" in result
    assert "artifacts" in result
    assert "summary_metrics" in result
    assert "invariance" in result["summary_metrics"]
