from __future__ import annotations

import json

import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.experiments.phase_ii_evaluation import run_phase_ii_evaluation
from capitalmarket.capitalselector.experiments.run_phase_ii import PhaseIIConfig, step_phase_ii, run_phase_ii_episode
from capitalmarket.capitalselector.interfaces import make_world_step_result
from capitalmarket.capitalselector.phase_ii_state import PhaseIIEconomicState, PhaseIISelectorState


class StaticAsymmetricWorld:
    def step(self, t: int, action):
        _ = t
        return make_world_step_result(
            r_vec=np.asarray([0.20, -0.05], dtype=float),
            c_total=0.02,
            action=action,
        )


class ImmediateDeathWorld:
    def step(self, t: int, action):
        _ = t
        _ = action
        return make_world_step_result(
            r_vec=np.asarray([0.0, 0.0], dtype=float),
            c_total=2.0,
            action=action,
        )


def _selector(weights: list[float]):
    selector = (
        CapitalSelectorBuilder()
        .with_K(len(weights))
        .with_selector_policy("myopic")
        .build()
    )
    selector.w = np.asarray(weights, dtype=float)
    selector.K = len(weights)
    selector.ensure_channel_state(len(weights))
    return selector


def test_additional_expected_inflows_pathway_should_change_next_action_with_h_near():
    world = StaticAsymmetricWorld()
    sel_h0 = _selector([0.5, 0.5])
    sel_h2 = _selector([0.5, 0.5])

    # Keep selector internals identical except for near-term horizon definition in economic state.
    custom_mu = np.asarray(
        [
            [0.0, 8.0, 8.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    sel_h0.mu_term = custom_mu.copy()
    sel_h2.mu_term = custom_mu.copy()

    econ_h0 = PhaseIIEconomicState(due_curve=np.asarray([5.0, 0.0, 0.0], dtype=float), h_near_idx=0)
    econ_h2 = PhaseIIEconomicState(due_curve=np.asarray([5.0, 0.0, 0.0], dtype=float), h_near_idx=2)

    sstate_h0 = PhaseIISelectorState()
    sstate_h2 = PhaseIISelectorState()

    _ = step_phase_ii(selector=sel_h0, world=world, t=0, economic_state=econ_h0, selector_state=sstate_h0)
    _ = step_phase_ii(selector=sel_h2, world=world, t=0, economic_state=econ_h2, selector_state=sstate_h2)

    rec_h0 = step_phase_ii(selector=sel_h0, world=world, t=1, economic_state=econ_h0, selector_state=sstate_h0)
    rec_h2 = step_phase_ii(selector=sel_h2, world=world, t=1, economic_state=econ_h2, selector_state=sstate_h2)

    # Sanierung 3.5/3.6 requirement: different expected inflow horizon should influence policy action.
    exp_h0 = float(
        econ_h0.compute_expected_inflows(
            weights=np.asarray(rec_h0.action.weights, dtype=float),
            mu_term=np.asarray(sel_h0.mu_term, dtype=float),
        )
    )
    exp_h2 = float(
        econ_h2.compute_expected_inflows(
            weights=np.asarray(rec_h2.action.weights, dtype=float),
            mu_term=np.asarray(sel_h2.mu_term, dtype=float),
        )
    )

    assert exp_h0 != exp_h2
    assert np.isclose(float(econ_h0.liquidity_mismatch), float(econ_h0.near_term_obligations() - exp_h0), rtol=0.0, atol=1e-12)
    assert np.isclose(float(econ_h2.liquidity_mismatch), float(econ_h2.near_term_obligations() - exp_h2), rtol=0.0, atol=1e-12)
    # Selector-side features expose expected inflows as a derived value.
    assert np.isclose(float(sstate_h0.derived_features()["expected_inflows"]), exp_h0, rtol=0.0, atol=1e-12)
    assert np.isclose(float(sstate_h2.derived_features()["expected_inflows"]), exp_h2, rtol=0.0, atol=1e-12)
    assert not np.allclose(
        np.asarray(rec_h0.action.weights, dtype=float),
        np.asarray(rec_h2.action.weights, dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


def test_additional_expected_inflows_source_of_truth_is_economic_state():
    world = StaticAsymmetricWorld()
    selector = _selector([0.5, 0.5])
    econ = PhaseIIEconomicState(due_curve=np.asarray([4.0, 1.0, 0.0], dtype=float), h_near_idx=1)
    sstate = PhaseIISelectorState(
        near_term_obligations=1e9,
        liquidity_mismatch=-1e9,
    )

    rec = step_phase_ii(selector=selector, world=world, t=0, economic_state=econ, selector_state=sstate)
    expected = float(
        econ.compute_expected_inflows(
            weights=np.asarray(rec.action.weights, dtype=float),
            mu_term=np.asarray(selector.mu_term, dtype=float),
        )
    )

    # Economic state is the canonical path; selector-side value is derived, not stored.
    assert np.isclose(float(rec.economic_observables["expected_inflows"]), expected, rtol=0.0, atol=1e-12)
    assert np.isclose(float(sstate.derived_features()["expected_inflows"]), expected, rtol=0.0, atol=1e-12)


def test_additional_h_near_definition_must_be_exposed_in_phase_ii_evaluation_artifacts(tmp_path):
    cfg = PhaseIIConfig(
        selector_policy="myopic",
        gross_exposure=1.0,
        leverage_limit=1.0,
        allow_short=False,
        h_near_idx=2,
    )

    result = run_phase_ii_evaluation(
        seeds=[0, 1, 2],
        selectors=["Selector-3", "Selector-4", "Selector-5"],
        worlds=["baseline_world"],
        steps=8,
        channels=5,
        backend="cpu",
        base_config=cfg,
        bootstrap_samples=64,
        bootstrap_seed=123,
        output_dir=tmp_path,
        enforce_min_runs=False,
    )

    summary_path = result["artifacts"]["summary_metrics_json"]
    summary = json.loads((tmp_path / "phase_ii_summary_metrics.json").read_text(encoding="utf-8"))

    # Sanierung 3.6 requirement: near-term horizon definition must be explicit and shared in evaluation artifacts.
    assert summary_path.endswith("phase_ii_summary_metrics.json")
    assert "h_near_idx" in summary["config"]
    assert int(summary["config"]["h_near_idx"]) == 2


def test_additional_episode_should_stop_after_first_death_event():
    out = run_phase_ii_episode(
        world=ImmediateDeathWorld(),
        steps=5,
        channels=2,
        seed=3,
        config=PhaseIIConfig(selector_policy="myopic", h_near_idx=0),
        backend="cpu",
    )

    ttd = int(out["evaluation_metrics"]["time_to_death"])
    history = out["history"]

    # Sanierung 3.7 recommendation: stop simulation at first terminal event.
    assert ttd == 1
    assert len(history) == ttd
    assert float(out["final"]["wealth"]) == float(history[ttd - 1]["wealth"])
