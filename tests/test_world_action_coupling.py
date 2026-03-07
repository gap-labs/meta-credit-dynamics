from __future__ import annotations

import numpy as np
import pytest

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.experiments.run_phase_ii import (
    PhaseIIConfig,
    run_phase_ii_episode,
    step_phase_ii,
)
from capitalmarket.capitalselector.interfaces import WorldStepResult, make_world_step_result


class ActionConditionedWorld:
    def __init__(self):
        self.calls = 0

    def step(self, t: int, action):
        _ = t
        self.calls += 1
        return make_world_step_result(
            r_vec=np.asarray([10.0, 1.0], dtype=float),
            c_total=0.5,
            action=action,
        )


class PrecomputedOutcomeWorld:
    def step(self, t: int, action):
        _ = t
        _ = action
        return WorldStepResult(
            realized_return=0.25,
            costs=0.10,
            channel_returns=np.asarray([100.0, -100.0], dtype=float),
            cost_by_channel=np.asarray([0.0, 0.0], dtype=float),
            freeze=False,
        )


class MinimalClosedLoopWorld:
    def step(self, t: int, action):
        _ = t
        return make_world_step_result(
            r_vec=np.asarray([0.2, -0.1], dtype=float),
            c_total=0.05,
            action=action,
        )


def _selector_with_weights(weights: list[float]):
    selector = CapitalSelectorBuilder().with_K(len(weights)).build()
    selector.w = np.asarray(weights, dtype=float)
    selector.K = len(weights)
    return selector


def test_phase_ii_coupling_books_action_conditioned_realized_return():
    selector = _selector_with_weights([1.0, 0.0])
    world = ActionConditionedWorld()

    wealth_before = float(selector.wealth)
    rec = step_phase_ii(selector=selector, world=world, t=0)

    expected_delta = 10.0 - 0.5
    np.testing.assert_allclose(rec.wealth_next - wealth_before, expected_delta, rtol=0.0, atol=1e-12)
    assert world.calls == 1


def test_phase_ii_kernel_does_not_recompute_exposure_mapping():
    selector = _selector_with_weights([0.9, 0.1])
    world = PrecomputedOutcomeWorld()

    wealth_before = float(selector.wealth)
    rec = step_phase_ii(selector=selector, world=world, t=0)

    expected_delta = 0.25 - 0.10
    np.testing.assert_allclose(rec.wealth_next - wealth_before, expected_delta, rtol=0.0, atol=1e-12)


def test_phase_ii_runner_exposes_state_and_boundary_contracts():
    out = run_phase_ii_episode(
        world=MinimalClosedLoopWorld(),
        steps=3,
        channels=2,
        seed=7,
        config=PhaseIIConfig(h_near_idx=0),
    )

    assert len(out["history"]) == 3
    for step in out["history"]:
        assert "events" in step
        assert "event_counts" in step
        assert "event_summary" in step
        assert "economic_observables" in step
        assert "strategic_credit_exposure" in step

        summary = step["event_summary"]
        assert set(summary.keys()) == {"event_counts", "last_event", "channel_event_vector"}
        assert int(sum(summary["event_counts"].values())) == int(len(step["events"]))

        if step["events"]:
            assert summary["last_event"] == step["events"][-1]["category"]

    assert "economic_state" in out["final"]
    assert "selector_state" in out["final"]

    metrics = out["evaluation_metrics"]
    assert set(metrics.keys()) == {
        "terminal_wealth",
        "time_to_death",
        "rollover_failure_frequency",
    }
    assert metrics["terminal_wealth"] == pytest.approx(float(out["final"]["wealth"]))

    expected_ttd = int(len(out["history"]))
    for step in out["history"]:
        is_dead_step = float(step["wealth"]) < 0.0 or int(step["event_counts"].get("FAIL", 0)) > 0
        if is_dead_step:
            expected_ttd = int(step["t"]) + 1
            break
    assert int(metrics["time_to_death"]) == expected_ttd

    rollover_count = sum(int(step["event_counts"].get("ROLLOVER", 0)) for step in out["history"])
    fail_count = sum(int(step["event_counts"].get("FAIL", 0)) for step in out["history"])
    expected_rollover_failure_frequency = float(fail_count / max(1, rollover_count + fail_count))
    assert metrics["rollover_failure_frequency"] == pytest.approx(expected_rollover_failure_frequency)
