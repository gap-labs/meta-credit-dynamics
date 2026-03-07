from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import WorldAction, WorldStepResult
from capitalmarket.capitalselector.phase_ii_events import (
    PHASE_II_EVENT_CATEGORIES,
    apply_phase_ii_event_mapping,
    event_counts,
)
from capitalmarket.capitalselector.phase_ii_state import PhaseIIEconomicState


def _selector_with_k(k: int):
    selector = CapitalSelectorBuilder().with_K(k).build()
    selector.ensure_channel_state(k)
    selector.w = np.ones(k, dtype=float) / float(max(1, k))
    return selector


def _serialize_events(events):
    return [
        (
            str(event.category),
            int(event.channel),
            int(event.horizon),
            float(event.amount),
            None if event.rollover_to_horizon is None else int(event.rollover_to_horizon),
        )
        for event in events
    ]


def test_phase_ii_event_contract_declares_required_categories():
    assert PHASE_II_EVENT_CATEGORIES == (
        "RETURN",
        "DUE_CASH",
        "COST",
        "ROLLOVER",
        "FAIL",
        "SETTLEMENT",
    )


def test_phase_ii_event_mapping_updates_due_curve_and_settlement_state():
    selector = _selector_with_k(2)
    action = WorldAction(weights=np.asarray([1.0, 0.0], dtype=float), gross_exposure=1.0, leverage_limit=1.0, allow_short=False)
    world_out = WorldStepResult(
        realized_return=0.2,
        costs=2.0,
        channel_returns=np.asarray([0.2, 0.0], dtype=float),
        cost_by_channel=np.asarray([0.0, 0.0], dtype=float),
        freeze=False,
    )
    econ = PhaseIIEconomicState.zeros(horizon_bins=3, h_near_idx=0)
    mu_term = np.zeros((2, 3), dtype=float)

    events = apply_phase_ii_event_mapping(
        selector=selector,
        action=action,
        world_out=world_out,
        economic_state=econ,
        mu_term=mu_term,
    )
    events_list, summary = events
    counts = event_counts(events_list)

    assert counts["RETURN"] == 1
    assert counts["COST"] == 1
    assert counts["DUE_CASH"] == 1
    assert counts["ROLLOVER"] == 1
    assert counts["SETTLEMENT"] == 1
    assert counts["FAIL"] == 0
    assert summary.last_event == "SETTLEMENT"
    assert summary.event_counts["ROLLOVER"] == 1

    np.testing.assert_allclose(econ.due_curve, np.asarray([0.0, 2.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(econ.liquidity_mismatch), 0.0, rtol=0.0, atol=1e-12)
    assert float(selector.stats.mu) > 0.0
    np.testing.assert_allclose(np.asarray(selector.phase_ii_cost_by_channel, dtype=float), np.asarray([2.0, 0.0], dtype=float), rtol=0.0, atol=1e-12)


def test_phase_ii_event_mapping_fail_updates_psi_when_no_rollover_capacity():
    selector = _selector_with_k(2)
    action = WorldAction(weights=np.asarray([1.0, 0.0], dtype=float), gross_exposure=1.0, leverage_limit=1.0, allow_short=False)
    world_out = WorldStepResult(
        realized_return=0.0,
        costs=1.0,
        channel_returns=np.asarray([0.0, 0.0], dtype=float),
        cost_by_channel=np.asarray([0.0, 0.0], dtype=float),
        freeze=False,
    )
    econ = PhaseIIEconomicState.zeros(horizon_bins=1, h_near_idx=0)
    mu_term = np.zeros((2, 1), dtype=float)

    events = apply_phase_ii_event_mapping(
        selector=selector,
        action=action,
        world_out=world_out,
        economic_state=econ,
        mu_term=mu_term,
    )
    events_list, summary = events
    counts = event_counts(events_list)

    assert counts["FAIL"] == 1
    assert counts["SETTLEMENT"] == 0
    assert float(selector.psi[0]) > 0.0
    assert summary.last_event == "FAIL"


def test_phase_ii_event_mapping_is_deterministic_for_same_inputs():
    action = WorldAction(weights=np.asarray([1.0, 0.0], dtype=float), gross_exposure=1.0, leverage_limit=1.0, allow_short=False)
    world_out = WorldStepResult(
        realized_return=0.1,
        costs=0.5,
        channel_returns=np.asarray([0.1, 0.0], dtype=float),
        cost_by_channel=np.asarray([0.0, 0.0], dtype=float),
        freeze=False,
    )
    mu_term = np.zeros((2, 2), dtype=float)

    sel_a = _selector_with_k(2)
    sel_b = _selector_with_k(2)
    econ_a = PhaseIIEconomicState.zeros(horizon_bins=2, h_near_idx=0)
    econ_b = PhaseIIEconomicState.zeros(horizon_bins=2, h_near_idx=0)

    events_a = apply_phase_ii_event_mapping(
        selector=sel_a,
        action=action,
        world_out=world_out,
        economic_state=econ_a,
        mu_term=mu_term,
    )
    events_b = apply_phase_ii_event_mapping(
        selector=sel_b,
        action=action,
        world_out=world_out,
        economic_state=econ_b,
        mu_term=mu_term,
    )

    events_a_list, summary_a = events_a
    events_b_list, summary_b = events_b

    assert _serialize_events(events_a_list) == _serialize_events(events_b_list)
    assert summary_a.last_event == summary_b.last_event
    assert summary_a.event_counts == summary_b.event_counts
    np.testing.assert_allclose(summary_a.channel_event_vector, summary_b.channel_event_vector, rtol=0.0, atol=0.0)
