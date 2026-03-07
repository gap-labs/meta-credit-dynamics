from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import WorldStepResult
from capitalmarket.capitalselector.phase_ii_events import (
    EventSummary,
    PhaseIIEvent,
    apply_phase_ii_selector_updates,
    build_event_summary,
)


def _selector_with_k(k: int):
    selector = CapitalSelectorBuilder().with_K(k).build()
    selector.ensure_channel_state(k)
    selector.w = np.ones(k, dtype=float) / float(max(1, k))
    return selector


def test_event_summary_is_built_deterministically_from_ordered_events():
    events = [
        PhaseIIEvent(category="RETURN", channel=1, horizon=0, amount=0.3),
        PhaseIIEvent(category="COST", channel=1, horizon=0, amount=0.1),
        PhaseIIEvent(category="FAIL", channel=1, horizon=0, amount=0.2),
    ]

    a = build_event_summary(events, channel_count=3)
    b = build_event_summary(events, channel_count=3)

    assert a.last_event == "FAIL"
    assert a.event_counts == b.event_counts
    np.testing.assert_allclose(a.channel_event_vector, b.channel_event_vector, rtol=0.0, atol=0.0)


def test_selector_updates_consume_event_summary_boundary():
    selector = _selector_with_k(3)
    world_out = WorldStepResult(
        realized_return=0.4,
        costs=0.3,
        channel_returns=np.asarray([0.4, 0.0, 0.0], dtype=float),
        cost_by_channel=np.asarray([0.0, 0.0, 0.0], dtype=float),
        freeze=False,
    )
    summary = EventSummary(
        event_counts={
            "RETURN": 1,
            "DUE_CASH": 0,
            "COST": 1,
            "ROLLOVER": 0,
            "FAIL": 1,
            "SETTLEMENT": 0,
        },
        last_event="FAIL",
        channel_event_vector=np.asarray([0.0, 2.0, 0.0], dtype=float),
    )

    apply_phase_ii_selector_updates(selector=selector, world_out=world_out, summary=summary)

    assert float(selector.stats.mu) > 0.0
    assert float(selector.psi[1]) > 0.0
    np.testing.assert_allclose(
        np.asarray(selector.phase_ii_cost_by_channel, dtype=float),
        np.asarray([0.0, 0.3, 0.0], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
