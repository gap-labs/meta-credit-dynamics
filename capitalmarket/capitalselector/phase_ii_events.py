from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .interfaces import WorldAction, WorldStepResult
from .phase_ii_state import PhaseIIEconomicState


PhaseIIEventCategory = Literal[
    "RETURN",
    "DUE_CASH",
    "COST",
    "ROLLOVER",
    "FAIL",
    "SETTLEMENT",
]

PHASE_II_EVENT_CATEGORIES: tuple[PhaseIIEventCategory, ...] = (
    "RETURN",
    "DUE_CASH",
    "COST",
    "ROLLOVER",
    "FAIL",
    "SETTLEMENT",
)


@dataclass(frozen=True)
class PhaseIIEvent:
    category: PhaseIIEventCategory
    channel: int
    horizon: int
    amount: float
    rollover_to_horizon: int | None = None


@dataclass(frozen=True)
class EventSummary:
    event_counts: dict[str, int]
    last_event: str | None
    channel_event_vector: np.ndarray


def _dominant_channel(weights: np.ndarray) -> int:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        return 0
    return int(np.argmax(np.abs(w)))


def _phase_ii_events(
    *,
    action: WorldAction,
    world_out: WorldStepResult,
    economic_state: PhaseIIEconomicState,
    mu_term: np.ndarray,
) -> list[PhaseIIEvent]:
    weights = np.asarray(action.weights, dtype=float)
    K = int(weights.shape[0])
    if K <= 0:
        return []

    dominant = _dominant_channel(weights)

    H = int(economic_state.due_curve.shape[0])
    horizon_map = np.minimum(np.arange(K, dtype=int), max(0, H - 1))

    channel_returns = np.asarray(world_out.channel_returns, dtype=float)
    if channel_returns.shape != weights.shape:
        raise ValueError("world_out.channel_returns shape must match action.weights")

    events: list[PhaseIIEvent] = [
        PhaseIIEvent(
            category="RETURN",
            channel=int(dominant),
            horizon=int(horizon_map[dominant]),
            amount=float(world_out.realized_return),
        )
    ]

    # Deterministic channel order for cost attribution.
    cost_vec = np.asarray(world_out.cost_by_channel, dtype=float)
    if cost_vec.ndim == 1 and cost_vec.shape[0] == K and np.any(np.abs(cost_vec) > 0.0):
        allocated_cost = np.maximum(0.0, cost_vec)
    else:
        # Fallback: allocate by absolute exposure share.
        exposure = np.abs(weights)
        denom = float(exposure.sum())
        if denom <= 0.0 or float(world_out.costs) <= 0.0:
            allocated_cost = np.zeros(K, dtype=float)
        else:
            allocated_cost = exposure * (float(world_out.costs) / denom)

    for c in range(K):
        amount = float(allocated_cost[c])
        if amount <= 0.0:
            continue
        h = int(horizon_map[c])
        events.append(
            PhaseIIEvent(
                category="COST",
                channel=int(c),
                horizon=h,
                amount=amount,
            )
        )
        events.append(
            PhaseIIEvent(
                category="DUE_CASH",
                channel=int(c),
                horizon=h,
                amount=amount,
            )
        )

    # Use a deterministic projected state to choose rollover/fail/settlement events.
    projected = economic_state.copy()
    for event in events:
        if event.category == "DUE_CASH" and float(event.amount) > 0.0:
            projected.apply_due_cash(horizon_bin=int(event.horizon), amount=float(event.amount))

    mismatch = float(projected.update_liquidity_mismatch(weights=weights, mu_term=mu_term))

    if mismatch > 0.0:
        near_idx = int(projected.h_near_idx)
        src = max(0, min(int(projected.due_curve.shape[0]) - 1, near_idx))
        dst = min(int(projected.due_curve.shape[0]) - 1, int(projected.h_near_idx) + 1)
        can_roll = int(projected.due_curve.shape[0]) > 1 and dst != src and float(projected.due_curve[src]) > 0.0
        if can_roll:
            rollover_amt = min(float(projected.due_curve[src]), float(mismatch))
            if rollover_amt > 0.0:
                events.append(
                    PhaseIIEvent(
                        category="ROLLOVER",
                        channel=int(dominant),
                        horizon=int(src),
                        amount=float(rollover_amt),
                        rollover_to_horizon=int(dst),
                    )
                )
                projected.apply_rollover(from_bin=int(src), to_bin=int(dst), amount=float(rollover_amt))
                mismatch = float(projected.update_liquidity_mismatch(weights=weights, mu_term=mu_term))

        if mismatch > 0.0:
            events.append(
                PhaseIIEvent(
                    category="FAIL",
                    channel=int(dominant),
                    horizon=int(src),
                    amount=float(mismatch),
                )
            )
        else:
            events.append(
                PhaseIIEvent(
                    category="SETTLEMENT",
                    channel=int(dominant),
                    horizon=int(src),
                    amount=0.0,
                )
            )
    else:
        events.append(
            PhaseIIEvent(
                category="SETTLEMENT",
                channel=int(dominant),
                horizon=int(max(0, min(int(projected.due_curve.shape[0]) - 1, int(projected.h_near_idx)))),
                amount=0.0,
            )
        )

    return events


def apply_phase_ii_event_mapping(
    *,
    selector: Any,
    action: WorldAction,
    world_out: WorldStepResult,
    economic_state: PhaseIIEconomicState,
    mu_term: np.ndarray,
) -> tuple[list[PhaseIIEvent], EventSummary]:
    """Apply deterministic Phase-II event mapping with EventSummary boundary."""
    events = _phase_ii_events(
        action=action,
        world_out=world_out,
        economic_state=economic_state,
        mu_term=mu_term,
    )

    K = int(np.asarray(action.weights, dtype=float).shape[0])
    summary = build_event_summary(events, channel_count=K)

    apply_phase_ii_economic_updates(
        action=action,
        economic_state=economic_state,
        mu_term=mu_term,
        events=events,
    )
    apply_phase_ii_selector_updates(
        selector=selector,
        action=action,
        world_out=world_out,
        events=events,
        summary=summary,
    )

    return events, summary


def apply_phase_ii_economic_updates(
    *,
    action: WorldAction,
    economic_state: PhaseIIEconomicState,
    mu_term: np.ndarray,
    events: list[PhaseIIEvent],
) -> None:
    """Apply deterministic Economic/Settlement-side event mapping updates."""
    for event in events:
        if event.category == "DUE_CASH":
            economic_state.apply_due_cash(horizon_bin=int(event.horizon), amount=float(event.amount))
            continue

        if event.category == "ROLLOVER":
            if event.rollover_to_horizon is None:
                raise ValueError("ROLLOVER event requires rollover_to_horizon")
            economic_state.apply_rollover(
                from_bin=int(event.horizon),
                to_bin=int(event.rollover_to_horizon),
                amount=float(event.amount),
            )
            continue

        if event.category == "SETTLEMENT":
            economic_state.update_liquidity_mismatch(
                weights=np.asarray(action.weights, dtype=float),
                mu_term=np.asarray(mu_term, dtype=float),
            )


def apply_phase_ii_selector_updates(
    *,
    selector: Any,
    action: WorldAction | None = None,
    world_out: WorldStepResult,
    events: list[PhaseIIEvent] | None = None,
    summary: EventSummary,
) -> None:
    """Apply deterministic Selector-side updates from EventSummary boundary."""
    K = int(np.asarray(summary.channel_event_vector, dtype=float).shape[0])
    if not hasattr(selector, "phase_ii_cost_by_channel") or len(np.asarray(selector.phase_ii_cost_by_channel, dtype=float)) != K:
        selector.phase_ii_cost_by_channel = np.zeros(K, dtype=float)

    counts = {name: int(summary.event_counts.get(name, 0)) for name in PHASE_II_EVENT_CATEGORIES}
    channel_vec = np.asarray(summary.channel_event_vector, dtype=float)

    if counts["RETURN"] > 0:
        selector.stats.update(float(world_out.realized_return))

    if counts["COST"] > 0 and float(world_out.costs) != 0.0:
        weight_norm = float(np.abs(channel_vec).sum())
        if weight_norm <= 0.0:
            cost_alloc = np.zeros_like(channel_vec, dtype=float)
        else:
            cost_alloc = np.abs(channel_vec) / weight_norm
        selector.phase_ii_cost_by_channel = np.asarray(selector.phase_ii_cost_by_channel, dtype=float) + (
            cost_alloc * float(world_out.costs)
        )

    if counts["FAIL"] > 0:
        updater = getattr(selector, "update_psi", None)
        if callable(updater) and K > 0:
            fail_channel = int(np.argmax(np.abs(channel_vec)))
            updater(fail_channel, 1.0)

    # Feed selector statistics from deterministic event stream.
    if events is None:
        events = []

    returns = np.asarray(world_out.channel_returns, dtype=float)
    update_term_mu = getattr(selector, "update_term_mu", None)
    update_rho = getattr(selector, "update_rho", None)
    update_psi = getattr(selector, "update_psi", None)

    # Keep per-channel term expectation updates deterministic and independent
    # from the event count contract (which emits a single RETURN event).
    if callable(update_term_mu) and returns.ndim == 1:
        horizon_count = int(max(1, getattr(selector, "horizon_count", 1)))
        for c in range(int(returns.shape[0])):
            h_idx = int(min(c, horizon_count - 1))
            update_term_mu(c, h_idx, float(returns[c]))

    for event in events:
        channel = int(event.channel)
        horizon = int(event.horizon)

        if event.category == "COST" and callable(update_rho):
            update_rho(channel, max(0.0, float(event.amount)))
            continue

        if event.category == "ROLLOVER" and callable(update_rho):
            update_rho(channel, max(0.0, float(event.amount)))
            continue

        if event.category == "SETTLEMENT" and callable(update_rho):
            update_rho(channel, 0.0)
            continue

        if event.category == "FAIL":
            if callable(update_psi):
                update_psi(channel, 1.0)
            if callable(update_rho):
                update_rho(channel, 1.0)

def event_counts(events: list[PhaseIIEvent]) -> dict[str, int]:
    counts = {name: 0 for name in PHASE_II_EVENT_CATEGORIES}
    for event in events:
        counts[str(event.category)] = int(counts[str(event.category)]) + 1
    return counts


def build_event_summary(events: list[PhaseIIEvent], *, channel_count: int) -> EventSummary:
    """Build deterministic EventSummary boundary artifact from step-local events."""
    K = max(0, int(channel_count))
    vector = np.zeros(K, dtype=float)

    for event in events:
        channel = int(event.channel)
        if K > 0 and 0 <= channel < K:
            vector[channel] = float(vector[channel]) + 1.0

    counts = event_counts(events)
    last_event = None if len(events) == 0 else str(events[-1].category)
    return EventSummary(
        event_counts=counts,
        last_event=last_event,
        channel_event_vector=vector,
    )
