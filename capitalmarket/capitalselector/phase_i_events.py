from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from .phase_i_bucket import DEFAULT_HORIZON_BUCKET_CONFIG, HorizonBucketConfig, phi


EventCategory = Literal["RETURN", "DUE_CASH", "COST", "ROLLOVER", "FAIL"]


@dataclass(frozen=True)
class AttributionEvent:
    category: EventCategory
    channel: int
    horizon: int
    pi: float
    r: float


def _channel_count(state: Any) -> int:
    w = getattr(state, "w", None)
    if w is not None:
        return int(len(w))
    return int(getattr(state, "K", 0))


def _clamp_channel(channel: int, channel_count: int) -> int:
    if channel_count <= 0:
        return 0
    return max(0, min(int(channel), int(channel_count) - 1))


def _dominant_trigger_channel(state: Any, r_vec: np.ndarray) -> int:
    channel_count = _channel_count(state)

    if r_vec.size > 0:
        idx = int(np.argmax(np.abs(r_vec)))
        return _clamp_channel(idx, channel_count)

    w = getattr(state, "w", None)
    if w is not None and len(w) > 0:
        return _clamp_channel(int(np.argmax(np.asarray(w, dtype=float))), channel_count)

    return _clamp_channel(0, channel_count)


def _offer_by_id(state: Any) -> dict[str, Any]:
    offers = list(getattr(state, "offers", []) or [])
    offers.sort(key=lambda item: str(getattr(item, "offer_id", "")))
    return {str(getattr(offer, "offer_id", "")): offer for offer in offers}


def _offer_channel(offer: Any) -> int | None:
    payload = dict(getattr(offer, "payload", {}) or {})
    for key in ("channel", "channel_id", "channel_idx", "origin_channel_id"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
    return None


def _claim_from_state(state: Any, claim_id: str | None) -> Any | None:
    if claim_id is None:
        return None
    ledger = getattr(state, "claim_ledger", None)
    if ledger is None:
        return None
    try:
        return ledger.get_claim(str(claim_id))
    except Exception:
        return None


def _channel_for_claim(state: Any, claim_id: str | None, fallback_channel: int, offer_index: Mapping[str, Any]) -> int:
    channel_count = _channel_count(state)
    claim = _claim_from_state(state, claim_id)
    if claim is None:
        return _clamp_channel(fallback_channel, channel_count)

    source_offer_id = getattr(claim, "source_offer_id", None)
    if source_offer_id is not None:
        offer = offer_index.get(str(source_offer_id))
        if offer is not None:
            offer_channel = _offer_channel(offer)
            if offer_channel is not None:
                return _clamp_channel(int(offer_channel), channel_count)

    return _clamp_channel(fallback_channel, channel_count)


def _horizon_for_claim_id(
    state: Any,
    claim_id: str | None,
    tau: int,
    buckets: HorizonBucketConfig,
) -> int:
    claim = _claim_from_state(state, claim_id)
    if claim is None:
        return phi(0, buckets)

    try:
        maturity = int(getattr(claim, "maturity_tau"))
    except (TypeError, ValueError):
        return phi(0, buckets)

    delta_tau = max(0, int(maturity) - int(tau))
    return phi(delta_tau, buckets)


def _horizon_for_due_item(item: Mapping[str, Any], tau: int, buckets: HorizonBucketConfig) -> int:
    if "due_time" not in item:
        return phi(0, buckets)
    try:
        due_time = int(item.get("due_time", tau))
    except (TypeError, ValueError):
        return phi(0, buckets)
    return phi(max(0, due_time - int(tau)), buckets)


def _status_value(raw_status: Any) -> str:
    return str(getattr(raw_status, "value", raw_status or "")).strip().lower()


def _iter_settlement_events(settlement_result: Mapping[str, Any]) -> Iterable[Any]:
    return list(settlement_result.get("events", []) or [])


def psi(
    *,
    state: Any,
    tau: int,
    input_events: Mapping[str, Any],
    due_returns: Mapping[str, Any],
    due_obligations: list[Mapping[str, Any]],
    settlement_result: Mapping[str, Any],
    bucket_config: HorizonBucketConfig = DEFAULT_HORIZON_BUCKET_CONFIG,
) -> list[AttributionEvent]:
    """Map existing runtime signals into local Phase-I attribution events."""

    events: list[AttributionEvent] = []
    r_vec = np.asarray(due_returns.get("r_vec", input_events.get("r_vec", [])), dtype=float)
    fallback_channel = _dominant_trigger_channel(state, r_vec)
    offer_index = _offer_by_id(state)

    # RETURN events: positive realized channel returns.
    for channel, value in enumerate(r_vec):
        amount = float(value)
        if amount <= 0.0:
            continue
        events.append(
            AttributionEvent(
                category="RETURN",
                channel=_clamp_channel(int(channel), _channel_count(state)),
                horizon=phi(0, bucket_config),
                pi=amount,
                r=0.0,
            )
        )

    # Due obligations are mapped to cash/cost cashflow events.
    for item in list(due_obligations or []):
        kind = str(item.get("kind", "")).strip().lower()
        amount_due = float(item.get("amount_due", 0.0))
        if amount_due <= 0.0:
            continue

        if kind == "legacy_cash_due":
            events.append(
                AttributionEvent(
                    category="DUE_CASH",
                    channel=_clamp_channel(fallback_channel, _channel_count(state)),
                    horizon=phi(0, bucket_config),
                    pi=-amount_due,
                    r=0.0,
                )
            )
            continue

        if kind == "claim_due":
            claim_id = item.get("claim_id")
            events.append(
                AttributionEvent(
                    category="COST",
                    channel=_channel_for_claim(state, None if claim_id is None else str(claim_id), fallback_channel, offer_index),
                    horizon=_horizon_for_claim_id(state, None if claim_id is None else str(claim_id), int(tau), bucket_config)
                    if claim_id is not None
                    else _horizon_for_due_item(item, int(tau), bucket_config),
                    pi=-amount_due,
                    r=0.0,
                )
            )

    # Settlement outcomes are mapped to risk impulses.
    fail_emitted = False
    for settlement_event in _iter_settlement_events(settlement_result):
        if isinstance(settlement_event, Mapping):
            raw_status = settlement_event.get("status")
            claim_id = settlement_event.get("claim_id")
            reason = settlement_event.get("reason")
            new_claim_ids = tuple(settlement_event.get("new_claim_ids", ()) or ())
        else:
            raw_status = getattr(settlement_event, "status", None)
            claim_id = getattr(settlement_event, "claim_id", None)
            reason = getattr(settlement_event, "reason", None)
            new_claim_ids = tuple(getattr(settlement_event, "new_claim_ids", ()) or ())

        status = _status_value(raw_status)
        claim_id_str = None if claim_id is None else str(claim_id)

        if status == "accepted" and len(new_claim_ids) > 0:
            first_new_claim_id = str(new_claim_ids[0])
            events.append(
                AttributionEvent(
                    category="ROLLOVER",
                    channel=_channel_for_claim(state, claim_id_str, fallback_channel, offer_index),
                    horizon=_horizon_for_claim_id(state, first_new_claim_id, int(tau), bucket_config),
                    pi=0.0,
                    r=1.0,
                )
            )

        if status == "rejected" and str(reason) == "rejected_and_insufficient_cash":
            fail_emitted = True
            events.append(
                AttributionEvent(
                    category="FAIL",
                    channel=_channel_for_claim(state, claim_id_str, fallback_channel, offer_index),
                    horizon=_horizon_for_claim_id(state, claim_id_str, int(tau), bucket_config),
                    pi=0.0,
                    r=1.0,
                )
            )

    if bool(settlement_result.get("settlement_failed", False)) and not fail_emitted:
        events.append(
            AttributionEvent(
                category="FAIL",
                channel=_clamp_channel(fallback_channel, _channel_count(state)),
                horizon=phi(0, bucket_config),
                pi=0.0,
                r=1.0,
            )
        )

    return events


def update_rho_from_events(state: Any, events: Iterable[AttributionEvent]) -> None:
    """Apply rho[channel] EWMA updates for all risk-bearing attribution events."""

    updater = getattr(state, "update_rho", None)
    if not callable(updater):
        return

    for event in events:
        if float(event.r) == 0.0:
            continue
        updater(int(event.channel), float(event.r))


def update_mu_from_events(state: Any, events: Iterable[AttributionEvent]) -> None:
    """Apply mu[channel, horizon] EWMA updates for all attributed cashflow events."""

    updater = getattr(state, "update_term_mu", None)
    if not callable(updater):
        return

    for event in events:
        updater(int(event.channel), int(event.horizon), float(event.pi))
