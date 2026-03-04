from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .ledger import ClaimCapacityExceeded


class SettlementStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SettlementEvent:
    claim_id: str | None
    status: SettlementStatus
    cash_paid: float
    new_claim_ids: tuple[str, ...]
    reason: str | None = None


def _sorted_offers(state: Any):
    offers = list(getattr(state, "offers", []) or [])
    return sorted(offers, key=lambda item: str(item.offer_id))


def _materialize_repayment_claims_from_expired_offers(state: Any, tau: int) -> None:
    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is None or process_id is None:
        return

    seen = set(getattr(state, "_processed_offer_ids", set()))
    generation_id = int(getattr(state, "generation_id", 0))

    for offer in _sorted_offers(state):
        offer_id = str(offer.offer_id)
        if offer_id in seen:
            continue

        payload = dict(offer.payload or {})
        borrow_window_end_tau = int(payload.get("borrow_window_end_tau", tau))
        if int(tau) <= borrow_window_end_tau:
            continue

        drawn_principal = float(payload.get("drawn_principal", 0.0))
        if drawn_principal > 0.0:
            ledger.create_claim(
                process_id=process_id,
                generation_id=generation_id,
                creditor_id=str(payload.get("creditor_id", "creditor")),
                debtor_id=str(payload.get("debtor_id", process_id)),
                nominal=drawn_principal,
                maturity_tau=int(payload.get("repayment_due_tau", tau)),
                claim_type="repayment",
                source_offer_id=offer_id,
                drawn_principal=drawn_principal,
            )
        seen.add(offer_id)

    state._processed_offer_ids = seen


def extract_due_obligations_at_tau(state: Any, input_events: Mapping[str, Any], tau: int):
    _materialize_repayment_claims_from_expired_offers(state, tau)

    obligations: list[dict[str, Any]] = []
    c_total = float(input_events.get("c_total", 0.0))
    if c_total > 0.0:
        obligations.append(
            {
                "kind": "legacy_cash_due",
                "claim_id": None,
                "amount_due": c_total,
                "due_time": int(tau),
            }
        )

    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is not None and process_id is not None:
        for claim in ledger.claims_for_process(process_id):
            if ledger.get_status(claim.claim_id) != "open":
                continue
            if int(claim.maturity_tau) != int(tau):
                continue
            if claim.claim_type == "repayment" and float(claim.drawn_principal) <= 0.0:
                continue

            obligations.append(
                {
                    "kind": "claim_due",
                    "claim_id": claim.claim_id,
                    "amount_due": float(claim.nominal),
                    "due_time": int(claim.maturity_tau),
                    "debtor_id": claim.debtor_id,
                    "creditor_id": claim.creditor_id,
                }
            )

    obligations.sort(key=lambda item: (int(item.get("due_time", tau)), str(item.get("claim_id") or ""), str(item.get("kind", ""))))
    return obligations


def settle_due_claims_at_tau(state: Any, tau: int, rng: Any = None, config: Mapping[str, Any] | None = None, due_obligations=None):
    cfg = dict(config or {})
    lambda_cash_share = float(cfg.get("lambda_cash_share", getattr(state, "lambda_cash_share", 0.5)))
    lambda_cash_share = max(0.0, min(1.0, lambda_cash_share))
    maturity_offset = int(cfg.get("future_maturity_offset", 1))
    accept_by_default = bool(cfg.get("accept_by_default", True))

    if due_obligations is None:
        due_obligations = extract_due_obligations_at_tau(state, {"c_total": 0.0}, tau)

    events: list[SettlementEvent] = []
    unresolved: list[dict[str, Any]] = []
    settlement_failed = False
    settled_amount = 0.0

    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    generation_id = int(getattr(state, "generation_id", 0))

    for obligation in due_obligations:
        amount_due = float(obligation.get("amount_due", 0.0))
        claim_id = obligation.get("claim_id")
        force_reject = bool(obligation.get("force_reject", False))

        if amount_due <= 0.0:
            continue

        available_cash = max(0.0, float(getattr(state, "wealth", 0.0)))
        cash_part = min(available_cash, lambda_cash_share * amount_due)
        remainder = max(0.0, amount_due - cash_part)

        proposal_status = SettlementStatus.PROPOSED
        if force_reject:
            accepted = False
        else:
            accepted = bool(accept_by_default)

        if accepted and remainder > 0.0 and (ledger is None or process_id is None or claim_id is None):
            accepted = False

        new_claim_ids: list[str] = []

        if accepted and remainder > 0.0:
            try:
                rewritten = ledger.rewrite_claim(
                    claim_id=str(claim_id),
                    generation_id=generation_id,
                    closed_at=int(tau),
                    nominal=float(remainder),
                    maturity_tau=int(tau) + maturity_offset,
                )
                new_claim_ids.append(rewritten.claim_id)
            except ClaimCapacityExceeded:
                accepted = False

        if accepted:
            proposal_status = SettlementStatus.ACCEPTED
            state.wealth = float(state.wealth) - float(cash_part)
            settled_amount += float(cash_part)

            if remainder <= 0.0 and claim_id is not None and ledger is not None:
                ledger.close_claim(claim_id=str(claim_id), closed_at=int(tau), status="consumed")

            events.append(
                SettlementEvent(
                    claim_id=None if claim_id is None else str(claim_id),
                    status=proposal_status,
                    cash_paid=float(cash_part),
                    new_claim_ids=tuple(new_claim_ids),
                    reason=None,
                )
            )
            continue

        proposal_status = SettlementStatus.REJECTED
        if float(getattr(state, "wealth", 0.0)) >= amount_due:
            state.wealth = float(state.wealth) - amount_due
            settled_amount += amount_due
            if claim_id is not None and ledger is not None:
                ledger.close_claim(claim_id=str(claim_id), closed_at=int(tau), status="consumed")
            events.append(
                SettlementEvent(
                    claim_id=None if claim_id is None else str(claim_id),
                    status=proposal_status,
                    cash_paid=float(amount_due),
                    new_claim_ids=tuple(),
                    reason="rejected_but_paid_full_cash",
                )
            )
        else:
            settlement_failed = True
            unresolved.append(obligation)
            events.append(
                SettlementEvent(
                    claim_id=None if claim_id is None else str(claim_id),
                    status=proposal_status,
                    cash_paid=0.0,
                    new_claim_ids=tuple(),
                    reason="rejected_and_insufficient_cash",
                )
            )

    state._last_settlement_failed = bool(settlement_failed)
    state._last_settlement_events = events
    return state, {getattr(state, "process_id", 0): bool(settlement_failed)}, {
        "obligations_after": unresolved,
        "settled_amount": float(settled_amount),
        "settlement_failed": bool(settlement_failed),
        "events": events,
    }
