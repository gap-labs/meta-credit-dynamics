from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Offer:
    offer_id: str
    process_id: int | str
    generation_id: int
    payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Claim:
    claim_id: str
    parent_claim_id: str | None
    process_id: int | str
    generation_id: int
    creditor_id: str
    debtor_id: str
    nominal: float
    maturity_tau: int
    claim_type: str = "generic"
    source_offer_id: str | None = None
    drawn_principal: float = 0.0


class IdAllocator:
    """Deterministic, process-local monotonic ID allocator."""

    def __init__(self):
        self._offer_seq: dict[int | str, int] = {}
        self._claim_seq: dict[int | str, int] = {}

    def next_offer_id(self, process_id: int | str) -> str:
        next_value = self._offer_seq.get(process_id, 0) + 1
        self._offer_seq[process_id] = next_value
        return f"{process_id}:offer:{next_value}"

    def next_claim_id(self, process_id: int | str) -> str:
        next_value = self._claim_seq.get(process_id, 0) + 1
        self._claim_seq[process_id] = next_value
        return f"{process_id}:claim:{next_value}"
