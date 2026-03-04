from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import torch

from .claims import Claim, IdAllocator


@dataclass(frozen=True)
class ClaimCapacityExceeded(Exception):
    process_id: int | str
    max_claims_per_process: int

    def __str__(self) -> str:
        return (
            f"claim capacity exceeded for process={self.process_id}; "
            f"max_claims_per_process={self.max_claims_per_process}"
        )


class ClaimLedger:
    """Append-only claim storage with lineage and controlled creation point."""

    def __init__(self, *, id_allocator: IdAllocator | None = None, max_claims_per_process: int = 1_000_000):
        if int(max_claims_per_process) <= 0:
            raise ValueError("max_claims_per_process must be > 0")

        self._ids = id_allocator or IdAllocator()
        self._max_claims_per_process = int(max_claims_per_process)

        self._claims_by_id: dict[str, Claim] = {}
        self._claims_by_process: dict[int | str, list[str]] = defaultdict(list)
        self._claim_slot_by_id: dict[str, int] = {}

        self._nominal_by_process: dict[int | str, torch.Tensor] = defaultdict(lambda: torch.empty((0,), dtype=torch.float64, device="cpu"))
        self._maturity_by_process: dict[int | str, torch.Tensor] = defaultdict(lambda: torch.empty((0,), dtype=torch.int32, device="cpu"))
        self._generation_by_process: dict[int | str, torch.Tensor] = defaultdict(lambda: torch.empty((0,), dtype=torch.int32, device="cpu"))
        self._open_mask_by_process: dict[int | str, torch.Tensor] = defaultdict(lambda: torch.empty((0,), dtype=torch.bool, device="cpu"))

        self._status_by_id: dict[str, str] = {}
        self._closed_at_by_id: dict[str, int | None] = {}

    @property
    def max_claims_per_process(self) -> int:
        return self._max_claims_per_process

    def create_claim(
        self,
        *,
        process_id: int | str,
        generation_id: int,
        creditor_id: str,
        debtor_id: str,
        nominal: float,
        maturity_tau: int,
        parent_claim_id: str | None = None,
        claim_type: str = "generic",
        source_offer_id: str | None = None,
        drawn_principal: float = 0.0,
    ) -> Claim:
        self._assert_capacity(process_id)

        claim = Claim(
            claim_id=self._ids.next_claim_id(process_id),
            parent_claim_id=parent_claim_id,
            process_id=process_id,
            generation_id=int(generation_id),
            creditor_id=str(creditor_id),
            debtor_id=str(debtor_id),
            nominal=float(nominal),
            maturity_tau=int(maturity_tau),
            claim_type=str(claim_type),
            source_offer_id=None if source_offer_id is None else str(source_offer_id),
            drawn_principal=float(drawn_principal),
        )
        self._register_claim(claim)
        return claim

    def rewrite_claim(
        self,
        *,
        claim_id: str,
        generation_id: int,
        closed_at: int,
        nominal: float,
        maturity_tau: int,
    ) -> Claim:
        old_claim = self._claims_by_id[claim_id]
        if self._status_by_id[claim_id] != "open":
            raise ValueError(f"claim is not open: {claim_id}")

        self._status_by_id[claim_id] = "consumed"
        self._closed_at_by_id[claim_id] = int(closed_at)
        self._mark_claim_closed_in_tensor_store(claim_id)

        return self.create_claim(
            process_id=old_claim.process_id,
            generation_id=generation_id,
            creditor_id=old_claim.creditor_id,
            debtor_id=old_claim.debtor_id,
            nominal=nominal,
            maturity_tau=maturity_tau,
            parent_claim_id=old_claim.claim_id,
            claim_type=old_claim.claim_type,
            source_offer_id=old_claim.source_offer_id,
            drawn_principal=old_claim.drawn_principal,
        )

    def close_claim(self, *, claim_id: str, closed_at: int, status: str = "consumed") -> None:
        if claim_id not in self._claims_by_id:
            raise KeyError(claim_id)
        if self._status_by_id[claim_id] != "open":
            raise ValueError(f"claim is not open: {claim_id}")
        self._status_by_id[claim_id] = str(status)
        self._closed_at_by_id[claim_id] = int(closed_at)
        self._mark_claim_closed_in_tensor_store(claim_id)

    def get_claim(self, claim_id: str) -> Claim:
        return self._claims_by_id[claim_id]

    def get_status(self, claim_id: str) -> str:
        return self._status_by_id[claim_id]

    def get_closed_at(self, claim_id: str) -> int | None:
        return self._closed_at_by_id[claim_id]

    def claims_for_process(self, process_id: int | str) -> list[Claim]:
        return [self._claims_by_id[cid] for cid in self._claims_by_process.get(process_id, [])]

    def claim_tensor_batch_for_process(
        self,
        *,
        process_id: int | str,
        start_index: int,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | int]:
        start = int(start_index)
        total = len(self._claims_by_process.get(process_id, []))
        if start < 0:
            start = 0
        if start >= total:
            empty_float = torch.empty((0,), device=device, dtype=float_dtype)
            empty_int = torch.empty((0,), device=device, dtype=torch.int32)
            empty_bool = torch.empty((0,), device=device, dtype=torch.bool)
            return {
                "batch_len": 0,
                "nominal": empty_float,
                "maturity_tau": empty_int,
                "generation_id": empty_int,
                "claim_target": empty_int,
                "is_open": empty_bool,
            }

        nominal = self._nominal_by_process[process_id][start:total].to(device=device, dtype=float_dtype)
        maturity_tau = self._maturity_by_process[process_id][start:total].to(device=device, dtype=torch.int32)
        generation_id = self._generation_by_process[process_id][start:total].to(device=device, dtype=torch.int32)
        is_open = self._open_mask_by_process[process_id][start:total].to(device=device, dtype=torch.bool)
        claim_target = torch.arange(start, total, device=device, dtype=torch.int32)
        return {
            "batch_len": int(total - start),
            "nominal": nominal,
            "maturity_tau": maturity_tau,
            "generation_id": generation_id,
            "claim_target": claim_target,
            "is_open": is_open,
        }

    def lineage_chain(self, claim_id: str) -> list[Claim]:
        chain: list[Claim] = []
        current = self._claims_by_id[claim_id]
        while True:
            chain.append(current)
            if current.parent_claim_id is None:
                break
            current = self._claims_by_id[current.parent_claim_id]
        chain.reverse()
        return chain

    def _register_claim(self, claim: Claim) -> None:
        process_id = claim.process_id
        slot = len(self._claims_by_process[process_id])
        self._claims_by_id[claim.claim_id] = claim
        self._claims_by_process[process_id].append(claim.claim_id)
        self._claim_slot_by_id[claim.claim_id] = int(slot)

        nominal_new = torch.as_tensor([float(claim.nominal)], dtype=torch.float64, device="cpu")
        maturity_new = torch.as_tensor([int(claim.maturity_tau)], dtype=torch.int32, device="cpu")
        generation_new = torch.as_tensor([int(claim.generation_id)], dtype=torch.int32, device="cpu")
        open_new = torch.as_tensor([True], dtype=torch.bool, device="cpu")

        self._nominal_by_process[process_id] = torch.cat((self._nominal_by_process[process_id], nominal_new), dim=0)
        self._maturity_by_process[process_id] = torch.cat((self._maturity_by_process[process_id], maturity_new), dim=0)
        self._generation_by_process[process_id] = torch.cat((self._generation_by_process[process_id], generation_new), dim=0)
        self._open_mask_by_process[process_id] = torch.cat((self._open_mask_by_process[process_id], open_new), dim=0)

        self._status_by_id[claim.claim_id] = "open"
        self._closed_at_by_id[claim.claim_id] = None

    def _mark_claim_closed_in_tensor_store(self, claim_id: str) -> None:
        claim = self._claims_by_id[claim_id]
        process_id = claim.process_id
        slot = int(self._claim_slot_by_id[claim_id])
        self._open_mask_by_process[process_id][slot] = False

    def _assert_capacity(self, process_id: int | str) -> None:
        if len(self._claims_by_process[process_id]) >= self._max_claims_per_process:
            raise ClaimCapacityExceeded(process_id, self._max_claims_per_process)
