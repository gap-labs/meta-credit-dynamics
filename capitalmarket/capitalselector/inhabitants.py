from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InhabitantEntry:
    process_id: int
    generation_id: int
    tau_dead: int
    fitness: float
    final_liquidity: float
    metadata: dict[str, Any]


class InhabitantsBook:
    """Append-only Einwohnerbuch for dead processes."""

    def __init__(self):
        self._entries: list[InhabitantEntry] = []

    def append_dead(
        self,
        *,
        process_id: int,
        generation_id: int,
        tau_dead: int,
        fitness: float,
        final_liquidity: float,
        metadata: dict[str, Any] | None = None,
    ) -> InhabitantEntry:
        entry = InhabitantEntry(
            process_id=int(process_id),
            generation_id=int(generation_id),
            tau_dead=int(tau_dead),
            fitness=float(fitness),
            final_liquidity=float(final_liquidity),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> list[InhabitantEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
