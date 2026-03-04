from __future__ import annotations

import math


class BurndownPool:
    def __init__(self, *, initial: float = 0.0, rounding_scale: int = 12):
        self.B_current = max(0.0, float(initial))
        self._scale = int(rounding_scale)
        self._unit = 10 ** (-self._scale)

    def apply_tau_inflows(self, *, burn: float, kappa: float, jackpot: float = 0.0) -> float:
        burn_term = max(0.0, float(kappa) * max(0.0, float(burn)))
        jackpot_term = max(0.0, float(jackpot))
        self.B_current = max(0.0, float(self.B_current) + burn_term + jackpot_term)
        return float(self.B_current)

    def allocate_fair_same_tau(
        self,
        *,
        requested: list[float],
        stable_keys: list[tuple[int, int]],
        epsilon: float = 1e-12,
    ) -> list[float]:
        if len(requested) != len(stable_keys):
            raise ValueError("requested and stable_keys must have equal length")

        if not requested:
            return []

        total_requested = float(sum(max(0.0, float(value)) for value in requested))
        if total_requested < float(epsilon):
            return [0.0 for _ in requested]

        indexed = list(enumerate(zip(stable_keys, requested)))
        indexed.sort(key=lambda item: item[1][0])

        if total_requested <= self.B_current:
            allocations = [max(0.0, float(value)) for value in requested]
        else:
            factor = 0.0 if self.B_current <= 0.0 else float(self.B_current) / total_requested
            raw = [max(0.0, float(value)) * factor for value in requested]
            floored = [math.floor(value * (10 ** self._scale)) * self._unit for value in raw]
            allocations = floored
            consumed = float(sum(allocations))
            residual = max(0.0, float(self.B_current) - consumed)
            quanta = int(round(residual * (10 ** self._scale)))
            if quanta > 0:
                order = [index for index, _pair in indexed]
                for offset in range(quanta):
                    target = order[offset % len(order)]
                    allocations[target] += self._unit

        total_alloc = float(sum(allocations))
        total_alloc = min(total_alloc, float(self.B_current))
        self.B_current = max(0.0, float(self.B_current) - total_alloc)
        return allocations
