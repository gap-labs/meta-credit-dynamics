from __future__ import annotations

import numpy as np

from ..interfaces import WorldAction, WorldStepResult, make_world_step_result


class DeterministicScriptWorld:
    """Deterministic world with fixed scripted returns (Profile A)."""

    def __init__(self, *, r: list[float] | None = None, c: float = 0.0):
        self._r = np.asarray(r if r is not None else [0.01, 0.0, -0.01], dtype=float)
        self._c = float(c)

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        _ = t  # unused; world is time-invariant
        return make_world_step_result(r_vec=self._r, c_total=self._c, action=action)
