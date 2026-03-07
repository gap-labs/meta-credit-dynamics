from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..interfaces import WorldAction, WorldStepResult, make_world_step_result


@dataclass
class DeterministicClusterWorld:
    """Deterministic world with a persistent clustered return signal."""

    r_vec: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.r_vec is None:
            self.r_vec = np.array([0.02, 0.02, 0.0, -0.01, -0.01], dtype=float)
        else:
            self.r_vec = np.asarray(self.r_vec, dtype=float)

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        _ = t
        return make_world_step_result(r_vec=self.r_vec.copy(), c_total=0.0, action=action)
