from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class GovernanceWorld:
    """Governance-style world with per-process KPI/cost events.

    This world emits process-indexed events of shape:
    `pid -> {"r_vec": np.ndarray[K], "c_total": float, "freeze": bool}`.

    It additionally tracks aggregate diagnostics over time:
    - `V_hist`: governance value process
    - `M_hist`: manipulation pressure process
    - `K_hist`: blended KPI aggregate
    """

    K_channels: int
    regime: dict[str, Any]
    seed: int = 0

    V_hist: list[float] = field(default_factory=list)
    M_hist: list[float] = field(default_factory=list)
    K_hist: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.K = int(self.K_channels)
        if self.K <= 0:
            raise ValueError("K_channels must be > 0")
        self.reg = dict(self.regime)
        self.rng = np.random.default_rng(int(self.seed))

        self.V = 0.0
        self.M = 0.0

    def step(self, t: int, process_ids: list[int]) -> dict[int, dict[str, Any]]:
        _ = t
        alpha = float(self.reg["alpha"])
        manipulability = float(self.reg["manipulability"])
        reweight_speed = float(self.reg.get("reweight_speed", 0.0))
        punishment = float(self.reg["punishment"])
        volatility = float(self.reg["volatility"])

        incident = bool(self.rng.random() < (0.02 + 0.10 * punishment))
        incident_cost = float(self.rng.normal(0.0, 0.8)) if incident else 0.0

        dM = float(self.rng.normal(0.10, 0.15) * manipulability)
        # Metric-gaming effort can increase dashboard KPI while eroding true value.
        gaming_drag = float(reweight_speed * manipulability * max(dM, 0.0))
        dV = float(self.rng.normal(0.25, 0.20) - gaming_drag + incident_cost)

        self.V += dV
        self.M += dM

        # KPI mixes true value with manipulation pressure; low alpha amplifies KPI drift.
        k_total = self.V + (1.0 - alpha) * manipulability * self.M + float(self.rng.normal(0.0, 0.2))

        self.V_hist.append(float(self.V))
        self.M_hist.append(float(self.M))
        self.K_hist.append(float(k_total))

        base = self.rng.dirichlet(np.ones(self.K, dtype=float))
        events: dict[int, dict[str, Any]] = {}

        for pid in process_ids:
            pid_noise = float(self.rng.normal(0.0, 0.15))
            r_total = max(0.0, float(k_total + pid_noise))

            r_vec = base * r_total + self.rng.normal(0.0, 0.30 + 0.50 * volatility, size=self.K)
            r_vec = np.asarray(r_vec, dtype=float)

            c_total = max(
                0.0,
                float(self.rng.normal(0.10, 0.10)) + (0.35 * punishment if incident else 0.0),
            )

            events[int(pid)] = {
                "r_vec": r_vec,
                "c_total": float(c_total),
                "freeze": False,
            }

        return events
