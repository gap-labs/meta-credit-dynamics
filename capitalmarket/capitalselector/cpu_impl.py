from __future__ import annotations

from .kernel_semantics import step_at_tau


class CpuCore:
    """CPU reference backend bound to explicit event-order kernel semantics."""

    def __init__(self, *, hooks=None, policy=None, start_tau: int = 0):
        self._tau = int(start_tau)
        self._hooks = hooks
        self._policy = policy

    def step(self, selector, r_vec, c_total, *, freeze: bool) -> None:
        input_events = {
            "r_vec": r_vec,
            "c_total": c_total,
            "freeze": freeze,
        }
        _, _, next_tau = step_at_tau(
            selector,
            input_events,
            self._policy,
            self._tau,
            hooks=self._hooks,
        )
        self._tau = next_tau
