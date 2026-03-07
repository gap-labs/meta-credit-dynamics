from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - defensive import guard
    torch = None

HAS_CUDA = bool(torch is not None and torch.cuda.is_available())

from capitalmarket.capitalselector.experiments.run_phase_ii import PhaseIIConfig, run_phase_ii_episode
from capitalmarket.capitalselector.interfaces import make_world_step_result

RTOL = 1e-7
ATOL = 1e-9


class DeterministicParityWorld:
    def step(self, t: int, action):
        t_f = float(int(t))
        r_vec = np.asarray([0.02 + 0.0007 * t_f, -0.005 + 0.0003 * t_f, 0.001], dtype=float)
        c_total = 0.01 + 0.00005 * t_f
        return make_world_step_result(r_vec=r_vec, c_total=c_total, action=action)


def _run(seed: int, backend: str) -> dict:
    return run_phase_ii_episode(
        world=DeterministicParityWorld(),
        steps=15,
        channels=3,
        seed=int(seed),
        config=PhaseIIConfig(selector_policy="term_risk", h_near_idx=1),
        backend=backend,
    )


def _assert_close(cpu: np.ndarray, cuda: np.ndarray) -> None:
    bound = ATOL + RTOL * np.abs(cpu)
    diff = np.abs(cpu - cuda)
    assert np.all(diff <= bound), f"max diff {float(diff.max())} exceeds bound {float(bound.max())}"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA unavailable")
def test_phase_ii_cpu_cuda_parity():
    cpu = _run(seed=23, backend="cpu")
    cuda = _run(seed=23, backend="cuda")

    # Exact-match parity requirements.
    assert cpu["exact_metrics"]["terminal_dead"] == cuda["exact_metrics"]["terminal_dead"]
    assert cpu["exact_metrics"]["event_counts"] == cuda["exact_metrics"]["event_counts"]
    assert cpu["exact_metrics"]["rollover_count"] == cuda["exact_metrics"]["rollover_count"]
    assert cpu["exact_metrics"]["last_timestep"] == cuda["exact_metrics"]["last_timestep"]

    cpu_t = np.asarray([int(step["t"]) for step in cpu["history"]], dtype=int)
    cuda_t = np.asarray([int(step["t"]) for step in cuda["history"]], dtype=int)
    np.testing.assert_array_equal(cpu_t, cuda_t)

    # Numerical parity requirements.
    _assert_close(
        np.asarray([float(step["wealth"]) for step in cpu["history"]], dtype=float),
        np.asarray([float(step["wealth"]) for step in cuda["history"]], dtype=float),
    )
    _assert_close(
        np.asarray([float(step["realized_return"]) for step in cpu["history"]], dtype=float),
        np.asarray([float(step["realized_return"]) for step in cuda["history"]], dtype=float),
    )
    _assert_close(
        np.asarray([float(step["strategic_credit_exposure"]) for step in cpu["history"]], dtype=float),
        np.asarray([float(step["strategic_credit_exposure"]) for step in cuda["history"]], dtype=float),
    )

    cpu_weights = np.asarray([np.asarray(step["weights"], dtype=float) for step in cpu["history"]], dtype=float)
    cuda_weights = np.asarray([np.asarray(step["weights"], dtype=float) for step in cuda["history"]], dtype=float)
    _assert_close(cpu_weights, cuda_weights)

    cpu_ev = np.asarray(
        [np.asarray(step["event_summary"]["channel_event_vector"], dtype=float) for step in cpu["history"]],
        dtype=float,
    )
    cuda_ev = np.asarray(
        [np.asarray(step["event_summary"]["channel_event_vector"], dtype=float) for step in cuda["history"]],
        dtype=float,
    )
    _assert_close(cpu_ev, cuda_ev)
