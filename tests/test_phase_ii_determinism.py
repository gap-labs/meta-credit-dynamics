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


class DeterministicPhaseIIWorld:
    def step(self, t: int, action):
        t_f = float(int(t))
        r_vec = np.asarray([0.03 + 0.001 * t_f, -0.01 + 0.0005 * t_f], dtype=float)
        c_total = 0.02 + 0.0001 * t_f
        return make_world_step_result(r_vec=r_vec, c_total=c_total, action=action)


def _history_signature(out: dict) -> dict:
    history = out["history"]
    return {
        "t": [int(step["t"]) for step in history],
        "event_counts": [dict(step["event_counts"]) for step in history],
        "last_event": [step["event_summary"]["last_event"] for step in history],
        "rollover_counts": [int(step["event_counts"].get("ROLLOVER", 0)) for step in history],
    }


def _numerics(out: dict) -> dict:
    history = out["history"]
    return {
        "wealth": np.asarray([float(step["wealth"]) for step in history], dtype=float),
        "strategic_credit_exposure": np.asarray(
            [float(step["strategic_credit_exposure"]) for step in history],
            dtype=float,
        ),
    }


def _run(seed: int, backend: str) -> dict:
    return run_phase_ii_episode(
        world=DeterministicPhaseIIWorld(),
        steps=12,
        channels=2,
        seed=int(seed),
        config=PhaseIIConfig(selector_policy="term_aware", h_near_idx=0),
        backend=backend,
    )


def test_phase_ii_cpu_determinism_reproducible():
    a = _run(seed=13, backend="cpu")
    b = _run(seed=13, backend="cpu")

    assert _history_signature(a) == _history_signature(b)
    num_a = _numerics(a)
    num_b = _numerics(b)
    np.testing.assert_allclose(num_a["wealth"], num_b["wealth"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        num_a["strategic_credit_exposure"],
        num_b["strategic_credit_exposure"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA unavailable")
def test_phase_ii_cuda_determinism_reproducible():
    a = _run(seed=13, backend="cuda")
    b = _run(seed=13, backend="cuda")

    assert _history_signature(a) == _history_signature(b)
    num_a = _numerics(a)
    num_b = _numerics(b)
    np.testing.assert_allclose(num_a["wealth"], num_b["wealth"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        num_a["strategic_credit_exposure"],
        num_b["strategic_credit_exposure"],
        rtol=0.0,
        atol=0.0,
    )
