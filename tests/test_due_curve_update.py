from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.phase_ii_state import (
    PhaseIIEconomicState,
    PhaseIIEconomicStateTensors,
)


def test_due_curve_update_due_cash_and_rollover_are_deterministic():
    state = PhaseIIEconomicState.zeros(horizon_bins=4, h_near_idx=1)

    state.apply_due_cash(horizon_bin=0, amount=2.0)
    state.apply_rollover(from_bin=0, to_bin=2, amount=1.25)

    np.testing.assert_allclose(state.due_curve, np.asarray([0.75, 0.0, 1.25, 0.0], dtype=float), rtol=0.0, atol=1e-12)


def test_due_curve_cuda_tensor_roundtrip_is_updateable():
    state = PhaseIIEconomicState.zeros(horizon_bins=3, h_near_idx=0)
    state.apply_due_cash(horizon_bin=1, amount=3.0)
    state.liquidity_mismatch = 1.5

    tensors = PhaseIIEconomicStateTensors.from_state(state, batch_size=1, device="cpu")
    restored = tensors.to_state(batch_index=0)

    np.testing.assert_allclose(restored.due_curve, state.due_curve, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.liquidity_mismatch, state.liquidity_mismatch, rtol=0.0, atol=1e-12)

    state.apply_rollover(from_bin=1, to_bin=2, amount=1.0)
    state.liquidity_mismatch = -0.25
    tensors.update_from_state(state, batch_index=0)

    updated = tensors.to_state(batch_index=0)
    np.testing.assert_allclose(updated.due_curve, state.due_curve, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(updated.liquidity_mismatch, state.liquidity_mismatch, rtol=0.0, atol=1e-12)
