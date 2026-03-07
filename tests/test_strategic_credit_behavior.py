from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.phase_ii_state import (
    PhaseIIEconomicState,
    PhaseIISelectorState,
    PhaseIISelectorStateTensors,
)


def test_strategic_credit_exposure_positive_when_gap_is_financed_by_long_term_inflows():
    economic = PhaseIIEconomicState(
        due_curve=np.asarray([10.0, 0.0, 0.0], dtype=float),
        h_near_idx=0,
    )
    selector_state = PhaseIISelectorState()

    weights = np.asarray([0.5, 0.5], dtype=float)
    mu_term = np.asarray(
        [
            [1.0, 20.0, 0.0],
            [1.0, 20.0, 0.0],
        ],
        dtype=float,
    )

    selector_state.update_from_economic(economic_state=economic, weights=weights, mu_term=mu_term)

    # near-term inflows = 1.0, near-term obligations = 10.0 => gap = 9.0
    # long-term inflows = 20.0 => exposure = min(gap, long-term inflows) = 9.0
    np.testing.assert_allclose(selector_state.strategic_credit_exposure, 9.0, rtol=0.0, atol=1e-12)


def test_strategic_credit_exposure_zero_without_positive_liquidity_gap():
    economic = PhaseIIEconomicState(
        due_curve=np.asarray([1.0, 0.0, 0.0], dtype=float),
        h_near_idx=0,
    )
    selector_state = PhaseIISelectorState()

    weights = np.asarray([0.5, 0.5], dtype=float)
    mu_term = np.asarray(
        [
            [2.0, 10.0, 0.0],
            [2.0, 10.0, 0.0],
        ],
        dtype=float,
    )

    selector_state.update_from_economic(economic_state=economic, weights=weights, mu_term=mu_term)
    np.testing.assert_allclose(selector_state.strategic_credit_exposure, 0.0, rtol=0.0, atol=1e-12)


def test_strategic_credit_selector_tensor_state_roundtrip():
    state = PhaseIISelectorState(
        strategic_credit_exposure=1.25,
        near_term_obligations=3.0,
        liquidity_mismatch=1.0,
        expected_long_term_inflows=4.0,
    )
    tensors = PhaseIISelectorStateTensors.from_state(state, batch_size=1, device="cpu")
    restored = tensors.to_state(batch_index=0)

    np.testing.assert_allclose(restored.strategic_credit_exposure, state.strategic_credit_exposure, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.near_term_obligations, state.near_term_obligations, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.liquidity_mismatch, state.liquidity_mismatch, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(restored.expected_long_term_inflows, state.expected_long_term_inflows, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        restored.derived_features()["expected_inflows"],
        state.derived_features()["expected_inflows"],
        rtol=0.0,
        atol=1e-12,
    )
