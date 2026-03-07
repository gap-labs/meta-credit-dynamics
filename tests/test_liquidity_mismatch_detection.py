from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.phase_ii_state import PhaseIIEconomicState


def test_liquidity_mismatch_formula_matches_spec():
    state = PhaseIIEconomicState(
        due_curve=np.asarray([1.0, 2.0, 4.0], dtype=float),
        h_near_idx=1,
    )
    weights = np.asarray([0.6, 0.4], dtype=float)
    mu_term = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
        ],
        dtype=float,
    )

    mismatch = state.update_liquidity_mismatch(weights=weights, mu_term=mu_term)

    expected_inflows = (0.6 * 1.0 + 0.4 * 2.0) + (0.6 * 10.0 + 0.4 * 20.0)
    near_term_obligations = 1.0 + 2.0
    expected_mismatch = near_term_obligations - expected_inflows

    np.testing.assert_allclose(mismatch, expected_mismatch, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.liquidity_mismatch, expected_mismatch, rtol=0.0, atol=1e-12)


def test_liquidity_mismatch_uses_fixed_deterministic_sum_order():
    state = PhaseIIEconomicState(
        due_curve=np.asarray([3.0, 1.0, 0.0], dtype=float),
        h_near_idx=1,
    )
    weights = np.asarray([0.25, 0.75], dtype=float)
    mu_term = np.asarray(
        [
            [4.0, 1.0, 9.0],
            [2.0, 3.0, 8.0],
        ],
        dtype=float,
    )

    a = state.compute_expected_inflows(weights=weights, mu_term=mu_term)
    b = state.compute_expected_inflows(weights=weights, mu_term=mu_term)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)
