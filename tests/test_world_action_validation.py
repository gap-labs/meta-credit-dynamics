from __future__ import annotations

import numpy as np
import pytest

from capitalmarket.capitalselector.interfaces import (
    WorldAction,
    validate_and_normalize_world_action,
)
from capitalmarket.capitalselector.selector_policy import build_world_action


def test_world_action_normalization_no_short_matches_gross_exposure():
    action = WorldAction(weights=np.asarray([2.0, 1.0, 1.0], dtype=float), gross_exposure=1.5, leverage_limit=2.0, allow_short=False)
    out = validate_and_normalize_world_action(action, expected_channels=3)

    np.testing.assert_allclose(out.weights, np.asarray([0.75, 0.375, 0.375], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(out.weights.sum()), 1.5, rtol=0.0, atol=1e-12)


def test_world_action_normalization_short_uses_l1_exposure():
    action = WorldAction(weights=np.asarray([2.0, -1.0, 1.0], dtype=float), gross_exposure=2.0, leverage_limit=2.5, allow_short=True)
    out = validate_and_normalize_world_action(action, expected_channels=3)

    np.testing.assert_allclose(out.weights, np.asarray([1.0, -0.5, 0.5], dtype=float), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(float(np.abs(out.weights).sum()), 2.0, rtol=0.0, atol=1e-12)


def test_world_action_invalid_fails_in_deterministic_order_finite_first():
    action = WorldAction(weights=np.asarray([np.nan, -1.0], dtype=float), gross_exposure=2.0, leverage_limit=1.0, allow_short=False)
    with pytest.raises(ValueError, match="weights must contain only finite values"):
        _ = validate_and_normalize_world_action(action)


def test_world_action_invalid_fails_in_deterministic_order_sign_before_leverage():
    action = WorldAction(weights=np.asarray([1.0, -0.1], dtype=float), gross_exposure=2.0, leverage_limit=1.0, allow_short=False)
    with pytest.raises(ValueError, match="negative weights are not allowed"):
        _ = validate_and_normalize_world_action(action)


def test_world_action_invalid_fails_in_deterministic_order_leverage_before_normalization():
    action = WorldAction(weights=np.asarray([0.0, 0.0], dtype=float), gross_exposure=1.1, leverage_limit=1.0, allow_short=False)
    with pytest.raises(ValueError, match="gross_exposure exceeds leverage_limit"):
        _ = validate_and_normalize_world_action(action)


def test_selector_policy_build_world_action_returns_normalized_action():
    out = build_world_action(
        weights=np.asarray([3.0, 1.0], dtype=float),
        gross_exposure=1.0,
        leverage_limit=1.0,
        allow_short=False,
        expected_channels=2,
    )
    np.testing.assert_allclose(out.weights, np.asarray([0.75, 0.25], dtype=float), rtol=0.0, atol=1e-12)
