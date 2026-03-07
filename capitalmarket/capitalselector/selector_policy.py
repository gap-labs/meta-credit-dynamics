from __future__ import annotations

from typing import Literal, cast
import numpy as np

from .interfaces import WorldAction, validate_and_normalize_world_action


SelectorPolicy = Literal["myopic", "term_aware", "term_risk"]

DEFAULT_SELECTOR_POLICY: SelectorPolicy = "myopic"

_ALLOWED_SELECTOR_POLICIES = {"myopic", "term_aware", "term_risk"}


def validate_selector_policy(policy: str) -> SelectorPolicy:
    normalized = str(policy).strip().lower()
    if normalized not in _ALLOWED_SELECTOR_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_SELECTOR_POLICIES))
        raise ValueError(f"unknown selector policy '{policy}', expected one of: {allowed}")
    return cast(SelectorPolicy, normalized)


def build_world_action(
    *,
    weights: np.ndarray,
    gross_exposure: float = 1.0,
    leverage_limit: float = 1.0,
    allow_short: bool = False,
    expected_channels: int | None = None,
) -> WorldAction:
    """Build a validated and normalized WorldAction from policy weights."""
    action = WorldAction(
        weights=np.asarray(weights, dtype=float),
        gross_exposure=float(gross_exposure),
        leverage_limit=float(leverage_limit),
        allow_short=bool(allow_short),
    )
    return validate_and_normalize_world_action(action, expected_channels=expected_channels)
