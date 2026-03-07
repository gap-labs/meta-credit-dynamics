from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonBucketConfig:
    """Deterministic threshold configuration for Phase-I horizon buckets."""

    t_short: int = 1
    t_long: int = 4
    h0: int = 0
    h1: int = 1
    h2: int = 2

    def __post_init__(self) -> None:
        if int(self.t_short) < 0:
            raise ValueError("t_short must be >= 0")
        if int(self.t_long) < int(self.t_short):
            raise ValueError("t_long must be >= t_short")


DEFAULT_HORIZON_BUCKET_CONFIG = HorizonBucketConfig()


def phi(delta_tau: int | float, config: HorizonBucketConfig = DEFAULT_HORIZON_BUCKET_CONFIG) -> int:
    """Map time-to-maturity to a deterministic horizon bucket id.

    Boundaries are inclusive: ``delta <= t_short`` maps to ``h0`` and
    ``delta <= t_long`` maps to ``h1``. Negative deltas are clamped to 0.
    """

    delta = max(0, int(delta_tau))
    if delta <= int(config.t_short):
        return int(config.h0)
    if delta <= int(config.t_long):
        return int(config.h1)
    return int(config.h2)
