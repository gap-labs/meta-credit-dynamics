from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..interfaces import WorldAction, WorldStepResult, make_world_step_result


@dataclass
class RegimeSwitchBanditWorld:
    """Regime-switching bandit world with deterministic RNG."""

    p: float = 0.05
    sigma: float = 0.01
    seed: int = 0
    c_high: float = 0.0
    q: float = 0.0
    c_spike: float = 0.05
    shock_times: set[int] | None = None
    shock_size: float = 0.0
    regime_sequence: list[str] | None = None
    noise_sequence: np.ndarray | None = None
    shock_sequence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._shock_rng = np.random.default_rng(int(self.seed) + 20000)
        # Initial regime: Bernoulli(0.5)
        self._regime = "A" if self._rng.random() < 0.5 else "B"

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)

        if self.noise_sequence is not None:
            noise = np.asarray(self.noise_sequence[int(t)], dtype=float)
        else:
            noise = self._rng.normal(0.0, float(self.sigma), size=mean.shape)
        r = mean + noise
        if self.shock_times and int(t) in self.shock_times and self.shock_size:
            active_idx = 0 if regime == "A" else 3
            r[active_idx] = float(r[active_idx]) - float(self.shock_size)
        if self.shock_sequence is not None:
            shock = bool(self.shock_sequence[int(t)])
        elif float(self.q) > 0.0:
            shock = bool(self._shock_rng.random() < float(self.q))
        else:
            shock = False

        c_regime = 0.0 if regime == "A" else float(self.c_high)
        c_shock = float(self.c_spike) if shock else 0.0
        c = c_regime + c_shock
        return make_world_step_result(r_vec=r, c_total=c, action=action)


@dataclass
class RuinRegimeBanditWorld:
    """Regime-switching world with a ruinous regime mean."""

    p: float = 0.05
    sigma: float = 0.01
    seed: int = 0
    regime_sequence: list[str] | None = None
    noise_sequence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._regime = "A" if self._rng.random() < 0.5 else "B"

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([-0.03, 0.0, 0.0, 0.0, 0.0], dtype=float)

        if self.noise_sequence is not None:
            noise = np.asarray(self.noise_sequence[int(t)], dtype=float)
        else:
            noise = self._rng.normal(0.0, float(self.sigma), size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class MarginalMatchedControlWorld:
    """Control world matching marginal means/variances without regime state."""

    sigma: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        _ = t
        high_idx = 0 if self._rng.random() < 0.5 else 3
        mean = np.zeros(5, dtype=float)
        mean[high_idx] = 0.02
        noise = self._rng.normal(0.0, float(self.sigma), size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class SubsetRegimeBanditWorld:
    """Regime-switching world with subset-based channel boosts."""

    p: float = 0.05
    sigma: float = 0.01
    seed: int = 0
    regime_sequence: list[str] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._regime = "A" if self._rng.random() < 0.5 else "B"

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.02, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.02], dtype=float)

        noise = self._rng.normal(0.0, float(self.sigma), size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class VolatilityRegimeBanditWorld:
    """Regime-switching world with regime-dependent volatility."""

    p: float = 0.05
    sigma_low: float = 0.005
    sigma_high: float = 0.02
    seed: int = 0
    regime_sequence: list[str] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._regime = "A" if self._rng.random() < 0.5 else "B"

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
            sigma = float(self.sigma_low)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)
            sigma = float(self.sigma_high)

        noise = self._rng.normal(0.0, sigma, size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class NonStationaryVolatilityBanditWorld:
    """Regime-switching world with time-varying volatility schedule."""

    p: float = 0.05
    sigma_stationary: float = 0.01
    sigma_low: float = 0.005
    sigma_high: float = 0.02
    volatility_mode: str = "stationary"
    seed: int = 0
    horizon: int = 500
    regime_sequence: list[str] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._regime = "A" if self._rng.random() < 0.5 else "B"

    def _sigma_t(self, t: int) -> float:
        mode = str(self.volatility_mode)
        if mode == "stationary":
            return float(self.sigma_stationary)
        if mode == "drift_up":
            denom = float(max(1, int(self.horizon) - 1))
            frac = float(t) / denom
            return float(self.sigma_low + (self.sigma_high - self.sigma_low) * frac)
        if mode == "asym_drift":
            return float(self.sigma_stationary)
        raise ValueError(f"Unsupported volatility_mode: {mode}")

    def _sigma_vec(self, t: int, regime: str) -> np.ndarray:
        mode = str(self.volatility_mode)
        if mode != "asym_drift":
            return np.full(5, self._sigma_t(t), dtype=float)

        denom = float(max(1, int(self.horizon) - 1))
        frac = float(t) / denom
        sigma_neutral = float(self.sigma_low + (self.sigma_high - self.sigma_low) * frac)
        sigma_active = float(self.sigma_stationary)

        sigma_vec = np.full(5, sigma_neutral, dtype=float)
        active_idx = 0 if regime == "A" else 3
        sigma_vec[active_idx] = sigma_active
        return sigma_vec

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)

        sigma_vec = self._sigma_vec(int(t), regime)
        noise = self._rng.normal(0.0, sigma_vec, size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class AdversarialPhaseShiftBanditWorld:
    """Two-phase world with adversarial SNR reversal under fixed means."""

    p: float = 0.001
    sigma_phase1: float = 0.005
    sigma_active_high: float = 0.02
    sigma_opposing_low: float = 0.005
    sigma_other: float = 0.01
    seed: int = 0
    horizon: int = 500
    phase_split: int | None = None
    regime_sequence: list[str] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._regime = "A" if self._rng.random() < 0.5 else "B"
        if self.phase_split is None:
            self.phase_split = int(self.horizon) // 2

    def _sigma_vec(self, t: int, regime: str) -> np.ndarray:
        if int(t) < int(self.phase_split):
            return np.full(5, float(self.sigma_phase1), dtype=float)

        sigma_vec = np.full(5, float(self.sigma_other), dtype=float)
        active_idx = 0 if regime == "A" else 3
        opposing_idx = 3 if regime == "A" else 0
        sigma_vec[active_idx] = float(self.sigma_active_high)
        sigma_vec[opposing_idx] = float(self.sigma_opposing_low)
        return sigma_vec

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.regime_sequence is not None:
            regime = self.regime_sequence[int(t)]
        else:
            if self._rng.random() < float(self.p):
                self._regime = "B" if self._regime == "A" else "A"
            regime = self._regime

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)

        sigma_vec = self._sigma_vec(int(t), regime)
        noise = self._rng.normal(0.0, sigma_vec, size=mean.shape)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


@dataclass
class ShuffledRegimeBanditWorld:
    """Regime world with shuffled regime sequence (same marginals)."""

    p: float = 0.05
    sigma: float = 0.01
    seed: int = 0
    shuffle_seed: int | None = None
    regime_sequence: list[str] | None = None
    noise_sequence: np.ndarray | None = None

    def __post_init__(self) -> None:
        base_seed = int(self.seed)
        self._shuffle_seed = int(self.shuffle_seed if self.shuffle_seed is not None else base_seed + 10000)
        if self.regime_sequence is None:
            raise ValueError("regime_sequence is required for ShuffledRegimeBanditWorld")
        self._rng = np.random.default_rng(self._shuffle_seed)
        self._perm = self._rng.permutation(len(self.regime_sequence))

    def step(self, t: int, action: WorldAction | None = None) -> WorldStepResult:
        if self.noise_sequence is None:
            raise ValueError("noise_sequence is required for ShuffledRegimeBanditWorld")
        idx = int(self._perm[int(t)])
        regime = self.regime_sequence[idx]

        if regime == "A":
            mean = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            mean = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)

        noise = np.asarray(self.noise_sequence[int(t)], dtype=float)
        r = mean + noise
        return make_world_step_result(r_vec=r, c_total=0.0, action=action)


def _generate_regime_sequence(*, p: float, seed: int, length: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    regime = "A" if rng.random() < 0.5 else "B"
    seq: list[str] = []
    for _ in range(int(length)):
        if rng.random() < float(p):
            regime = "B" if regime == "A" else "A"
        seq.append(regime)
    return seq


def _generate_shock_sequence(*, q: float, seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 20000)
    return rng.random(int(length)) < float(q)
