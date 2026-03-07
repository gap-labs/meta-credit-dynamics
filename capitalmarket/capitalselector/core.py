from __future__ import annotations

import numpy as np
from .stats import EWMAStats
from .rebirth import RebirthPolicy
from .phase_i_state import (
    DEFAULT_LAMBDA_RISK,
    DEFAULT_TERM_GAMMA,
    allocate_term_mu,
    ewma_update,
    term_channel_score,
    term_risk_channel_score,
    validate_lambda_risk,
)
from .reweight import simplex_normalize
from .selector_policy import DEFAULT_SELECTOR_POLICY, SelectorPolicy, validate_selector_policy

from abc import ABC, abstractmethod
from typing import Tuple


class Channel(ABC):
    """
    Minimaler ökonomischer Kanal:
    nimmt Kapitalgewicht und liefert (r, c) zurück.
    """

    @abstractmethod
    def step(self, weight: float) -> Tuple[float, float]:
        pass


class CapitalSelector(Channel):
    """
    Der kanonische, stackbare CapitalSelector.
    """

    def __init__(
        self,
        *,
        wealth: float,
        rebirth_threshold: float,
        stats: EWMAStats,
        reweight_fn,
        kind: str = "entrepreneur",
        rebirth_policy: RebirthPolicy | None = None,
        channels: list[Channel] | None = None,
        selector_policy: SelectorPolicy = DEFAULT_SELECTOR_POLICY,
        lambda_risk: float = DEFAULT_LAMBDA_RISK,
    ):
        # G2 inventory: see docs/phase_g_g2_state_inventory.md
        self.wealth = wealth
        self.rebirth_threshold = rebirth_threshold
        self.stats = stats
        self.reweight_fn = reweight_fn
        self.kind = kind
        self.rebirth_policy = rebirth_policy
        self.selector_policy = validate_selector_policy(selector_policy)
        self.lambda_risk = validate_lambda_risk(lambda_risk)
        # Short alias for debugging/snapshots.
        self.policy = self.selector_policy

        self.channels = channels or []
        self.K = len(self.channels)

        self.w = np.ones(self.K) / self.K if self.K > 0 else None
        self.gamma = np.asarray(DEFAULT_TERM_GAMMA, dtype=float).copy()
        self.horizon_count = int(self.gamma.shape[0])
        self.beta_term = float(getattr(self.stats, "beta", 0.01))
        self.mu_term = allocate_term_mu(self.K, self.horizon_count)
        self.beta_r = float(getattr(self.stats, "beta", 0.01))
        self.rho = np.zeros(self.K, dtype=float)
        self.psi = np.zeros(self.K, dtype=float)

        self._last_r = 0.0
        self._last_c = 0.0

    def ensure_channel_state(self, K: int) -> None:
        """Keep per-channel state tensors aligned with the active channel count."""
        K = int(K)
        if K < 0:
            raise ValueError("K must be >= 0")

        if not hasattr(self, "gamma"):
            self.gamma = np.asarray(DEFAULT_TERM_GAMMA, dtype=float).copy()
        if not hasattr(self, "horizon_count"):
            self.horizon_count = int(np.asarray(self.gamma, dtype=float).shape[0])
        if not hasattr(self, "mu_term"):
            self.mu_term = allocate_term_mu(0, self.horizon_count)
        if not hasattr(self, "rho"):
            self.rho = np.zeros(0, dtype=float)
        if not hasattr(self, "psi"):
            self.psi = np.zeros(0, dtype=float)
        if not hasattr(self, "beta_term"):
            self.beta_term = float(getattr(self.stats, "beta", 0.01))
        if not hasattr(self, "beta_r"):
            self.beta_r = float(getattr(self.stats, "beta", 0.01))
        if not hasattr(self, "lambda_risk"):
            self.lambda_risk = validate_lambda_risk(DEFAULT_LAMBDA_RISK)

        self.K = K
        if K == 0:
            self.w = None
            self.mu_term = allocate_term_mu(0, self.horizon_count)
            self.rho = np.zeros(0, dtype=float)
            self.psi = np.zeros(0, dtype=float)
            return

        if self.w is None or len(self.w) != K:
            self.w = np.ones(K, dtype=float) / float(K)

        if len(self.rho) != K:
            self.rho = np.zeros(K, dtype=float)

        if len(self.psi) != K:
            self.psi = np.zeros(K, dtype=float)

        if self.mu_term.shape[0] != K:
            self.mu_term = allocate_term_mu(K, self.horizon_count)

    def update_rho(self, channel: int, risk_signal: float) -> None:
        """Apply one Phase-I risk impulse update to rho[channel]."""
        if self.rho is None or len(self.rho) == 0:
            return

        idx = int(channel)
        if idx < 0 or idx >= len(self.rho):
            return

        beta_r = float(self.beta_r)
        risk = float(risk_signal)
        self.rho[idx] = (1.0 - beta_r) * float(self.rho[idx]) + beta_r * risk

    def update_term_mu(self, channel: int, horizon: int, pi: float) -> None:
        """Apply one Phase-I EWMA update to mu[channel, horizon]."""
        if self.mu_term.ndim != 2 or self.mu_term.shape[0] == 0:
            return

        i = int(channel)
        h = int(horizon)
        if i < 0 or i >= self.mu_term.shape[0]:
            return
        if h < 0:
            return

        h_idx = min(h, self.mu_term.shape[1] - 1)
        previous = float(self.mu_term[i, h_idx])
        self.mu_term[i, h_idx] = ewma_update(previous=previous, value=float(pi), beta=self.beta_term)

    def update_psi(self, channel: int, fail_signal: float) -> None:
        """Apply one deterministic FAIL impulse update to psi[channel]."""
        if self.psi is None or len(self.psi) == 0:
            return

        idx = int(channel)
        if idx < 0 or idx >= len(self.psi):
            return

        beta_r = float(self.beta_r)
        impulse = float(fail_signal)
        self.psi[idx] = (1.0 - beta_r) * float(self.psi[idx]) + beta_r * impulse

    def compute_term_score(self) -> np.ndarray:
        """Compute term-aware per-channel score from horizon expectations."""
        if self.mu_term.ndim != 2 or self.mu_term.shape[0] == 0:
            return np.zeros(0, dtype=float)
        return term_channel_score(self.mu_term, self.gamma)

    def compute_term_risk_score(self) -> np.ndarray:
        """Compute term-aware score with per-channel risk penalty."""
        if self.mu_term.ndim != 2 or self.mu_term.shape[0] == 0:
            return np.zeros(0, dtype=float)
        return term_risk_channel_score(
            mu_term=self.mu_term,
            gamma=self.gamma,
            rho=self.rho,
            lambda_risk=self.lambda_risk,
        )

    def compute_advantage(self, pi_vec: np.ndarray) -> np.ndarray:
        policy = self.selector_policy
        if policy == "myopic":
            return np.asarray(pi_vec, dtype=float) - float(self.stats.mu)
        if policy == "term_aware":
            return self.compute_term_score()
        if policy == "term_risk":
            return self.compute_term_risk_score()
        raise ValueError(f"Unknown selector policy: {policy}")

    # ---------- Allocation ----------

    def allocate(self) -> np.ndarray:
        return None if self.w is None else self.w.copy()

    # ---------- Channel Interface ----------

    def step(self, weight: float) -> tuple[float, float]:
        """
        Exportiert diesen Selector als Kanal.
        """
        return weight * self._last_r, weight * self._last_c

    # ---------- Stack Step ----------

    def stack_step(self):
        if not self.channels:
            return

        rs, cs = [], []
        w = self.allocate()

        for wi, ch in zip(w, self.channels):
            r_i, c_i = ch.step(wi)
            rs.append(r_i)
            cs.append(c_i)

        self.feedback(sum(rs), sum(cs))

    # ---------- Feedback ----------


    def feedback(self, r: float, c: float):
        """Scalar feedback for standalone selectors (K==0).

        For stacked selectors with sub-channels, use `feedback_vector(r_vec, c)`.
        """
        r = float(r); c = float(c)
        self._last_r = r
        self._last_c = c
        self.wealth += r - c
        self.stats.update(r - c)
        if self.wealth < self.rebirth_threshold:
            self.rebirth()

    def feedback_vector(self, r_vec: np.ndarray, c: float, trace: list[str] | None = None, freeze: bool = False):
        r_total = r_vec.sum()

        if freeze:
            if trace is not None:
                trace.append("freeze")
            self._enforce_invariants()
            if trace is not None:
                trace.append("invariants")
            return

        self._last_r = r_total
        self._last_c = c
        self.wealth += r_total - c

        if trace is not None:
            trace.append("compute_pi")
        _, _, pi_total, pi_vec = self.compute_pi(r_vec, c)
        self.stats.update(pi_total)
        if trace is not None:
            trace.append("update_stats")

        adv = self.compute_advantage(pi_vec)
        self.w = self.reweight_fn(self.w, adv)
        if trace is not None:
            trace.append("reweight")

        if self.wealth < self.rebirth_threshold:
            self.rebirth()
            if trace is not None:
                trace.append("rebirth")

        self._enforce_invariants()
        if trace is not None:
            trace.append("invariants")

    def compute_pi(self, r_vec: np.ndarray, c_total: float):
        """Compute canonical net-flow aggregates.

        Returns (R, C, Pi, pi_vec) with Pi = R - C and
        pi_vec_k = r_vec_k - w_k * C_total (Profile A).
        Assumes weights sum to 1.
        """
        r_vec = np.asarray(r_vec, dtype=float)
        R = float(r_vec.sum())
        C = float(c_total)
        Pi = R - C
        if self.w is None:
            pi_vec = r_vec.copy()
        else:
            pi_vec = r_vec - self.w * C
        return R, C, Pi, pi_vec

    def _enforce_invariants(self):
        """Ensure weight simplex constraints after updates."""
        if self.w is not None:
            self.w = simplex_normalize(self.w)

    # ---------- Rebirth ----------

    def rebirth(self):
        if self.rebirth_policy:
            self.rebirth_policy.on_rebirth(self)

        self.wealth = max(self.wealth, self.rebirth_threshold)
        if self.w is not None:
            self.w = np.ones(self.K) / self.K
        self.mu_term = allocate_term_mu(self.K, self.horizon_count)
        self.rho = np.zeros(self.K, dtype=float)
        self.psi = np.zeros(self.K, dtype=float)
        self.stats.reset()

    # ---------- Introspection ----------

    def state(self):
        return {
            "wealth": self.wealth,
            "kind": self.kind,
            "selector_policy": self.selector_policy,
            "mu": self.stats.mu,
            "var": self.stats.var,
            "dd": self.stats.dd,
            "cum_pi": self.stats.cum_pi,
            "peak_cum_pi": self.stats.peak_cum_pi,
            "weights": None if self.w is None else self.w.copy(),
            "mu_term": self.mu_term.copy(),
            "gamma": self.gamma.copy(),
            "rho": self.rho.copy(),
            "psi": self.psi.copy(),
            "lambda_risk": float(self.lambda_risk),
        }
