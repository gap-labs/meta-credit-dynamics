from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PhaseIIEconomicState:
    """Economic source-of-truth state for Phase II observables."""

    due_curve: np.ndarray
    expected_inflows: float = 0.0
    liquidity_mismatch: float = 0.0
    h_near_idx: int = 0

    def __post_init__(self) -> None:
        self.due_curve = np.asarray(self.due_curve, dtype=float)
        if self.due_curve.ndim != 1 or self.due_curve.size <= 0:
            raise ValueError("due_curve must be a non-empty 1D array")
        self.h_near_idx = int(self.h_near_idx)
        if self.h_near_idx < 0 or self.h_near_idx >= int(self.due_curve.shape[0]):
            raise ValueError("h_near_idx must satisfy 0 <= h_near_idx < len(due_curve)")
        self.expected_inflows = float(self.expected_inflows)
        self.liquidity_mismatch = float(self.liquidity_mismatch)

    @classmethod
    def zeros(cls, *, horizon_bins: int, h_near_idx: int = 0) -> "PhaseIIEconomicState":
        bins = int(horizon_bins)
        if bins <= 0:
            raise ValueError("horizon_bins must be > 0")
        return cls(
            due_curve=np.zeros(bins, dtype=float),
            expected_inflows=0.0,
            liquidity_mismatch=0.0,
            h_near_idx=int(h_near_idx),
        )

    def copy(self) -> "PhaseIIEconomicState":
        return PhaseIIEconomicState(
            due_curve=self.due_curve.copy(),
            expected_inflows=float(self.expected_inflows),
            liquidity_mismatch=float(self.liquidity_mismatch),
            h_near_idx=int(self.h_near_idx),
        )

    def apply_due_cash(self, *, horizon_bin: int, amount: float) -> None:
        idx = int(horizon_bin)
        if idx < 0 or idx >= int(self.due_curve.shape[0]):
            raise ValueError("horizon_bin out of range")
        amt = float(amount)
        self.due_curve[idx] = float(self.due_curve[idx]) + amt
        if self.due_curve[idx] < 0.0:
            raise ValueError("due_curve entries must remain non-negative")

    def apply_rollover(self, *, from_bin: int, to_bin: int, amount: float) -> None:
        src = int(from_bin)
        dst = int(to_bin)
        amt = float(amount)
        if amt < 0.0:
            raise ValueError("rollover amount must be >= 0")
        if src < 0 or src >= int(self.due_curve.shape[0]):
            raise ValueError("from_bin out of range")
        if dst < 0 or dst >= int(self.due_curve.shape[0]):
            raise ValueError("to_bin out of range")
        if float(self.due_curve[src]) < amt:
            raise ValueError("rollover amount exceeds source due_curve bucket")

        self.due_curve[src] = float(self.due_curve[src]) - amt
        self.due_curve[dst] = float(self.due_curve[dst]) + amt

    def compute_expected_inflows(self, *, weights: np.ndarray, mu_term: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        mu = np.asarray(mu_term, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if mu.ndim != 2:
            raise ValueError("mu_term must be a 2D array [channel, horizon]")
        if mu.shape[0] != w.shape[0]:
            raise ValueError("mu_term channel axis must match weights length")
        if mu.shape[1] != self.due_curve.shape[0]:
            raise ValueError("mu_term horizon axis must match due_curve length")

        # Deterministic sum order: horizon-major, then channel-major.
        total = 0.0
        for h in range(0, int(self.h_near_idx) + 1):
            for c in range(0, int(w.shape[0])):
                total += float(w[c]) * float(mu[c, h])
        return float(total)

    def expected_long_term_inflows(self, *, weights: np.ndarray, mu_term: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        mu = np.asarray(mu_term, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if mu.ndim != 2:
            raise ValueError("mu_term must be a 2D array [channel, horizon]")
        if mu.shape[0] != w.shape[0]:
            raise ValueError("mu_term channel axis must match weights length")
        if mu.shape[1] != self.due_curve.shape[0]:
            raise ValueError("mu_term horizon axis must match due_curve length")

        total = 0.0
        for h in range(int(self.h_near_idx) + 1, int(self.due_curve.shape[0])):
            for c in range(0, int(w.shape[0])):
                total += float(w[c]) * float(mu[c, h])
        return float(total)

    def near_term_obligations(self) -> float:
        total = 0.0
        for h in range(0, int(self.h_near_idx) + 1):
            total += float(self.due_curve[h])
        return float(total)

    def update_liquidity_mismatch(self, *, weights: np.ndarray, mu_term: np.ndarray) -> float:
        expected_inflows = self.compute_expected_inflows(weights=weights, mu_term=mu_term)
        self.expected_inflows = float(expected_inflows)
        near_obligations = self.near_term_obligations()
        self.liquidity_mismatch = float(near_obligations - expected_inflows)
        return float(self.liquidity_mismatch)

    def selector_derived_features(self, *, weights: np.ndarray, mu_term: np.ndarray) -> dict[str, float]:
        expected_inflows = self.compute_expected_inflows(weights=weights, mu_term=mu_term)
        near_obligations = self.near_term_obligations()
        liquidity_mismatch = float(near_obligations - expected_inflows)
        return {
            "expected_inflows": float(expected_inflows),
            "near_term_obligations": float(near_obligations),
            "liquidity_mismatch": float(liquidity_mismatch),
        }


@dataclass
class PhaseIISelectorState:
    """Selector-owned Phase II state (learning/control side)."""

    strategic_credit_exposure: float = 0.0
    near_term_obligations: float = 0.0
    liquidity_mismatch: float = 0.0
    expected_long_term_inflows: float = 0.0

    def update_from_economic(
        self,
        *,
        economic_state: PhaseIIEconomicState,
        weights: np.ndarray,
        mu_term: np.ndarray,
    ) -> None:
        features = economic_state.selector_derived_features(weights=weights, mu_term=mu_term)
        long_term_inflows = economic_state.expected_long_term_inflows(weights=weights, mu_term=mu_term)

        funding_gap = max(0.0, float(features["liquidity_mismatch"]))
        self.strategic_credit_exposure = float(min(funding_gap, max(0.0, float(long_term_inflows))))

        self.near_term_obligations = float(features["near_term_obligations"])
        self.liquidity_mismatch = float(features["liquidity_mismatch"])
        self.expected_long_term_inflows = float(long_term_inflows)

    def derived_features(self) -> dict[str, float]:
        expected_inflows = float(self.near_term_obligations - self.liquidity_mismatch)
        return {
            "expected_inflows": expected_inflows,
            "near_term_obligations": float(self.near_term_obligations),
            "liquidity_mismatch": float(self.liquidity_mismatch),
            "expected_long_term_inflows": float(self.expected_long_term_inflows),
        }


@dataclass
class PhaseIIEconomicStateTensors:
    """Updateable tensor container for Economic state (CPU/CUDA)."""

    due_curve: torch.Tensor
    expected_inflows: torch.Tensor
    liquidity_mismatch: torch.Tensor
    h_near_idx: int

    @classmethod
    def from_state(
        cls,
        state: PhaseIIEconomicState,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "PhaseIIEconomicStateTensors":
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        due_base = torch.as_tensor(state.due_curve, dtype=dtype, device=dev)
        due_curve = due_base.unsqueeze(0).repeat(batch, 1)
        expected_inflows = torch.full((batch,), float(state.expected_inflows), dtype=dtype, device=dev)
        mismatch = torch.full((batch,), float(state.liquidity_mismatch), dtype=dtype, device=dev)
        return cls(
            due_curve=due_curve,
            expected_inflows=expected_inflows,
            liquidity_mismatch=mismatch,
            h_near_idx=int(state.h_near_idx),
        )

    def to_state(self, *, batch_index: int = 0) -> PhaseIIEconomicState:
        idx = int(batch_index)
        if idx < 0 or idx >= int(self.due_curve.shape[0]):
            raise ValueError("batch_index out of range")
        return PhaseIIEconomicState(
            due_curve=self.due_curve[idx].detach().cpu().numpy().astype(float, copy=True),
            expected_inflows=float(self.expected_inflows[idx].item()),
            liquidity_mismatch=float(self.liquidity_mismatch[idx].item()),
            h_near_idx=int(self.h_near_idx),
        )

    def update_from_state(self, state: PhaseIIEconomicState, *, batch_index: int = 0) -> None:
        idx = int(batch_index)
        if idx < 0 or idx >= int(self.due_curve.shape[0]):
            raise ValueError("batch_index out of range")
        if int(self.due_curve.shape[1]) != int(state.due_curve.shape[0]):
            raise ValueError("due_curve horizon length mismatch")
        self.due_curve[idx] = torch.as_tensor(
            state.due_curve,
            dtype=self.due_curve.dtype,
            device=self.due_curve.device,
        )
        self.expected_inflows[idx] = torch.as_tensor(
            float(state.expected_inflows),
            dtype=self.expected_inflows.dtype,
            device=self.expected_inflows.device,
        )
        self.liquidity_mismatch[idx] = torch.as_tensor(
            float(state.liquidity_mismatch),
            dtype=self.liquidity_mismatch.dtype,
            device=self.liquidity_mismatch.device,
        )


@dataclass
class PhaseIISelectorStateTensors:
    """Updateable tensor container for Selector phase-II state (CPU/CUDA)."""

    strategic_credit_exposure: torch.Tensor
    near_term_obligations: torch.Tensor
    liquidity_mismatch: torch.Tensor
    expected_long_term_inflows: torch.Tensor

    @classmethod
    def from_state(
        cls,
        state: PhaseIISelectorState,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "PhaseIISelectorStateTensors":
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be > 0")
        dev = torch.device(device) if not isinstance(device, torch.device) else device

        def _vec(value: float) -> torch.Tensor:
            return torch.full((batch,), float(value), dtype=dtype, device=dev)

        return cls(
            strategic_credit_exposure=_vec(state.strategic_credit_exposure),
            near_term_obligations=_vec(state.near_term_obligations),
            liquidity_mismatch=_vec(state.liquidity_mismatch),
            expected_long_term_inflows=_vec(state.expected_long_term_inflows),
        )

    def to_state(self, *, batch_index: int = 0) -> PhaseIISelectorState:
        idx = int(batch_index)
        n = int(self.strategic_credit_exposure.shape[0])
        if idx < 0 or idx >= n:
            raise ValueError("batch_index out of range")
        return PhaseIISelectorState(
            strategic_credit_exposure=float(self.strategic_credit_exposure[idx].item()),
            near_term_obligations=float(self.near_term_obligations[idx].item()),
            liquidity_mismatch=float(self.liquidity_mismatch[idx].item()),
            expected_long_term_inflows=float(self.expected_long_term_inflows[idx].item()),
        )

    def update_from_state(self, state: PhaseIISelectorState, *, batch_index: int = 0) -> None:
        idx = int(batch_index)
        n = int(self.strategic_credit_exposure.shape[0])
        if idx < 0 or idx >= n:
            raise ValueError("batch_index out of range")

        self.strategic_credit_exposure[idx] = torch.as_tensor(
            float(state.strategic_credit_exposure),
            dtype=self.strategic_credit_exposure.dtype,
            device=self.strategic_credit_exposure.device,
        )
        self.near_term_obligations[idx] = torch.as_tensor(
            float(state.near_term_obligations),
            dtype=self.near_term_obligations.dtype,
            device=self.near_term_obligations.device,
        )
        self.liquidity_mismatch[idx] = torch.as_tensor(
            float(state.liquidity_mismatch),
            dtype=self.liquidity_mismatch.dtype,
            device=self.liquidity_mismatch.device,
        )
        self.expected_long_term_inflows[idx] = torch.as_tensor(
            float(state.expected_long_term_inflows),
            dtype=self.expected_long_term_inflows.dtype,
            device=self.expected_long_term_inflows.device,
        )
