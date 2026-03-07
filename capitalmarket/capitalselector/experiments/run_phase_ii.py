from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import WorldAction, WorldStepResult, validate_world_step_result
from capitalmarket.capitalselector.phase_ii_events import apply_phase_ii_event_mapping, event_counts
from capitalmarket.capitalselector.phase_ii_state import (
    PhaseIIEconomicState,
    PhaseIIEconomicStateTensors,
    PhaseIISelectorState,
    PhaseIISelectorStateTensors,
)
from capitalmarket.capitalselector.selector_policy import build_world_action, validate_selector_policy


TERM_RISK_HEADROOM_BONUS = 0.50
TERM_RISK_STRESS_PENALTY = 0.25


@dataclass(frozen=True)
class PhaseIIStepRecord:
    t: int
    wealth_prev: float
    wealth_next: float
    action: WorldAction
    world_out: WorldStepResult
    events: list[dict[str, Any]]
    event_counts: dict[str, int]
    event_summary: dict[str, Any]
    economic_observables: dict[str, float]
    strategic_credit_exposure: float


@dataclass(frozen=True)
class PhaseIIConfig:
    selector_policy: str = "myopic"
    gross_exposure: float = 1.0
    leverage_limit: float = 1.0
    allow_short: bool = False
    h_near_idx: int = 0
    coupling_alpha: float = 0.30
    coupling_beta: float = 0.30
    coupling_gamma: float = 0.00
    coupling_eta: float = 0.20
    coupling_eps: float = 1e-15
    channel_horizon_map: tuple[int, ...] | None = None


def _resolve_backend(backend: str) -> tuple[str, torch.device | None]:
    normalized = str(backend).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"invalid backend '{backend}', expected 'cpu' or 'cuda'")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=cuda requested but torch.cuda.is_available() is False")
        return normalized, torch.device("cuda")
    return normalized, None


def _current_selector_weights(selector: Any) -> np.ndarray:
    weights = np.asarray(getattr(selector, "w", None), dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Phase-II path requires initialized selector weights")
    return weights


def _time_to_death(history: list[dict[str, Any]], *, steps: int) -> int:
    """Return first 1-based timestep with FAIL event or negative wealth; otherwise full horizon."""
    for step in history:
        has_fail = int(step["event_counts"].get("FAIL", 0)) > 0
        has_negative_wealth = float(step["wealth"]) < 0.0
        if has_fail or has_negative_wealth:
            return int(step["t"]) + 1
    return int(steps)


def _resolve_channel_horizon_map(
    *,
    channels: int,
    horizon_bins: int,
    configured: tuple[int, ...] | None,
) -> np.ndarray:
    c = int(channels)
    h = int(horizon_bins)
    if c <= 0 or h <= 0:
        raise ValueError("channels and horizon_bins must be > 0")

    if configured is None:
        # Deterministic default: clip channel index into available horizon bins.
        idx = np.arange(c, dtype=int)
        return np.minimum(idx, h - 1)

    arr = np.asarray(configured, dtype=int)
    if arr.ndim != 1 or int(arr.shape[0]) != c:
        raise ValueError("channel_horizon_map must be a 1D tuple with length == channels")
    if np.any(arr < 0) or np.any(arr >= h):
        raise ValueError("channel_horizon_map values must satisfy 0 <= value < horizon_bins")
    return arr


def _apply_phase_ii_policy_coupling(
    *,
    base_weights: np.ndarray,
    economic_state: PhaseIIEconomicState,
    selector_state: PhaseIISelectorState,
    cfg: PhaseIIConfig,
    rho: np.ndarray | None = None,
    psi: np.ndarray | None = None,
) -> np.ndarray:
    w = np.asarray(base_weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("base_weights must be a non-empty 1D array")

    horizon_map = _resolve_channel_horizon_map(
        channels=int(w.shape[0]),
        horizon_bins=int(economic_state.due_curve.shape[0]),
        configured=cfg.channel_horizon_map,
    )

    eps = max(float(cfg.coupling_eps), 1e-18)
    base = np.maximum(eps, np.abs(w))

    stress_t = max(0.0, float(economic_state.liquidity_mismatch))
    near_term_obligations_t = float(economic_state.near_term_obligations())
    expected_inflows_t = float(economic_state.expected_inflows)
    credit_headroom_t = max(0.0, expected_inflows_t - near_term_obligations_t)
    strategic_credit_exposure_t = max(0.0, float(selector_state.strategic_credit_exposure))
    selector_lag_headroom_t = -float(selector_state.liquidity_mismatch)

    near_mask = (horizon_map <= int(economic_state.h_near_idx)).astype(float)
    long_mask = (horizon_map > int(economic_state.h_near_idx)).astype(float)
    near_due_t = np.asarray(economic_state.due_curve, dtype=float)[horizon_map]

    u = np.log(base)
    u -= float(cfg.coupling_alpha) * stress_t * near_mask
    u -= float(cfg.coupling_beta) * near_due_t
    u += float(cfg.coupling_gamma) * credit_headroom_t * long_mask
    u += float(cfg.coupling_eta) * strategic_credit_exposure_t * long_mask
    # Keep selector-state mirrors causally relevant without dominating normative terms.
    u += 1e-3 * selector_lag_headroom_t * long_mask

    if str(cfg.selector_policy).strip().lower() == "term_risk":
        # Strategic credit actor: stronger long-book support under headroom and
        # stronger near-term de-risking under stress.
        u += float(TERM_RISK_HEADROOM_BONUS) * credit_headroom_t * long_mask
        u -= float(TERM_RISK_STRESS_PENALTY) * stress_t * near_mask

    # Stabilized softmax mapping to keep deterministic numeric behavior.
    z = np.exp(u - float(np.max(u)))
    denom = float(np.sum(z))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(w, dtype=float) / float(w.shape[0])
    return z / denom


def step_phase_ii(
    *,
    selector: Any,
    world: Any,
    t: int,
    gross_exposure: float = 1.0,
    leverage_limit: float = 1.0,
    allow_short: bool = False,
    coupling_alpha: float = 0.30,
    coupling_beta: float = 0.30,
    coupling_gamma: float = 0.00,
    coupling_eta: float = 0.20,
    coupling_eps: float = 1e-15,
    channel_horizon_map: tuple[int, ...] | None = None,
    economic_state: PhaseIIEconomicState | None = None,
    selector_state: PhaseIISelectorState | None = None,
) -> PhaseIIStepRecord:
    """Execute one closed-loop Phase-II step.

    Binding coupling rule:
      world_out = world.step(t, action)
      wealth_next = wealth_prev + world_out.realized_return - world_out.costs
    """
    base_weights = _current_selector_weights(selector)

    mu_term = np.asarray(getattr(selector, "mu_term", np.zeros((len(base_weights), 1), dtype=float)), dtype=float)
    if mu_term.ndim != 2 or mu_term.shape[0] != int(base_weights.shape[0]):
        raise ValueError("Phase-II path requires selector.mu_term shape [channel, horizon]")

    econ = (
        economic_state
        if economic_state is not None
        else PhaseIIEconomicState.zeros(horizon_bins=int(mu_term.shape[1]), h_near_idx=0)
    )
    sel_state = selector_state if selector_state is not None else PhaseIISelectorState()

    policy_weights = _apply_phase_ii_policy_coupling(
        base_weights=base_weights,
        economic_state=econ,
        selector_state=sel_state,
        rho=np.asarray(getattr(selector, "rho", np.zeros_like(base_weights)), dtype=float),
        psi=np.asarray(getattr(selector, "psi", np.zeros_like(base_weights)), dtype=float),
        cfg=PhaseIIConfig(
            selector_policy=str(getattr(selector, "selector_policy", "myopic")),
            gross_exposure=float(gross_exposure),
            leverage_limit=float(leverage_limit),
            allow_short=bool(allow_short),
            h_near_idx=int(econ.h_near_idx),
            coupling_alpha=float(coupling_alpha),
            coupling_beta=float(coupling_beta),
            coupling_gamma=float(coupling_gamma),
            coupling_eta=float(coupling_eta),
            coupling_eps=float(coupling_eps),
            channel_horizon_map=channel_horizon_map,
        ),
    )

    action = build_world_action(
        weights=policy_weights,
        gross_exposure=float(gross_exposure),
        leverage_limit=float(leverage_limit),
        allow_short=bool(allow_short),
        expected_channels=int(base_weights.shape[0]),
    )

    world_out = validate_world_step_result(world.step(int(t), action))
    channel_returns = np.asarray(world_out.channel_returns, dtype=float)
    if channel_returns.shape != action.weights.shape:
        raise ValueError(
            "Phase-II path requires world_out.channel_returns to match action.weights shape"
        )

    events, summary = apply_phase_ii_event_mapping(
        selector=selector,
        action=action,
        world_out=world_out,
        economic_state=econ,
        mu_term=mu_term,
    )

    econ.update_liquidity_mismatch(weights=action.weights, mu_term=mu_term)
    sel_state.update_from_economic(economic_state=econ, weights=action.weights, mu_term=mu_term)

    wealth_prev = float(selector.wealth)
    wealth_next = wealth_prev + float(world_out.realized_return) - float(world_out.costs)
    selector.wealth = float(wealth_next)
    selector._last_r = float(world_out.realized_return)
    selector._last_c = float(world_out.costs)
    selector.strategic_credit_exposure = float(sel_state.strategic_credit_exposure)
    selector.phase_ii_features = sel_state.derived_features()

    if not bool(world_out.freeze):
        # Policy/state updates must be based on the executed action weights.
        selector.w = np.asarray(action.weights, dtype=float).copy()
        _, _, _, pi_vec = selector.compute_pi(channel_returns, float(world_out.costs))
        adv = selector.compute_advantage(np.asarray(pi_vec, dtype=float))
        selector.w = selector.reweight_fn(np.asarray(selector.w, dtype=float), adv)
        selector._enforce_invariants()

    return PhaseIIStepRecord(
        t=int(t),
        wealth_prev=wealth_prev,
        wealth_next=float(selector.wealth),
        action=action,
        world_out=world_out,
        events=[
            {
                "category": str(event.category),
                "channel": int(event.channel),
                "horizon": int(event.horizon),
                "amount": float(event.amount),
                "rollover_to_horizon": None if event.rollover_to_horizon is None else int(event.rollover_to_horizon),
            }
            for event in events
        ],
        event_counts=event_counts(events),
        event_summary={
            "event_counts": dict(summary.event_counts),
            "last_event": summary.last_event,
            "channel_event_vector": np.asarray(summary.channel_event_vector, dtype=float).copy(),
        },
        economic_observables=econ.selector_derived_features(weights=action.weights, mu_term=mu_term),
        strategic_credit_exposure=float(sel_state.strategic_credit_exposure),
    )


def run_phase_ii_episode(
    *,
    world: Any,
    steps: int,
    channels: int,
    seed: int = 0,
    config: PhaseIIConfig | None = None,
    backend: str = "cpu",
) -> dict[str, Any]:
    """Additive Phase-II runner preserving the existing Phase-I runtime path."""
    backend_norm, tensor_device = _resolve_backend(backend)
    cfg = config or PhaseIIConfig()
    selector = (
        CapitalSelectorBuilder()
        .with_K(int(channels))
        .with_selector_policy(validate_selector_policy(cfg.selector_policy))
        .build()
    )

    rng = np.random.default_rng(int(seed))
    selector.w = np.asarray(rng.dirichlet(np.ones(int(channels), dtype=float)), dtype=float)
    selector.K = int(channels)

    horizon_bins = int(getattr(selector, "horizon_count", 1))
    economic_state = PhaseIIEconomicState.zeros(horizon_bins=horizon_bins, h_near_idx=int(cfg.h_near_idx))
    selector_state = PhaseIISelectorState()

    economic_state_tensors: PhaseIIEconomicStateTensors | None = None
    selector_state_tensors: PhaseIISelectorStateTensors | None = None
    if backend_norm == "cuda":
        economic_state_tensors = PhaseIIEconomicStateTensors.from_state(
            economic_state,
            batch_size=1,
            device=tensor_device,
        )
        selector_state_tensors = PhaseIISelectorStateTensors.from_state(
            selector_state,
            batch_size=1,
            device=tensor_device,
        )

    history: list[dict[str, Any]] = []
    for t in range(int(steps)):
        if economic_state_tensors is not None and selector_state_tensors is not None:
            economic_state = economic_state_tensors.to_state(batch_index=0)
            selector_state = selector_state_tensors.to_state(batch_index=0)

        rec = step_phase_ii(
            selector=selector,
            world=world,
            t=int(t),
            gross_exposure=float(cfg.gross_exposure),
            leverage_limit=float(cfg.leverage_limit),
            allow_short=bool(cfg.allow_short),
            coupling_alpha=float(cfg.coupling_alpha),
            coupling_beta=float(cfg.coupling_beta),
            coupling_gamma=float(cfg.coupling_gamma),
            coupling_eta=float(cfg.coupling_eta),
            coupling_eps=float(cfg.coupling_eps),
            channel_horizon_map=cfg.channel_horizon_map,
            economic_state=economic_state,
            selector_state=selector_state,
        )

        if economic_state_tensors is not None and selector_state_tensors is not None:
            economic_state_tensors.update_from_state(economic_state, batch_index=0)
            selector_state_tensors.update_from_state(selector_state, batch_index=0)

        history.append(
            {
                "t": rec.t,
                "wealth": rec.wealth_next,
                "realized_return": float(rec.world_out.realized_return),
                "costs": float(rec.world_out.costs),
                "weights": np.asarray(selector.w, dtype=float).copy(),
                "events": list(rec.events),
                "event_counts": dict(rec.event_counts),
                "event_summary": {
                    "event_counts": dict(rec.event_summary["event_counts"]),
                    "last_event": rec.event_summary["last_event"],
                    "channel_event_vector": np.asarray(rec.event_summary["channel_event_vector"], dtype=float).copy(),
                },
                "economic_observables": dict(rec.economic_observables),
                "strategic_credit_exposure": float(rec.strategic_credit_exposure),
            }
        )

        # Episode termination rule: stop immediately after first terminal event.
        if int(rec.event_counts.get("FAIL", 0)) > 0 or float(rec.wealth_next) < 0.0:
            break

    if economic_state_tensors is not None and selector_state_tensors is not None:
        economic_state = economic_state_tensors.to_state(batch_index=0)
        selector_state = selector_state_tensors.to_state(batch_index=0)

    event_totals = {key: 0 for key in ("RETURN", "DUE_CASH", "COST", "ROLLOVER", "FAIL", "SETTLEMENT")}
    for step in history:
        counts = step["event_counts"]
        for key in event_totals:
            event_totals[key] += int(counts.get(key, 0))

    terminal_wealth = float(selector.wealth)
    time_to_death = _time_to_death(history, steps=int(steps))
    settlement_event_count = int(event_totals["ROLLOVER"] + event_totals["FAIL"])
    rollover_failure_frequency = float(int(event_totals["FAIL"]) / max(1, settlement_event_count))

    terminal_dead = bool(terminal_wealth < 0.0 or int(event_totals["FAIL"]) > 0)
    last_timestep = int(history[-1]["t"]) if history else -1

    return {
        "history": history,
        "final": {
            "wealth": terminal_wealth,
            "weights": np.asarray(selector.w, dtype=float).copy(),
            "economic_state": {
                "due_curve": economic_state.due_curve.copy(),
                "expected_inflows": float(economic_state.expected_inflows),
                "liquidity_mismatch": float(economic_state.liquidity_mismatch),
                "h_near_idx": int(economic_state.h_near_idx),
            },
            "selector_state": {
                "strategic_credit_exposure": float(selector_state.strategic_credit_exposure),
                "derived_features": selector_state.derived_features(),
            },
            "config": asdict(cfg),
        },
        "evaluation_metrics": {
            "terminal_wealth": terminal_wealth,
            "time_to_death": int(time_to_death),
            "rollover_failure_frequency": rollover_failure_frequency,
        },
        "runtime": {
            "requested_backend": str(backend),
            "effective_backend": str(backend_norm),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "exact_metrics": {
            "terminal_dead": terminal_dead,
            "event_counts": event_totals,
            "rollover_count": int(event_totals["ROLLOVER"]),
            "last_timestep": int(last_timestep),
        },
    }


def run_phase_ii_evaluation(*args, **kwargs):
    """Lazy import to avoid circular imports with the evaluation module."""
    from .phase_ii_evaluation import run_phase_ii_evaluation as _run_phase_ii_evaluation

    return _run_phase_ii_evaluation(*args, **kwargs)
