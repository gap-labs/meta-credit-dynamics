from __future__ import annotations

from typing import Any, Callable, Mapping
import numpy as np

from .settlement import extract_due_obligations_at_tau, settle_due_claims_at_tau


HookLike = Mapping[str, Callable[..., None]] | Any | None


def _call_hook(hooks: HookLike, name: str, *args: Any) -> None:
    if hooks is None:
        return
    if isinstance(hooks, Mapping):
        fn = hooks.get(name)
        if callable(fn):
            fn(*args)
        return
    fn = getattr(hooks, name, None)
    if callable(fn):
        fn(*args)


def _default_due_extractor(state: Any, input_events: Mapping[str, Any], tau: int):
    r_vec = np.asarray(input_events.get("r_vec", []), dtype=float)
    due_returns = {
        "r_vec": r_vec,
        "total": float(r_vec.sum()),
    }
    due_obligations = extract_due_obligations_at_tau(state, input_events, tau)
    return due_obligations, due_returns


def _default_returns_booker(state: Any, due_returns: Mapping[str, Any], tau: int, input_events: Mapping[str, Any]):
    liquidity_before = float(state.wealth)
    total_returns = float(due_returns.get("total", 0.0))
    state.wealth = liquidity_before + total_returns
    state._last_r = total_returns
    state._last_c = float(input_events.get("c_total", 0.0))
    return liquidity_before, float(state.wealth)


def _default_settlement_processor(state: Any, due_obligations: list[dict[str, float]], tau: int):
    _, _, settlement_result = settle_due_claims_at_tau(
        state,
        tau,
        rng=None,
        config=getattr(state, "settlement_config", None),
        due_obligations=due_obligations,
    )
    return settlement_result


def _default_wealth_computer(state: Any, settlement_result: Mapping[str, Any], tau: int):
    obligations_after = settlement_result.get("obligations_after", [])
    due_total = float(sum(float(item.get("amount_due", 0.0)) for item in obligations_after))
    state.wealth = float(state.wealth) - due_total
    return float(state.wealth)


def _default_dead_decider(state: Any, wealth_value: float, tau: int) -> bool:
    settlement_failed = bool(getattr(state, "_last_settlement_failed", False))
    return settlement_failed or float(wealth_value) < 0.0


def _default_offer_publisher(state: Any, due_returns: Mapping[str, Any], input_events: Mapping[str, Any], tau: int):
    r_vec = np.asarray(due_returns.get("r_vec", []), dtype=float)
    c_total = float(input_events.get("c_total", 0.0))

    _, _, pi_total, pi_vec = state.compute_pi(r_vec, c_total)
    state.stats.update(pi_total)

    adv = pi_vec - state.stats.mu
    state.w = state.reweight_fn(state.w, adv)

    state._enforce_invariants()
    return []


def step_at_tau(
    state: Any,
    input_events: Mapping[str, Any],
    policy: Mapping[str, Callable[..., Any]] | None,
    tau: int,
    hooks: HookLike = None,
):
    """Execute one explicit Phase-H semantic step at time tau.

    Ordered phases:
      1) due extraction
      2) returns booking
      3) settlement
      4) wealth computation
      5) dead decision
      6) offer publication (only if alive)
    """

    freeze = bool(input_events.get("freeze", False))
    if freeze:
        state._enforce_invariants()
        next_tau = int(tau) + 1
        return state, [], next_tau

    policy = policy or {}
    due_extractor = policy.get("due_extractor", _default_due_extractor)
    returns_booker = policy.get("returns_booker", _default_returns_booker)
    settlement_processor = policy.get("settlement_processor", _default_settlement_processor)
    wealth_computer = policy.get("wealth_computer", _default_wealth_computer)
    dead_decider = policy.get("dead_decider", _default_dead_decider)
    offer_publisher = policy.get("offer_publisher", _default_offer_publisher)
    next_tau_fn = policy.get("next_tau_fn")

    due_obligations, due_returns = due_extractor(state, input_events, tau)
    _call_hook(hooks, "on_due_extracted", due_obligations, due_returns)

    liquidity_before, liquidity_after = returns_booker(state, due_returns, tau, input_events)
    _call_hook(hooks, "on_returns_booked", liquidity_before, liquidity_after, due_returns)

    settlement_result = settlement_processor(state, due_obligations, tau)
    _call_hook(hooks, "on_settlement_completed", settlement_result)

    wealth_value = float(wealth_computer(state, settlement_result, tau))
    _call_hook(hooks, "on_wealth_computed", wealth_value)

    is_dead = bool(dead_decider(state, wealth_value, tau))
    _call_hook(hooks, "on_dead_decided", is_dead, tau if is_dead else None)

    output_offers = []
    if not is_dead:
        output_offers = offer_publisher(state, due_returns, input_events, tau)
    _call_hook(hooks, "on_offers_published", output_offers)

    next_tau = next_tau_fn(state, tau) if callable(next_tau_fn) else int(tau) + 1
    return state, output_offers, int(next_tau)
