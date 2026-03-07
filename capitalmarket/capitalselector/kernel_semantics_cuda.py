from __future__ import annotations

from typing import Any, Mapping

import torch

from .cuda_state import DeviceState


def _as_vector(value: torch.Tensor | float, *, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.reshape(1).expand(n)
        if t.ndim == 1 and t.shape[0] == n:
            return t
        raise ValueError(f"expected scalar or shape [N], got {tuple(t.shape)}")
    return torch.full((n,), float(value), device=device, dtype=dtype)


def _compact_claim_slots(state: DeviceState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = state.claim_active_mask
    n, c = active.shape
    device = active.device

    slot_idx = torch.arange(c, device=device, dtype=torch.int64).unsqueeze(0).expand(n, c)
    active_i64 = active.to(dtype=torch.int64)
    inactive_i64 = (~active).to(dtype=torch.int64)

    active_count_i64 = active_i64.sum(dim=1, keepdim=True)
    active_pos = torch.cumsum(active_i64, dim=1) - 1
    inactive_pos = torch.cumsum(inactive_i64, dim=1) - 1 + active_count_i64
    dest_pos = torch.where(active, active_pos, inactive_pos)

    order = torch.empty((n, c), device=device, dtype=torch.int64)
    order.scatter_(1, dest_pos, slot_idx)

    gather_idx_float = order
    gather_idx_bool = order
    gather_idx_int = order

    claim_amount = torch.gather(state.claim_amount, 1, gather_idx_float)
    claim_interest = torch.gather(state.claim_interest, 1, gather_idx_float)
    claim_target = torch.gather(state.claim_target, 1, gather_idx_int)
    claim_maturity_tau = torch.gather(state.claim_maturity_tau, 1, gather_idx_int)
    claim_active_mask = torch.gather(state.claim_active_mask, 1, gather_idx_bool)

    claim_generation_id = None
    if state.claim_generation_id is not None:
        claim_generation_id = torch.gather(state.claim_generation_id, 1, gather_idx_int)

    claim_parent_id = None
    if state.claim_parent_id is not None:
        claim_parent_id = torch.gather(state.claim_parent_id, 1, gather_idx_int)

    claim_count = claim_active_mask.sum(dim=1).to(dtype=torch.int32)

    fields: dict[str, torch.Tensor] = {
        "claim_amount": claim_amount,
        "claim_interest": claim_interest,
        "claim_target": claim_target,
        "claim_maturity_tau": claim_maturity_tau,
        "claim_active_mask": claim_active_mask,
        "claim_count": claim_count,
    }
    if claim_generation_id is not None:
        fields["claim_generation_id"] = claim_generation_id
    if claim_parent_id is not None:
        fields["claim_parent_id"] = claim_parent_id
    return claim_count, fields


def step_at_tau_cuda(
    state: DeviceState,
    input_events: Mapping[str, Any],
    *,
    tau: int,
) -> dict[str, Any]:
    freeze = bool(input_events.get("freeze", False))
    n = int(state.wealth.shape[0])
    device = state.device
    dtype = state.wealth.dtype

    c_total = _as_vector(input_events.get("c_total", 0.0), n=n, device=device, dtype=dtype)
    returns_total = _as_vector(input_events.get("returns_total", 0.0), n=n, device=device, dtype=dtype)
    k = int(state.weights.shape[1])

    returns_vec_in = input_events.get("returns_vec")
    if isinstance(returns_vec_in, torch.Tensor):
        returns_vec = returns_vec_in.to(device=device, dtype=dtype)
    else:
        returns_vec = torch.zeros((n, k), device=device, dtype=dtype)

    if returns_vec.ndim == 1 and returns_vec.shape[0] == k:
        returns_vec = returns_vec.unsqueeze(0).expand(n, k)
    if returns_vec.ndim != 2 or returns_vec.shape != (n, k):
        raise ValueError(f"returns_vec must have shape [N, K], got {tuple(returns_vec.shape)}")

    lambda_cash_share_in = input_events.get("lambda_cash_share", 0.5)
    if isinstance(lambda_cash_share_in, torch.Tensor):
        lambda_cash_share_t = lambda_cash_share_in.to(device=device, dtype=dtype).reshape(())
    else:
        lambda_cash_share_t = torch.scalar_tensor(float(lambda_cash_share_in), device=device, dtype=dtype)

    accept_by_default_in = input_events.get("accept_by_default", True)
    if isinstance(accept_by_default_in, torch.Tensor):
        accept_by_default_t = accept_by_default_in.to(device=device, dtype=torch.bool).reshape(())
    else:
        accept_by_default_t = torch.scalar_tensor(bool(accept_by_default_in), device=device, dtype=torch.bool)

    stats_beta_in = input_events.get("stats_beta", 0.0)
    if isinstance(stats_beta_in, torch.Tensor):
        stats_beta_t = stats_beta_in.to(device=device, dtype=dtype).reshape(())
    else:
        stats_beta_t = torch.scalar_tensor(float(stats_beta_in), device=device, dtype=dtype)

    phase_i_beta_in = input_events.get("phase_i_beta", stats_beta_t)
    if isinstance(phase_i_beta_in, torch.Tensor):
        phase_i_beta_t = phase_i_beta_in.to(device=device, dtype=dtype).reshape(())
    else:
        phase_i_beta_t = torch.scalar_tensor(float(phase_i_beta_in), device=device, dtype=dtype)

    phase_i_beta_r_in = input_events.get("phase_i_beta_r", stats_beta_t)
    if isinstance(phase_i_beta_r_in, torch.Tensor):
        phase_i_beta_r_t = phase_i_beta_r_in.to(device=device, dtype=dtype).reshape(())
    else:
        phase_i_beta_r_t = torch.scalar_tensor(float(phase_i_beta_r_in), device=device, dtype=dtype)

    maturity_offset_in = input_events.get("future_maturity_offset", 1)
    if isinstance(maturity_offset_in, torch.Tensor):
        maturity_offset_i32 = maturity_offset_in.to(device=device, dtype=torch.int32).reshape(())
    else:
        maturity_offset_i32 = torch.scalar_tensor(int(maturity_offset_in), device=device, dtype=torch.int32)

    cuda_ops_count = 0

    if freeze:
        return {
            "state": state,
            "is_dead": torch.zeros((n,), device=device, dtype=torch.bool),
            "settlement_failed": torch.zeros((n,), device=device, dtype=torch.bool),
            "offer_publication_mask": torch.zeros((n,), device=device, dtype=torch.bool),
            "claim_cash_paid": torch.zeros((n,), device=device, dtype=dtype),
            "claim_remainder": torch.zeros((n,), device=device, dtype=dtype),
            "legacy_cash_paid": torch.zeros((n,), device=device, dtype=dtype),
            "legacy_unresolved": torch.zeros((n,), device=device, dtype=dtype),
            "cuda_ops_count": cuda_ops_count,
            "tau": int(tau),
        }

    maturity = state.claim_maturity_tau
    due_mask = state.claim_active_mask & (maturity == int(tau))
    due_amount = torch.where(due_mask, state.claim_amount, torch.zeros_like(state.claim_amount))
    due_claim_amount = torch.where(due_mask, due_amount, torch.zeros_like(due_amount))
    claim_due_total = due_claim_amount.sum(dim=1)
    has_due_claim = claim_due_total > 0
    cuda_ops_count += 1

    liquidity_before = state.liquidity
    liquidity_after_returns = liquidity_before + returns_total
    wealth_after_returns = state.wealth + returns_total
    cuda_ops_count += 1

    available_cash = torch.clamp(wealth_after_returns, min=0.0)
    claim_cash_candidate = torch.minimum(available_cash, lambda_cash_share_t * claim_due_total)
    claim_remainder = torch.clamp(claim_due_total - claim_cash_candidate, min=0.0)

    safe_due_total = torch.clamp(claim_due_total, min=torch.as_tensor(1e-12, device=device, dtype=dtype))
    slot_share = torch.where(due_mask, due_amount / safe_due_total.unsqueeze(1), torch.zeros_like(due_amount))
    slot_cash_paid = claim_cash_candidate.unsqueeze(1) * slot_share
    slot_remainder = torch.where(due_mask, torch.clamp(due_amount - slot_cash_paid, min=0.0), torch.zeros_like(due_amount))

    accepted_claim = has_due_claim & accept_by_default_t
    rejected_claim = has_due_claim & (~accepted_claim)

    can_pay_rejected_claim_full = wealth_after_returns >= claim_due_total
    rejected_claim_paid_full = rejected_claim & can_pay_rejected_claim_full
    rejected_claim_failed = rejected_claim & (~can_pay_rejected_claim_full)

    wealth_after_claim = torch.where(
        accepted_claim,
        wealth_after_returns - claim_cash_candidate,
        torch.where(rejected_claim_paid_full, wealth_after_returns - claim_due_total, wealth_after_returns),
    )

    unresolved_claim = torch.where(rejected_claim_failed, claim_due_total, torch.zeros_like(claim_due_total))
    cuda_ops_count += 1

    eps_t = torch.scalar_tensor(1e-12, device=device, dtype=dtype)
    close_due_claim = (accepted_claim.unsqueeze(1) & due_mask & (slot_remainder <= eps_t)) | (
        rejected_claim_paid_full.unsqueeze(1) & due_mask
    )
    rewrite_due_claim = accepted_claim.unsqueeze(1) & due_mask & (slot_remainder > eps_t)

    claim_amount_after = state.claim_amount.clone()
    claim_active_after = state.claim_active_mask.clone()
    claim_maturity_after = state.claim_maturity_tau.clone()

    claim_amount_after = torch.where(close_due_claim, torch.zeros_like(claim_amount_after), claim_amount_after)
    claim_active_after = torch.where(close_due_claim, torch.zeros_like(claim_active_after), claim_active_after)
    claim_maturity_after = torch.where(close_due_claim, torch.full_like(claim_maturity_after, -1), claim_maturity_after)

    claim_amount_after = torch.where(rewrite_due_claim, slot_remainder, claim_amount_after)
    claim_maturity_after = torch.where(
        rewrite_due_claim,
        torch.full_like(claim_maturity_after, int(tau) + int(maturity_offset_i32.item())),
        claim_maturity_after,
    )

    state_tmp = DeviceState(
        **{
            **state.__dict__,
            "claim_amount": claim_amount_after,
            "claim_active_mask": claim_active_after,
            "claim_maturity_tau": claim_maturity_after,
        }
    )
    claim_count_compact, compact_fields = _compact_claim_slots(state_tmp)
    cuda_ops_count += 1

    has_legacy_due = c_total > 0
    can_pay_legacy_full = wealth_after_claim >= c_total
    legacy_cash_paid = torch.where(has_legacy_due & can_pay_legacy_full, c_total, torch.zeros_like(c_total))
    legacy_unresolved = torch.where(has_legacy_due & (~can_pay_legacy_full), c_total, torch.zeros_like(c_total))

    wealth_after_settlement = wealth_after_claim - legacy_cash_paid
    settlement_failed = (rejected_claim_failed) | (legacy_unresolved > 0)
    cuda_ops_count += 1

    wealth_after_wealth_phase = wealth_after_settlement - unresolved_claim - legacy_unresolved
    is_dead = settlement_failed | (wealth_after_wealth_phase < 0.0)
    cuda_ops_count += 1

    alive_mask = ~is_dead
    pi_total = returns_total - c_total

    beta = stats_beta_t
    one = torch.as_tensor(1.0, device=device, dtype=dtype)

    mean_new = (one - beta) * state.mean + beta * pi_total
    var_new = torch.clamp((one - beta) * state.var + beta * torch.square(pi_total - state.mean), min=0.0)
    cum_pi_new = state.cum_pi + pi_total
    peak_cum_pi_new = torch.maximum(state.peak_cum_pi, cum_pi_new)
    drawdown_new = peak_cum_pi_new - cum_pi_new
    cuda_ops_count += 1

    # Phase-I deterministic updates on device state.
    mu_term_new = state.mu_term.clone()
    rho_new = state.rho.clone()

    if k > 0:
        batch_idx = torch.arange(n, device=device, dtype=torch.int64)
        dominant_channel = torch.argmax(torch.abs(returns_vec), dim=1)

        # RETURN events (horizon h0): one EWMA update per channel with positive return.
        positive_returns = torch.clamp(returns_vec, min=0.0)
        mu_h0 = mu_term_new[:, :, 0]
        mu_h0_updated = (1.0 - phase_i_beta_t) * mu_h0 + phase_i_beta_t * positive_returns
        mu_h0 = torch.where(positive_returns > 0.0, mu_h0_updated, mu_h0)

        # DUE_CASH events (horizon h0): aggregated legacy cash signal on deterministic fallback channel.
        has_legacy_due = c_total > 0.0
        if bool(torch.any(has_legacy_due)):
            old_due = mu_h0[batch_idx, dominant_channel]
            due_pi = -c_total
            new_due = (1.0 - phase_i_beta_t) * old_due + phase_i_beta_t * due_pi
            mu_h0[batch_idx, dominant_channel] = torch.where(has_legacy_due, new_due, old_due)

        # COST events (horizon h0): aggregated claim due signal on deterministic fallback channel.
        has_claim_due = claim_due_total > 0.0
        if bool(torch.any(has_claim_due)):
            old_cost = mu_h0[batch_idx, dominant_channel]
            cost_pi = -claim_due_total
            new_cost = (1.0 - phase_i_beta_t) * old_cost + phase_i_beta_t * cost_pi
            mu_h0[batch_idx, dominant_channel] = torch.where(has_claim_due, new_cost, old_cost)

        mu_term_new[:, :, 0] = mu_h0

        # Risk impulses from ROLLOVER (claim remainder) and FAIL (settlement_failed).
        risk_impulse = ((claim_remainder > 0.0) | settlement_failed).to(dtype=dtype)
        old_rho = rho_new[batch_idx, dominant_channel]
        new_rho = (1.0 - phase_i_beta_r_t) * old_rho + phase_i_beta_r_t * risk_impulse
        rho_new[batch_idx, dominant_channel] = new_rho

    state_next = DeviceState(
        **{
            **state.__dict__,
            "liquidity": wealth_after_wealth_phase,
            "wealth": wealth_after_wealth_phase,
            **compact_fields,
            "mean": torch.where(alive_mask, mean_new, state.mean),
            "var": torch.where(alive_mask, var_new, state.var),
            "cum_pi": torch.where(alive_mask, cum_pi_new, state.cum_pi),
            "peak_cum_pi": torch.where(alive_mask, peak_cum_pi_new, state.peak_cum_pi),
            "drawdown": torch.where(alive_mask, drawdown_new, state.drawdown),
            "mu_term": mu_term_new,
            "rho": rho_new,
            "dead_mask": is_dead,
            "due_mask": due_mask,
            "due_amount": due_amount,
            "claim_count": claim_count_compact,
        }
    )

    return {
        "state": state_next,
        "is_dead": is_dead,
        "settlement_failed": settlement_failed,
        "offer_publication_mask": alive_mask,
        "claim_cash_paid": claim_cash_candidate,
        "claim_remainder": claim_remainder,
        "legacy_cash_paid": legacy_cash_paid,
        "legacy_unresolved": legacy_unresolved,
        "cuda_ops_count": cuda_ops_count,
        "tau": int(tau),
    }


def batch_core_step(
    state: DeviceState,
    *,
    input_events: Mapping[str, Any],
    tau: int,
) -> dict[str, Any]:
    """Single batch state->state CUDA core entrypoint for one step."""
    return step_at_tau_cuda(state, input_events, tau=int(tau))
