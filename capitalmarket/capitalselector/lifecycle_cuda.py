from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CudaLifecycleOutput:
    dead_mask: torch.Tensor
    burn_by_process: torch.Tensor
    pool_after_inflow: torch.Tensor
    rebirth_allocations_dead: torch.Tensor
    pool_final: torch.Tensor


def compute_lifecycle_cuda(
    *,
    wealth: torch.Tensor,
    dead_mask_semantic: torch.Tensor,
    dead_keys: torch.Tensor,
    pool_before: float,
    jackpot: float,
    rebirth_enabled: bool,
    base_liquidity: float,
    eta: float,
    kappa: float,
    epsilon: float,
    rounding_scale: int = 12,
) -> CudaLifecycleOutput:
    if wealth.ndim != 1:
        raise ValueError(f"wealth must be [N], got {tuple(wealth.shape)}")
    if dead_mask_semantic.ndim != 1 or dead_mask_semantic.shape[0] != wealth.shape[0]:
        raise ValueError("dead_mask_semantic must be [N]")
    if dead_keys.ndim != 2 or dead_keys.shape[0] != wealth.shape[0] or dead_keys.shape[1] != 2:
        raise ValueError("dead_keys must be [N,2] = (process_id,generation_id)")

    device = wealth.device
    dtype = wealth.dtype

    zero = torch.scalar_tensor(0.0, device=device, dtype=dtype)
    unit = torch.scalar_tensor(10.0 ** (-int(rounding_scale)), device=device, dtype=dtype)
    scale = torch.scalar_tensor(10.0 ** int(rounding_scale), device=device, dtype=dtype)
    base_liquidity_t = torch.scalar_tensor(float(base_liquidity), device=device, dtype=dtype)
    eta_t = torch.scalar_tensor(float(eta), device=device, dtype=dtype)
    kappa_t = torch.scalar_tensor(float(kappa), device=device, dtype=dtype)
    epsilon_t = torch.scalar_tensor(float(epsilon), device=device, dtype=dtype)

    dead_mask = dead_mask_semantic.to(dtype=torch.bool)
    burn_by_process = torch.where(dead_mask, torch.clamp(-wealth, min=zero), torch.zeros_like(wealth))

    pool0 = torch.scalar_tensor(float(pool_before), device=device, dtype=dtype)
    jackpot_t = torch.scalar_tensor(max(0.0, float(jackpot)), device=device, dtype=dtype)
    inflow = torch.clamp(kappa_t, min=zero) * burn_by_process.sum()
    pool_after_inflow = torch.clamp(pool0 + inflow + jackpot_t, min=zero)

    dead_idx = torch.nonzero(dead_mask, as_tuple=False).flatten()
    d = int(dead_idx.numel())
    if (not rebirth_enabled) or d == 0:
        return CudaLifecycleOutput(
            dead_mask=dead_mask,
            burn_by_process=burn_by_process,
            pool_after_inflow=pool_after_inflow,
            rebirth_allocations_dead=torch.zeros((d,), device=device, dtype=dtype),
            pool_final=pool_after_inflow,
        )

    requested_value = base_liquidity_t + eta_t * pool_after_inflow
    requested = torch.full((d,), requested_value, device=device, dtype=dtype)
    requested = torch.clamp(requested, min=zero)
    total_requested = requested.sum()

    if bool(total_requested <= epsilon_t):
        return CudaLifecycleOutput(
            dead_mask=dead_mask,
            burn_by_process=burn_by_process,
            pool_after_inflow=pool_after_inflow,
            rebirth_allocations_dead=torch.zeros_like(requested),
            pool_final=pool_after_inflow,
        )

    if bool(total_requested <= pool_after_inflow):
        alloc_dead = requested
        return CudaLifecycleOutput(
            dead_mask=dead_mask,
            burn_by_process=burn_by_process,
            pool_after_inflow=pool_after_inflow,
            rebirth_allocations_dead=alloc_dead,
            pool_final=torch.clamp(pool_after_inflow - alloc_dead.sum(), min=zero),
        )

    factor = torch.where(pool_after_inflow > zero, pool_after_inflow / total_requested, zero)
    raw = requested * factor
    floored = torch.floor(raw * scale) / scale

    residual = torch.clamp(pool_after_inflow - floored.sum(), min=zero)
    quanta = int(torch.round(residual * scale).item())

    dead_keys_only = dead_keys.index_select(0, dead_idx).to(dtype=torch.int64)
    max_gen = int(dead_keys_only[:, 1].max().item()) if d > 0 else 0
    stride = max(1, max_gen + 1)
    lex = dead_keys_only[:, 0] * stride + dead_keys_only[:, 1]
    order = torch.argsort(lex, dim=0, stable=True)

    add = torch.zeros((d,), device=device, dtype=dtype)
    if quanta > 0 and d > 0:
        base_q = quanta // d
        rem_q = quanta % d
        if base_q > 0:
            add = add + (float(base_q) * unit)
        if rem_q > 0:
            rem = torch.zeros((d,), device=device, dtype=dtype)
            rem_indices = order[:rem_q]
            rem[rem_indices] = unit
            add = add + rem

    alloc_dead = floored + add
    pool_final = torch.clamp(pool_after_inflow - alloc_dead.sum(), min=zero)

    return CudaLifecycleOutput(
        dead_mask=dead_mask,
        burn_by_process=burn_by_process,
        pool_after_inflow=pool_after_inflow,
        rebirth_allocations_dead=alloc_dead,
        pool_final=pool_final,
    )