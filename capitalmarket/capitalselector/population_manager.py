from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .builder import CapitalSelectorBuilder
from .cpu_impl import CpuCore
from .cuda_impl import CudaCore
from .inhabitants import InhabitantsBook
from .kernel_semantics import step_at_tau
from .lifecycle_cuda import compute_lifecycle_cuda
from .ledger import ClaimLedger
from .world_burndown import BurndownPool


@dataclass(frozen=True)
class RebirthConfig:
    enabled: bool = True
    base_liquidity: float = 0.0
    eta: float = 0.0
    kappa: float = 1.0
    selection_epsilon: float = 1e-12
    max_claims_per_process: int = 1_000_000


class PopulationManager:
    """CPU meta-layer for dead archival and rebirth instantiation."""

    def __init__(self, *, processes: dict[int, Any], rebirth_config: RebirthConfig | None = None, backend: str = "cpu"):
        self.processes: dict[int, Any] = dict(processes)
        self.rebirth_config = rebirth_config or RebirthConfig()
        self.backend = str(backend)
        self.inhabitants = InhabitantsBook()
        self.burndown = BurndownPool()
        self._selection_cursor = 0
        self._next_process_id = (max(self.processes.keys()) + 1) if self.processes else 0
        self._cores: dict[int, Any] = {}

        for process_id, selector in self.processes.items():
            self._ensure_selector_meta(selector, process_id)
            self._cores[process_id] = self._build_core(start_tau=0)

    @classmethod
    def single(cls, selector: Any, *, process_id: int = 0, rebirth_config: RebirthConfig | None = None, backend: str = "cpu"):
        manager = cls(processes={int(process_id): selector}, rebirth_config=rebirth_config, backend=backend)
        return manager

    def step_tau(self, *, tau: int, process_events: Mapping[int, Mapping[str, Any]], jackpot: float = 0.0) -> dict[str, Any]:
        dead_now: list[Any] = []
        dead_keys: list[tuple[int, int]] = []
        active_cuda_selectors: list[Any] = []
        active_cuda_cores: list[Any] = []

        for process_id in sorted(self.processes.keys()):
            selector = self.processes[process_id]
            if bool(getattr(selector, "dead", False)):
                continue

            event = dict(process_events.get(process_id, {}))
            r_vec = np.asarray(event.get("r_vec", []), dtype=float)
            c_total = float(event.get("c_total", 0.0))
            freeze = bool(event.get("freeze", False))

            if selector.w is None or len(selector.w) != len(r_vec):
                selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
                selector.K = len(r_vec)

            core = self._cores[process_id]
            if hasattr(core, "step_with_tau"):
                core.step_with_tau(selector, r_vec, c_total, freeze=freeze, tau=int(tau))
            else:
                step_at_tau(
                    selector,
                    {"r_vec": r_vec, "c_total": c_total, "freeze": freeze},
                    policy=None,
                    tau=int(tau),
                    hooks=None,
                )

            selector._fitness_integral = float(getattr(selector, "_fitness_integral", 0.0)) + float(selector.wealth)

            if self.backend == "cuda" and hasattr(core, "lifecycle_snapshot"):
                active_cuda_selectors.append(selector)
                active_cuda_cores.append(core)
            else:
                is_dead = bool(getattr(selector, "_last_settlement_failed", False)) or float(selector.wealth) < 0.0
                if is_dead:
                    selector.dead = True
                    selector.tau_dead = int(tau)
                    dead_now.append(selector)
                    dead_keys.append((int(selector.process_id), int(selector.generation_id)))

        allocations_by_pid: dict[int, float] = {}
        if self.backend == "cuda" and active_cuda_selectors:
            snaps = [core.lifecycle_snapshot(selector) for selector, core in zip(active_cuda_selectors, active_cuda_cores)]
            wealth_t = torch.cat([snap["wealth"] for snap in snaps], dim=0)
            dead_t = torch.cat([snap["dead_mask"] for snap in snaps], dim=0)
            pid_t = torch.cat([snap["process_id"] for snap in snaps], dim=0)
            gen_t = torch.cat([snap["generation_id"] for snap in snaps], dim=0)
            dead_keys_t = torch.stack([pid_t, gen_t], dim=1)

            lifecycle = compute_lifecycle_cuda(
                wealth=wealth_t,
                dead_mask_semantic=dead_t,
                dead_keys=dead_keys_t,
                pool_before=float(self.burndown.B_current),
                jackpot=float(jackpot),
                rebirth_enabled=bool(self.rebirth_config.enabled),
                base_liquidity=float(self.rebirth_config.base_liquidity),
                eta=float(self.rebirth_config.eta),
                kappa=float(self.rebirth_config.kappa),
                epsilon=float(self.rebirth_config.selection_epsilon),
            )

            self.burndown.B_current = float(lifecycle.pool_final.item())
            dead_idx = torch.nonzero(lifecycle.dead_mask, as_tuple=False).flatten().tolist()

            for dead_offset, idx in enumerate(dead_idx):
                selector = active_cuda_selectors[int(idx)]
                selector.dead = True
                selector.tau_dead = int(tau)
                dead_now.append(selector)
                dead_keys.append((int(selector.process_id), int(selector.generation_id)))
                if int(dead_offset) < int(lifecycle.rebirth_allocations_dead.shape[0]):
                    allocations_by_pid[int(selector.process_id)] = float(lifecycle.rebirth_allocations_dead[int(dead_offset)].item())
        else:
            burn_total = float(sum(max(0.0, -float(selector.wealth)) for selector in dead_now))
            self.burndown.apply_tau_inflows(burn=burn_total, kappa=self.rebirth_config.kappa, jackpot=jackpot)

        dead_now.sort(key=lambda selector: (int(selector.process_id), int(selector.generation_id)))
        for selector in dead_now:
            self.inhabitants.append_dead(
                process_id=int(selector.process_id),
                generation_id=int(selector.generation_id),
                tau_dead=int(selector.tau_dead),
                fitness=float(getattr(selector, "_fitness_integral", 0.0)),
                final_liquidity=float(selector.wealth),
                metadata={"kind": getattr(selector, "kind", "unknown")},
            )

        newborn_ids: list[int] = []
        if self.rebirth_config.enabled and dead_now:
            if self.backend == "cuda":
                allocations = [float(allocations_by_pid.get(int(selector.process_id), 0.0)) for selector in dead_now]
            else:
                B_tau = float(self.burndown.B_current)
                requested = [float(self.rebirth_config.base_liquidity + self.rebirth_config.eta * B_tau) for _ in dead_now]
                allocations = self.burndown.allocate_fair_same_tau(
                    requested=requested,
                    stable_keys=dead_keys,
                    epsilon=self.rebirth_config.selection_epsilon,
                )

            for selector, allocation in zip(dead_now, allocations):
                parent = self._select_parent_entry()
                generation_id = int(parent.generation_id + 1)
                new_id = self._allocate_process_id()
                newborn = self._instantiate_new_process(
                    process_id=new_id,
                    generation_id=generation_id,
                    liquidity=float(allocation),
                    template=selector,
                )
                self.processes[new_id] = newborn
                self._cores[new_id] = self._build_core(start_tau=int(tau) + 1)
                newborn_ids.append(new_id)

        return {
            "dead_ids": [int(selector.process_id) for selector in dead_now],
            "newborn_ids": newborn_ids,
            "pool": float(self.burndown.B_current),
        }

    def _select_parent_entry(self):
        entries = self.inhabitants.entries()
        entries.sort(key=lambda entry: (entry.process_id, entry.generation_id))
        if not entries:
            raise ValueError("cannot select parent from empty inhabitants book")

        positives = [entry for entry in entries if max(entry.fitness, 0.0) > self.rebirth_config.selection_epsilon]
        if positives:
            weights = [max(entry.fitness, 0.0) for entry in positives]
            weight_sum = float(sum(weights))
            if weight_sum > self.rebirth_config.selection_epsilon:
                target = (self._selection_cursor % 10_000) / 10_000.0
                self._selection_cursor += 1
                cumulative = 0.0
                for entry, weight in zip(positives, weights):
                    cumulative += weight / weight_sum
                    if target <= cumulative:
                        return entry
                return positives[-1]

        entry = entries[self._selection_cursor % len(entries)]
        self._selection_cursor += 1
        return entry

    def _instantiate_new_process(self, *, process_id: int, generation_id: int, liquidity: float, template: Any):
        builder = (
            CapitalSelectorBuilder()
            .with_K(int(getattr(template, "K", 0)))
            .with_kind(str(getattr(template, "kind", "entrepreneur")))
            .with_initial_wealth(float(liquidity))
            .with_rebirth_threshold(float(getattr(template, "rebirth_threshold", 0.0)))
        )
        selector = builder.build()
        selector.process_id = int(process_id)
        selector.generation_id = int(generation_id)
        selector.dead = False
        selector.tau_dead = None
        selector._fitness_integral = 0.0
        selector.claim_ledger = ClaimLedger(max_claims_per_process=self.rebirth_config.max_claims_per_process)
        selector.offers = []
        return selector

    def _allocate_process_id(self) -> int:
        process_id = int(self._next_process_id)
        self._next_process_id += 1
        return process_id

    def _ensure_selector_meta(self, selector: Any, process_id: int) -> None:
        selector.process_id = int(getattr(selector, "process_id", process_id))
        selector.generation_id = int(getattr(selector, "generation_id", 0))
        selector.dead = bool(getattr(selector, "dead", False))
        selector.tau_dead = getattr(selector, "tau_dead", None)
        selector._fitness_integral = float(getattr(selector, "_fitness_integral", 0.0))
        if not hasattr(selector, "claim_ledger"):
            selector.claim_ledger = ClaimLedger(max_claims_per_process=self.rebirth_config.max_claims_per_process)
        if not hasattr(selector, "offers"):
            selector.offers = []

    def _build_core(self, *, start_tau: int):
        if self.backend == "cuda":
            return CudaCore(start_tau=int(start_tau), device="cuda")
        return CpuCore(start_tau=int(start_tau))
