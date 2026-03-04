from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch


DEFAULT_MAX_CLAIMS_PER_PROCESS = 8
_ALLOWED_FLOAT_DTYPES = {torch.float32, torch.float64}


@dataclass(frozen=True)
class CudaState:
    """Passive CUDA representation (tensors + structural metadata)."""

    tensors: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DeviceState:
    """CPU/GPU tensor state contract for Phase H (process axis = dimension 0)."""

    device: torch.device
    max_claims_per_process: int
    claim_slot_index: torch.Tensor

    weights: torch.Tensor

    liquidity: torch.Tensor
    wealth: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor
    drawdown: torch.Tensor
    cum_pi: torch.Tensor
    peak_cum_pi: torch.Tensor
    rebirth_threshold: torch.Tensor

    claim_amount: torch.Tensor
    claim_interest: torch.Tensor
    claim_target: torch.Tensor
    claim_maturity_tau: torch.Tensor
    claim_active_mask: torch.Tensor
    claim_generation_id: torch.Tensor | None = None
    claim_parent_id: torch.Tensor | None = None
    claim_count: torch.Tensor | None = None

    due_mask: torch.Tensor | None = None
    due_amount: torch.Tensor | None = None
    returns_tensor: torch.Tensor | None = None

    dead_mask: torch.Tensor | None = None
    generation_id: torch.Tensor | None = None

    burndown_pool: torch.Tensor | None = None
    rebirth_budget: torch.Tensor | None = None
    rebirth_buffer: torch.Tensor | None = None
    inhabitants_staging: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "DeviceState":
        target_device = torch.device(device) if not isinstance(device, torch.device) else device

        def _to_optional(t: torch.Tensor | None):
            return None if t is None else t.to(target_device)

        return DeviceState(
            device=target_device,
            max_claims_per_process=int(self.max_claims_per_process),
            claim_slot_index=self.claim_slot_index.to(target_device),
            weights=self.weights.to(target_device),
            liquidity=self.liquidity.to(target_device),
            wealth=self.wealth.to(target_device),
            mean=self.mean.to(target_device),
            var=self.var.to(target_device),
            drawdown=self.drawdown.to(target_device),
            cum_pi=self.cum_pi.to(target_device),
            peak_cum_pi=self.peak_cum_pi.to(target_device),
            rebirth_threshold=self.rebirth_threshold.to(target_device),
            claim_amount=self.claim_amount.to(target_device),
            claim_interest=self.claim_interest.to(target_device),
            claim_target=self.claim_target.to(target_device),
            claim_maturity_tau=self.claim_maturity_tau.to(target_device),
            claim_active_mask=self.claim_active_mask.to(target_device),
            claim_generation_id=_to_optional(self.claim_generation_id),
            claim_parent_id=_to_optional(self.claim_parent_id),
            claim_count=_to_optional(self.claim_count),
            due_mask=_to_optional(self.due_mask),
            due_amount=_to_optional(self.due_amount),
            returns_tensor=_to_optional(self.returns_tensor),
            dead_mask=_to_optional(self.dead_mask),
            generation_id=_to_optional(self.generation_id),
            burndown_pool=_to_optional(self.burndown_pool),
            rebirth_budget=_to_optional(self.rebirth_budget),
            rebirth_buffer=_to_optional(self.rebirth_buffer),
            inhabitants_staging=_to_optional(self.inhabitants_staging),
        )

    def to_cpu(self) -> "DeviceState":
        return self.to("cpu")

    @property
    def claim_ids(self) -> torch.Tensor:
        return self.claim_target

    @property
    def parent_claim_ids(self) -> torch.Tensor | None:
        return self.claim_parent_id

    @property
    def generation_ids(self) -> torch.Tensor | None:
        if self.generation_id is None:
            return None
        return self.generation_id.unsqueeze(-1)

    @property
    def dead_flags(self) -> torch.Tensor | None:
        if self.dead_mask is None:
            return None
        return self.dead_mask.to(dtype=self.wealth.dtype).unsqueeze(-1)

    def _named_tensors(self) -> list[tuple[str, torch.Tensor]]:
        items: list[tuple[str, torch.Tensor]] = [
            ("claim_slot_index", self.claim_slot_index),
            ("weights", self.weights),
            ("liquidity", self.liquidity),
            ("wealth", self.wealth),
            ("mean", self.mean),
            ("var", self.var),
            ("drawdown", self.drawdown),
            ("cum_pi", self.cum_pi),
            ("peak_cum_pi", self.peak_cum_pi),
            ("rebirth_threshold", self.rebirth_threshold),
            ("claim_amount", self.claim_amount),
            ("claim_interest", self.claim_interest),
            ("claim_target", self.claim_target),
            ("claim_maturity_tau", self.claim_maturity_tau),
            ("claim_active_mask", self.claim_active_mask),
        ]
        optional = [
            ("claim_generation_id", self.claim_generation_id),
            ("claim_parent_id", self.claim_parent_id),
            ("claim_count", self.claim_count),
            ("due_mask", self.due_mask),
            ("due_amount", self.due_amount),
            ("returns_tensor", self.returns_tensor),
            ("dead_mask", self.dead_mask),
            ("generation_id", self.generation_id),
            ("burndown_pool", self.burndown_pool),
            ("rebirth_budget", self.rebirth_budget),
            ("rebirth_buffer", self.rebirth_buffer),
            ("inhabitants_staging", self.inhabitants_staging),
        ]
        for name, tensor in optional:
            if tensor is not None:
                items.append((name, tensor))
        return items

    def validate_shapes(self) -> None:
        n = int(self.liquidity.shape[0]) if self.liquidity.ndim == 1 else -1
        if n <= 0:
            raise ValueError("liquidity must have shape [N] with N > 0")

        if self.max_claims_per_process <= 0:
            raise ValueError("max_claims_per_process must be positive")

        c = int(self.max_claims_per_process)

        vector_fields = {
            "liquidity": self.liquidity,
            "wealth": self.wealth,
            "mean": self.mean,
            "var": self.var,
            "drawdown": self.drawdown,
            "cum_pi": self.cum_pi,
            "peak_cum_pi": self.peak_cum_pi,
            "rebirth_threshold": self.rebirth_threshold,
        }
        for name, tensor in vector_fields.items():
            if tensor.ndim != 1 or tensor.shape[0] != n:
                raise ValueError(f"{name} must have shape [N], got {tuple(tensor.shape)}")

        if self.weights.ndim != 2 or self.weights.shape[0] != n:
            raise ValueError(f"weights must have shape [N, K], got {tuple(self.weights.shape)}")

        if self.claim_slot_index.ndim != 1 or self.claim_slot_index.shape[0] != c:
            raise ValueError(f"claim_slot_index must have shape [C], got {tuple(self.claim_slot_index.shape)}")

        claim_matrix_fields = {
            "claim_amount": self.claim_amount,
            "claim_interest": self.claim_interest,
            "claim_target": self.claim_target,
            "claim_maturity_tau": self.claim_maturity_tau,
            "claim_active_mask": self.claim_active_mask,
        }
        for name, tensor in claim_matrix_fields.items():
            if tensor.ndim != 2 or tensor.shape != (n, c):
                raise ValueError(f"{name} must have shape [N, C], got {tuple(tensor.shape)}")

        optional_matrix_fields = {
            "claim_generation_id": self.claim_generation_id,
            "claim_parent_id": self.claim_parent_id,
            "due_mask": self.due_mask,
            "due_amount": self.due_amount,
            "returns_tensor": self.returns_tensor,
        }
        for name, tensor in optional_matrix_fields.items():
            if tensor is not None and (tensor.ndim != 2 or tensor.shape != (n, c)):
                raise ValueError(f"{name} must have shape [N, C], got {tuple(tensor.shape)}")

        optional_vector_fields = {
            "claim_count": self.claim_count,
            "dead_mask": self.dead_mask,
            "generation_id": self.generation_id,
            "burndown_pool": self.burndown_pool,
            "rebirth_budget": self.rebirth_budget,
        }
        for name, tensor in optional_vector_fields.items():
            if tensor is not None and (tensor.ndim != 1 or tensor.shape[0] != n):
                raise ValueError(f"{name} must have shape [N], got {tuple(tensor.shape)}")

    def validate_dtypes(self) -> None:
        float_fields = {
            "weights": self.weights,
            "liquidity": self.liquidity,
            "wealth": self.wealth,
            "mean": self.mean,
            "var": self.var,
            "drawdown": self.drawdown,
            "cum_pi": self.cum_pi,
            "peak_cum_pi": self.peak_cum_pi,
            "rebirth_threshold": self.rebirth_threshold,
            "claim_amount": self.claim_amount,
            "claim_interest": self.claim_interest,
        }
        if self.due_amount is not None:
            float_fields["due_amount"] = self.due_amount
        if self.returns_tensor is not None:
            float_fields["returns_tensor"] = self.returns_tensor
        if self.burndown_pool is not None:
            float_fields["burndown_pool"] = self.burndown_pool
        if self.rebirth_budget is not None:
            float_fields["rebirth_budget"] = self.rebirth_budget

        for name, tensor in float_fields.items():
            if tensor.dtype not in _ALLOWED_FLOAT_DTYPES:
                raise ValueError(f"{name} must be float32 or float64, got {tensor.dtype}")

        int_fields = {
            "claim_slot_index": self.claim_slot_index,
            "claim_target": self.claim_target,
                "claim_maturity_tau": self.claim_maturity_tau,
        }
        if self.claim_generation_id is not None:
            int_fields["claim_generation_id"] = self.claim_generation_id
        if self.claim_parent_id is not None:
            int_fields["claim_parent_id"] = self.claim_parent_id
        if self.claim_count is not None:
            int_fields["claim_count"] = self.claim_count
        if self.generation_id is not None:
            int_fields["generation_id"] = self.generation_id

        for name, tensor in int_fields.items():
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32, got {tensor.dtype}")

        bool_fields = {
            "claim_active_mask": self.claim_active_mask,
        }
        if self.due_mask is not None:
            bool_fields["due_mask"] = self.due_mask
        if self.dead_mask is not None:
            bool_fields["dead_mask"] = self.dead_mask

        for name, tensor in bool_fields.items():
            if tensor.dtype != torch.bool:
                raise ValueError(f"{name} must be bool, got {tensor.dtype}")

    def validate_device(self, *, expected_backend: str | None = None) -> None:
        expected = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        for name, tensor in self._named_tensors():
            same_type = tensor.device.type == expected.type
            same_index = expected.index is None or tensor.device.index == expected.index
            if not (same_type and same_index):
                raise ValueError(f"{name} is on {tensor.device}, expected {expected}")

        if expected_backend is not None:
            backend = str(expected_backend)
            if backend == "cuda" and expected.type != "cuda":
                raise ValueError(f"expected cuda backend but state device is {expected}")
            if backend == "cpu" and expected.type != "cpu":
                raise ValueError(f"expected cpu backend but state device is {expected}")

    def validate_determinism_ready(self) -> None:
        self.validate_shapes()
        self.validate_dtypes()

        slot_expected = torch.arange(
            int(self.max_claims_per_process),
            dtype=torch.int32,
            device=self.claim_slot_index.device,
        )
        if not torch.equal(self.claim_slot_index, slot_expected):
            raise ValueError("claim_slot_index must be a stable ascending sequence [0..C-1]")

        for name, tensor in self._named_tensors():
            if tensor.requires_grad:
                raise ValueError(f"{name} has requires_grad=True; state must be inference-only")
            if tensor.is_sparse:
                raise ValueError(f"{name} is sparse; dense tensors are required for deterministic slot layout")

        if self.claim_count is not None:
            if torch.any(self.claim_count < 0) or torch.any(self.claim_count > int(self.max_claims_per_process)):
                raise ValueError("claim_count out of bounds")
            c = int(self.max_claims_per_process)
            idx = torch.arange(c, device=self.claim_count.device).unsqueeze(0).expand(self.claim_count.shape[0], c)
            invalid_active = self.claim_active_mask & (idx >= self.claim_count.unsqueeze(1))
            if bool(torch.any(invalid_active)):
                raise ValueError("claim_active_mask violates stable prefix ordering implied by claim_count")


def to_device_state(
    selector,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    max_claims_per_process: int = DEFAULT_MAX_CLAIMS_PER_PROCESS,
) -> DeviceState:
    """CPU selector -> DeviceState with batch dimension B=1."""
    if selector.w is None:
        raise ValueError("selector.w must be initialized before to_device_state()")

    target_device = torch.device(device) if not isinstance(device, torch.device) else device

    w_np = np.asarray(selector.w, dtype=float)
    w_t = torch.as_tensor(w_np, device=target_device)
    base_dtype = dtype if dtype is not None else w_t.dtype
    w_t = w_t.to(dtype=base_dtype).unsqueeze(0)

    def _vector(val: float) -> torch.Tensor:
        t = torch.as_tensor(np.array(val), device=target_device, dtype=base_dtype)
        return t.reshape(1)

    generation_id = int(getattr(selector, "generation_id", 0))
    dead_flag = bool(getattr(selector, "dead", False))
    c = int(max_claims_per_process)
    if c <= 0:
        raise ValueError("max_claims_per_process must be positive")

    claim_amount = torch.zeros((1, c), dtype=base_dtype, device=target_device)
    claim_interest = torch.zeros((1, c), dtype=base_dtype, device=target_device)
    claim_target = torch.full((1, c), -1, dtype=torch.int32, device=target_device)
    claim_maturity_tau = torch.full((1, c), -1, dtype=torch.int32, device=target_device)
    claim_active_mask = torch.zeros((1, c), dtype=torch.bool, device=target_device)
    claim_generation_id = torch.full((1, c), generation_id, dtype=torch.int32, device=target_device)
    claim_parent_id = torch.full((1, c), -1, dtype=torch.int32, device=target_device)
    claim_count = torch.zeros((1,), dtype=torch.int32, device=target_device)

    due_mask = torch.zeros((1, c), dtype=torch.bool, device=target_device)
    due_amount = torch.zeros((1, c), dtype=base_dtype, device=target_device)
    returns_tensor = torch.zeros((1, c), dtype=base_dtype, device=target_device)

    dead_mask = torch.as_tensor([dead_flag], dtype=torch.bool, device=target_device)
    generation = torch.as_tensor([generation_id], dtype=torch.int32, device=target_device)

    burndown_pool = torch.zeros((1,), dtype=base_dtype, device=target_device)
    rebirth_budget = torch.zeros((1,), dtype=base_dtype, device=target_device)
    rebirth_buffer = torch.zeros((1, 4), dtype=base_dtype, device=target_device)
    inhabitants_staging = torch.zeros((1, 4), dtype=base_dtype, device=target_device)

    state = DeviceState(
        device=target_device,
        max_claims_per_process=c,
        claim_slot_index=torch.arange(c, dtype=torch.int32, device=target_device),
        weights=w_t,
        liquidity=_vector(float(selector.wealth)),
        wealth=_vector(float(selector.wealth)),
        mean=_vector(float(selector.stats.mu)),
        var=_vector(float(selector.stats.var)),
        drawdown=_vector(float(selector.stats.dd)),
        cum_pi=_vector(float(selector.stats.cum_pi)),
        peak_cum_pi=_vector(float(selector.stats.peak_cum_pi)),
        rebirth_threshold=_vector(float(selector.rebirth_threshold)),
        claim_amount=claim_amount,
        claim_interest=claim_interest,
        claim_target=claim_target,
        claim_maturity_tau=claim_maturity_tau,
        claim_active_mask=claim_active_mask,
        claim_generation_id=claim_generation_id,
        claim_parent_id=claim_parent_id,
        claim_count=claim_count,
        due_mask=due_mask,
        due_amount=due_amount,
        returns_tensor=returns_tensor,
        dead_mask=dead_mask,
        generation_id=generation,
        burndown_pool=burndown_pool,
        rebirth_budget=rebirth_budget,
        rebirth_buffer=rebirth_buffer,
        inhabitants_staging=inhabitants_staging,
    )

    state.validate_shapes()
    state.validate_dtypes()
    state.validate_device(expected_backend=target_device.type)
    state.validate_determinism_ready()

    return state


def canonical_state_dump(
    selector,
    *,
    stack_manager: Any | None = None,
    sediment: Any | None = None,
) -> Dict[str, Any]:
    """Return a canonical, next-step-sufficient dump for test equivalence."""
    stats = selector.stats
    return {
        "selector": {
            "wealth": float(selector.wealth),
            "rebirth_threshold": float(selector.rebirth_threshold),
            "kind": str(selector.kind),
            "K": int(selector.K),
            "w": None if selector.w is None else np.asarray(selector.w, dtype=float),
            "_last_r": float(selector._last_r),
            "_last_c": float(selector._last_c),
        },
        "stats": {
            "beta": float(stats.beta),
            "mu": float(stats.mu),
            "var": float(stats.var),
            "dd": float(stats.dd),
            "cum_pi": float(stats.cum_pi),
            "peak_cum_pi": float(stats.peak_cum_pi),
            "seed_var": float(stats.seed_var),
        },
        "stack_manager": _dump_stack_manager(stack_manager),
        "sediment": _dump_sediment(sediment),
    }


def _dump_stack_manager(stack_manager: Any | None) -> Dict[str, Any] | None:
    if stack_manager is None:
        return None
    stacks_dump = []
    for sid, st in stack_manager.stacks.items():
        stacks_dump.append(
            {
                "stack_id": str(sid),
                "members": list(st.members.keys()),
                "_w_internal_keys": list(st._w_internal.keys()),
                "_w_internal_vals": [float(v) for v in st._w_internal.values()],
                "_alive": bool(st._alive),
                "_dt": float(st._dt),
                "_mu": float(st._mu),
                "_var": float(st._var),
                "_beta": float(st._beta),
                "_dd_peak": float(st._dd_peak),
                "_dd_val": float(st._dd_val),
                "_dd_max": float(st._dd_max),
                "_cvar": float(st._cvar),
            }
        )
    return {
        "stack_cfg": {
            "C_agg": float(stack_manager.stack_cfg.C_agg),
            "min_size": int(stack_manager.stack_cfg.min_size),
            "max_size": int(stack_manager.stack_cfg.max_size),
            "stack_weighting": str(stack_manager.stack_cfg.stack_weighting),
            "tau_mu": float(stack_manager.stack_cfg.tau_mu),
            "tau_vol": float(stack_manager.stack_cfg.tau_vol),
            "tau_cvar": float(stack_manager.stack_cfg.tau_cvar),
            "tau_dd": float(stack_manager.stack_cfg.tau_dd),
            "use_cvar": bool(stack_manager.stack_cfg.use_cvar),
        },
        "thresholds": {
            "tau_mu": float(stack_manager.thresholds.tau_mu),
            "tau_vol": float(stack_manager.thresholds.tau_vol),
            "tau_cvar": float(stack_manager.thresholds.tau_cvar),
            "tau_surv": float(stack_manager.thresholds.tau_surv),
            "tau_corr": float(stack_manager.thresholds.tau_corr),
            "min_size": int(stack_manager.thresholds.min_size),
            "max_size": int(stack_manager.thresholds.max_size),
        },
        "stacks": stacks_dump,
        "world_id": str(stack_manager.world_id),
        "phase_id": str(stack_manager.phase_id),
        "run_id": str(stack_manager.run_id),
        "_counter": int(stack_manager._counter),
        "_t_counter": int(stack_manager._t_counter),
        "_last_reject_t": int(stack_manager._last_reject_t),
    }


def _dump_sediment(sediment: Any | None) -> Dict[str, Any] | None:
    if sediment is None:
        return None
    nodes = []
    for node in sediment.nodes():
        nodes.append(
            {
                "node_id": int(node.node_id),
                "members": list(node.members),
                "mask": dict(node.mask),
                "world_id": str(node.world_id),
                "phase_id": str(node.phase_id),
                "t": int(node.t),
                "run_id": str(node.run_id),
            }
        )
    return {
        "forbid_pairs": bool(sediment.forbid_pairs),
        "_last_node_id_by_run": dict(sediment._last_node_id_by_run),
        "_next_node_id": int(sediment._next_node_id),
        "nodes": nodes,
    }


def toCuda(state_dump: Dict[str, Any], device: str, dtype: torch.dtype | None = None) -> CudaState:
    """CPU/Python -> CUDA tensor representation. No semantic changes."""
    selector = state_dump["selector"]
    stats = state_dump["stats"]

    tensors: Dict[str, torch.Tensor] = {}

    w = selector["w"]
    if w is not None:
        w_arr = np.asarray(w)
        w_t = torch.as_tensor(w_arr, device=device)
        if dtype is not None:
            w_t = w_t.to(dtype)
        tensors["selector.w"] = w_t
        if (w_t < 0).any() or not torch.isclose(w_t.sum(), torch.tensor(1.0, device=device, dtype=w_t.dtype)):
            raise ValueError("Invariant violation: weights not on simplex")

    for key in ("wealth", "rebirth_threshold", "_last_r", "_last_c"):
        val = np.array(selector[key])
        t = torch.as_tensor(val, device=device)
        if dtype is not None:
            t = t.to(dtype)
        tensors[f"selector.{key}"] = t

    for key in ("beta", "mu", "var", "dd", "cum_pi", "peak_cum_pi", "seed_var"):
        val = np.array(stats[key])
        t = torch.as_tensor(val, device=device)
        if dtype is not None:
            t = t.to(dtype)
        tensors[f"stats.{key}"] = t
        if key in ("var", "dd", "seed_var") and t.item() < 0:
            raise ValueError(f"Invariant violation: {key} < 0")

    meta = {
        "selector.kind": selector["kind"],
        "selector.K": int(selector["K"]),
        "stack_manager": state_dump.get("stack_manager"),
        "sediment": state_dump.get("sediment"),
    }

    # Stack internal state (if present)
    sm = state_dump.get("stack_manager")
    if sm is not None:
        meta["stack_manager"] = dict(sm)
        for st in sm.get("stacks", []):
            sid = st["stack_id"]
            meta[f"stack.{sid}.members"] = list(st["members"])
            meta[f"stack.{sid}._w_internal_keys"] = list(st["_w_internal_keys"])

            for key in ("_w_internal_vals", "_dt", "_mu", "_var", "_beta", "_dd_peak", "_dd_val", "_dd_max", "_cvar"):
                val = np.asarray(st[key], dtype=float)
                t = torch.as_tensor(val, device=device)
                if dtype is not None:
                    t = t.to(dtype)
                tensors[f"stack.{sid}.{key}"] = t

            alive_val = np.asarray(1.0 if st["_alive"] else 0.0, dtype=float)
            t_alive = torch.as_tensor(alive_val, device=device)
            if dtype is not None:
                t_alive = t_alive.to(dtype)
            tensors[f"stack.{sid}._alive"] = t_alive

    return CudaState(tensors=tensors, meta=meta)


def fromCuda(cuda_state: CudaState) -> Dict[str, Any]:
    """CUDA tensor representation -> CPU/Python state dump."""
    t = cuda_state.tensors
    meta = cuda_state.meta

    w_t = t.get("selector.w")
    w = None if w_t is None else w_t.detach().cpu().numpy()

    selector = {
        "wealth": float(t["selector.wealth"].item()),
        "rebirth_threshold": float(t["selector.rebirth_threshold"].item()),
        "kind": str(meta["selector.kind"]),
        "K": int(meta["selector.K"]),
        "w": w,
        "_last_r": float(t["selector._last_r"].item()),
        "_last_c": float(t["selector._last_c"].item()),
    }

    stats = {
        "beta": float(t["stats.beta"].item()),
        "mu": float(t["stats.mu"].item()),
        "var": float(t["stats.var"].item()),
        "dd": float(t["stats.dd"].item()),
        "cum_pi": float(t["stats.cum_pi"].item()),
        "peak_cum_pi": float(t["stats.peak_cum_pi"].item()),
        "seed_var": float(t["stats.seed_var"].item()),
    }

    stack_manager = meta.get("stack_manager")
    if stack_manager is not None:
        stacks = []
        for st in stack_manager.get("stacks", []):
            sid = st["stack_id"]
            keys = meta.get(f"stack.{sid}._w_internal_keys", [])
            vals_t = t.get(f"stack.{sid}._w_internal_vals")
            vals = [] if vals_t is None else [float(x) for x in vals_t.detach().cpu().numpy().tolist()]
            w_internal = dict(zip(keys, vals))
            stacks.append(
                {
                    "stack_id": str(sid),
                    "members": list(meta.get(f"stack.{sid}.members", [])),
                    "_w_internal_keys": list(keys),
                    "_w_internal_vals": vals,
                    "_alive": bool(t[f"stack.{sid}._alive"].item() >= 0.5),
                    "_dt": float(t[f"stack.{sid}._dt"].item()),
                    "_mu": float(t[f"stack.{sid}._mu"].item()),
                    "_var": float(t[f"stack.{sid}._var"].item()),
                    "_beta": float(t[f"stack.{sid}._beta"].item()),
                    "_dd_peak": float(t[f"stack.{sid}._dd_peak"].item()),
                    "_dd_val": float(t[f"stack.{sid}._dd_val"].item()),
                    "_dd_max": float(t[f"stack.{sid}._dd_max"].item()),
                    "_cvar": float(t[f"stack.{sid}._cvar"].item()),
                }
            )
        stack_manager = dict(stack_manager)
        stack_manager["stacks"] = stacks

    return {
        "selector": selector,
        "stats": stats,
        "stack_manager": stack_manager,
        "sediment": meta.get("sediment"),
    }
