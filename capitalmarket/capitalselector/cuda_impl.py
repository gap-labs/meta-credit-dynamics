from __future__ import annotations

import os
from typing import Any
import numpy as np
import torch

from .cuda_state import DeviceState, to_device_state
from .kernel_semantics_cuda import batch_core_step


_PUBLISH_POLICY_BY_PROFILE = {
    "test": "full",
    "parity": "full",
    "benchmark": "minimal",
    "masssim": "minimal",
    "prod": "minimal",
    "runtime": "minimal",
}


def _resolve_publish_policy() -> tuple[str, str]:
    profile = str(os.environ.get("CAPM_CUDA_PROFILE", "test")).strip().lower()
    if profile not in _PUBLISH_POLICY_BY_PROFILE:
        allowed_profiles = ", ".join(sorted(_PUBLISH_POLICY_BY_PROFILE.keys()))
        raise ValueError(f"invalid CAPM_CUDA_PROFILE='{profile}', expected one of: {allowed_profiles}")

    explicit_policy = os.environ.get("CAPM_CUDA_PUBLISH_POLICY")
    if explicit_policy is None:
        return _PUBLISH_POLICY_BY_PROFILE[profile], profile

    policy = str(explicit_policy).strip().lower()
    if policy not in {"minimal", "full"}:
        raise ValueError("invalid CAPM_CUDA_PUBLISH_POLICY, expected 'minimal' or 'full'")
    return policy, profile

class CudaCore:
    """CUDA backend bound to Phase-H event-order semantics.

    This keeps CPU as semantic oracle and mirrors the same ordered phases.
    """

    def __init__(self, *, hooks=None, policy=None, start_tau: int = 0, device: str | torch.device = "cuda"):
        self._tau = int(start_tau)
        self._hooks = hooks
        self._policy = policy
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._mode = str(os.environ.get("CAPM_MODE", "deterministic"))
        self._publish_policy, self._publish_profile = _resolve_publish_policy()
        self._state_by_selector: dict[int, DeviceState] = {}
        self._imported_claims_by_selector: dict[int, int] = {}
        self._scalar_cache_by_selector: dict[int, dict[str, torch.Tensor]] = {}
        self._metrics = {
            "effective_backend": "cuda",
            "CAPM_MODE": self._mode,
            "CAPM_CUDA_PROFILE": self._publish_profile,
            "CAPM_CUDA_PUBLISH_POLICY": self._publish_policy,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_ops_count": 0,
            "cpu_fallback_used": 0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "overflow_events": 0,
            "steps": 0,
        }
        self._snapshot_stride = int(os.environ.get("CAPM_CUDA_SNAPSHOT_EVERY", "1000"))

    def step(self, selector, r_vec, c_total, *, freeze: bool) -> None:
        self.step_with_tau(selector, r_vec, c_total, freeze=freeze, tau=self._tau)
        self._tau += 1

    def step_with_tau(self, selector: Any, r_vec, c_total, *, freeze: bool, tau: int) -> None:
        if self._device.type != "cuda":
            raise RuntimeError("CudaCore requires a cuda device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is False")

        selector_id = id(selector)
        state = self._state_by_selector.get(selector_id)
        if state is None:
            max_claims = int(getattr(getattr(selector, "claim_ledger", None), "max_claims_per_process", 8))
            state = to_device_state(
                selector,
                device=self._device,
                max_claims_per_process=max_claims,
            )
            self._state_by_selector[selector_id] = state
            self._imported_claims_by_selector[selector_id] = 0
            self._scalar_cache_by_selector[selector_id] = {
                "returns_total": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "c_total": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "lambda_cash_share": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "stats_beta": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "accept_by_default": torch.zeros((), device=self._device, dtype=torch.bool),
                "future_maturity_offset": torch.zeros((), device=self._device, dtype=torch.int32),
                "phase_i_beta": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "phase_i_beta_r": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
            }

        r_arr = np.asarray(r_vec, dtype=float)
        if hasattr(selector, "ensure_channel_state"):
            selector.ensure_channel_state(len(r_arr))
        elif selector.w is None or len(selector.w) != len(r_arr):
            selector.w = np.ones(len(r_arr), dtype=float) / max(1, len(r_arr))
            selector.K = len(r_arr)

        state_prev = state
        state = self._ingest_new_claims(selector=selector, state=state)
        state.validate_shapes()
        state.validate_dtypes()
        state.validate_device(expected_backend="cuda")
        state.validate_determinism_ready()

        scalar_cache = self._scalar_cache_by_selector[selector_id]
        returns_total = scalar_cache["returns_total"].fill_(float(np.asarray(r_arr, dtype=float).sum()))
        c_total_tensor = scalar_cache["c_total"].fill_(float(c_total))
        returns_vec_tensor = torch.as_tensor(np.asarray(r_arr, dtype=float), device=self._device, dtype=state.wealth.dtype).unsqueeze(0)
        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int((returns_total.element_size() + c_total_tensor.element_size()))
        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(returns_vec_tensor.numel() * returns_vec_tensor.element_size())

        settlement_cfg = dict(getattr(selector, "settlement_config", {}) or {})
        lambda_cash_share = float(settlement_cfg.get("lambda_cash_share", getattr(selector, "lambda_cash_share", 0.5)))
        accept_by_default = bool(settlement_cfg.get("accept_by_default", True))
        lambda_cash_share_t = scalar_cache["lambda_cash_share"].fill_(lambda_cash_share)
        accept_by_default_t = scalar_cache["accept_by_default"].fill_(accept_by_default)
        stats_beta_t = scalar_cache["stats_beta"].fill_(float(getattr(selector.stats, "beta", 0.0)))
        future_maturity_offset_t = scalar_cache["future_maturity_offset"].fill_(int(settlement_cfg.get("future_maturity_offset", 1)))
        phase_i_beta_t = scalar_cache["phase_i_beta"].fill_(float(getattr(selector, "beta_term", getattr(selector.stats, "beta", 0.0))))
        phase_i_beta_r_t = scalar_cache["phase_i_beta_r"].fill_(float(getattr(selector, "beta_r", getattr(selector.stats, "beta", 0.0))))

        out = batch_core_step(
            state,
            input_events={
                "returns_total": returns_total,
                "returns_vec": returns_vec_tensor,
                "c_total": c_total_tensor,
                "freeze": bool(freeze),
                "lambda_cash_share": lambda_cash_share_t,
                "accept_by_default": accept_by_default_t,
                "stats_beta": stats_beta_t,
                "phase_i_beta": phase_i_beta_t,
                "phase_i_beta_r": phase_i_beta_r_t,
                "future_maturity_offset": future_maturity_offset_t,
            },
            tau=int(tau),
        )
        state_next = out["state"]
        self._metrics["cuda_ops_count"] = int(self._metrics["cuda_ops_count"]) + int(out.get("cuda_ops_count", 0))

        self._publish_selector_runtime(
            selector=selector,
            state_prev=state_prev,
            state=state_next,
            out=out,
            r_vec=r_arr,
            c_total=float(c_total),
            freeze=bool(freeze),
            tau=int(tau),
        )

        self._state_by_selector[selector_id] = state_next
        self._metrics["steps"] = int(self._metrics["steps"]) + 1

    def metrics_snapshot(self) -> dict[str, Any]:
        return dict(self._metrics)

    def outstanding_claim_count(self, selector: Any) -> int:
        state = self._state_by_selector.get(id(selector))
        if state is None or state.claim_count is None:
            return 0
        return int(state.claim_count[0].item())

    def lifecycle_snapshot(self, selector: Any) -> dict[str, torch.Tensor]:
        state = self._state_by_selector.get(id(selector))
        if state is None:
            raise KeyError("selector state not initialized")

        process_id = int(getattr(selector, "process_id", 0))
        generation_id = int(getattr(selector, "generation_id", 0))
        return {
            "wealth": state.wealth,
            "dead_mask": state.dead_mask if state.dead_mask is not None else torch.zeros_like(state.wealth, dtype=torch.bool),
            "process_id": torch.as_tensor([process_id], device=state.device, dtype=torch.int64),
            "generation_id": torch.as_tensor([generation_id], device=state.device, dtype=torch.int64),
        }

    def _ingest_new_claims(self, *, selector: Any, state: DeviceState) -> DeviceState:
        ledger = getattr(selector, "claim_ledger", None)
        if ledger is None or not hasattr(ledger, "claim_tensor_batch_for_process"):
            return state

        selector_id = id(selector)
        process_id = int(getattr(selector, "process_id", 0))
        imported = int(self._imported_claims_by_selector.get(selector_id, 0))
        batch = ledger.claim_tensor_batch_for_process(
            process_id=process_id,
            start_index=imported,
            device=state.device,
            float_dtype=state.claim_amount.dtype,
        )

        batch_len = int(batch["batch_len"])
        if batch_len <= 0:
            return state

        imported_after = imported + batch_len
        self._imported_claims_by_selector[selector_id] = imported_after

        open_mask = batch["is_open"]
        if open_mask.numel() == 0:
            return state

        open_amount = batch["nominal"][open_mask]
        if open_amount.numel() == 0:
            return state

        open_target = batch["claim_target"][open_mask]
        open_maturity = batch["maturity_tau"][open_mask]
        open_generation = batch["generation_id"][open_mask]

        free_mask = ~state.claim_active_mask[0]
        free_slots = torch.nonzero(free_mask, as_tuple=False).flatten()

        requested = int(open_amount.numel())
        available = int(free_slots.numel())
        insert_count = min(requested, available)

        if requested > available:
            if self._mode == "deterministic":
                raise RuntimeError(
                    f"claim capacity exceeded in deterministic mode: process={process_id}, max_claims_per_process={state.max_claims_per_process}"
                )
            self._metrics["overflow_events"] = int(self._metrics["overflow_events"]) + int(requested - available)

        if insert_count <= 0:
            return state

        insert_slots = free_slots[:insert_count]

        state.claim_amount[0, insert_slots] = open_amount[:insert_count]
        state.claim_interest[0, insert_slots] = torch.zeros((insert_count,), device=state.device, dtype=state.claim_interest.dtype)
        state.claim_target[0, insert_slots] = open_target[:insert_count].to(dtype=state.claim_target.dtype)
        state.claim_maturity_tau[0, insert_slots] = open_maturity[:insert_count].to(dtype=state.claim_maturity_tau.dtype)
        state.claim_active_mask[0, insert_slots] = True

        if state.claim_generation_id is not None:
            state.claim_generation_id[0, insert_slots] = open_generation[:insert_count].to(dtype=state.claim_generation_id.dtype)
        if state.claim_parent_id is not None:
            state.claim_parent_id[0, insert_slots] = torch.full(
                (insert_count,),
                -1,
                device=state.device,
                dtype=state.claim_parent_id.dtype,
            )
        if state.claim_count is not None:
            state.claim_count[0] = state.claim_active_mask[0].sum().to(dtype=state.claim_count.dtype)

        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(
            (insert_count * state.claim_amount.element_size())
            + (insert_count * state.claim_target.element_size())
            + (insert_count * state.claim_maturity_tau.element_size())
            + (insert_count * state.claim_active_mask.element_size())
        )

        return state

    def _publish_selector_runtime(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
    ) -> None:
        if self._publish_policy == "minimal":
            self._publish_selector_runtime_minimal(
                selector=selector,
                state_prev=state_prev,
                state=state,
                out=out,
                r_vec=r_vec,
                c_total=c_total,
                freeze=freeze,
                tau=tau,
            )
            return

        self._publish_selector_runtime_full(
            selector=selector,
            state_prev=state_prev,
            state=state,
            out=out,
            r_vec=r_vec,
            c_total=c_total,
            freeze=freeze,
            tau=tau,
        )

    def _publish_selector_runtime_minimal(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
    ) -> None:
        # Minimal Sync Surface (System Contract)
        # Required by CPU meta/control path only:
        # - selector.wealth: fitness accumulation in PopulationManager
        # - selector.stats.mu: CPU reweight advantage (pi_vec - mu)
        # - offer_publication_mask: determines whether CPU reweight is executed
        # - settlement_failed: deterministic dead/settlement decisions
        # - selector.w (full vector) only when offer_publication_mask is true
        #   because compute_pi/reweight_fn are CPU-side today.
        # Everything else remains device-resident in minimal mode.
        scalar_sync = torch.stack(
            [
                state.wealth[0],
                state.mean[0],
                out["offer_publication_mask"][0].to(dtype=state.wealth.dtype),
                out["settlement_failed"][0].to(dtype=state.wealth.dtype),
            ]
        ).detach().cpu()

        selector.wealth = float(scalar_sync[0])
        selector.stats.mu = float(scalar_sync[1])

        settlement_failed = bool(scalar_sync[3])
        selector._last_settlement_failed = settlement_failed
        selector.dead = settlement_failed or float(selector.wealth) < 0.0

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(4 * state.wealth.element_size())

        if freeze:
            return

        if str(getattr(selector, "selector_policy", "myopic")) in {"term_aware", "term_risk"}:
            self._sync_selector_phase_i_state(selector=selector, state=state)

        offer_mask = bool(scalar_sync[2])
        if offer_mask and not bool(selector.dead):
            w_host = state.weights[0].detach().cpu().numpy().astype(float, copy=True)
            selector.w = w_host
            selector.K = int(w_host.shape[0])
            self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
                state.weights[0].numel() * state.weights[0].element_size()
            )

            _, _, _, pi_vec = selector.compute_pi(r_vec, float(c_total))
            adv = selector.compute_advantage(np.asarray(pi_vec, dtype=float))
            selector.w = selector.reweight_fn(np.asarray(selector.w, dtype=float), adv)
            selector._enforce_invariants()
            state.weights[0] = torch.as_tensor(selector.w, device=state.device, dtype=state.weights.dtype)
            self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(
                state.weights[0].numel() * state.weights[0].element_size()
            )

        stride = max(1, int(getattr(selector, "cuda_snapshot_every", self._snapshot_stride)))
        if int(tau) % stride == 0:
            selector._cuda_state_snapshot = {
                "claim_count": int(state.claim_count[0].item()) if state.claim_count is not None else 0,
                "active_due": int(state.due_mask.sum().item()) if state.due_mask is not None else 0,
            }

    def _publish_selector_runtime_full(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
    ) -> None:
        stats_vec = torch.stack(
            [
                state.wealth[0],
                state.mean[0],
                state.var[0],
                state.drawdown[0],
                state.cum_pi[0],
                state.peak_cum_pi[0],
                out["legacy_cash_paid"][0],
                out["legacy_unresolved"][0],
            ]
        ).detach().cpu().tolist()

        selector.wealth = float(stats_vec[0])
        selector.stats.mu = float(stats_vec[1])
        selector.stats.var = float(stats_vec[2])
        selector.stats.dd = float(stats_vec[3])
        selector.stats.cum_pi = float(stats_vec[4])
        selector.stats.peak_cum_pi = float(stats_vec[5])
        selector._last_r = float(stats_vec[6])
        selector._last_c = float(stats_vec[7])

        w_host = state.weights[0].detach().cpu().numpy().astype(float, copy=True)
        selector.w = w_host
        selector.K = int(w_host.shape[0])

        bool_vec = torch.stack(
            [
                state.dead_mask[0] if state.dead_mask is not None else torch.as_tensor(bool(getattr(selector, "dead", False)), device=state.device),
                out["settlement_failed"][0],
            ]
        ).detach().cpu().tolist()
        selector.dead = bool(bool_vec[0])
        selector._last_settlement_failed = bool(bool_vec[1])

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
            (9 * state.wealth.element_size()) + state.weights[0].numel() * state.weights[0].element_size()
        )

        if freeze:
            return

        if str(getattr(selector, "selector_policy", "myopic")) in {"term_aware", "term_risk"}:
            self._sync_selector_phase_i_state(selector=selector, state=state)

        offer_mask = bool(out["offer_publication_mask"][0].item())
        if offer_mask and not bool(selector.dead):
            _, _, _, pi_vec = selector.compute_pi(np.asarray(r_vec, dtype=float), float(c_total))
            adv = selector.compute_advantage(np.asarray(pi_vec, dtype=float))
            selector.w = selector.reweight_fn(np.asarray(selector.w, dtype=float), adv)
            selector._enforce_invariants()
            state.weights[0] = torch.as_tensor(selector.w, device=state.device, dtype=state.weights.dtype)
            self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(state.weights[0].numel() * state.weights[0].element_size())

        stride = max(1, int(getattr(selector, "cuda_snapshot_every", self._snapshot_stride)))
        if int(tau) % stride == 0:
            selector._cuda_state_snapshot = {
                "claim_count": int(state.claim_count[0].item()) if state.claim_count is not None else 0,
                "active_due": int(state.due_mask.sum().item()) if state.due_mask is not None else 0,
            }

    def _sync_selector_phase_i_state(self, *, selector: Any, state: DeviceState) -> None:
        mu_host = state.mu_term[0].detach().cpu().numpy().astype(float, copy=True)
        rho_host = state.rho[0].detach().cpu().numpy().astype(float, copy=True)

        selector.mu_term = mu_host
        selector.rho = rho_host
        selector.horizon_count = int(mu_host.shape[1]) if mu_host.ndim == 2 else int(getattr(selector, "horizon_count", 0))

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
            state.mu_term[0].numel() * state.mu_term[0].element_size()
            + state.rho[0].numel() * state.rho[0].element_size()
        )
