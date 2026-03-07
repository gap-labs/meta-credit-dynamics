from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import os
import numpy as np
import torch

from .config import ProfileAConfig
from .interfaces import World, Curriculum, Teacher, validate_world_output
from .builder import CapitalSelectorBuilder
from .cpu_impl import CpuCore
from .cuda_impl import CudaCore
from .determinism import enable_determinism
from .phase_i_state import DEFAULT_LAMBDA_RISK
from .population_manager import PopulationManager, RebirthConfig
from .selector_policy import DEFAULT_SELECTOR_POLICY, SelectorPolicy


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str = "A"
    freeze: bool = False
    mode: str = "A"
    deterministic: bool = False
    seed: int | None = 0
    backend: str | None = None
    capm_mode: str | None = None
    config_backend: str | None = None
    config_mode: str | None = None
    max_claims_per_process: int = 1_000_000
    enable_meta_rebirth: bool = False
    rebirth_base_liquidity: float = 0.0
    rebirth_eta: float = 0.0
    rebirth_kappa: float = 1.0
    selector_policy: SelectorPolicy = DEFAULT_SELECTOR_POLICY
    lambda_risk: float = DEFAULT_LAMBDA_RISK


def _resolve_backend(cfg: RuntimeConfig) -> tuple[str, str]:
    env_backend = os.environ.get("CAPM_BACKEND", os.environ.get("CAPM_DEVICE", None))
    requested = cfg.backend if cfg.backend is not None else (env_backend if env_backend is not None else (cfg.config_backend if cfg.config_backend is not None else "cpu"))
    requested_norm = str(requested).strip().lower()
    if requested_norm == "gpu":
        requested_norm = "cuda"
    if requested_norm not in {"cpu", "cuda"}:
        raise RuntimeError(f"invalid backend '{requested}'; expected 'cpu' or 'cuda'")

    if requested_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("backend=cuda requested but torch.cuda.is_available() is False")

    return requested_norm, requested_norm


def _resolve_capm_mode(cfg: RuntimeConfig) -> str:
    env_mode = os.environ.get("CAPM_MODE", None)
    mode = cfg.capm_mode if cfg.capm_mode is not None else (env_mode if env_mode is not None else (cfg.config_mode if cfg.config_mode is not None else "deterministic"))
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"deterministic", "fast"}:
        raise RuntimeError(f"invalid CAPM_MODE '{mode}'; expected 'deterministic' or 'fast'")
    return mode_norm


def _validate_builder_runtime(cfg: RuntimeConfig, *, effective_backend: str, effective_mode: str) -> None:
    if int(cfg.max_claims_per_process) <= 0:
        raise RuntimeError("max_claims_per_process must be > 0")

    dtype_env = os.environ.get("CAPM_DTYPE", "").strip().lower()
    if dtype_env and dtype_env not in {"float32", "float64"}:
        raise RuntimeError("CAPM_DTYPE must be 'float32' or 'float64' when set")

    if effective_mode == "deterministic" and cfg.seed is None:
        raise RuntimeError("deterministic mode requires an explicit seed")

    if effective_backend == "cuda" and torch.cuda.is_available() is False:
        raise RuntimeError("backend/device mismatch: cuda backend without available cuda device")


def run_population(
    *,
    world: World,
    steps: int,
    config: RuntimeConfig,
    backend: str,
    capm_mode: str,
) -> Dict[str, Any]:
    os.environ["CAPM_MODE"] = str(capm_mode)

    selector = (
        CapitalSelectorBuilder()
        .with_K(0)
        .with_selector_policy(config.selector_policy)
        .with_lambda_risk(config.lambda_risk)
        .build()
    )
    manager = PopulationManager.single(
        selector,
        process_id=0,
        backend=str(backend),
        rebirth_config=RebirthConfig(
            enabled=True,
            base_liquidity=float(config.rebirth_base_liquidity),
            eta=float(config.rebirth_eta),
            kappa=float(config.rebirth_kappa),
            max_claims_per_process=int(config.max_claims_per_process),
        ),
    )

    trace: list[str] = []
    history: list[dict[str, Any]] = []
    population_history: list[dict[int, dict[str, Any]]] = []

    for t in range(int(steps)):
        out = world.step(t)

        jackpot = float(out.get("jackpot", 0.0)) if isinstance(out, dict) else 0.0
        process_events: dict[int, dict[str, Any]] = {}

        if isinstance(out, dict) and "population" in out:
            for item in out.get("population", []):
                process_id = int(item["process_id"])
                r_vec = np.asarray(item.get("r", []), dtype=float)
                c_total = float(item.get("c", 0.0))
                process_events[process_id] = {"r_vec": r_vec, "c_total": c_total, "freeze": bool(config.freeze)}
        else:
            r_vec, c_total = validate_world_output(out)
            process_events[0] = {"r_vec": r_vec, "c_total": c_total, "freeze": bool(config.freeze)}

        manager.step_tau(tau=t, process_events=process_events, jackpot=jackpot)

        snapshot = {pid: sel.state() for pid, sel in sorted(manager.processes.items())}
        population_history.append(snapshot)
        history.append(snapshot.get(0, {"wealth": float("nan")}))
        trace.append("step")

    return {
        "history": history,
        "trace": trace,
        "population_history": population_history,
        "inhabitants": manager.inhabitants.entries(),
        "pool": float(manager.burndown.B_current),
        "runtime": {
            "requested_backend": str(backend),
            "effective_backend": str(backend),
            "CAPM_MODE": str(capm_mode),
            "seed": int(config.seed) if config.seed is not None else None,
            "deterministic": bool(config.deterministic),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def run(
    *,
    world: World,
    steps: int,
    config: RuntimeConfig | None = None,
    profile: ProfileAConfig | None = None,
) -> Dict[str, Any]:
    """Canonical runtime entry point (Profile A).

    This is a minimal runner for deterministic Profile A semantics.
    """
    cfg = config or RuntimeConfig()
    if cfg.profile != "A":
        raise ValueError("Only Profile A is supported in v1")

    requested_backend, effective_backend = _resolve_backend(cfg)
    effective_mode = _resolve_capm_mode(cfg)
    _validate_builder_runtime(cfg, effective_backend=effective_backend, effective_mode=effective_mode)

    os.environ["CAPM_MODE"] = str(effective_mode)

    if cfg.deterministic:
        enable_determinism(0 if cfg.seed is None else int(cfg.seed))

    prof = profile or ProfileAConfig()
    _ = prof

    if cfg.enable_meta_rebirth:
        return run_population(world=world, steps=steps, config=cfg, backend=effective_backend, capm_mode=effective_mode)

    core = CudaCore() if effective_backend == "cuda" else CpuCore()

    # initialize selector from Profile A defaults
    selector = (
        CapitalSelectorBuilder()
        .with_K(0)
        .with_selector_policy(cfg.selector_policy)
        .with_lambda_risk(cfg.lambda_risk)
        .build()
    )

    trace = []
    history = []
    for t in range(int(steps)):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        if hasattr(selector, "ensure_channel_state"):
            selector.ensure_channel_state(len(r_vec))
        elif selector.w is None or len(selector.w) != len(r_vec):
            selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
            selector.K = len(r_vec)
        core.step(selector, r_vec, c_total, freeze=cfg.freeze)
        history.append(selector.state())
        trace.append("step")

    return {
        "history": history,
        "trace": trace,
        "runtime": {
            "requested_backend": str(requested_backend),
            "effective_backend": str(effective_backend),
            "CAPM_MODE": str(effective_mode),
            "seed": int(cfg.seed) if cfg.seed is not None else None,
            "deterministic": bool(cfg.deterministic),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
