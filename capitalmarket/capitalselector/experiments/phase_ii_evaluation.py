from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from capitalmarket.capitalselector.interfaces import make_world_step_result
from capitalmarket.capitalselector.worlds import GovernanceWorld, RegimeSwitchBanditWorld

from .run_phase_ii import PhaseIIConfig, run_phase_ii_episode

DEFAULT_SELECTOR_TO_POLICY: dict[str, str] = {
    "Selector-3": "myopic",
    "Selector-4": "term_aware",
    "Selector-5": "term_risk",
}
DEFAULT_SELECTORS: tuple[str, ...] = tuple(DEFAULT_SELECTOR_TO_POLICY.keys())
DEFAULT_WORLDS: tuple[str, ...] = ("baseline_world", "governance_world", "stress_world")
DEFAULT_SEEDS_PATH = Path(__file__).with_name("seeds_phase_i.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results_phase_ii")


@dataclass(frozen=True)
class PhaseIIRunMetrics:
    seed: int
    world: str
    selector: str
    terminal_wealth: float
    time_to_death: int
    rollover_failure_frequency: float
    rollover_count: int
    fail_count: int
    settlement_event_count: int
    dead: bool


class _GovernanceWorldPhaseIIAdapter:
    """Adapter to expose GovernanceWorld through the Phase-II WorldStepResult contract."""

    def __init__(self, *, channels: int, seed: int) -> None:
        regime = {
            "alpha": 0.70,
            "manipulability": 0.45,
            "punishment": 0.35,
            "volatility": 0.30,
        }
        self._inner = GovernanceWorld(K_channels=int(channels), regime=regime, seed=int(seed))

    def step(self, t: int, action):
        events = self._inner.step(int(t), [0])
        payload = events.get(0, {})
        r_vec = np.asarray(payload.get("r_vec", np.zeros(len(action.weights), dtype=float)), dtype=float)
        c_total = float(payload.get("c_total", 0.0))
        freeze = bool(payload.get("freeze", False))
        return make_world_step_result(r_vec=r_vec, c_total=c_total, action=action, freeze=freeze)


def _parse_csv_list(text: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not items:
        raise ValueError("Expected a non-empty comma-separated list")
    return items


def load_seeds(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("seeds", [])
    if not isinstance(payload, list):
        raise ValueError(f"Seed file must contain a list or a dict with 'seeds': {path}")

    seeds: list[int] = []
    for item in payload:
        value = int(item)
        if value < 0:
            raise ValueError("Seeds must be non-negative integers")
        seeds.append(value)

    if not seeds:
        raise ValueError(f"No seeds found in: {path}")
    return seeds


def _normalize_seed_list(seeds: Sequence[int]) -> list[int]:
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        raise ValueError("seeds must be non-empty")
    if any(seed < 0 for seed in normalized):
        raise ValueError("seeds must be non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique for paired evaluation")
    return normalized


def _build_world(*, world_name: str, seed: int, channels: int):
    if world_name == "baseline_world":
        return RegimeSwitchBanditWorld(
            p=0.05,
            sigma=0.01,
            seed=int(seed),
            c_high=0.005,
            q=0.02,
            c_spike=0.02,
        )

    if world_name == "governance_world":
        return _GovernanceWorldPhaseIIAdapter(channels=int(channels), seed=int(seed))

    if world_name == "stress_world":
        return RegimeSwitchBanditWorld(
            p=0.20,
            sigma=0.03,
            seed=int(seed),
            c_high=0.04,
            q=0.15,
            c_spike=0.10,
            shock_size=0.10,
            shock_times={75, 150, 225, 300},
        )

    raise ValueError(f"Unknown world '{world_name}'. Allowed: {', '.join(DEFAULT_WORLDS)}")


def run_phase_ii_single(
    *,
    seed: int,
    world_name: str,
    selector: str,
    steps: int,
    channels: int,
    backend: str,
    base_config: PhaseIIConfig | None,
) -> PhaseIIRunMetrics:
    if selector not in DEFAULT_SELECTOR_TO_POLICY:
        raise ValueError(f"Unknown selector '{selector}'. Allowed: {list(DEFAULT_SELECTOR_TO_POLICY.keys())}")

    policy = DEFAULT_SELECTOR_TO_POLICY[str(selector)]
    cfg = base_config or PhaseIIConfig()
    selector_cfg = replace(cfg, selector_policy=str(policy))

    if str(selector) == "Selector-5":
        # Strategic credit profile: higher deployment with stronger due-curve and
        # headroom coupling, but lower stress drag than the default profile.
        selector_cfg = replace(
            selector_cfg,
            gross_exposure=1.40,
            leverage_limit=max(float(selector_cfg.leverage_limit), 1.40),
            coupling_alpha=0.02,
            coupling_beta=0.70,
            coupling_gamma=1.30,
            coupling_eta=0.08,
        )

    world = _build_world(world_name=str(world_name), seed=int(seed), channels=int(channels))
    out = run_phase_ii_episode(
        world=world,
        steps=int(steps),
        channels=int(channels),
        seed=int(seed),
        config=selector_cfg,
        backend=str(backend),
    )

    eval_metrics = dict(out["evaluation_metrics"])
    exact = dict(out["exact_metrics"])
    rollover_count = int(exact["event_counts"].get("ROLLOVER", 0))
    fail_count = int(exact["event_counts"].get("FAIL", 0))
    settlement_event_count = int(rollover_count + fail_count)

    return PhaseIIRunMetrics(
        seed=int(seed),
        world=str(world_name),
        selector=str(selector),
        terminal_wealth=float(eval_metrics["terminal_wealth"]),
        time_to_death=int(eval_metrics["time_to_death"]),
        rollover_failure_frequency=float(eval_metrics["rollover_failure_frequency"]),
        rollover_count=rollover_count,
        fail_count=fail_count,
        settlement_event_count=settlement_event_count,
        dead=bool(exact["terminal_dead"]),
    )


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if deltas.ndim != 1:
        raise ValueError("deltas must be a 1D array")
    if deltas.size == 0:
        raise ValueError("paired bootstrap requires at least one paired sample")

    n = int(deltas.shape[0])
    draw_idx = rng.integers(0, n, size=(int(bootstrap_samples), n), endpoint=False)
    boot_means = deltas[draw_idx].mean(axis=1)

    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _vector_for(
    rows: Sequence[PhaseIIRunMetrics],
    *,
    world: str,
    selector: str,
    metric: str,
    paired_seeds: Sequence[int],
) -> np.ndarray:
    values_by_seed = {
        int(row.seed): float(getattr(row, metric))
        for row in rows
        if row.world == world and row.selector == selector
    }
    return np.asarray([values_by_seed[int(seed)] for seed in paired_seeds], dtype=float)


def _paired_seed_list(
    *,
    rows: Sequence[PhaseIIRunMetrics],
    world: str,
    baseline_selector: str,
    candidate_selector: str,
    seeds: Sequence[int],
) -> list[int]:
    baseline_seed_set = {
        int(row.seed)
        for row in rows
        if row.world == world and row.selector == baseline_selector
    }
    candidate_seed_set = {
        int(row.seed)
        for row in rows
        if row.world == world and row.selector == candidate_selector
    }
    paired = baseline_seed_set.intersection(candidate_seed_set)
    return [int(seed) for seed in seeds if int(seed) in paired]


def compute_phase_ii_hypotheses(
    *,
    rows: Sequence[PhaseIIRunMetrics],
    seeds: Sequence[int],
    worlds: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(bootstrap_seed))
    results: list[dict[str, Any]] = []

    for world in worlds:
        h1_seeds = _paired_seed_list(
            rows=rows,
            world=str(world),
            baseline_selector="Selector-3",
            candidate_selector="Selector-4",
            seeds=seeds,
        )
        if h1_seeds:
            s3 = _vector_for(
                rows,
                world=str(world),
                selector="Selector-3",
                metric="time_to_death",
                paired_seeds=h1_seeds,
            )
            s4 = _vector_for(
                rows,
                world=str(world),
                selector="Selector-4",
                metric="time_to_death",
                paired_seeds=h1_seeds,
            )
            h1_stats = _paired_bootstrap_ci(s4 - s3, bootstrap_samples=int(bootstrap_samples), rng=rng)
            results.append(
                {
                    "hypothesis": "H1",
                    "description": "Selector-4 robustness",
                    "world": str(world),
                    "metric": "time_to_death",
                    "delta_definition": "time_to_death(Selector-4) - time_to_death(Selector-3)",
                    "paired_seed_count": int(len(h1_seeds)),
                    "ci_low": float(h1_stats["ci_low"]),
                    "ci_high": float(h1_stats["ci_high"]),
                    "delta_mean": float(h1_stats["delta_mean"]),
                    "decision_rule": "CI_lower(Delta) > 0",
                    "supported": bool(float(h1_stats["ci_low"]) > 0.0),
                }
            )

        h2_seeds = _paired_seed_list(
            rows=rows,
            world=str(world),
            baseline_selector="Selector-4",
            candidate_selector="Selector-5",
            seeds=seeds,
        )
        if h2_seeds:
            s4 = _vector_for(
                rows,
                world=str(world),
                selector="Selector-4",
                metric="rollover_failure_frequency",
                paired_seeds=h2_seeds,
            )
            s5 = _vector_for(
                rows,
                world=str(world),
                selector="Selector-5",
                metric="rollover_failure_frequency",
                paired_seeds=h2_seeds,
            )
            h2_stats = _paired_bootstrap_ci(s5 - s4, bootstrap_samples=int(bootstrap_samples), rng=rng)
            results.append(
                {
                    "hypothesis": "H2",
                    "description": "Selector-5 credit stability",
                    "world": str(world),
                    "metric": "rollover_failure_frequency",
                    "delta_definition": "rollover_failure_frequency(Selector-5) - rollover_failure_frequency(Selector-4)",
                    "paired_seed_count": int(len(h2_seeds)),
                    "ci_low": float(h2_stats["ci_low"]),
                    "ci_high": float(h2_stats["ci_high"]),
                    "delta_mean": float(h2_stats["delta_mean"]),
                    "decision_rule": "CI_upper(Delta) < 0",
                    "supported": bool(float(h2_stats["ci_high"]) < 0.0),
                }
            )

    return results


def _aggregate_summary(
    rows: Sequence[PhaseIIRunMetrics],
    *,
    worlds: Sequence[str],
    selectors: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for world in worlds:
        world_summary: dict[str, Any] = {}
        for selector in selectors:
            subset = [row for row in rows if row.world == world and row.selector == selector]
            if not subset:
                continue
            world_summary[str(selector)] = {
                "run_count": int(len(subset)),
                "terminal_wealth_mean": float(np.mean([row.terminal_wealth for row in subset])),
                "time_to_death_mean": float(np.mean([row.time_to_death for row in subset])),
                "rollover_failure_frequency_mean": float(
                    np.mean([row.rollover_failure_frequency for row in subset])
                ),
                "dead_rate": float(np.mean([1.0 if row.dead else 0.0 for row in subset])),
            }
        summary[str(world)] = world_summary
    return summary


def _write_csv(rows: Sequence[PhaseIIRunMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "world",
        "selector",
        "terminal_wealth",
        "time_to_death",
        "rollover_failure_frequency",
        "rollover_count",
        "fail_count",
        "settlement_event_count",
        "dead",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_phase_ii_evaluation(
    *,
    seeds: Sequence[int] | None = None,
    selectors: Sequence[str] = DEFAULT_SELECTORS,
    worlds: Sequence[str] = DEFAULT_WORLDS,
    steps: int = 400,
    channels: int = 5,
    backend: str = "cpu",
    base_config: PhaseIIConfig | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 2026,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds_file: Path = DEFAULT_SEEDS_PATH,
    enforce_min_runs: bool = True,
) -> dict[str, Any]:
    if int(steps) <= 0:
        raise ValueError("steps must be > 0")
    if int(channels) <= 0:
        raise ValueError("channels must be > 0")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be > 0")

    selectors_norm = [str(selector) for selector in selectors]
    unknown_selectors = [selector for selector in selectors_norm if selector not in DEFAULT_SELECTOR_TO_POLICY]
    if unknown_selectors:
        raise ValueError(f"Unknown selectors: {unknown_selectors}")

    worlds_norm = [str(world) for world in worlds]
    unknown_worlds = [world for world in worlds_norm if world not in set(DEFAULT_WORLDS)]
    if unknown_worlds:
        raise ValueError(f"Unknown worlds: {unknown_worlds}")

    seeds_list = _normalize_seed_list(load_seeds(Path(seeds_file)) if seeds is None else seeds)
    if bool(enforce_min_runs) and len(seeds_list) < 100:
        raise ValueError("Phase-II evaluation requires at least 100 seeds per selector/world")

    rows: list[PhaseIIRunMetrics] = []
    for world_name in worlds_norm:
        for selector in selectors_norm:
            for seed in seeds_list:
                rows.append(
                    run_phase_ii_single(
                        seed=int(seed),
                        world_name=str(world_name),
                        selector=str(selector),
                        steps=int(steps),
                        channels=int(channels),
                        backend=str(backend),
                        base_config=base_config,
                    )
                )

    hypotheses = compute_phase_ii_hypotheses(
        rows=rows,
        seeds=seeds_list,
        worlds=worlds_norm,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )

    output_dir = Path(output_dir)
    csv_path = output_dir / "phase_ii_selector_world_results.csv"
    hypothesis_path = output_dir / "phase_ii_hypotheses.json"
    summary_path = output_dir / "phase_ii_summary_metrics.json"

    _write_csv(rows, csv_path)
    _write_json(hypothesis_path, hypotheses)
    cfg = base_config or PhaseIIConfig()

    _write_json(
        summary_path,
        {
            "config": {
                "seed_count": int(len(seeds_list)),
                "steps": int(steps),
                "channels": int(channels),
                "selectors": selectors_norm,
                "worlds": worlds_norm,
                "h_near_idx": int(cfg.h_near_idx),
                "coupling_alpha": float(cfg.coupling_alpha),
                "coupling_beta": float(cfg.coupling_beta),
                "coupling_gamma": float(cfg.coupling_gamma),
                "coupling_eta": float(cfg.coupling_eta),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
                "backend": str(backend),
                "confidence_interval": 0.95,
                "bootstrap_mode": "paired",
                "enforce_min_runs": bool(enforce_min_runs),
            },
            "aggregate_by_world_selector": _aggregate_summary(
                rows,
                worlds=worlds_norm,
                selectors=selectors_norm,
            ),
            "hypothesis_results": hypotheses,
        },
    )

    return {
        "rows": rows,
        "hypotheses": hypotheses,
        "summary_metrics": {
            "aggregate_by_world_selector": _aggregate_summary(
                rows,
                worlds=worlds_norm,
                selectors=selectors_norm,
            ),
            "hypothesis_results": hypotheses,
        },
        "artifacts": {
            "selector_world_results_csv": str(csv_path),
            "hypotheses_json": str(hypothesis_path),
            "summary_metrics_json": str(summary_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-II paired-bootstrap evaluation protocol")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selectors", type=str, default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--worlds", type=str, default=",".join(DEFAULT_WORLDS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--backend", type=str, default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--h-near-idx", type=int, default=0)
    parser.add_argument("--gross-exposure", type=float, default=1.0)
    parser.add_argument("--leverage-limit", type=float, default=1.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument(
        "--allow-small-seed-matrix",
        action="store_true",
        help="Allow fewer than 100 seeds for smoke runs.",
    )
    args = parser.parse_args()

    seeds = load_seeds(Path(args.seeds_file))
    selectors = _parse_csv_list(args.selectors)
    worlds = _parse_csv_list(args.worlds)

    base_config = PhaseIIConfig(
        selector_policy="myopic",
        gross_exposure=float(args.gross_exposure),
        leverage_limit=float(args.leverage_limit),
        allow_short=bool(args.allow_short),
        h_near_idx=int(args.h_near_idx),
    )

    result = run_phase_ii_evaluation(
        seeds=seeds,
        selectors=selectors,
        worlds=worlds,
        steps=int(args.steps),
        channels=int(args.channels),
        backend=str(args.backend),
        base_config=base_config,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        output_dir=Path(args.output_dir),
        enforce_min_runs=not bool(args.allow_small_seed_matrix),
    )

    print(json.dumps({"artifacts": result["artifacts"], "hypotheses": result["hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()
