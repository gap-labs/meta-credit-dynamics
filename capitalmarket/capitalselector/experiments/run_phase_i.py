from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.cpu_impl import CpuCore
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.ledger import ClaimLedger
from capitalmarket.capitalselector.selector_policy import validate_selector_policy
from capitalmarket.capitalselector.worlds import GovernanceWorld, RegimeSwitchBanditWorld

DEFAULT_SELECTORS = ("random", "myopic", "term_aware", "term_risk")
DEFAULT_WORLDS = ("baseline_world", "governance_world", "stress_world")
DEFAULT_SEEDS_PATH = Path(__file__).with_name("seeds_phase_i.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")

METRIC_DIRECTIONS: dict[str, bool] = {
    "terminal_wealth": True,
    "time_to_death": True,
    "rollover_failure_frequency": False,
    "transaction_cost_total": False,
}


@dataclass(frozen=True)
class RunMetrics:
    seed: int
    world: str
    selector: str
    terminal_wealth: float
    time_to_death: int
    rollover_failure_frequency: float
    transaction_cost_total: float
    transaction_cost_mean: float
    rollover_count: int
    fail_count: int
    settlement_event_count: int
    dead: bool


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


def _build_world(world_name: str, seed: int):
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
        regime = {
            "alpha": 0.70,
            "manipulability": 0.45,
            "punishment": 0.35,
            "volatility": 0.30,
        }
        return GovernanceWorld(K_channels=5, regime=regime, seed=int(seed))

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


def _step_world(world: Any, t: int) -> tuple[np.ndarray, float, bool]:
    if isinstance(world, GovernanceWorld):
        events = world.step(int(t), [0])
        payload = events.get(0, {})
        r_vec = np.asarray(payload.get("r_vec", []), dtype=float)
        c_total = float(payload.get("c_total", 0.0))
        freeze = bool(payload.get("freeze", False))
        return r_vec, c_total, freeze

    out = world.step(int(t))
    r_vec, c_total = validate_world_output(out)
    return r_vec, c_total, False


def _build_selector(selector_name: str, seed: int, lambda_risk: float):
    if selector_name == "random":
        selector = (
            CapitalSelectorBuilder()
            .with_K(0)
            .with_selector_policy("myopic")
            .with_lambda_risk(lambda_risk)
            .build()
        )
        rng = np.random.default_rng(int(seed) + 91_731)

        def _random_reweight(weights: np.ndarray, _advantage: np.ndarray) -> np.ndarray:
            if weights is None or len(weights) == 0:
                return weights
            return rng.dirichlet(np.ones(len(weights), dtype=float))

        selector.reweight_fn = _random_reweight
        return selector

    policy = validate_selector_policy(selector_name)
    return (
        CapitalSelectorBuilder()
        .with_K(0)
        .with_selector_policy(policy)
        .with_lambda_risk(lambda_risk)
        .build()
    )


def _attach_initial_rollover_claim(selector: Any, seed: int) -> None:
    """Attach deterministic claim state so rollover/failure metrics are observable."""
    selector.process_id = int(getattr(selector, "process_id", 0))
    selector.generation_id = int(getattr(selector, "generation_id", 0))
    selector.claim_ledger = ClaimLedger(max_claims_per_process=10_000)
    selector.offers = []

    nominal = 2.0 + 0.05 * float(int(seed) % 5)
    selector.claim_ledger.create_claim(
        process_id=int(selector.process_id),
        generation_id=int(selector.generation_id),
        creditor_id="phase_i_creditor",
        debtor_id=str(selector.process_id),
        nominal=float(nominal),
        maturity_tau=0,
        claim_type="repayment",
        source_offer_id="phase_i_seed_claim",
        drawn_principal=float(nominal),
    )


def run_single(
    *,
    seed: int,
    world_name: str,
    selector_name: str,
    steps: int,
    lambda_risk: float,
) -> RunMetrics:
    world = _build_world(world_name, int(seed))
    selector = _build_selector(selector_name, int(seed), float(lambda_risk))
    _attach_initial_rollover_claim(selector, int(seed))
    core = CpuCore()

    transaction_cost_total = 0.0
    rollover_count = 0
    fail_count = 0
    time_to_death: int | None = None
    dead = False

    for t in range(int(steps)):
        if dead:
            break

        r_vec, c_total, freeze = _step_world(world, int(t))
        selector.ensure_channel_state(len(r_vec))

        core.step(selector, r_vec, c_total, freeze=freeze)

        transaction_cost_total += float(c_total)
        events = list(getattr(selector, "_last_phase_i_events", []) or [])
        rollover_count += sum(1 for event in events if str(getattr(event, "category", "")) == "ROLLOVER")
        fail_count += sum(1 for event in events if str(getattr(event, "category", "")) == "FAIL")

        is_dead = bool(getattr(selector, "_last_settlement_failed", False)) or float(selector.wealth) < 0.0
        if is_dead:
            dead = True
            if time_to_death is None:
                time_to_death = int(t) + 1
            break

    if time_to_death is None:
        time_to_death = int(steps)

    settlement_event_count = int(rollover_count + fail_count)
    rollover_failure_frequency = float(fail_count / max(1, settlement_event_count))

    return RunMetrics(
        seed=int(seed),
        world=str(world_name),
        selector=str(selector_name),
        terminal_wealth=float(selector.wealth),
        time_to_death=int(time_to_death),
        rollover_failure_frequency=rollover_failure_frequency,
        transaction_cost_total=float(transaction_cost_total),
        transaction_cost_mean=float(transaction_cost_total) / float(max(1, int(steps))),
        rollover_count=int(rollover_count),
        fail_count=int(fail_count),
        settlement_event_count=settlement_event_count,
        dead=bool(dead),
    )


def _paired_bootstrap_ci(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    higher_is_better: bool,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape:
        raise ValueError("Paired bootstrap requires equal-length baseline/candidate vectors")
    if baseline.size == 0:
        raise ValueError("Paired bootstrap requires at least one paired sample")

    diffs = candidate - baseline if bool(higher_is_better) else baseline - candidate
    n = int(diffs.shape[0])

    draw_idx = rng.integers(0, n, size=(int(bootstrap_samples), n), endpoint=False)
    boot_means = diffs[draw_idx].mean(axis=1)

    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    improvement_mean = float(diffs.mean())

    return {
        "improvement_mean": improvement_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_one_sided": float(np.mean(boot_means <= 0.0)),
        "supports_improvement": bool(float(ci_low) > 0.0),
    }


def _vector_for(
    rows: Sequence[RunMetrics],
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


def compute_bootstrap_results(
    *,
    rows: Sequence[RunMetrics],
    seeds: Sequence[int],
    worlds: Sequence[str],
    selectors: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    comparisons = [
        ("myopic", "random"),
        ("myopic", "term_aware"),
        ("myopic", "term_risk"),
        ("term_aware", "term_risk"),
    ]
    available_selectors = set(selectors)
    rng = np.random.default_rng(int(bootstrap_seed))

    results: list[dict[str, Any]] = []
    for world in worlds:
        for baseline_selector, candidate_selector in comparisons:
            if baseline_selector not in available_selectors or candidate_selector not in available_selectors:
                continue

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
            paired_seeds = [
                int(seed)
                for seed in seeds
                if int(seed) in baseline_seed_set and int(seed) in candidate_seed_set
            ]
            if not paired_seeds:
                continue

            for metric, higher_is_better in METRIC_DIRECTIONS.items():
                baseline = _vector_for(
                    rows,
                    world=world,
                    selector=baseline_selector,
                    metric=metric,
                    paired_seeds=paired_seeds,
                )
                candidate = _vector_for(
                    rows,
                    world=world,
                    selector=candidate_selector,
                    metric=metric,
                    paired_seeds=paired_seeds,
                )
                stats = _paired_bootstrap_ci(
                    baseline,
                    candidate,
                    higher_is_better=bool(higher_is_better),
                    bootstrap_samples=int(bootstrap_samples),
                    rng=rng,
                )
                results.append(
                    {
                        "world": str(world),
                        "metric": str(metric),
                        "higher_is_better": bool(higher_is_better),
                        "baseline_selector": str(baseline_selector),
                        "candidate_selector": str(candidate_selector),
                        "paired_seed_count": int(len(paired_seeds)),
                        **stats,
                    }
                )

    return results


def _aggregate_summary(rows: Sequence[RunMetrics], worlds: Sequence[str], selectors: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for world in worlds:
        world_summary: dict[str, Any] = {}
        for selector in selectors:
            subset = [row for row in rows if row.world == world and row.selector == selector]
            if not subset:
                continue
            world_summary[selector] = {
                "run_count": int(len(subset)),
                "terminal_wealth_mean": float(np.mean([row.terminal_wealth for row in subset])),
                "terminal_wealth_std": float(np.std([row.terminal_wealth for row in subset], ddof=0)),
                "time_to_death_mean": float(np.mean([row.time_to_death for row in subset])),
                "rollover_failure_frequency_mean": float(np.mean([row.rollover_failure_frequency for row in subset])),
                "transaction_cost_total_mean": float(np.mean([row.transaction_cost_total for row in subset])),
                "dead_rate": float(np.mean([1.0 if row.dead else 0.0 for row in subset])),
            }
        summary[world] = world_summary
    return summary


def compute_policy_invariance(
    *,
    rows: Sequence[RunMetrics],
    seeds: Sequence[int],
    worlds: Sequence[str],
    selectors: Sequence[str],
) -> dict[str, Any]:
    metrics = ("terminal_wealth", "time_to_death", "rollover_failure_frequency")
    by_world_metric: list[dict[str, Any]] = []

    global_max_policy_deviation = 0.0
    global_policy_outcome_variance = 0.0

    for world in worlds:
        seed_sets: list[set[int]] = []
        available_selectors: list[str] = []
        for selector in selectors:
            selector_seed_set = {
                int(row.seed)
                for row in rows
                if row.world == world and row.selector == selector
            }
            if selector_seed_set:
                seed_sets.append(selector_seed_set)
                available_selectors.append(selector)

        if len(seed_sets) < 2:
            continue

        common_seed_set = set.intersection(*seed_sets)
        common_seeds = [int(seed) for seed in seeds if int(seed) in common_seed_set]
        if not common_seeds:
            continue

        for metric in metrics:
            aligned = np.stack(
                [
                    _vector_for(
                        rows,
                        world=world,
                        selector=selector,
                        metric=metric,
                        paired_seeds=common_seeds,
                    )
                    for selector in available_selectors
                ],
                axis=0,
            )

            per_seed_deviation = np.max(aligned, axis=0) - np.min(aligned, axis=0)
            max_policy_deviation = float(np.max(np.abs(per_seed_deviation)))
            # Phase-I invariance looks at policy-induced spread for the same seed.
            # We therefore compute selector-variance per seed and aggregate it.
            per_seed_policy_variance = np.var(aligned, axis=0, ddof=0)
            policy_outcome_variance = float(np.max(per_seed_policy_variance))

            global_max_policy_deviation = max(global_max_policy_deviation, max_policy_deviation)
            global_policy_outcome_variance = max(global_policy_outcome_variance, policy_outcome_variance)

            by_world_metric.append(
                {
                    "world": str(world),
                    "metric": str(metric),
                    "paired_seed_count": int(len(common_seeds)),
                    "selector_count": int(len(available_selectors)),
                    "selectors": list(available_selectors),
                    "max_policy_deviation": max_policy_deviation,
                    "policy_outcome_variance": policy_outcome_variance,
                }
            )

    return {
        "expected_phase_i": "~0",
        "max_policy_deviation": float(global_max_policy_deviation),
        "policy_outcome_variance": float(global_policy_outcome_variance),
        "by_world_metric": by_world_metric,
    }


def _write_csv(rows: Sequence[RunMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "world",
        "selector",
        "terminal_wealth",
        "time_to_death",
        "rollover_failure_frequency",
        "transaction_cost_total",
        "transaction_cost_mean",
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


def run_phase_i_evaluation(
    *,
    seeds: Sequence[int],
    selectors: Sequence[str],
    worlds: Sequence[str],
    steps: int,
    lambda_risk: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if int(steps) <= 0:
        raise ValueError("steps must be > 0")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be > 0")

    unknown_selectors = [
        selector
        for selector in selectors
        if selector != "random" and selector not in {"myopic", "term_aware", "term_risk"}
    ]
    if unknown_selectors:
        raise ValueError(f"Unknown selectors: {unknown_selectors}")

    unknown_worlds = [world for world in worlds if world not in set(DEFAULT_WORLDS)]
    if unknown_worlds:
        raise ValueError(f"Unknown worlds: {unknown_worlds}")

    rows: list[RunMetrics] = []
    for world_name in worlds:
        for selector_name in selectors:
            for seed in seeds:
                rows.append(
                    run_single(
                        seed=int(seed),
                        world_name=str(world_name),
                        selector_name=str(selector_name),
                        steps=int(steps),
                        lambda_risk=float(lambda_risk),
                    )
                )

    bootstrap_results = compute_bootstrap_results(
        rows=rows,
        seeds=seeds,
        worlds=worlds,
        selectors=selectors,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )
    invariance = compute_policy_invariance(
        rows=rows,
        seeds=seeds,
        worlds=worlds,
        selectors=selectors,
    )

    output_dir = Path(output_dir)
    csv_path = output_dir / "selector_world_results.csv"
    bootstrap_path = output_dir / "bootstrap_results.json"
    summary_path = output_dir / "summary_metrics.json"

    _write_csv(rows, csv_path)
    _write_json(bootstrap_path, bootstrap_results)
    _write_json(
        summary_path,
        {
            "config": {
                "seed_count": int(len(seeds)),
                "steps": int(steps),
                "selectors": list(selectors),
                "worlds": list(worlds),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(bootstrap_seed),
                "lambda_risk": float(lambda_risk),
            },
            "aggregate_by_world_selector": _aggregate_summary(rows, worlds, selectors),
            "invariance": invariance,
        },
    )

    return {
        "rows": rows,
        "bootstrap_results": bootstrap_results,
        "summary_metrics": {
            "aggregate_by_world_selector": _aggregate_summary(rows, worlds, selectors),
            "invariance": invariance,
        },
        "artifacts": {
            "selector_world_results_csv": str(csv_path),
            "bootstrap_results_json": str(bootstrap_path),
            "summary_metrics_json": str(summary_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-I selector evaluation protocol")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selectors", type=str, default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--worlds", type=str, default=",".join(DEFAULT_WORLDS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lambda-risk", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2025)
    args = parser.parse_args()

    seeds = load_seeds(Path(args.seeds_file))
    selectors = _parse_csv_list(args.selectors)
    worlds = _parse_csv_list(args.worlds)

    result = run_phase_i_evaluation(
        seeds=seeds,
        selectors=selectors,
        worlds=worlds,
        steps=int(args.steps),
        lambda_risk=float(args.lambda_risk),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        output_dir=Path(args.output_dir),
    )

    print(json.dumps({"artifacts": result["artifacts"], "invariance": result["summary_metrics"]["invariance"]}, indent=2))


if __name__ == "__main__":
    main()
