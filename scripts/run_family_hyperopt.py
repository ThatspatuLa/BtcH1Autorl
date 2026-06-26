#!/usr/bin/env python3
"""run_family_hyperopt.py — Stage 10 family-budgeted hyperopt (3-phase).

Phase 1: Discovery sweep — 22 pure-axis families × 500 epochs each.
Phase 2: Deep optimisation — top-5 families × 5,000 epochs each.
Phase 3: Combo deep-dive — top-10 triples × (10 iterations × 500 epochs).

Usage:
    # Phase 1 — run one family
    python3 scripts/run_family_hyperopt.py --phase 1 --family pure_atr

    # Phase 2 — run one family (deep)
    python3 scripts/run_family_hyperopt.py --phase 2 --family pure_atr

    # Phase 3 — run one combo iteration
    python3 scripts/run_family_hyperopt.py --phase 3 --family combo_X_Y_Z --iteration 1

    # Phase 1 — run ALL families (orchestrated by cron)
    python3 scripts/run_family_hyperopt.py --phase 1 --all-families

Locked: reward weights, market, timeframe, direction, shorting, safety, TP method.
Only DCA accumulation params + simple fixed-TP pct mutate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolution.config import EvolutionConfig
from evolution.harness import EvolutionHarness
from evolution.hyperopt_config import (
    PHASE1_EPOCHS_PER_FAMILY,
    PHASE2_EPOCHS_PER_FAMILY,
    PHASE2_MUTATION_RATE,
    PHASE2_CROSSOVER_RATE,
    PHASE2_RANDOM_INJECTION,
    PHASE3_EPOCHS_PER_ITERATION,
    PHASE3_ITERATIONS_PER_COMBO,
    PHASE3_TOP_N_COMBOS,
    PHASE2_TOP_N_FAMILIES,
    SMART_ADJUST_TIGHTEN_FACTOR,
    FamilySpec,
    HyperoptRunConfig,
    build_family_specs,
    build_triple_combos,
)
from evolution.population_builder import (
    build_population,
    set_family_constraints,
    clear_family_constraints,
)
from genome.schema import GridMethod, AllocationMethod, ConfirmationIndicator


# ============================================================
# Output structure
# ============================================================

def phase1_output_dir(family_name: str) -> Path:
    return ROOT / "runs" / "hyperopt" / "discovery" / family_name


def phase2_output_dir(family_name: str) -> Path:
    return ROOT / "runs" / "hyperopt" / "deep" / family_name


def phase3_output_dir(combo_name: str, iteration: int) -> Path:
    return ROOT / "runs" / "hyperopt" / "combo" / combo_name / f"iteration_{iteration:02d}"


def phase_summary_path(phase: int) -> Path:
    return ROOT / "runs" / "hyperopt" / "phase_summaries" / f"phase{phase}_results.json"


# ============================================================
# Family DNA → population builder constraints
# ============================================================

def apply_family_constraints(family: FamilySpec) -> None:
    """Set family DNA constraints on the population builder module.

    This restricts ALL subsequent candidate generation to the family's DNA axis.
    Must be called before EvolutionHarness.run(). Call clear_family_constraints()
    after the run completes.
    """
    # Convert enum tuples to the format expected by population_builder
    forced_grid = tuple(family.forced_grid_methods) if family.forced_grid_methods else None
    forced_alloc = family.forced_allocation if family.forced_allocation else None
    forced_conf = tuple(family.forced_confirmations) if family.forced_confirmations else None

    set_family_constraints(
        forced_grid_methods=forced_grid,
        forced_allocation=forced_alloc,
        forced_confirmations=forced_conf,
        max_dca_layers_cap=family.max_dca_layers_cap,
    )


# ============================================================
# Phase runners
# ============================================================

def _run_evolution(config: EvolutionConfig, output_dir: Path) -> dict[str, Any]:
    """Run evolution and return result dict with deployment-passing totals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    harness = EvolutionHarness(config)
    summary_obj = harness.run()
    result = summary_obj.to_dict()

    # Sum deployment-passing across all generations
    history_path = output_dir / "generation_history.json"
    total_deploy_passing = 0
    total_passed = 0
    if history_path.exists():
        hist = json.loads(history_path.read_text())
        for gen in hist.get("generations", []):
            total_deploy_passing += gen.get("n_deployment_passing", 0)
            total_passed += gen.get("n_passed", 0)

    result["n_deployment_passing"] = total_deploy_passing
    result["n_passed"] = total_passed
    return result


def run_phase1_family(family: FamilySpec) -> dict[str, Any]:
    """Run Phase 1 discovery sweep for one family."""
    output_dir = phase1_output_dir(family.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete":
            return existing

    # Set family constraints BEFORE creating config
    apply_family_constraints(family)

    config = EvolutionConfig(
        candidates_per_gen=500,
        elite_count=20,
        random_injection=120,
        mutation_rate=0.30,
        crossover_rate=0.50,
        max_generations=PHASE1_EPOCHS_PER_FAMILY,
        stagnation_generations=5,
        all_rejected_generations=3,
        parallel_workers=8,
        base_seed=family.deterministic_seed,
        output_dir=str(output_dir),
        experiment_id=f"hyperopt_p1_{family.name}",
        island_mode=False,
        retirement_enabled=False,
        force_retire_after_gens=999,
        checkpoint_interval_minutes=20,
    )

    try:
        result = _run_evolution(config, output_dir)
    finally:
        clear_family_constraints()

    summary = {
        "phase": 1,
        "family": family.name,
        "group": family.group,
        "status": "complete",
        "max_generations": PHASE1_EPOCHS_PER_FAMILY,
        "best_fitness": result.get("best_fitness_ever", 0.0),
        "n_deployment_passing": result.get("n_deployment_passing", 0),
        "n_passed": result.get("n_passed", 0),
        "generations_completed": result.get("generations_completed", 0),
        "family_dna": {
            "forced_grid_methods": [g.value for g in family.forced_grid_methods] if family.forced_grid_methods else None,
            "forced_allocation": family.forced_allocation.value if family.forced_allocation else None,
            "forced_confirmations": [c.value for c in family.forced_confirmations] if family.forced_confirmations else None,
            "max_dca_layers_cap": family.max_dca_layers_cap,
        },
        "seed": family.deterministic_seed,
        "completed_at": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_phase2_family(family: FamilySpec, phase1_summary: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 2 deep optimisation for one family."""
    output_dir = phase2_output_dir(family.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete":
            return existing

    apply_family_constraints(family)

    config = EvolutionConfig(
        candidates_per_gen=500,
        elite_count=20,
        random_injection=PHASE2_RANDOM_INJECTION,
        mutation_rate=PHASE2_MUTATION_RATE,
        crossover_rate=PHASE2_CROSSOVER_RATE,
        max_generations=PHASE2_EPOCHS_PER_FAMILY,
        stagnation_generations=200,
        all_rejected_generations=3,
        parallel_workers=8,
        base_seed=family.deterministic_seed,
        output_dir=str(output_dir),
        experiment_id=f"hyperopt_p2_{family.name}",
        island_mode=False,
        retirement_enabled=False,
        force_retire_after_gens=999,
        checkpoint_interval_minutes=20,
    )

    try:
        result = _run_evolution(config, output_dir)
    finally:
        clear_family_constraints()

    summary = {
        "phase": 2,
        "family": family.name,
        "group": family.group,
        "status": "complete",
        "max_generations": PHASE2_EPOCHS_PER_FAMILY,
        "best_fitness": result.get("best_fitness_ever", 0.0),
        "n_deployment_passing": result.get("n_deployment_passing", 0),
        "n_passed": result.get("n_passed", 0),
        "generations_completed": result.get("generations_completed", 0),
        "phase1_fitness": phase1_summary.get("best_fitness", 0.0),
        "fitness_improvement": result.get("best_fitness_ever", 0.0) - phase1_summary.get("best_fitness", 0.0),
        "completed_at": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_phase3_combo_iteration(
    combo: dict[str, str],
    iteration: int,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Phase 3 combo iteration with smart adjustment."""
    combo_name = combo["name"]
    output_dir = phase3_output_dir(combo_name, iteration)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete":
            return existing

    families_in_combo = combo["families"]
    layer_split = combo.get("layer_split", {})

    # For combo runs, we DON'T set family constraints — the combo
    # inherits DNA from multiple families via layer_split.
    # The population builder generates full-range candidates,
    # but the combo mutator (TODO) will respect layer boundaries.

    config = EvolutionConfig(
        candidates_per_gen=500,
        elite_count=20,
        random_injection=PHASE2_RANDOM_INJECTION,
        mutation_rate=PHASE2_MUTATION_RATE,
        crossover_rate=PHASE2_CROSSOVER_RATE,
        max_generations=PHASE3_EPOCHS_PER_ITERATION,
        stagnation_generations=100,
        all_rejected_generations=3,
        parallel_workers=8,
        base_seed=int(hashlib.sha256(combo_name.encode()).hexdigest()[:8], 16) + iteration,
        output_dir=str(output_dir),
        experiment_id=f"hyperopt_p3_{combo_name}_iter{iteration:02d}",
        island_mode=False,
        retirement_enabled=False,
        force_retire_after_gens=999,
        checkpoint_interval_minutes=20,
    )

    result = _run_evolution(config, output_dir)

    summary = {
        "phase": 3,
        "combo": combo_name,
        "families": families_in_combo,
        "layer_split": layer_split,
        "iteration": iteration,
        "status": "complete",
        "max_generations": PHASE3_EPOCHS_PER_ITERATION,
        "best_fitness": result.get("best_fitness_ever", 0.0),
        "n_deployment_passing": result.get("n_deployment_passing", 0),
        "n_passed": result.get("n_passed", 0),
        "generations_completed": result.get("generations_completed", 0),
        "completed_at": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


# ============================================================
# Phase orchestration helpers
# ============================================================

def collect_phase1_results() -> list[dict[str, Any]]:
    """Collect all Phase 1 results and rank by best_fitness."""
    families = build_family_specs()
    results = []
    for family in families:
        summary_path = phase1_output_dir(family.name) / "run_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            if data.get("status") == "complete":
                results.append(data)
    results.sort(key=lambda r: r.get("best_fitness", 0.0), reverse=True)
    return results


def select_top5_families() -> list[str]:
    """Select top-5 families by Phase 1 fitness."""
    results = collect_phase1_results()
    summary_path = phase_summary_path(1)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "phase": 1,
        "ranking": results,
        "top5": [r["family"] for r in results[:PHASE2_TOP_N_FAMILIES]],
        "generated_at": time.time(),
    }, indent=2))
    return [r["family"] for r in results[:PHASE2_TOP_N_FAMILIES]]


def collect_phase2_results() -> list[dict[str, Any]]:
    """Collect all Phase 2 results."""
    top5 = select_top5_families()
    results = []
    for family_name in top5:
        summary_path = phase2_output_dir(family_name) / "run_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            if data.get("status") == "complete":
                results.append(data)
    results.sort(key=lambda r: r.get("best_fitness", 0.0), reverse=True)
    return results


def collect_phase3_results() -> list[dict[str, Any]]:
    """Collect all Phase 3 combo results."""
    summary_path = phase_summary_path(2)
    if not summary_path.exists():
        return []
    p2_data = json.loads(summary_path.read_text())
    top5 = p2_data.get("top5", [])

    combos = build_triple_combos(top5)
    results = []
    for combo in combos:
        combo_name = combo["name"]
        best_fitness = 0.0
        total_deploy_passing = 0
        for iteration in range(1, PHASE3_ITERATIONS_PER_COMBO + 1):
            iter_path = phase3_output_dir(combo_name, iteration) / "run_summary.json"
            if iter_path.exists():
                data = json.loads(iter_path.read_text())
                if data.get("status") == "complete":
                    best_fitness = max(best_fitness, data.get("best_fitness", 0.0))
                    total_deploy_passing += data.get("n_deployment_passing", 0)
        results.append({
            "combo": combo_name,
            "families": combo["families"],
            "best_fitness": best_fitness,
            "total_deployment_passing": total_deploy_passing,
        })
    results.sort(key=lambda r: r.get("best_fitness", 0.0), reverse=True)
    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 10 family-budgeted hyperopt")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--family", type=str, help="Family name to run")
    parser.add_argument("--all-families", action="store_true", help="Run all families in this phase")
    parser.add_argument("--iteration", type=int, default=1, help="Phase 3 iteration number (1-10)")
    parser.add_argument("--status", action="store_true", help="Show phase status and exit")
    parser.add_argument("--rank", action="store_true", help="Show ranking and exit")
    args = parser.parse_args()

    if args.status:
        if args.phase == 1:
            results = collect_phase1_results()
            families = build_family_specs()
            completed = len(results)
            total = len(families)
            print(f"Phase 1: {completed}/{total} families complete")
            for r in results:
                print(f"  {r['family']:30s}  fitness={r['best_fitness']:.6f}  deploy_pass={r['n_deployment_passing']}")
        elif args.phase == 2:
            top5 = select_top5_families()
            print(f"Phase 2: top-5 = {top5}")
            results = collect_phase2_results()
            for r in results:
                print(f"  {r['family']:30s}  fitness={r['best_fitness']:.6f}  improvement={r.get('fitness_improvement', 0):+.6f}")
        elif args.phase == 3:
            results = collect_phase3_results()
            print(f"Phase 3: {len(results)} combos evaluated")
            for r in results:
                print(f"  {r['combo']:50s}  fitness={r['best_fitness']:.6f}  total_deploy={r['total_deployment_passing']}")
        return

    if args.rank:
        if args.phase in (1, 2):
            results = collect_phase1_results() if args.phase == 1 else collect_phase2_results()
            for i, r in enumerate(results):
                print(f"{i+1:2d}. {r.get('family', r.get('combo', '?')):30s}  fitness={r['best_fitness']:.6f}")
        elif args.phase == 3:
            results = collect_phase3_results()
            for i, r in enumerate(results):
                print(f"{i+1:2d}. {r['combo']:50s}  fitness={r['best_fitness']:.6f}")
        return

    if args.phase == 1:
        families = {f.name: f for f in build_family_specs()}
        if args.all_families:
            results = []
            for family in families.values():
                print(f"[Phase 1] Running family: {family.name}")
                result = run_phase1_family(family)
                results.append(result)
                print(f"[Phase 1] {family.name}: fitness={result['best_fitness']:.6f}")
            select_top5_families()
            print(f"[Phase 1] Complete. {len(results)}/{len(families)} families done.")
        elif args.family:
            if args.family not in families:
                print(f"Unknown family: {args.family}")
                print(f"Available: {list(families.keys())}")
                sys.exit(1)
            result = run_phase1_family(families[args.family])
            print(json.dumps(result, indent=2))
        else:
            print("Specify --family NAME or --all-families")
            sys.exit(1)

    elif args.phase == 2:
        top5 = select_top5_families()
        families = {f.name: f for f in build_family_specs()}
        phase1_results = {r["family"]: r for r in collect_phase1_results()}

        if args.all_families:
            for family_name in top5:
                family = families.get(family_name)
                if family is None:
                    print(f"[Phase 2] Family {family_name} not in specs, skipping")
                    continue
                p1 = phase1_results.get(family_name, {})
                print(f"[Phase 2] Running deep: {family_name}")
                result = run_phase2_family(family, p1)
                print(f"[Phase 2] {family_name}: fitness={result['best_fitness']:.6f}  improvement={result.get('fitness_improvement', 0):+.6f}")
            print(f"[Phase 2] Complete.")
        elif args.family:
            if args.family not in top5:
                print(f"Warning: {args.family} not in top-5 ({top5})")
            family = families.get(args.family)
            if family is None:
                print(f"Unknown family: {args.family}")
                sys.exit(1)
            p1 = phase1_results.get(args.family, {})
            result = run_phase2_family(family, p1)
            print(json.dumps(result, indent=2))
        else:
            print("Specify --family NAME or --all-families")
            sys.exit(1)

    elif args.phase == 3:
        top5 = select_top5_families()
        combos = build_triple_combos(top5)
        combos = combos[:PHASE3_TOP_N_COMBOS]

        if args.all_families:
            for combo in combos:
                combo_name = combo["name"]
                print(f"[Phase 3] Running combo: {combo_name}")
                previous_result = None
                for iteration in range(1, PHASE3_ITERATIONS_PER_COMBO + 1):
                    result = run_phase3_combo_iteration(combo, iteration, previous_result)
                    previous_result = result
                    print(f"[Phase 3] {combo_name} iter{iteration:02d}: fitness={result['best_fitness']:.6f}")
                    if result.get("generations_completed", 0) < PHASE3_EPOCHS_PER_ITERATION * 0.5:
                        print(f"[Phase 3] {combo_name} converged early at iter{iteration}")
                        break
            print(f"[Phase 3] Complete.")
        elif args.family:
            matching = [c for c in combos if c["name"] == args.family]
            if not matching:
                print(f"Unknown combo: {args.family}")
                print(f"Available: {[c['name'] for c in combos]}")
                sys.exit(1)
            combo = matching[0]
            result = run_phase3_combo_iteration(combo, args.iteration)
            print(json.dumps(result, indent=2))
        else:
            print("Specify --family COMBO_NAME or --all-families")
            sys.exit(1)


if __name__ == "__main__":
    main()
