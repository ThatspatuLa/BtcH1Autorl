#!/usr/bin/env python3
"""run_continuous_evolution.py — Stage 10 DCA evolution with lock, resume, and Discord reporting.

Usage:
    python3 scripts/run_continuous_evolution.py [--output-dir runs/evo_continuous] [--wall-time 7200]

Features:
    - Lock file prevents overlapping runs
    - Timestamped output directories per run
    - Auto-resume from generation_history.json
    - Reports to Discord channel 1500437358934233219 on: start, new best deployment,
      completion, failure, and every 5 generations
    - PopulationBuilder 250/150/75/25 split
    - Only generates valid executable genomes (all grid methods wired in OrderManager)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from evolution.config import EvolutionConfig
from evolution.harness import EvolutionHarness, HarnessHooks
from evolution.population_builder import build_population


# ============================================================
# Lock file — prevents overlapping runs
# ============================================================

LOCK_FILE_PATH = ROOT / "runs" / "evolution.lock"


def acquire_lock() -> bool:
    """Try to acquire the evolution lock. Returns True if acquired."""
    LOCK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(LOCK_FILE_PATH, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return True
    except (IOError, OSError):
        return False


def release_lock() -> None:
    """Release the evolution lock."""
    try:
        if LOCK_FILE_PATH.exists():
            LOCK_FILE_PATH.unlink()
    except OSError:
        pass


# ============================================================
# Discord reporting
# ============================================================

DISCORD_CHANNEL_ID = "1500437358934233219"


def send_discord(message: str) -> None:
    """Send a message to the Discord channel via Hermes send_message tool.

    This writes to a file that the cron agent picks up, since we can't call
    Hermes tools directly from a Python subprocess.
    """
    queue_dir = ROOT / "runs" / "discord_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    msg_file = queue_dir / f"msg_{ts}.json"
    payload = {
        "channel": DISCORD_CHANNEL_ID,
        "message": message,
        "timestamp": time.time(),
    }
    msg_file.write_text(json.dumps(payload))


# ============================================================
# Approved queue — only run experiments in the approved list
# ============================================================

APPROVED_EXPERIMENTS: list[str] = [
    "stage10_continuous",
    "stage10_seeded",
    "stage10_explore",
]


def is_approved(experiment_id: str) -> bool:
    """Check if the experiment is in the approved queue."""
    return experiment_id in APPROVED_EXPERIMENTS


# ============================================================
# Main runner
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 10 Continuous DCA Evolution")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (default: runs/evo_continuous_<timestamp>)",
    )
    parser.add_argument(
        "--wall-time-seconds",
        type=int,
        default=7200,
        help="Wall-time cap per run in seconds (default 7200 = 2h)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=40,
        help="Max generations per run (default 40 for island mode)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=500,
        help="Candidates per generation (default 500)",
    )
    parser.add_argument(
        "--island-mode",
        action="store_true",
        help="Use 8-island sub-population model (Plan B, default False)",
    )
    parser.add_argument(
        "--n-islands",
        type=int,
        default=8,
        help="Number of islands when --island-mode is on (default 8)",
    )
    parser.add_argument(
        "--migration-every",
        type=int,
        default=5,
        help="Migrate top-K between islands every N generations (default 5)",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=20,
        help="Elite count (default 20)",
    )
    parser.add_argument(
        "--random-injection",
        type=int,
        default=220,
        help="Fresh random candidates per generation (default 220 — push harder from 180)",
    )
    parser.add_argument(
        "--retirement-enabled",
        action="store_true",
        help="Enable island retirement: archive islands whose top fitness crosses --retirement-threshold",
    )
    parser.add_argument(
        "--retirement-threshold",
        type=float,
        default=0.80,
        help="Per-island top fitness that triggers retirement (default 0.80)",
    )
    parser.add_argument(
        "--retirement-archive-dir",
        type=str,
        default="runs/retired_islands",
        help="Root directory for archived islands (default runs/retired_islands)",
    )
    parser.add_argument(
        "--checkpoint-interval-min",
        type=int,
        default=20,
        help="Write a checkpoint snapshot every N minutes (default 20). Set 0 to disable.",
    )
    parser.add_argument(
        "--force-retire-after-gens",
        type=int,
        default=8,
        help="Force-retire an island if its top fitness doesn't improve for this many gens "
             "(default 8). Skip if its top fitness is already >= --force-retire-min-fitness.",
    )
    parser.add_argument(
        "--force-retire-min-fitness",
        type=float,
        default=0.70,
        help="Don't force-retire islands whose top fitness is already at or above this "
             "(default 0.70). Lets near-passing islands keep trying.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.45,
        help="Per-param mutation rate, 0..1 (default 0.45 — 'push harder' from 0.30)",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.40,
        help="Fraction of children from crossover (default 0.40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel evaluation workers (default 8)",
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=5,
        help="Stagnation guard: stop after N gens with no improvement (default 5)",
    )
    parser.add_argument(
        "--all-rejected-generations",
        type=int,
        default=3,
        help="All-rejected guard: stop after N gens of 0 passing (default 3)",
    )
    parser.add_argument(
        "--experiment-id",
        default="stage10_continuous",
        help="Experiment ID (must be in approved queue)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "processed" / "btc_h1_5y.feather"),
        help="Path to BTC H1 feather data",
    )
    args = parser.parse_args()

    # Check approved queue
    if not is_approved(args.experiment_id):
        print(f"[evo] ERROR: experiment '{args.experiment_id}' not in approved queue: {APPROVED_EXPERIMENTS}")
        return 1

    # Acquire lock
    if not acquire_lock():
        print("[evo] ERROR: Another evolution run is already running (lock file exists). Exiting.")
        return 1

    try:
        return _run_evolution(args)
    finally:
        release_lock()


def resolve_output_dir(args_output_dir: str | None, ts: str) -> tuple[Path, bool]:
    """Resolve the actual output directory for a run.

    Behavior:
      - If --output-dir is None: create runs/evo_continuous_<ts>/ and return it.
      - If --output-dir is "runs" or ends with "/runs" (the cron convention):
          create runs/evo_continuous_<ts>/ inside it, update runs/latest symlink,
          and return that timestamped subdir.
      - If --output-dir is any other path: use it as-is.

    Returns:
      (output_dir, created_subdir_bool)
      created_subdir_bool=True means we created a timestamped subdir (so the
      caller knows the dir is fresh per cycle).
    """
    if args_output_dir is None:
        out = Path(f"runs/evo_continuous_{ts}")
        out.mkdir(parents=True, exist_ok=True)
        return out, True

    requested = Path(args_output_dir)
    # The cron convention: --output-dir runs (literal) or runs/ (trailing slash)
    is_cron_runs = (
        requested.name == "runs"
        and (requested.parent == Path(".") or str(requested.parent) == "")
    )

    if is_cron_runs:
        # Create a timestamped subdir so each cycle has its own output.
        out = requested / f"evo_continuous_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        # Update runs/latest -> this cycle (symlink for easy "what's running now").
        latest_link = requested / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(out.name)
        except OSError:
            # Symlinks can fail on some FS (e.g. CIF without privs); non-fatal.
            pass
        return out, True

    # Custom output dir (e.g. runs/evo_continuous_<ts>) — use as-is.
    requested.mkdir(parents=True, exist_ok=True)
    return requested, False


def _run_evolution(args: argparse.Namespace) -> int:
    """Run one evolution pass. Returns exit code."""

    # Timestamped output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir, _created_subdir = resolve_output_dir(args.output_dir, ts)
    print(f"[evo] Output dir: {output_dir}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        msg = f"[evo] ERROR: data file not found: {data_path}"
        print(msg)
        send_discord(f"❌ {msg}")
        return 1

    print(f"[evo] Loading data: {data_path}")
    df = pd.read_feather(data_path)
    print(f"[evo] Loaded {len(df):,} rows")

    # RNG
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)

    # Config
    config = EvolutionConfig(
        candidates_per_gen=args.candidates,
        elite_count=args.elite_count,
        random_injection=args.random_injection,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_generations=args.max_generations,
        wall_time_seconds=args.wall_time_seconds,
        stagnation_generations=args.stagnation_generations,
        all_rejected_generations=args.all_rejected_generations,
        parallel_workers=args.workers,
        base_seed=args.seed,
        output_dir=str(output_dir),
        experiment_id=args.experiment_id,
        leaderboard_top_n=20,
        island_mode=args.island_mode,
        n_islands=args.n_islands,
        migration_every_n_gens=args.migration_every,
        retirement_enabled=args.retirement_enabled,
        retirement_threshold=args.retirement_threshold,
        retirement_archive_dir=args.retirement_archive_dir,
        checkpoint_interval_minutes=args.checkpoint_interval_min,
        force_retire_after_gens=args.force_retire_after_gens,
        force_retire_min_fitness=args.force_retire_min_fitness,
    )
    print(
        f"[evo] GA config: cands={args.candidates} elites={args.elite_count} "
        f"random={args.random_injection} mut_rate={args.mutation_rate:.2f} "
        f"cx_rate={args.crossover_rate:.2f} workers={args.workers} "
        f"islands={args.n_islands if args.island_mode else 'off'}"
    )

    # Build seeded population for gen 0
    print(f"[evo] Building seeded population of {args.candidates}")
    if args.island_mode:
        from evolution.islands import get_island_specs
        from evolution.population_builder import build_island_population
        specs = get_island_specs()[:args.n_islands]
        seeded_pop = build_island_population(
            rng=rng,
            generation_index=0,
            island_specs=specs,
            gid_start=0,
            random_count=4,
        )
    else:
        seeded_pop = build_population(rng, generation_index=0)
    print(f"[evo] Seeded population built: {len(seeded_pop)} candidates")

    # Population split report
    pop_report = _analyze_population(seeded_pop)
    print(f"[evo] Population split: {json.dumps(pop_report, indent=2)}")

    # Discord: start notification
    start_msg = (
        f"🧬 **Stage 10 Evolution Started**\n"
        f"Experiment: `{args.experiment_id}`\n"
        f"Output: `{output_dir.name}`\n"
        f"Config: {args.candidates} cands/gen, max {args.max_generations} gens, "
        f"wall-time {args.wall_time_seconds}s\n"
        f"Seed: {seed}\n"
        f"Population: {pop_report['exploit']} exploit + {pop_report['explore']} explore + "
        f"{pop_report['hybrid']} hybrid + {pop_report['random']} random = {len(seeded_pop)}\n"
        f"Grid methods: {', '.join(pop_report['grid_methods_used'])}\n"
        f"Confirmations: {', '.join(pop_report['confirmations_used'])}"
    )
    send_discord(start_msg)
    print(f"[evo] Discord notification sent")

    # Track best deployment fitness for reporting
    best_deploy_fitness = 0.0
    best_deploy_genome_id = ""
    n_deploy_passing_total = 0

    # Hooks for Discord reporting
    def on_gen_end(record: Any) -> None:
        nonlocal best_deploy_fitness, best_deploy_genome_id, n_deploy_passing_total
        n_deploy_passing_total += record.n_deployment_passing

        # Check for new best deployment candidate
        if record.deployment_leaderboard:
            top = record.deployment_leaderboard[0]
            if top["deployment_fitness"] > best_deploy_fitness:
                best_deploy_fitness = top["deployment_fitness"]
                best_deploy_genome_id = top["genome_id"]
                send_discord(
                    f"🏆 **New Best Deployment Candidate** (Gen {record.generation_index})\n"
                    f"Fitness: {top['deployment_fitness']:.6f} | "
                    f"Consistency: {top['consistency_ratio']:.4f}"
                )

        # Per-generation summary (User directive 2026-06-23: "after every generation").
        # Was previously every-5-gens; switched to every-gen for visibility.
        # 2026-06-25: enriched with per-island top fitness + stagnation warnings.
        # Six asked: "what island has the top candidate, and all other islands' top fitness".

        # Build per-island table sorted by best fitness desc
        island_bias_lookup = {}
        try:
            from evolution.islands import get_island_specs
            for spec in get_island_specs()[:8]:
                island_bias_lookup[spec.island_id] = spec.name
        except Exception:
            pass

        per_island = record.per_island_best_fitness or {}
        if per_island:
            # Sort islands by best fitness desc
            ranked = sorted(per_island.items(), key=lambda kv: kv[1], reverse=True)
            medals = ["🥇", "🥈", "🥉"]
            island_lines = []
            for rank, (iid, fit) in enumerate(ranked):
                medal = medals[rank] if rank < 3 else "  "
                bias_name = island_bias_lookup.get(iid, f"island_{iid}")
                island_lines.append(f"{medal} I{iid} ({bias_name}): {fit:.4f}")
            island_block = "\n".join(island_lines)
        else:
            island_block = "_no per-island data this gen_"

        # Stagnation warnings (force-retire threshold from config — was hardcoded 8)
        # Six's fix 2026-06-25: cron now passes --force-retire-after-gens 15 (was 8).
        # The threshold must come from config so the warning matches the trigger.
        stagnation = record.per_island_stagnation_counter or {}
        # `config` is in the enclosing main() scope (closure variable)
        force_threshold = getattr(config, 'force_retire_after_gens', 8)
        warn_threshold = max(1, force_threshold - 10)  # warn ~10 gens before force-retire
        stagnation_warnings = []
        for iid, counter in sorted(stagnation.items()):
            island_fit = per_island.get(iid, 0.0)
            if counter >= force_threshold and island_fit < 0.70:
                bias_name = island_bias_lookup.get(iid, f"island_{iid}")
                stagnation_warnings.append(
                    f"⚠️ I{iid} ({bias_name}): {counter} gens stagnant @ {island_fit:.4f} → force-retire imminent"
                )
            elif counter >= warn_threshold:
                bias_name = island_bias_lookup.get(iid, f"island_{iid}")
                stagnation_warnings.append(
                    f"⚡ I{iid} ({bias_name}): {counter} gens stagnant"
                )

        stagnation_block = ""
        if stagnation_warnings:
            stagnation_block = "\n" + "\n".join(stagnation_warnings)

        # Retirement / force-retire events this gen
        retirement_block = ""
        if record.retired_islands:
            ret_lines = []
            for rec_dict in record.retired_islands:
                ret_lines.append(
                    f"🏝️ Archived I{rec_dict.get('island_id', '?')} "
                    f"({rec_dict.get('bias_name', '?')}): {rec_dict.get('reason', '?')}"
                )
            retirement_block = "\n" + "\n".join(ret_lines)

        # New all-time best detection (vs record.best_fitness this gen)
        new_best_marker = ""
        if record.best_fitness >= best_deploy_fitness and record.best_fitness > 0:
            # Treat global best fitness as the marker; only flag if it's a clear jump
            pass  # we already notify on deployment_pass leaderboard above

        send_discord(
            f"📊 **Gen {record.generation_index} Summary** — Cap 10\n"
            f"Passed: {record.n_passed}/{record.n_candidates} | "
            f"Deploy-passing: {record.n_deployment_passing} | "
            f"Best: {record.best_fitness:.6f} | Median: {record.median_fitness:.6f}\n"
            f"Total deploy-passing so far: {n_deploy_passing_total}\n\n"
            f"🏝️ **Per-Island Top Fitness:**\n"
            f"{island_block}"
            f"{stagnation_block}"
            f"{retirement_block}"
        )

    hooks = HarnessHooks(on_generation_end=on_gen_end)

    # Run evolution
    print(f"[evo] Starting evolution...")
    harness = EvolutionHarness(
        config=config,
        df=df,
        hooks=hooks,
        seeded_population=seeded_pop,
        rng=rng,
    )

    t0 = time.time()
    summary = harness.run(resume=True)
    elapsed = time.time() - t0

    # Final report
    final_msg = (
        f"✅ **Stage 10 Evolution Complete**\n"
        f"Experiment: `{args.experiment_id}`\n"
        f"Status: {summary.termination_reason}\n"
        f"Generations: {summary.generations_completed}/{summary.generations_planned}\n"
        f"Total candidates: {summary.total_candidates_evaluated}\n"
        f"Best fitness: {summary.best_fitness_ever:.6f}\n"
        f"Best genome: `{summary.best_genome_id_ever}`\n"
        f"Total deploy-passing: {n_deploy_passing_total}\n"
        f"Runtime: {elapsed:.1f}s\n"
        f"Output: `{output_dir.name}`"
    )
    send_discord(final_msg)

    # Write final status
    status = {
        "termination_reason": summary.termination_reason,
        "generations_completed": summary.generations_completed,
        "generations_planned": summary.generations_planned,
        "total_candidates_evaluated": summary.total_candidates_evaluated,
        "best_fitness_ever": summary.best_fitness_ever,
        "best_genome_id_ever": summary.best_genome_id_ever,
        "best_candidate_id_ever": summary.best_candidate_id_ever,
        "n_deployment_passing_total": n_deploy_passing_total,
        "total_runtime_seconds": elapsed,
        "output_dir": str(output_dir),
        "seed": seed,
        "population_split": pop_report,
    }
    (output_dir / "final_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(f"[evo] Wrote final_status.json")
    print(f"[evo] Done. Status: {summary.termination_reason}")

    # Post-cycle Obsidian sync (deterministic, idempotent).
    # Skipped silently if the post_cycle_obsidian_update.py script isn't available.
    try:
        import subprocess
        post_result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "post_cycle_obsidian_update.py"),
             "--cycle-dir", str(output_dir)],
            capture_output=True, text=True, timeout=30,
        )
        if post_result.returncode == 0:
            print(f"[evo] Obsidian post-cycle sync: OK")
        else:
            print(f"[evo] Obsidian post-cycle sync: SKIPPED (rc={post_result.returncode})")
    except Exception as e:
        print(f"[evo] Obsidian post-cycle sync: ERROR ({e})")

    return 0


def _analyze_population(population: list[Any]) -> dict[str, Any]:
    """Analyze the population composition for reporting."""
    grid_methods: dict[str, int] = {}
    confirmations: dict[str, int] = {}
    allocation_methods: dict[str, int] = {}

    for g in population:
        gm = g.dca_genome.grid_method.value
        grid_methods[gm] = grid_methods.get(gm, 0) + 1
        am = g.dca_genome.allocation_method.value
        allocation_methods[am] = allocation_methods.get(am, 0) + 1
        for c in g.dca_genome.confirmation_indicators:
            cv = c.value
            confirmations[cv] = confirmations.get(cv, 0) + 1

    # Count by family (exploit = has volhigh, explore = no volhigh, hybrid = crossover)
    n_exploit = sum(1 for g in population if any(
        c.value == "volatility_high" for c in g.dca_genome.confirmation_indicators
    ))

    return {
        "total": len(population),
        "exploit": n_exploit,
        "explore": len(population) - n_exploit,
        "hybrid": sum(1 for g in population if g.lineage.parent_b_id is not None),
        "random": sum(1 for g in population if g.lineage.parent_a_id is None and g.lineage.parent_b_id is None),
        "grid_methods_used": sorted(grid_methods.keys()),
        "confirmations_used": sorted(confirmations.keys()),
        "allocation_methods_used": sorted(allocation_methods.keys()),
        "grid_method_counts": grid_methods,
        "confirmation_counts": confirmations,
    }


if __name__ == "__main__":
    sys.exit(main())
