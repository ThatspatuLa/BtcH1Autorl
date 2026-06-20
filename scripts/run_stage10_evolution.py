"""Stage 10 — Full DCA evolution run.

500 candidates per generation, 20 max generations, 8h wall-time cap,
bounded by stagnation/all-rejected stops. TP stays simple/fixed (Stage 9).

Resumable via `evolution/GenerationHistory` re-loaded from <output_dir>/generation_history.json
when resume=True and the file exists.

Output: runs/stage10_<timestamp>/{dca_leaderboard.json, generation_history.json,
best_genome.json, rejection_reasons.json, final_status.json, per-gen/}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root on path so `evolution` and `dca_engine` resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolution.config import EvolutionConfig  # noqa: E402
from evolution.harness import EvolutionHarness  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "btc_h1_5y.feather"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10 DCA evolution")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write per-gen + final artifacts "
             "(default: runs/stage10_<timestamp>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing state in --output-dir if generation_history.json exists",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=500,
        help="Candidates per generation (default 500)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Hard cap on generations (default 20)",
    )
    parser.add_argument(
        "--wall-time-seconds",
        type=int,
        default=8 * 3600,
        help="Wall-time cap in seconds (default 28800 = 8h)",
    )
    parser.add_argument(
        "--stagnation-gens",
        type=int,
        default=5,
        help="Stop after N gens with no improvement (default 5)",
    )
    parser.add_argument(
        "--all-rejected-gens",
        type=int,
        default=3,
        help="Stop after N consecutive gens with 0 passing candidates (default 3)",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=20,
        help="Elite count carried into next gen (default 20). Must be <= --candidates.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Process pool size (default 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to BTC H1 feather data file (default: data/processed/btc_h1_5y.feather)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[stage10] ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1
    print(f"[stage10] Loading data: {data_path}")
    df = pd.read_feather(data_path)
    print(f"[stage10] Loaded {len(df):,} rows ({df['date'].min()} → {df['date'].max()})")

    output_dir = Path(args.output_dir or f"runs/stage10_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage10] Output dir: {output_dir}")
    print(f"[stage10] Config: {args.candidates} cands/gen, max {args.max_generations} gens, "
          f"wall-time {args.wall_time_seconds}s, stagnation {args.stagnation_gens}, "
          f"all-rejected {args.all_rejected_gens}, workers {args.parallel_workers}")
    if args.seed is not None:
        print(f"[stage10] Seed: {args.seed}")
    print(f"[stage10] Resume: {args.resume}")

    config = EvolutionConfig(
        candidates_per_gen=args.candidates,
        max_generations=args.max_generations,
        wall_time_seconds=args.wall_time_seconds,
        stagnation_generations=args.stagnation_gens,
        all_rejected_generations=args.all_rejected_gens,
        parallel_workers=args.parallel_workers,
        elite_count=args.elite_count,
        output_dir=str(output_dir),
        base_seed=args.seed if args.seed is not None else 42,
    )

    harness = EvolutionHarness(config, df)
    summary = harness.run(resume=args.resume)

    # Print the final summary
    print("\n" + "=" * 60)
    print("STAGE 10 EVOLUTION SUMMARY")
    print("=" * 60)
    print(f"Status:                 {summary.termination_reason}")
    print(f"Generations completed:  {summary.generations_completed} / {summary.generations_planned}")
    print(f"Total candidates:       {summary.total_candidates_evaluated}")
    print(f"Best fitness:           {summary.best_fitness_ever}")
    print(f"Best genome id:         {summary.best_genome_id_ever}")
    print(f"Best candidate id:      {summary.best_candidate_id_ever}")
    print(f"Total runtime:          {summary.total_runtime_seconds:.1f}s")
    print(f"Output dir:             {output_dir}")
    print("=" * 60)

    # Final status file (flat dict for easy reading)
    final_status = {
        "termination_reason": summary.termination_reason,
        "generations_completed": summary.generations_completed,
        "generations_planned": summary.generations_planned,
        "total_candidates_evaluated": summary.total_candidates_evaluated,
        "best_fitness_ever": summary.best_fitness_ever,
        "best_genome_id_ever": summary.best_genome_id_ever,
        "best_candidate_id_ever": summary.best_candidate_id_ever,
        "total_runtime_seconds": summary.total_runtime_seconds,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "output_dir": summary.output_dir,
    }
    (output_dir / "final_status.json").write_text(json.dumps(final_status, indent=2, default=str))
    print("\n[stage10] Wrote final_status.json")

    return 0  # user inspects summary for termination_reason


if __name__ == "__main__":
    sys.exit(main())
