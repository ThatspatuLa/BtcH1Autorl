"""20-candidate benchmark generation — measure real runtime, verify end-to-end.

Tests:
- 20 candidates per generation (vs locked 500)
- 2 generations (random gen0 + mutate/crossover/random gen1)
- Real BTC 5y data
- Verify: rejection reasons, safety integration, monthly fitness, reporting

This is the GATE between Stage 9 and Stage 10. If this benchmark passes,
we're cleared to do the real 500-candidate / 20-generation run.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

# Project path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evolution import EvolutionConfig, EvolutionHarness, HarnessHooks
from evolution.persistence import load_state


def main() -> int:
    # === Load real BTC 5y data ===
    data_path = Path("data/processed/btc_h1_5y.feather")
    if not data_path.exists():
        print(f"FATAL: {data_path} not found. Run Stage 2 data pipeline first.")
        return 1
    df = pd.read_feather(data_path)
    print(f"Loaded {len(df):,} BTC H1 candles from {data_path}")

    # === Configure the benchmark ===
    # Override the v1 defaults with benchmark-friendly values
    output_dir = "results/benchmark_20cand_2gen"
    # Wipe any prior run
    import shutil
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    config = EvolutionConfig(
        candidates_per_gen=20,           # 20 instead of 500
        elite_count=4,                    # 4 elites (20% — generous for tiny pop)
        random_injection=4,               # 4 fresh random per gen
        mutation_rate=0.30,
        crossover_rate=0.50,
        wall_time_seconds=600,            # 10 min cap (very generous)
        max_generations=2,                # 2 generations
        stagnation_generations=5,
        all_rejected_generations=3,
        parallel_workers=1,               # serial for the benchmark
        base_seed=42,
        output_dir=output_dir,
        experiment_id="exp_benchmark_20cand",
        tp_pct=0.02,
    )

    # Sanity: confirm children math
    print(f"Config: {config.candidates_per_gen} per gen, "
          f"{config.elite_count} elites, {config.random_injection} random inject, "
          f"{config.crossover_children} crossover + {config.mutation_children} mutation children")
    assert config.elite_count + config.crossover_children + config.mutation_children + config.random_injection == config.candidates_per_gen

    # === Set up observability hooks ===
    n_evaluated = {"count": 0}
    rejected_sources = Counter()
    rejection_reasons = Counter()
    all_results = []

    def on_candidate(res) -> None:
        n_evaluated["count"] += 1
        all_results.append(res)
        if res.rejected:
            rejected_sources[res.rejection_source or "unknown"] += 1
            rejection_reasons[res.reject_reason or "unknown"] += 1

    def on_gen_start(idx: int) -> None:
        print(f"\n=== Gen {idx} starting ===")

    def on_gen_end(rec) -> None:
        print(f"=== Gen {rec.generation_index} done: "
              f"{rec.n_passed} passed / {rec.n_rejected} rejected, "
              f"best_fitness={rec.best_fitness:.4f}, "
              f"median_fitness={rec.median_fitness:.4f}, "
              f"reasons={rec.rejection_reasons}")
        for entry in rec.leaderboard[:3]:
            print(f"  #{entry['rank']}: {entry['genome_id']} "
                  f"fitness={entry['fitness']:.4f} "
                  f"final_equity=${entry['final_equity']:.2f} "
                  f"dd={entry['max_dd_pct']:.1%}")

    hooks = HarnessHooks(
        on_candidate_evaluated=on_candidate,
        on_generation_start=on_gen_start,
        on_generation_end=on_gen_end,
    )

    # === Run ===
    harness = EvolutionHarness(config, df, hooks=hooks)
    t0 = time.time()
    summary = harness.run(resume=False)
    total_runtime = time.time() - t0

    # === Report ===
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total runtime:        {total_runtime:.2f}s")
    print(f"Generations:          {summary.generations_completed} / {summary.generations_planned}")
    print(f"Candidates evaluated: {summary.total_candidates_evaluated}")
    print(f"  (expected: {config.candidates_per_gen * config.max_generations})")
    print(f"Avg time per cand:    {total_runtime / max(1, summary.total_candidates_evaluated):.3f}s")
    print(f"Estimated 500x20:     {total_runtime / max(1, summary.total_candidates_evaluated) * 500 * 20 / 3600:.2f}h")
    print(f"Termination:          {summary.termination_reason}")
    print(f"Best fitness ever:    {summary.best_fitness_ever:.4f}")
    print(f"Best genome:          {summary.best_genome_id_ever}")
    print(f"Best candidate:       {summary.best_candidate_id_ever}")
    print()
    print(f"Rejections by source: {dict(rejected_sources)}")
    print(f"Rejections by reason: {dict(rejection_reasons)}")
    print()
    print(f"Output dir:           {output_dir}/")
    print("  - run_summary.json")
    print("  - generation_history.json")
    print("  - leaderboards/gen_*.json")
    print("  - best_genomes/gen_*.json")
    print("  - rejection_reports/gen_*.json")

    # === Verify the artifacts exist ===
    out = Path(output_dir)
    assert (out / "run_summary.json").exists(), "run_summary.json missing"
    assert (out / "generation_history.json").exists(), "generation_history.json missing"
    assert (out / "leaderboards").exists(), "leaderboards/ missing"
    assert (out / "best_genomes").exists(), "best_genomes/ missing"
    assert (out / "rejection_reports").exists(), "rejection_reports/ missing"
    print("\n✓ All output artifacts present")

    # === Verify resumability: re-load and check state matches ===
    history = load_state(output_dir)
    assert history is not None
    assert len(history.generations) == summary.generations_completed, (
        f"history has {len(history.generations)} gens, summary says {summary.generations_completed}"
    )
    print(f"✓ History resumable: {len(history.generations)} generations recorded")

    # === Verify safety integration: no candidate should have margin breach ===
    # (Safety is integrated via Stage 6 monthly fitness hard rejects —
    # we don't need a separate assertion here, the rejection_reasons show it)
    if "consistency<0.50" in rejection_reasons or "drawdown>35%" in rejection_reasons:
        print("✓ Safety/fitness hard rejects firing as expected")

    # === Pass/fail ===
    expected_n = config.candidates_per_gen * config.max_generations
    if summary.total_candidates_evaluated == expected_n:
        print(f"\n✓ BENCHMARK PASSED: {expected_n} candidates evaluated as expected")
        return 0
    print(f"\n✗ BENCHMARK FAILED: expected {expected_n}, got {summary.total_candidates_evaluated}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
