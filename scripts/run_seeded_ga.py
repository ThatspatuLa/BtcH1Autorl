#!/usr/bin/env python3
"""Seeded GA run around the volhigh + cooldown family.

Seeds the initial population with variations around the known-good
volatility_high(1.3) + cooldown 3 + 0.5% grid + 0.4% TPorganism,
then lets evolution explore from there.
"""
import sys
import time
import random

sys.path.insert(0, '.')

import pandas as pd
from genome.schema import (
    AllocationMethod, CandidateGenome, ComboMethod, ConfirmationIndicator,
    DcaGenome, GridMethod, LineageMetadata, TpExitMethod, TpGenome, TriggerMode,
)
from evolution.config import EvolutionConfig
from evolution.harness import EvolutionHarness
from evolution.operators import ALL_CONFIRMATION_INDICATORS, INDICATOR_DEFAULT_PARAMS

df = pd.read_feather('data/processed/btc_h1_5y.feather')
print(f"[seeded_ga] Loaded {len(df)} rows")

# Base organism: volhigh(1.3) + cd3 + gp05 + tp04
BASE_GRID_PCT = 0.005
BASE_TP_PCT = 0.004
BASE_MAX_LAYERS = 12
BASE_COOLDOWN = 3
BASE_INDICATORS = [ConfirmationIndicator.VOLATILITY_HIGH]
BASE_IND_PARAMS = {"volatility_high": {"threshold": 1.3}}

def make_seeded_candidate(rng, genome_id, generation_index, gid,
                          grid_pct, tp_pct, max_layers, cooldown,
                          indicators, ind_params):
    dca = DcaGenome(
        grid_method=GridMethod.FIXED_PCT,
        grid_params={"pct": grid_pct, "max_layers": max_layers, "tp_pct": tp_pct},
        allocation_method=AllocationMethod.EQUAL,
        allocation_params={},
        combo_method=ComboMethod.WEIGHTED_AVERAGE,
        combo_params={},
        trigger_mode=TriggerMode.PRICE_ONLY,
        confirmation_indicators=indicators,
        indicator_params=ind_params,
        max_dca_layers=max_layers,
    )
    tp = TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": tp_pct})
    return CandidateGenome(
        genome_id=f"genome_G{generation_index}_{gid:06d}",
        dca_genome=dca,
        tp_genome=tp,
        lineage=LineageMetadata(
            parent_a_id=None, parent_b_id=None,
            generation_index=generation_index,
            mutation_seed=rng.randint(0, 2**31 - 1),
        ),
    )

def build_seeded_population(n_total, rng):
    """Build a population seeded around the known-good organism."""
    candidates = []
    gid_counter = 0

    # 1. The exact base organism (1 candidate)
    candidates.append(make_seeded_candidate(
        rng, f"genome_G0_{gid_counter:06d}", 0, gid_counter,
        BASE_GRID_PCT, BASE_TP_PCT, BASE_MAX_LAYERS, BASE_COOLDOWN,
        list(BASE_INDICATORS), dict(BASE_IND_PARAMS),
    ))
    gid_counter += 1

    # 2. Variations around the base: grid_pct ±20%, tp_pct ±20%, cooldown ±1, vol_threshold ±0.2 (15 candidates)
    for _ in range(15):
        gp = BASE_GRID_PCT * rng.uniform(0.8, 1.2)
        tp = BASE_TP_PCT * rng.uniform(0.8, 1.2)
        ml = BASE_MAX_LAYERS + rng.choice([-2, -1, 0, 1, 2])
        ml = max(2, min(15, ml))
        cd = BASE_COOLDOWN + rng.choice([-1, 0, 1])
        cd = max(0, min(10, cd))
        vt = BASE_IND_PARAMS["volatility_high"]["threshold"] + rng.uniform(-0.2, 0.2)
        vt = max(1.0, min(2.0, vt))
        ind_params = {"volatility_high": {"threshold": round(vt, 2)}}
        candidates.append(make_seeded_candidate(
            rng, f"genome_G0_{gid_counter:06d}", 0, gid_counter,
            round(gp, 4), round(tp, 4), ml, cd,
            [ConfirmationIndicator.VOLATILITY_HIGH], ind_params,
        ))
        gid_counter += 1

    # 3. Add rsi_below + volhigh combos (10 candidates)
    for _ in range(10):
        gp = BASE_GRID_PCT * rng.uniform(0.7, 1.3)
        tp = BASE_TP_PCT * rng.uniform(0.7, 1.3)
        ml = BASE_MAX_LAYERS + rng.choice([-2, -1, 0, 1, 2])
        ml = max(2, min(15, ml))
        cd = BASE_COOLDOWN + rng.choice([-1, 0, 1])
        cd = max(0, min(10, cd))
        vt = rng.uniform(1.1, 1.8)
        rsi_thresh = rng.uniform(25.0, 40.0)
        inds = [ConfirmationIndicator.VOLATILITY_HIGH, ConfirmationIndicator.RSI_BELOW]
        ind_params = {
            "volatility_high": {"threshold": round(vt, 2)},
            "rsi_below": {"threshold": round(rsi_thresh, 1)},
        }
        candidates.append(make_seeded_candidate(
            rng, f"genome_G0_{gid_counter:06d}", 0, gid_counter,
            round(gp, 4), round(tp, 4), ml, cd, inds, ind_params,
        ))
        gid_counter += 1

    # 4. Add volhigh + ma_below combos (10 candidates)
    for _ in range(10):
        gp = BASE_GRID_PCT * rng.uniform(0.7, 1.3)
        tp = BASE_TP_PCT * rng.uniform(0.7, 1.3)
        ml = BASE_MAX_LAYERS + rng.choice([-2, -1, 0, 1, 2])
        ml = max(2, min(15, ml))
        cd = BASE_COOLDOWN + rng.choice([-1, 0, 1])
        cd = max(0, min(10, cd))
        vt = rng.uniform(1.1, 1.8)
        inds = [ConfirmationIndicator.VOLATILITY_HIGH, ConfirmationIndicator.MA_BELOW]
        ind_params = {"volatility_high": {"threshold": round(vt, 2)}}
        candidates.append(make_seeded_candidate(
            rng, f"genome_G0_{gid_counter:06d}", 0, gid_counter,
            round(gp, 4), round(tp, 4), ml, cd, inds, ind_params,
        ))
        gid_counter += 1

    # 5. Fill remaining with random candidates from the full search space
    from evolution.operators import random_candidate_genome
    while len(candidates) < n_total:
        candidates.append(random_candidate_genome(rng=rng, generation_index=0))
        gid_counter += 1

    return candidates[:n_total]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seeded GA around volhigh+cooldown family")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--wall-time-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=str, default="runs/stage10_seeded_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"[seeded_ga] Building seeded population of {args.candidates}")
    seeded_pop = build_seeded_population(args.candidates, rng)
    print(f"[seeded_ga] Seeded population built: {len(seeded_pop)} candidates")

    config = EvolutionConfig(
        candidates_per_gen=args.candidates,
        elite_count=args.elite_count,
        max_generations=args.max_generations,
        wall_time_seconds=args.wall_time_seconds,
        stagnation_generations=5,
        all_rejected_generations=3,
        parallel_workers=8,
        output_dir=args.output_dir,
        experiment_id="seeded_v1",
    )

    harness = EvolutionHarness(
        config=config,
        df=df,
        seeded_population=seeded_pop,
        rng=rng,
    )

    t0 = time.time()
    result = harness.run()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SEEDED GA COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Status:           {result.termination_reason}")
    print(f"  Generations:      {result.generations_completed} / {config.max_generations}")
    print(f"  Total candidates: {result.total_candidates_evaluated}")
    print(f"  Best fitness:     {result.best_fitness_ever:.6f}")
    print(f"  Best genome:      {result.best_genome_id_ever}")
    print(f"  Runtime:          {elapsed:.1f}s")
    print(f"  Output dir:       {config.output_dir}")


if __name__ == "__main__":
    main()
