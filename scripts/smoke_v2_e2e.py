#!/usr/bin/env python3
"""Phase E — End-to-end smoke test of v2 wiring on a single candidate.

Runs ONE candidate through backtest → compute_score → compute_monthly_fitness
and verifies all v2 fields are populated correctly.

Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from evolution.evaluator import CandidateEvaluator
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ComboMethod,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    SafetyGenome,
    SettingsOverrides,
    TpExitMethod,
    TpGenome,
    TriggerMode,
)
from reward.scoring import compute_score


def build_smoke_genome() -> CandidateGenome:
    """Build a realistic Phase E smoke-test genome (medium-aggressive fixed grid DCA)."""
    return CandidateGenome(
        genome_id="smoke_v2_e2e",
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"grid_pct": 0.005, "cooldown_candles": 3},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={"max_layers": 8, "volatility_threshold": 1.5},
            combo_method=ComboMethod.WEIGHTED_AVERAGE,
            combo_params={},
            trigger_mode=TriggerMode.PRICE_ONLY,
            confirmation_indicators=[],
            indicator_params={},
            max_dca_layers=8,
        ),
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"fixed_tp_pct": 0.005},
        ),
        safety_genome=SafetyGenome(
            max_dca_layers=8,
            overlap_allowed=False,
            min_break_even_for_overlap_pct=0.0,
            require_buffer_pct=0.20,
        ),
        settings_overrides=SettingsOverrides(),
        lineage=LineageMetadata(generation_index=0),
    )


def main() -> int:
    print("=" * 60)
    print("Phase E — Discovery Fitness v2 end-to-end smoke test")
    print("=" * 60)

    # 1. Load real BTC data
    data_path = PROJECT_ROOT / "data" / "processed" / "btc_h1_5y.feather"
    df = pd.read_feather(data_path)
    print(f"\n[1] Loaded {len(df):,} candles from {data_path.name}")

    # 2. Build genome
    genome = build_smoke_genome()
    print(f"[2] Built smoke genome: {genome.genome_id}")
    print(f"    DCA params: grid={genome.dca_genome.grid_method.value} "
          f"pct={genome.dca_genome.grid_params.get('grid_pct')} "
          f"max_layers={genome.safety_genome.max_dca_layers}")

    # 3-5. Test full evaluator (Stage 10 wrapper does backtest + score + monthly_fitness)
    print(f"\n[3-5] Running CandidateEvaluator (full wrapper, 1 candidate)...")
    evaluator = CandidateEvaluator(df=df, experiment_slug="smoke")
    er = evaluator.evaluate(genome, "cand_smoke_v2")
    print(f"    Backtest result: {er.n_cycles_closed} cycles, "
          f"final=${er.final_equity:,.2f}, max_dd={er.max_dd_pct*100:.2f}%")

    # Stage 5
    bt_equity = er.monthly_fitness.monthly_scores[0] if er.monthly_fitness.monthly_scores else None
    score_repr = er.score_breakdown
    if score_repr:
        print(f"\n[4] Stage 5 — score: {score_repr.get('final_score', 0):.4f}")
        if 'dd_quality' in score_repr:
            print(f"    DD quality: {score_repr['dd_quality'].get('normalised', 0):.4f}")

    # Stage 6 — MonthlyFitnessResult (v2)
    mf = er.monthly_fitness
    print(f"\n[5] Stage 6 — MonthlyFitnessResult (v2):")
    print(f"    discovery_fitness:     {mf.discovery_fitness:.4f}")
    print(f"    full_period_base_score (60%): {mf.full_period_base_score:.4f}")
    print(f"    recovery_score        (20%): {mf.recovery_score:.4f}")
    print(f"    stability_score        (5%): {mf.stability_score:.4f}")
    print(f"    concentration_score    (5%): {mf.concentration_score:.4f}")
    print(f"    consistency_ratio:            {mf.consistency_ratio:.4f}")
    print(f"    consistency_multiplier:       {mf.consistency_multiplier:.4f}")
    print(f"    deployment_fitness:           {mf.deployment_fitness:.4f}")
    print(f"    deployment_pass:              {mf.deployment_pass}")
    print(f"    rejected:                     {mf.rejected} (reason={mf.reject_reason})")
    print(f"    Recovery sub-metrics:")
    for k, v in mf.recovery_breakdown.items():
        print(f"      {k:35s} = {v:.4f}")

    # 6. Assertions
    print(f"\n[6] Verifying v2 contract...")
    assert 0.0 <= mf.discovery_fitness <= 1.0, "discovery_fitness out of [0,1]"
    assert 0.0 <= mf.full_period_base_score <= 1.0, "full_period_base_score out of [0,1]"
    assert 0.0 <= mf.recovery_score <= 1.0, "recovery_score out of [0,1]"
    assert 0.0 <= mf.stability_score <= 1.0, "stability_score out of [0,1]"
    assert 0.0 <= mf.concentration_score <= 1.0, "concentration_score out of [0,1]"
    assert len(mf.recovery_breakdown) == 4, (
        f"recovery_breakdown should have 4 entries, got {len(mf.recovery_breakdown)}"
    )
    for k in ["drawdown_recovery_speed", "post_loss_month_bounce_rate",
              "equity_high_reclaim_rate", "cycle_recovery_health"]:
        assert k in mf.recovery_breakdown, f"recovery_breakdown missing {k}"
    # Verify v2 weight sum holds (approximately)
    manual_v2 = (
        0.60 * mf.full_period_base_score
        + 0.20 * mf.recovery_score
        + 0.10 * mf.consistency_ratio
        + 0.05 * mf.stability_score
        + 0.05 * mf.concentration_score
    )
    print(f"    Manual v2 recompute: {manual_v2:.6f} vs reported {mf.discovery_fitness:.6f}")
    assert abs(manual_v2 - mf.discovery_fitness) < 1e-6, (
        f"v2 formula mismatch: recompute={manual_v2}, reported={mf.discovery_fitness}"
    )
    print(f"    ✓ All assertions passed")
    print(f"    ✓ v2 formula verified: 0.60·fpbs + 0.20·rec + 0.10·cons + 0.05·stab + 0.05·conc")

    # 7. Test full evaluator (the wrapper Stage 10 calls)
    print(f"\n[7] Testing CandidateEvaluator (full wrapper)...")
    evaluator = CandidateEvaluator(df=df, experiment_slug="smoke")
    er = evaluator.evaluate(genome, "cand_smoke_v2")
    print(f"    EvaluationResult:")
    print(f"      discovery_fitness: {er.discovery_fitness:.4f}")
    print(f"      deployment_fitness: {er.deployment_fitness:.4f}")
    print(f"      deployment_pass: {er.deployment_pass}")
    print(f"      full_period_base_score: {er.full_period_base_score:.4f}")
    print(f"      recovery_score: {er.recovery_score:.4f}")
    print(f"      stability_score: {er.stability_score:.4f}")
    print(f"      concentration_score: {er.concentration_score:.4f}")
    print(f"      rejected: {er.rejected} ({er.reject_reason})")
    print(f"      n_cycles_closed: {er.n_cycles_closed}")
    print(f"      elapsed_seconds: {er.elapsed_seconds:.3f}")
    assert er.discovery_fitness == mf.discovery_fitness, (
        "evaluator.discovery_fitness != monthly_fitness.discovery_fitness"
    )
    print(f"    ✓ Evaluator matches monthly_fitness")

    # 8. to_dict() includes v2 fields
    d = er.to_dict()
    for k in ["full_period_base_score", "recovery_score", "stability_score",
              "concentration_score", "recovery_breakdown"]:
        assert k in d, f"to_dict() missing {k}"
    print(f"    ✓ to_dict() includes all v2 fields")

    print(f"\n{'=' * 60}")
    print(f"PHASE E SMOKE TEST PASSED ✓")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
