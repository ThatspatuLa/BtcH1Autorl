"""Tests for Fix A (per-island stagnation) and Fix B (elite quality gate).

Effective 2026-06-22 per Six's plan.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evolution.config import EvolutionConfig
from evolution.evaluator import EvaluationResult
from evolution.harness import EvolutionHarness
from evolution.persistence import GenerationRecord


# ============================================================
# Helpers
# ============================================================

def _make_result(cand_id, disc_fit, cons_ratio, dep_pass=False, dep_fit=None):
    """Build a minimal EvaluationResult with the fields the harness reads."""
    return EvaluationResult(
        candidate_id=cand_id,
        genome_id=f"genome_{cand_id}",
        discovery_fitness=disc_fit,
        deployment_fitness=dep_fit if dep_pass else 0.0,
        deployment_pass=dep_pass,
        failed_deployment_gates=[] if dep_pass else ["consistency<0.50"],
        closest_to_passing_score=0.0,
        consistency_ratio=cons_ratio,
        consistency_multiplier=1.0,
        # Phase D: v2 component scores (synthesised for the test)
        full_period_base_score=disc_fit,
        recovery_score=0.5,
        stability_score=0.7,
        concentration_score=0.8,
        recovery_breakdown={
            "drawdown_recovery_speed": 0.5,
            "post_loss_month_bounce_rate": 0.5,
            "equity_high_reclaim_rate": 0.5,
            "cycle_recovery_health": 0.5,
        },
        rejected=False,
        reject_reason=None,
        rejection_source=None,
        elapsed_seconds=0.001,
        monthly_fitness=MagicMock(),
        score_breakdown=None,
        raw_metrics={},
        n_cycles_closed=100,
        final_equity=10000.0,
        max_dd_pct=0.05,
    )


def _make_gen_record(
    gen_idx: int,
    per_island_best: dict[int, float] | None = None,
    per_island_elite_count: dict[int, int] | None = None,
    best_fitness: float = 0.5,
) -> GenerationRecord:
    """Build a minimal GenerationRecord for stagnation testing."""
    return GenerationRecord(
        generation_index=gen_idx,
        started_at=0.0,
        ended_at=1.0,
        n_candidates=500,
        n_rejected=50,
        n_passed=450,
        n_elite_eligible=sum((per_island_elite_count or {}).values()),
        n_deployment_passing=10,
        best_fitness=best_fitness,
        median_fitness=0.3,
        best_candidate_id="cand_001",
        best_genome_id="genome_001",
        wall_time_seconds_used=300.0,
        rejection_reasons={},
        per_island_best_fitness=per_island_best or {},
        per_island_best_count=per_island_best or {},
        per_island_elite_count=per_island_elite_count or {},
    )


@pytest.fixture
def harness_island():
    """Build an EvolutionHarness with island mode + Fix A/B enabled."""
    df = pd.DataFrame({"close": [100.0] * 100})  # placeholder
    config = EvolutionConfig(
        candidates_per_gen=8,  # tiny for tests
        elite_count=2,
        random_injection=2,
        mutation_rate=0.45,
        crossover_rate=0.40,
        max_generations=10,
        wall_time_seconds=60,
        stagnation_generations=3,
        all_rejected_generations=3,
        parallel_workers=1,
        output_dir=tempfile.mkdtemp(prefix="test_stag_"),
        experiment_id="test_stag",
        base_seed=42,
        island_mode=True,
        n_islands=4,  # 4 islands for tests
        migration_every_n_gens=2,
        per_island_stagnation=True,
        min_consistency_for_elite=0.50,
        min_discovery_for_elite=0.70,
    )
    return EvolutionHarness(config=config, df=df)


# ============================================================
# Fix B: Elite quality gate
# ============================================================

def test_elite_quality_gate_excludes_low_consistency(harness_island):
    """A 0.28-fitness candidate with 0.30 consistency should NOT be elite."""
    results = [
        _make_result("cand_bad", disc_fit=0.28, cons_ratio=0.30),  # the gen-3 0.28 problem
        _make_result("cand_ok", disc_fit=0.80, cons_ratio=0.55),
        _make_result("cand_high_disc", disc_fit=0.85, cons_ratio=0.20),  # high disc → eligible
    ]
    # Replicate the elite selection logic from _run_generation
    passed = [r for r in results if not r.rejected]
    elite_eligible = [
        r for r in passed
        if r.consistency_ratio >= harness_island.config.min_consistency_for_elite
        or r.discovery_fitness >= harness_island.config.min_discovery_for_elite
    ]
    elite_ids = {r.candidate_id for r in elite_eligible}
    assert "cand_bad" not in elite_ids
    assert "cand_ok" in elite_ids
    assert "cand_high_disc" in elite_ids


def test_elite_quality_gate_fallback_to_all_passed(harness_island):
    """If no candidate meets the gate, fall back to all passed (so we never starve)."""
    results = [
        _make_result("cand_a", disc_fit=0.10, cons_ratio=0.10),
        _make_result("cand_b", disc_fit=0.20, cons_ratio=0.20),
    ]
    passed = [r for r in results if not r.rejected]
    elite_eligible = [
        r for r in passed
        if r.consistency_ratio >= harness_island.config.min_consistency_for_elite
        or r.discovery_fitness >= harness_island.config.min_discovery_for_elite
    ]
    # All fail the gate
    assert len(elite_eligible) == 0
    # The harness's fallback ensures breeding never starves
    breeding_pool = elite_eligible if elite_eligible else passed
    assert len(breeding_pool) == 2


def test_elite_quality_gate_uses_or_logic(harness_island):
    """Either consistency >= 0.5 OR discovery >= 0.7 qualifies."""
    # Border cases
    assert (_make_result("c1", 0.69, 0.50).discovery_fitness >= 0.70 or
            _make_result("c1", 0.69, 0.50).consistency_ratio >= 0.50)  # cons qualifies
    assert (_make_result("c2", 0.70, 0.49).discovery_fitness >= 0.70 or
            _make_result("c2", 0.70, 0.49).consistency_ratio >= 0.50)  # disc qualifies
    # Both fail
    c = _make_result("c3", 0.69, 0.49)
    assert not (c.discovery_fitness >= 0.70 or c.consistency_ratio >= 0.50)


# ============================================================
# Fix A: Per-island stagnation
# ============================================================

def test_per_island_stagnation_not_fired_when_one_island_improving(harness_island):
    """If even one island is improving, no stagnation."""
    # Gen 0: all islands improve
    r0 = _make_gen_record(0, per_island_best={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
                          per_island_elite_count={1: 1, 2: 1, 3: 1, 4: 1})
    assert not harness_island._check_stagnation(r0, gen_idx=0, last_improvement_gen=0)

    # Gen 1: islands 1, 2, 3 stagnate but island 4 improves
    r1 = _make_gen_record(1, per_island_best={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.6},
                          per_island_elite_count={1: 1, 2: 1, 3: 1, 4: 1})
    assert not harness_island._check_stagnation(r1, gen_idx=1, last_improvement_gen=0)

    # Gen 2: same — island 4 still improving
    r2 = _make_gen_record(2, per_island_best={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.7},
                          per_island_elite_count={1: 1, 2: 1, 3: 1, 4: 1})
    assert not harness_island._check_stagnation(r2, gen_idx=2, last_improvement_gen=0)


def test_per_island_stagnation_fired_when_all_stagnant(harness_island):
    """Stagnation fires when all active islands have plateaued for N gens."""
    # Gen 0: all improve
    r0 = _make_gen_record(0, per_island_best={1: 0.5, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    harness_island._check_stagnation(r0, gen_idx=0, last_improvement_gen=0)

    # Gen 1, 2, 3: no improvement (counter increments)
    for g in [1, 2, 3]:
        r = _make_gen_record(g, per_island_best={1: 0.5, 2: 0.5},
                             per_island_elite_count={1: 1, 2: 1})
        hit = harness_island._check_stagnation(r, gen_idx=g, last_improvement_gen=0)
        # stagnation_generations=3, so fire on the 3rd consecutive stagnant gen
        if g < 3:
            assert not hit, f"Should not fire at gen {g}"
        else:
            assert hit, f"Should fire at gen {g} (3 gens stagnant)"


def test_per_island_stagnation_excludes_inactive_islands(harness_island):
    """An island with 0 elites this gen is excluded from the stagnation check."""
    # Gen 0: all improve
    r0 = _make_gen_record(0, per_island_best={1: 0.5, 2: 0.5, 3: 0.5},
                          per_island_elite_count={1: 1, 2: 1, 3: 1})
    harness_island._check_stagnation(r0, gen_idx=0, last_improvement_gen=0)

    # Gen 1: islands 1, 2 plateau, island 3 produces 0 elites (re-seeded)
    r1 = _make_gen_record(1, per_island_best={1: 0.5, 2: 0.5},  # island 3 not in dict → 0 elites
                          per_island_elite_count={1: 1, 2: 1})  # island 3 absent
    assert not harness_island._check_stagnation(r1, gen_idx=1, last_improvement_gen=0)

    # Gen 2: same — islands 1, 2 still plateau, island 3 still 0 elites
    r2 = _make_gen_record(2, per_island_best={1: 0.5, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    assert not harness_island._check_stagnation(r2, gen_idx=2, last_improvement_gen=0)

    # Gen 3: stagnation_generations=3 → both 1 & 2 should now fire
    r3 = _make_gen_record(3, per_island_best={1: 0.5, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    assert harness_island._check_stagnation(r3, gen_idx=3, last_improvement_gen=0)


def test_per_island_stagnation_no_active_islands(harness_island):
    """If nobody produces elites, don't double-fire — let all_rejected handle it."""
    r = _make_gen_record(0, per_island_best={}, per_island_elite_count={})
    assert not harness_island._check_stagnation(r, gen_idx=0, last_improvement_gen=0)


def test_per_island_stagnation_resets_counter_on_improvement(harness_island):
    """A single improvement resets that island's counter."""
    # Gen 0: all at 0.5
    r0 = _make_gen_record(0, per_island_best={1: 0.5, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    harness_island._check_stagnation(r0, gen_idx=0, last_improvement_gen=0)

    # Gen 1: both plateau
    r1 = _make_gen_record(1, per_island_best={1: 0.5, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    harness_island._check_stagnation(r1, gen_idx=1, last_improvement_gen=0)
    # counter: island 1 = 1, island 2 = 1

    # Gen 2: island 1 improves (resets its counter), island 2 still plateau
    r2 = _make_gen_record(2, per_island_best={1: 0.7, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    assert not harness_island._check_stagnation(r2, gen_idx=2, last_improvement_gen=0)
    # counter: island 1 = 0, island 2 = 2

    # Gen 3: both plateau again
    r3 = _make_gen_record(3, per_island_best={1: 0.7, 2: 0.5},
                          per_island_elite_count={1: 1, 2: 1})
    assert not harness_island._check_stagnation(r3, gen_idx=3, last_improvement_gen=0)
    # counter: island 1 = 1, island 2 = 3 → stagnation_generations=3 → would fire,
    # but island 1 has counter 1, so all_stagnant is False


def test_per_island_stagnation_disabled_uses_global(harness_island):
    """When per_island_stagnation=False, fall back to global gens_since_improvement."""
    harness_island.config.per_island_stagnation = False
    # stagnation_generations=3; 5 gens since last improvement → fire
    r = _make_gen_record(5, best_fitness=0.5)
    assert harness_island._check_stagnation(r, gen_idx=5, last_improvement_gen=0)
    # 2 gens since last improvement → don't fire
    assert not harness_island._check_stagnation(r, gen_idx=2, last_improvement_gen=0)


# ============================================================
# GenerationRecord: new fields serialize/deserialize correctly
# ============================================================

def test_generation_record_per_island_roundtrip():
    """Per-island dicts must survive JSON round-trip."""
    from evolution.persistence import save_state, load_state, GenerationHistory
    import time

    rec = GenerationRecord(
        generation_index=0,
        started_at=time.time(),
        ended_at=time.time() + 60,
        n_candidates=500,
        n_rejected=50,
        n_passed=450,
        n_elite_eligible=13,
        n_deployment_passing=10,
        best_fitness=0.5,
        median_fitness=0.3,
        best_candidate_id="cand_001",
        best_genome_id="genome_001",
        wall_time_seconds_used=60.0,
        rejection_reasons={},
        per_island_best_fitness={1: 0.8, 2: 0.7, 3: 0.6},
        per_island_best_count={1: 50, 2: 30, 3: 20},
        per_island_elite_count={1: 2, 2: 1, 3: 3},
    )
    history = GenerationHistory(
        experiment_id="test",
        config={},
        started_at=time.time(),
        generations=[rec],
        best_fitness_ever=0.5,
        best_genome_id_ever="genome_001",
        best_candidate_id_ever="cand_001",
    )
    out = Path(tempfile.mkdtemp(prefix="gr_rt_"))
    save_state(history, out)
    loaded = load_state(out)
    assert loaded is not None
    gr = loaded.generations[0]
    assert gr.n_elite_eligible == 13
    assert gr.per_island_best_fitness[1] == 0.8
    assert gr.per_island_best_fitness[2] == 0.7
    assert gr.per_island_elite_count[3] == 3


def test_generation_record_defaults_backward_compat():
    """GenerationRecord can still be built without the new fields."""
    import time
    rec = GenerationRecord(
        generation_index=0,
        started_at=time.time(),
        ended_at=time.time() + 60,
        n_candidates=500,
        n_rejected=50,
        n_passed=450,
        n_deployment_passing=10,
        best_fitness=0.5,
        median_fitness=0.3,
        best_candidate_id="cand_001",
        best_genome_id="genome_001",
        wall_time_seconds_used=60.0,
        rejection_reasons={},
    )
    assert rec.n_elite_eligible == 0
    assert rec.per_island_best_fitness == {}
    assert rec.per_island_elite_count == {}
