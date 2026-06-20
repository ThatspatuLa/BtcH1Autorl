"""Tests for evolution harness — config, operators, evaluator, harness, persistence."""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from evolution.config import (
    DEFAULT_ALL_REJECTED_GENERATIONS,
    DEFAULT_CANDIDATES_PER_GEN,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_STAGNATION_GENERATIONS,
    DEFAULT_WALL_TIME_SECONDS,
    EvolutionConfig,
)
from evolution.evaluator import CandidateEvaluator, EvaluationResult
from evolution.harness import EvolutionHarness, HarnessHooks
from evolution.operators import (
    DCA_PARAM_RANGES,
    crossover,
    mutate,
    random_candidate_genome,
    random_dca_genome,
)
from evolution.persistence import (
    GenerationHistory,
    GenerationRecord,
    UnfinishedStatus,
    load_state,
    save_state,
    save_unfinished_status,
)

# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def temp_output_dir():
    """Provide a temporary output directory that's cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="evo_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def small_ohlcv_df() -> pd.DataFrame:
    """Tiny OHLCV that triggers the placeholder DCA at least once."""
    # 500 hourly candles, oscillating around 100
    idx = pd.date_range("2021-06-01", periods=500, freq="h")
    prices = [100.0 + (i % 50) * 0.3 - 7.5 for i in range(500)]  # 92.5-107.5
    return pd.DataFrame({
        "date": idx,
        "open": prices,
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "close": prices,
        "volume": [1000.0] * 500,
    })


# ============================================================
# Test: EvolutionConfig
# ============================================================

def test_config_defaults_locked():
    c = EvolutionConfig()
    assert c.candidates_per_gen == DEFAULT_CANDIDATES_PER_GEN == 500
    assert c.wall_time_seconds == DEFAULT_WALL_TIME_SECONDS == 28800
    assert c.max_generations == DEFAULT_MAX_GENERATIONS == 20
    assert c.stagnation_generations == DEFAULT_STAGNATION_GENERATIONS == 5
    assert c.all_rejected_generations == DEFAULT_ALL_REJECTED_GENERATIONS == 3


def test_config_children_math():
    c = EvolutionConfig(candidates_per_gen=500, elite_count=20, random_injection=120,
                        crossover_rate=0.5)
    # children = 500 - 20 - 120 = 360
    assert c.children_per_gen == 360
    # crossover = 360 * 0.5 = 180
    assert c.crossover_children == 180
    assert c.mutation_children == 180


def test_config_rejects_bad_elite_count():
    with pytest.raises(ValueError):
        EvolutionConfig(candidates_per_gen=10, elite_count=20)


def test_config_rejects_bad_mutation_rate():
    with pytest.raises(ValueError):
        EvolutionConfig(mutation_rate=1.5)


def test_config_rejects_bad_wall_time():
    with pytest.raises(ValueError):
        EvolutionConfig(wall_time_seconds=10)


def test_config_to_from_dict():
    c = EvolutionConfig(candidates_per_gen=100, base_seed=99)
    d = c.to_dict()
    c2 = EvolutionConfig.from_dict(d)
    assert c2.candidates_per_gen == 100
    assert c2.base_seed == 99
    assert c2.output_dir == "results/evolution"  # default; from_dict doesn't store it


# ============================================================
# Test: operators — random genome
# ============================================================

def test_random_dca_genome_in_range():
    rng = random.Random(42)
    for _ in range(50):
        g = random_dca_genome(rng=rng)
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        assert lo <= g.grid_params["pct"] <= hi
        lo_i, hi_i = DCA_PARAM_RANGES["max_layers"]
        assert lo_i <= g.max_dca_layers <= hi_i
        assert g.grid_method.value == "fixed_pct"


def test_random_candidate_genome_has_tp():
    rng = random.Random(42)
    g = random_candidate_genome(rng=rng, generation_index=0, tp_pct=0.025)
    assert g.tp_genome.exit_method.value == "fixed"
    assert g.tp_genome.exit_params["tp_pct"] == 0.025


def test_random_candidate_genome_ids_unique():
    rng = random.Random(42)
    ids = {random_candidate_genome(rng=rng, generation_index=0).genome_id for _ in range(100)}
    assert len(ids) == 100


# ============================================================
# Test: operators — expanded search space (Stage 10 v2)
# ============================================================

def test_random_dca_genome_search_space_includes_tp_pct():
    """The expanded search space: grid_pct, max_layers, AND tp_pct."""
    rng = random.Random(42)
    samples = [random_dca_genome(rng=rng, generation_index=0) for _ in range(50)]
    pcts = [s.grid_params["pct"] for s in samples]
    layers = [s.max_dca_layers for s in samples]
    tps = [s.grid_params["tp_pct"] for s in samples]
    # grid_pct range: 0.003..0.08
    assert all(0.003 <= p <= 0.08 for p in pcts)
    assert max(pcts) - min(pcts) > 0.04  # actually varying
    # max_layers: 2..12
    assert all(2 <= n_layers <= 12 for n_layers in layers)
    # tp_pct: 0.005..0.05 (new dimension)
    assert all(0.005 <= t <= 0.05 for t in tps)
    assert max(tps) - min(tps) > 0.02  # actually varying


def test_random_dca_tp_synced_with_tp_genome():
    """The tp_pct in dca_genome.grid_params must match tp_genome.exit_params."""
    rng = random.Random(7)
    for _ in range(20):
        g = random_candidate_genome(rng=rng, generation_index=0)
        assert g.dca_genome.grid_params["tp_pct"] == g.tp_genome.exit_params["tp_pct"]


def test_crossover_keeps_tp_synced():
    """Crossover must keep dca_genome.tp_pct == tp_genome.tp_pct."""
    rng = random.Random(13)
    a = random_candidate_genome(rng=rng, generation_index=0)
    b = random_candidate_genome(rng=rng, generation_index=1)
    for _ in range(20):
        child = crossover(a, b, rng=rng)
        assert child.dca_genome.grid_params["tp_pct"] == child.tp_genome.exit_params["tp_pct"]


def test_mutate_can_change_tp_pct():
    """With the expanded search space, mutate SHOULD sometimes change tp_pct."""
    rng = random.Random(99)
    parent = random_candidate_genome(rng=rng, generation_index=0)
    parent_tp = parent.dca_genome.grid_params["tp_pct"]
    changed = 0
    for _ in range(50):
        child = mutate(parent, rng=rng, mutation_rate=1.0)
        if abs(child.dca_genome.grid_params["tp_pct"] - parent_tp) > 1e-9:
            changed += 1
    # At high mutation_rate (1.0) we expect most to change
    assert changed > 25, f"Expected >25 changes in 50 mutations, got {changed}"


def test_extract_dca_params_returns_three_values():
    """extract_dca_params_from_genome now returns grid_pct, max_layers, tp_pct."""
    from dca_engine.tp_baseline import extract_dca_params_from_genome
    g = random_candidate_genome(rng=random.Random(1), generation_index=0)
    params = extract_dca_params_from_genome(g)
    assert "grid_pct" in params
    assert "max_layers" in params
    assert "tp_pct" in params
    assert params["tp_pct"] == g.dca_genome.grid_params["tp_pct"]


# ============================================================
# Test: operators — mutate
# ============================================================

def test_mutate_changes_params_sometimes():
    rng = random.Random(42)
    parent = random_candidate_genome(rng=rng, generation_index=0)
    # Run mutation many times; at least some should change
    changed = 0
    for _ in range(50):
        child = mutate(parent, rng=rng, mutation_rate=1.0)  # force mutation
        if (child.dca_genome.grid_params["pct"] != parent.dca_genome.grid_params["pct"]
                or child.dca_genome.max_dca_layers != parent.dca_genome.max_dca_layers):
            changed += 1
    assert changed > 0


def test_mutate_keeps_tp_synced_with_dca():
    """Stage 10: tp_pct is in BOTH dca_genome.grid_params and tp_genome.exit_params.
    Mutate must keep them in sync (otherwise the Stage 9 baseline reads from
    tp_genome and ignores the dca value)."""
    rng = random.Random(42)
    parent = random_candidate_genome(rng=rng, generation_index=0)
    for _ in range(20):
        child = mutate(parent, rng=rng, mutation_rate=1.0)
        dca_tp = child.dca_genome.grid_params["tp_pct"]
        tp_genome_tp = child.tp_genome.exit_params["tp_pct"]
        assert dca_tp == tp_genome_tp, f"TP desynced: dca={dca_tp} tp={tp_genome_tp}"


def test_mutate_clamps_to_range():
    """Mutation can't push params outside DCA_PARAM_RANGES."""
    rng = random.Random(42)
    parent = random_candidate_genome(rng=rng, generation_index=0)
    for _ in range(100):
        child = mutate(parent, rng=rng, mutation_rate=1.0)
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        assert lo <= child.dca_genome.grid_params["pct"] <= hi
        lo_i, hi_i = DCA_PARAM_RANGES["max_layers"]
        assert lo_i <= child.dca_genome.max_dca_layers <= hi_i


def test_mutate_lineage():
    rng = random.Random(42)
    parent = random_candidate_genome(rng=rng, generation_index=0)
    child = mutate(parent, rng=rng)
    assert child.lineage.parent_a_id == parent.genome_id
    assert child.lineage.parent_b_id is None
    assert child.lineage.mutation_ops == [{"op": "mutate", "parent_id": parent.genome_id}]


# ============================================================
# Test: operators — crossover
# ============================================================

def test_crossover_inherits_one_parent_per_param():
    rng = random.Random(42)
    a = random_candidate_genome(rng=rng, generation_index=0)
    b = random_candidate_genome(rng=rng, generation_index=0)
    child = crossover(a, b, rng=rng)
    # Either grid_pct from A or from B
    assert child.dca_genome.grid_params["pct"] in (
        a.dca_genome.grid_params["pct"], b.dca_genome.grid_params["pct"]
    )
    # Either max_layers from A or from B
    assert child.dca_genome.max_dca_layers in (
        a.dca_genome.max_dca_layers, b.dca_genome.max_dca_layers
    )


def test_crossover_keeps_tp_from_a():
    rng = random.Random(42)
    a = random_candidate_genome(rng=rng, generation_index=0, tp_pct=0.03)
    b = random_candidate_genome(rng=rng, generation_index=0, tp_pct=0.07)
    child = crossover(a, b, rng=rng)
    # TP comes from A in Stage 10
    assert child.tp_genome.exit_params["tp_pct"] == 0.03


def test_crossover_lineage():
    rng = random.Random(42)
    a = random_candidate_genome(rng=rng, generation_index=0)
    b = random_candidate_genome(rng=rng, generation_index=0)
    child = crossover(a, b, rng=rng)
    assert child.lineage.parent_a_id == a.genome_id
    assert child.lineage.parent_b_id == b.genome_id
    assert child.lineage.mutation_ops[0]["op"] == "crossover"


# ============================================================
# Test: evaluator
# ============================================================

def test_evaluator_smoke(small_ohlcv_df):
    ev = CandidateEvaluator(small_ohlcv_df)
    rng = random.Random(42)
    g = random_candidate_genome(rng=rng, generation_index=0, tp_pct=0.02)
    res = ev.evaluate(g, "cand_test")
    assert isinstance(res, EvaluationResult)
    assert res.candidate_id == "cand_test"
    assert res.elapsed_seconds > 0
    assert res.error is None  # not an evaluation error
    # 500 candles is a synthetic curve — the oscillating price triggers
    # many cycles. The candidate may or may not be rejected depending on
    # params; we just check the evaluator returned a valid result.
    assert isinstance(res.fitness, float)
    assert res.n_cycles_closed >= 0


def test_evaluator_never_raises(small_ohlcv_df):
    """Even with broken input, the evaluator should not throw."""
    ev = CandidateEvaluator(small_ohlcv_df)
    rng = random.Random(42)
    # Force a weird grid_pct via direct mutation
    g = random_candidate_genome(rng=rng, generation_index=0)
    g.dca_genome.grid_params["pct"] = -0.5  # negative
    res = ev.evaluate(g, "cand_weird")
    assert res is not None


# ============================================================
# Test: persistence
# ============================================================

def test_save_load_state_roundtrip(temp_output_dir):
    history = GenerationHistory(
        experiment_id="exp_test",
        config={"candidates_per_gen": 100},
        started_at=time.time(),
        best_fitness_ever=0.5,
        best_genome_id_ever="genome_001",
        best_candidate_id_ever="cand_001",
        candidate_counter=42,
    )
    history.generations.append(GenerationRecord(
        generation_index=0,
        started_at=time.time(),
        ended_at=time.time() + 60,
        n_candidates=100,
        n_rejected=80,
        n_passed=20,
        n_deployment_passing=2,
        best_fitness=0.5,
        median_fitness=0.3,
        best_candidate_id="cand_001",
        best_genome_id="genome_001",
        wall_time_seconds_used=60.0,
        rejection_reasons={"consistency<0.50": 80},
        evaluated_candidate_ids=["cand_001", "cand_002"],
    ))
    save_state(history, temp_output_dir)
    loaded = load_state(temp_output_dir)
    assert loaded is not None
    assert loaded.experiment_id == "exp_test"
    assert loaded.best_fitness_ever == 0.5
    assert len(loaded.generations) == 1
    assert loaded.generations[0].n_passed == 20


def test_save_unfinished_status(temp_output_dir):
    status = UnfinishedStatus(
        reason="wall_time",
        generations_completed=5,
        max_generations=20,
        wall_time_seconds_used=28800.0,
        wall_time_seconds_cap=28800,
        best_fitness_ever=0.7,
        best_genome_id_ever="genome_xyz",
        best_candidate_id_ever="cand_xyz",
        finished_at=time.time(),
    )
    save_unfinished_status(status, temp_output_dir)
    path = Path(temp_output_dir) / "unfinished_status.json"
    assert path.exists()
    with open(path) as f:
        d = json.load(f)
    assert d["reason"] == "wall_time"
    assert d["generations_completed"] == 5


def test_load_state_missing_returns_none(temp_output_dir):
    assert load_state(temp_output_dir) is None


# ============================================================
# Test: harness — full integration
# ============================================================

def test_harness_runs_gen0_random_only(small_ohlcv_df, temp_output_dir):
    """Gen 0: 5 random candidates, gen 1+: 5 candidates even with 0 elites."""
    config = EvolutionConfig(
        candidates_per_gen=5,
        elite_count=1,
        random_injection=1,
        mutation_rate=0.5,
        crossover_rate=0.5,
        wall_time_seconds=60,
        max_generations=2,
        stagnation_generations=5,
        all_rejected_generations=3,
        base_seed=42,
        output_dir=temp_output_dir,
        experiment_id="exp_test_small",
        tp_pct=0.02,
    )
    harness = EvolutionHarness(config, small_ohlcv_df)
    summary = harness.run(resume=False)
    # 5 × 2 = 10 candidates
    assert summary.total_candidates_evaluated == 10
    assert summary.generations_completed == 2
    # Even if all rejected, all artifacts should exist
    assert Path(temp_output_dir, "run_summary.json").exists()
    assert Path(temp_output_dir, "generation_history.json").exists()
    assert Path(temp_output_dir, "leaderboards").exists()
    assert Path(temp_output_dir, "best_genomes").exists()
    assert Path(temp_output_dir, "rejection_reports").exists()


def test_harness_resume_continues_from_saved(small_ohlcv_df, temp_output_dir):
    """After a 1-gen run, resume should start at gen 1."""
    config = EvolutionConfig(
        candidates_per_gen=3,
        elite_count=1,
        random_injection=1,
        wall_time_seconds=60,
        max_generations=2,
        stagnation_generations=5,
        all_rejected_generations=3,
        base_seed=42,
        output_dir=temp_output_dir,
        experiment_id="exp_resume",
        tp_pct=0.02,
    )
    # First run: 1 generation
    h1 = EvolutionHarness(config, small_ohlcv_df)
    # Force 1-gen by mutating max_generations to 1
    config.max_generations = 1
    s1 = h1.run(resume=False)
    assert s1.generations_completed == 1
    assert s1.total_candidates_evaluated == 3

    # Second run: resume, should add 1 more gen
    config.max_generations = 2
    h2 = EvolutionHarness(config, small_ohlcv_df)
    s2 = h2.run(resume=True)
    assert s2.generations_completed == 2
    assert s2.total_candidates_evaluated == 6


def test_harness_hooks_fire(small_ohlcv_df, temp_output_dir):
    """Hooks should fire on every candidate and every gen."""
    config = EvolutionConfig(
        candidates_per_gen=3,
        elite_count=1,
        random_injection=1,
        wall_time_seconds=60,
        max_generations=1,
        stagnation_generations=5,
        all_rejected_generations=3,
        base_seed=42,
        output_dir=temp_output_dir,
        experiment_id="exp_hooks",
        tp_pct=0.02,
    )
    n_evals = {"n": 0}
    n_gens = {"n": 0}

    def on_cand(r):
        n_evals["n"] += 1

    def on_gen_start(i):
        n_gens["n"] += 1

    hooks = HarnessHooks(on_candidate_evaluated=on_cand, on_generation_start=on_gen_start)
    harness = EvolutionHarness(config, small_ohlcv_df, hooks=hooks)
    harness.run(resume=False)
    assert n_evals["n"] == 3
    assert n_gens["n"] == 1


def test_harness_no_elites_falls_back_to_random(small_ohlcv_df, temp_output_dir):
    """When gen 0 has 0 elites, gen 1 should still produce candidates_per_gen."""
    config = EvolutionConfig(
        candidates_per_gen=4,
        elite_count=1,
        random_injection=1,
        wall_time_seconds=60,
        max_generations=2,
        stagnation_generations=5,
        all_rejected_generations=3,
        base_seed=42,
        output_dir=temp_output_dir,
        experiment_id="exp_fallback",
        tp_pct=0.02,
    )
    harness = EvolutionHarness(config, small_ohlcv_df)
    summary = harness.run(resume=False)
    # Should still get 4 × 2 = 8 even if all rejected
    assert summary.total_candidates_evaluated == 8
