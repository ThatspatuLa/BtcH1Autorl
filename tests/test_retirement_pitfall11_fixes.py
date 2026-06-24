"""Tests for the Pitfall #11 retirement logic fixes (2026-06-25).

Three fixes in one PR:
1. Duplicate-defense: force-retire skips islands already retired by fitness
   path in the SAME generation (prevents duplicate archive events).
2. Lowered threshold: retirement_threshold default 0.80 → 0.75 (cap-10 era).
3. Grace period parity: fitness retirement now respects the same 3-gen grace
   period as force-retire (prevents immediate re-fire after reseed).

Six's recommendation 2026-06-25: "do what you recommend."
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from evolution.config import EvolutionConfig
from evolution.persistence import GenerationRecord
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    TpExitMethod,
    TpGenome,
)


@pytest.fixture
def tmp_archive_dir():
    """Provide a clean temp dir for archive writes; auto-cleanup after."""
    d = Path(tempfile.mkdtemp(prefix="test_retirement_pf11_"))
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_record(
    gen_idx=1,
    per_island_best_fitness=None,
    per_island_stagnation_counter=None,
    retired_islands=None,
    leaderboard=None,
):
    """Helper to build a GenerationRecord with sensible defaults."""
    return GenerationRecord(
        generation_index=gen_idx,
        started_at=time.time() - 60,
        ended_at=time.time(),
        n_candidates=500,
        n_rejected=100,
        n_passed=400,
        n_deployment_passing=10,
        best_fitness=0.68,
        median_fitness=0.62,
        best_candidate_id="",
        best_genome_id="",
        wall_time_seconds_used=60.0,
        rejection_reasons={},
        per_island_best_fitness=per_island_best_fitness or {},
        per_island_stagnation_counter=per_island_stagnation_counter or {},
        retired_islands=retired_islands or [],
        leaderboard=leaderboard or [],
    )


def _make_genome(gid: str, iid: int) -> CandidateGenome:
    """Build a minimal CandidateGenome with the given island tag."""
    return CandidateGenome(
        genome_id=gid,
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"pct": 0.01, "max_layers": 5, "tp_pct": 0.005, "cooldown_candles": 0},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=5,
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.005}),
        lineage=LineageMetadata(mutation_ops=[
            {"op": "island_assign", "island_id": iid},
            {"op": "mutate", "parent_id": "p1"},
        ]),
    )


class _MockConfig:
    """Bare-minimum config object matching the harness's attribute access."""
    retirement_enabled: bool = True
    retirement_threshold: float = 0.75
    retirement_archive_dir: str = "runs/retired_islands"
    max_retired_per_cycle: int = 999
    force_retire_after_gens: int = 8
    force_retire_min_fitness: float = 0.70
    output_dir: str = "/tmp/test_retirement_archive"


# ============================================================
# Fix 2: Lowered threshold default
# ============================================================

def test_default_retirement_threshold_is_075():
    """Pitfall #11 fix #2: lowered default 0.80 → 0.75 for cap-10 era."""
    cfg = EvolutionConfig()
    assert cfg.retirement_threshold == 0.75, (
        f"Expected default retirement_threshold=0.75, got {cfg.retirement_threshold}"
    )


def test_config_from_dict_uses_075_default():
    """When no retirement_threshold key is present, default is 0.75."""
    cfg = EvolutionConfig.from_dict({})
    assert cfg.retirement_threshold == 0.75


def test_config_from_dict_respects_explicit_threshold():
    """Backward compat: explicit threshold value is honored."""
    cfg = EvolutionConfig.from_dict({"retirement_threshold": 0.85})
    assert cfg.retirement_threshold == 0.85


# ============================================================
# Fix 3: Fitness retirement grace period
# ============================================================

def _make_harness_with_grace(iid: int, reseeded_at_gen: int | None, archive_dir: str) -> "EvolutionHarness":
    """Build a bare-minimum harness with controlled grace state for island iid."""
    from evolution.harness import EvolutionHarness

    harness = EvolutionHarness.__new__(EvolutionHarness)
    harness._island_family_bias = {iid: {"name": "fixed_pct"}}
    harness._island_best_fitness = {iid: 0.80}
    harness._island_stagnation_counter = {iid: 0}
    harness._force_retired_at_gen = (
        {iid: reseeded_at_gen} if reseeded_at_gen is not None else {}
    )
    harness._any_retired_at_gen = {}
    harness._recent_bias_names = []
    harness._retired_records = []
    harness._cycle_id = "test_cycle"
    harness.config = _MockConfig()
    harness.config.retirement_archive_dir = archive_dir
    return harness


def test_fitness_retirement_respects_grace_period(tmp_archive_dir):
    """An island reseeded at gen 5 should NOT be fitness-retired at gen 6."""
    from evolution.harness import EvolutionHarness
    import random

    harness = _make_harness_with_grace(iid=1, reseeded_at_gen=5, archive_dir=tmp_archive_dir)

    record = _make_record(
        gen_idx=6,
        per_island_best_fitness={1: 0.80},
    )

    cand = _make_genome("gen1", iid=1)
    record.leaderboard = [{"genome_id": "gen1", "discovery_fitness": 0.80}]
    candidates = [cand]

    retired_dicts, bias_overrides = harness._check_retirement(
        record, candidates, random.Random(42),
    )

    # Should be empty due to grace period
    assert retired_dicts == [], (
        f"Island 1 should NOT retire during 3-gen grace period, "
        f"got {retired_dicts}"
    )
    assert bias_overrides == {}


def test_fitness_retirement_works_after_grace_period(tmp_archive_dir):
    """After 3+ gens since reseed, fitness retirement fires normally."""
    from evolution.harness import EvolutionHarness
    import random

    harness = _make_harness_with_grace(iid=1, reseeded_at_gen=5, archive_dir=tmp_archive_dir)

    record = _make_record(
        gen_idx=8,
        per_island_best_fitness={1: 0.80},
    )
    # Phase F7: deployment_passing_count must be >= min_deployment_passing (1)
    record.per_island_best_count = {1: 5}

    cand = _make_genome("gen1", iid=1)
    record.leaderboard = [{"genome_id": "gen1", "discovery_fitness": 0.80}]
    candidates = [cand]

    retired_dicts, bias_overrides = harness._check_retirement(
        record, candidates, random.Random(42),
    )

    # Should retire now (grace period over, fitness above 0.75)
    assert len(retired_dicts) == 1
    assert retired_dicts[0]["island_id"] == 1


def test_fitness_retirement_island_never_retired_works_immediately(tmp_archive_dir):
    """An island with NO prior reseed history can retire on the first attempt."""
    from evolution.harness import EvolutionHarness
    import random

    harness = _make_harness_with_grace(iid=1, reseeded_at_gen=None, archive_dir=tmp_archive_dir)

    record = _make_record(
        gen_idx=10,
        per_island_best_fitness={1: 0.80},
    )
    record.per_island_best_count = {1: 5}

    cand = _make_genome("gen1", iid=1)
    record.leaderboard = [{"genome_id": "gen1", "discovery_fitness": 0.80}]
    candidates = [cand]

    retired_dicts, bias_overrides = harness._check_retirement(
        record, candidates, random.Random(42),
    )

    assert len(retired_dicts) == 1


# ============================================================
# Fix 1: Duplicate-defense between fitness and force-retire
# ============================================================

def _make_harness_force_retire_setup():
    """Build harness where I1 fitness-retired this gen AND force-eligible."""
    from evolution.harness import EvolutionHarness

    harness = EvolutionHarness.__new__(EvolutionHarness)
    harness._island_family_bias = {
        1: {"name": "fixed_pct"},
        2: {"name": "atr"},
    }
    harness._island_best_fitness = {1: 0.85, 2: 0.65}  # I2 below 0.70
    harness._island_stagnation_counter = {1: 10, 2: 10}  # Both force-eligible
    harness._force_retired_at_gen = {}
    harness._any_retired_at_gen = {1: 6}  # Only I1 fitness-retired this gen
    harness._recent_bias_names = []
    harness._retired_records = []
    harness._cycle_id = "test_cycle"
    harness.config = _MockConfig()
    return harness


def test_force_retire_skips_island_already_retired_by_fitness():
    """Force-retire must NOT archive an island that fitness path already did."""
    import random

    harness = _make_harness_force_retire_setup()

    record = _make_record(
        gen_idx=6,
        per_island_best_fitness={1: 0.85, 2: 0.65},
        per_island_stagnation_counter={1: 10, 2: 10},
    )

    force_dicts, force_bias_overrides = harness._check_force_retire(
        record, gen_idx=6, candidates=[], rng=random.Random(42),
    )

    # Should be empty: fitness path already retired I1 this gen
    assert force_dicts == [], (
        f"Force-retire should skip I1 (already retired by fitness this gen), "
        f"got {force_dicts}"
    )


def test_force_retire_fires_for_different_island():
    """Force-retire still works for islands NOT already retired by fitness."""
    import random

    harness = _make_harness_force_retire_setup()

    record = _make_record(
        gen_idx=6,
        per_island_best_fitness={1: 0.85, 2: 0.65},
        per_island_stagnation_counter={1: 10, 2: 10},
    )

    # Build elite for I2 (so force-retire has something to archive).
    # Force-retire reads from gen_record.leaderboard, not candidates list.
    cand2 = _make_genome("gen2", iid=2)
    record.leaderboard = [
        {"genome_id": "gen2", "discovery_fitness": 0.65}
    ]
    candidates = [cand2]

    force_dicts, force_bias_overrides = harness._check_force_retire(
        record, gen_idx=6, candidates=candidates, rng=random.Random(42),
    )

    # Should fire for I2 only
    assert len(force_dicts) == 1
    assert force_dicts[0]["island_id"] == 2
    assert force_dicts[0]["reason"] == "stagnation_force"
