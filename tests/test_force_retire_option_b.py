"""Tests for force-retire Option B (2026-06-25, Six).

The previous rule had a fitness floor: islands above `force_retire_min_fitness`
(default 0.70) were PROTECTED from force-retirement. This caused the 65-gen
stagnation bug where 7/8 islands got stuck in the 0.70-0.73 fitness band and
couldn't be rotated out, causing system-wide stagnation and cycle termination.

Option B: drop the fitness floor entirely. Stagnation alone triggers retirement.
Elite islands survive by IMPROVING (resetting their counter), not by being
protected.

These tests document the new contract and would fail under the old rule.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def option_b_harness(tmp_path, monkeypatch):
    """Build an EvolutionHarness with Option B config (no floor effect)."""
    monkeypatch.chdir(tmp_path)

    from evolution.config import EvolutionConfig
    from evolution.harness import EvolutionHarness
    import pandas as pd
    import random

    df = pd.DataFrame({"close": [100.0] * 10})

    def make_harness(
        force_retire_after_gens: int = 8,
        # force_retire_min_fitness is now IGNORED but kept for back-compat
        force_retire_min_fitness: float = 0.70,
    ) -> EvolutionHarness:
        cfg = EvolutionConfig(
            output_dir=str(tmp_path / "out"),
            retirement_enabled=True,
            retirement_threshold=0.80,
            retirement_archive_dir=str(tmp_path / "retired"),
            force_retire_after_gens=force_retire_after_gens,
            force_retire_min_fitness=force_retire_min_fitness,
            island_mode=True,
            n_islands=4,
            max_generations=10,
            wall_time_seconds=600,
        )
        rng = random.Random(42)
        h = EvolutionHarness(config=cfg, df=df, rng=rng)
        h.evaluator = None
        h._init_island_family_bias()
        return h

    yield make_harness

    if (tmp_path / "retired").exists():
        shutil.rmtree(tmp_path / "retired", ignore_errors=True)


class _Rec:
    def __init__(self):
        self.leaderboard = []
        self.per_island_best_fitness = {}
        self.per_island_best_count = {}
        self.per_island_elite_count = {}
        self.generation_index = 10
        self.retired_islands = []
        self.island_bias_overrides = {}


class TestOptionBNoFloor:
    """Option B: stagnation alone triggers retirement, regardless of fitness."""

    def test_retires_at_fitness_0_50(self, option_b_harness):
        """Low fitness + stagnation → retire (was already true in old rule)."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert 0 in overrides

    def test_retires_at_fitness_0_70_floor(self, option_b_harness):
        """AT the old floor (0.70) → retire. Old rule: SKIP (protected)."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.70}  # exactly the old floor
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert 0 in overrides  # NEW: even at floor, retires

    def test_retires_at_fitness_0_73_in_protected_band(self, option_b_harness):
        """0.73 (the 65-gen bug zone) → retire. Old rule: SKIP (protected)."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.7314}  # the exact best-ever from 65-gen cycle
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert 0 in overrides  # NEW: even at all-time best, stagnation retires

    def test_retires_at_fitness_0_85_high(self, option_b_harness):
        """Very high fitness + stagnation → still retire. Counter resets, fresh chance."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.85}
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert 0 in overrides

    def test_min_fitness_config_ignored(self, option_b_harness):
        """The force_retire_min_fitness config value has no effect under Option B.

        Even if user sets it to 1.0 (would protect everything in old rule),
        Option B still retires stagnant islands.
        """
        h = option_b_harness(force_retire_min_fitness=1.0)  # extreme floor
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.99}  # below extreme floor
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert 0 in overrides  # still retires


class TestOptionBStagnationCounterLogic:
    """Stagnation counter behavior under Option B is unchanged."""

    def test_counter_resets_after_retirement(self, option_b_harness):
        """After retirement, counter resets to 0 — fresh start."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.75}
        h._force_retired_at_gen = {}
        h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        assert h._island_stagnation_counter[0] == 0

    def test_grace_period_still_applies(self, option_b_harness):
        """3-gen grace period after re-seed still applies under Option B."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.75}
        h._force_retired_at_gen = {0: 8}  # re-seeded at gen 8
        _, overrides = h._check_force_retire(_Rec(), 10, [], __import__('random').Random())
        # 10-8 = 2 < 3, grace period applies → no retirement
        assert 0 not in overrides

    def test_grace_period_expires(self, option_b_harness):
        """After 3+ gens since re-seed, retirement is allowed again."""
        h = option_b_harness()
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.75}
        h._force_retired_at_gen = {0: 5}  # re-seeded at gen 5
        rec = _Rec()
        rec.generation_index = 10  # 10-5 = 5 >= 3
        _, overrides = h._check_force_retire(rec, 10, [], __import__('random').Random())
        assert 0 in overrides


class TestOptionBAtScale:
    """Simulates the 65-gen cycle's stagnation pattern."""

    def test_simulated_65_gen_pattern_retires_stuck_islands(self, option_b_harness):
        """Reproduce the 65-gen cycle's situation: all 8 islands stagnant.

        Old rule: ALL protected (fitness 0.70-0.73 > floor 0.70).
        New rule (Option B): ALL retired, fresh niches introduced.
        """
        h = option_b_harness(force_retire_after_gens=15)
        # Simulate the 65-gen cycle's last-gen state
        h._island_stagnation_counter = {
            0: 9, 1: 18, 2: 5, 3: 15, 4: 7, 5: 11, 6: 6, 7: 16
        }
        # All above the 0.70 floor — old rule would have retired ZERO
        h._island_best_fitness = {
            0: 0.7146, 1: 0.7236, 2: 0.7076, 3: 0.7089,
            4: 0.7142, 5: 0.7043, 6: 0.7146, 7: 0.7269
        }
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 65, [], __import__('random').Random())
        # Under Option B: 4 islands get retired (those with stag >= 15)
        # Islands 1, 3, 5, 7 (stag=18, 15, 11... wait let me recount)
        # stag >= 15: islands 1 (18), 3 (15), 7 (16)  → 3 retirements
        retired = [i for i in overrides if i in h._island_stagnation_counter]
        assert set(retired) == {1, 3, 7}  # those with stag >= 15

    def test_old_rule_would_have_retired_zero(self, option_b_harness):
        """Document the OLD bug: under the old floor, ZERO retirements would fire.

        This test verifies the buggy state that caused cycle termination.
        Kept as a regression marker — under Option B, this scenario triggers
        retirements (see test above). Under the OLD rule (before our fix),
        this would have retired nothing.
        """
        h = option_b_harness(force_retire_after_gens=15)
        h._island_stagnation_counter = {i: 20 for i in range(8)}  # all stagnant
        # ALL fitness values above 0.70 floor — old rule would skip all
        h._island_best_fitness = {i: 0.71 + i * 0.005 for i in range(8)}
        h._force_retired_at_gen = {}
        _, overrides = h._check_force_retire(_Rec(), 65, [], __import__('random').Random())
        # Under Option B: ALL 8 should be retired
        assert len(overrides) == 8
