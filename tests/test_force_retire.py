"""Tests for force-retire on per-island stagnation.

Plan: 2026-06-24 per Six's spec. Force-retire kills individual dead islands
(their stagnation_counter >= force_retire_after_gens AND top fitness is
below force_retire_min_fitness) so the rest of the population can keep
improving without dead islands holding slots.

This complements `_check_stagnation` (terminates WHOLE run when ALL
islands stagnate) and `_check_retirement` (archives on fitness ≥ threshold).

Tests exercise `_check_force_retire` in isolation with a mock harness.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def mock_harness_factory(tmp_path, monkeypatch):
    """Build an EvolutionHarness-like object with just enough state for force-retire tests."""
    monkeypatch.chdir(tmp_path)

    from evolution.config import EvolutionConfig
    from evolution.harness import EvolutionHarness
    import pandas as pd
    import random

    # Minimal DataFrame — just needs to construct
    df = pd.DataFrame({"close": [100.0] * 10})

    def make_harness(
        force_retire_after_gens: int = 8,
        force_retire_min_fitness: float = 0.70,
        retirement_enabled: bool = True,
        retirement_threshold: float = 0.80,
    ) -> EvolutionHarness:
        cfg = EvolutionConfig(
            output_dir=str(tmp_path / "out"),
            retirement_enabled=retirement_enabled,
            retirement_threshold=retirement_threshold,
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
        # Don't init the heavy evaluator — replace with a stub
        h.evaluator = None
        # Pre-seed 4 islands with biases
        h._init_island_family_bias()
        return h

    yield make_harness

    if (tmp_path / "retired").exists():
        shutil.rmtree(tmp_path / "retired", ignore_errors=True)


class _FakeGenRecord:
    """Minimal stand-in for GenerationRecord — just what _check_force_retire reads."""
    def __init__(
        self,
        leaderboard: list[dict] | None = None,
        per_island_best_fitness: dict[int, float] | None = None,
        per_island_best_count: dict[int, int] | None = None,
        per_island_elite_count: dict[int, int] | None = None,
        generation_index: int = 5,
    ):
        self.leaderboard = leaderboard or []
        self.per_island_best_fitness = per_island_best_fitness or {}
        self.per_island_best_count = per_island_best_count or {}
        self.per_island_elite_count = per_island_elite_count or {}
        self.generation_index = generation_index
        self.retired_islands = []
        self.island_bias_overrides = {}


class TestForceRetireDisabled:
    def test_returns_empty_when_retirement_disabled(self, mock_harness_factory):
        h = mock_harness_factory(retirement_enabled=False)
        h._island_stagnation_counter = {0: 100, 1: 100}
        h._island_best_fitness = {0: 0.5, 1: 0.5}
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert retired == []
        assert overrides == {}


class TestForceRetireTriggerConditions:
    def test_fires_when_stagnation_at_threshold(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8, force_retire_min_fitness=0.70)
        h._island_stagnation_counter = {0: 8}
        h._island_best_fitness = {0: 0.50}  # well below 0.70
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        # island 0 should have been force-retired (bias override present;
        # archive only created if elites exist — none here)
        assert 0 in overrides

    def test_skipped_when_stagnation_below_threshold(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 7}  # one short
        h._island_best_fitness = {0: 0.50}
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert retired == []
        assert overrides == {}

    def test_skipped_when_fitness_above_min(self, mock_harness_factory):
        """Near-passing islands should be given a chance — don't kill them."""
        h = mock_harness_factory(force_retire_after_gens=8, force_retire_min_fitness=0.70)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.75}  # above 0.70
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert retired == []
        assert overrides == {}

    def test_skipped_when_recently_reseeded(self, mock_harness_factory):
        """3-gen grace period after a re-seed."""
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {0: 8}  # re-seeded at gen 8
        rec = _FakeGenRecord(generation_index=10)
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        # 10 - 8 = 2 < 3, grace period still applies
        assert retired == []
        assert overrides == {}

    def test_fires_after_grace_period(self, mock_harness_factory):
        """Once 3+ gens have passed since re-seed, force-retire is allowed again."""
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {0: 5}  # re-seeded at gen 5
        rec = _FakeGenRecord(generation_index=10)  # 10-5 = 5 >= 3
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert 0 in overrides


class TestForceRetireStateUpdates:
    def test_resets_stagnation_counter(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        h._check_force_retire(rec, 10, [], random.Random())
        # After force-retire, counter should be reset to 0
        assert h._island_stagnation_counter[0] == 0

    def test_clears_island_best_fitness(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        h._check_force_retire(rec, 10, [], random.Random())
        # After force-retire, fitness should be cleared (fresh start)
        assert 0 not in h._island_best_fitness

    def test_records_force_retired_at_gen(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        h._check_force_retire(rec, 10, [], random.Random())
        assert h._force_retired_at_gen[0] == 10

    def test_assigns_fresh_bias(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        # Manually seed island 0 with a known bias (init doesn't always populate)
        h._island_family_bias[0] = {"name": "atr"}
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        from evolution.retirement import BIAS_POOL
        h._check_force_retire(rec, 10, [], random.Random())
        new_bias_name = h._island_family_bias[0]["name"]
        assert new_bias_name in [b["name"] for b in BIAS_POOL]  # came from the pool


class TestForceRetireMultipleIslands:
    def test_force_retires_multiple_at_once(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        # Two islands stagnated, one not
        h._island_stagnation_counter = {0: 100, 1: 100, 2: 2}
        h._island_best_fitness = {0: 0.50, 1: 0.60, 2: 0.65}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        _, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert 0 in overrides
        assert 1 in overrides
        assert 2 not in overrides

    def test_independent_island_state(self, mock_harness_factory):
        """Force-retiring one island shouldn't affect another."""
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {0: 100, 1: 1}
        h._island_best_fitness = {0: 0.50, 1: 0.50}
        h._force_retired_at_gen = {}
        rec = _FakeGenRecord()
        import random
        h._check_force_retire(rec, 10, [], random.Random())
        # island 1 untouched
        assert h._island_stagnation_counter[1] == 1
        assert h._island_best_fitness[1] == 0.50


class TestForceRetireEdgeCases:
    def test_zero_threshold_disables(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=0)
        h._island_stagnation_counter = {0: 100}
        h._island_best_fitness = {0: 0.50}
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert retired == []
        assert overrides == {}

    def test_no_islands_no_op(self, mock_harness_factory):
        h = mock_harness_factory(force_retire_after_gens=8)
        h._island_stagnation_counter = {}
        h._island_best_fitness = {}
        rec = _FakeGenRecord()
        import random
        retired, overrides = h._check_force_retire(rec, 10, [], random.Random())
        assert retired == []
        assert overrides == {}
