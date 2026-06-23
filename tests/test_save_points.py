"""Tests for checkpoint save/load (every-N-min snapshots for restart safety).

Plan: 2026-06-24 per Six. Added so a computer restart mid-cycle can resume
from the latest snapshot instead of starting the cycle over.

Checkpoints live under <project_root>/checkpoints/. They contain:
- cycle_id
- generation_index
- wall_time_used
- per_island_best_fitness
- per_island_stagnation_counter
- retired_so_far (list of dicts)
- rng_state (optional, for reproducibility)
- saved_at (unix timestamp)

This file tests the persistence layer in isolation (no harness needed).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest


@pytest.fixture
def clean_checkpoint_dir(tmp_path, monkeypatch):
    """Point CHECKPOINT_ROOT at a tmp dir so tests don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    from evolution import persistence
    persistence.CHECKPOINT_ROOT = tmp_path / "checkpoints"
    yield persistence.CHECKPOINT_ROOT
    if persistence.CHECKPOINT_ROOT.exists():
        shutil.rmtree(persistence.CHECKPOINT_ROOT, ignore_errors=True)


class TestSaveCheckpoint:
    def test_writes_file_to_checkpoint_dir(self, clean_checkpoint_dir):
        from evolution.persistence import save_checkpoint
        path = save_checkpoint(
            cycle_id="20260624_120000",
            gen_idx=5,
            wall_time_used=600.0,
            per_island_best_fitness={0: 0.65, 1: 0.71},
            per_island_stagnation_counter={0: 3, 1: 1},
            retired_so_far=[],
        )
        assert path.exists()
        assert path.name.startswith("checkpoint_20260624_120000_")
        assert path.suffix == ".json"

    def test_writes_latest_json_alias(self, clean_checkpoint_dir):
        from evolution.persistence import save_checkpoint, load_latest_checkpoint
        save_checkpoint(
            cycle_id="20260624_120000",
            gen_idx=3,
            wall_time_used=300.0,
            per_island_best_fitness={0: 0.55},
            per_island_stagnation_counter={0: 1},
            retired_so_far=[],
        )
        latest = load_latest_checkpoint()
        assert latest is not None
        assert latest["cycle_id"] == "20260624_120000"
        assert latest["generation_index"] == 3
        assert latest["wall_time_used"] == 300.0

    def test_round_trips_per_island_data(self, clean_checkpoint_dir):
        from evolution.persistence import save_checkpoint, load_latest_checkpoint
        save_checkpoint(
            cycle_id="20260624_120000",
            gen_idx=10,
            wall_time_used=1200.0,
            per_island_best_fitness={0: 0.65, 1: 0.71, 2: 0.80},
            per_island_stagnation_counter={0: 3, 1: 1, 2: 0},
            retired_so_far=[{"island_id": 2, "reason": "fitness_>=0.80"}],
        )
        latest = load_latest_checkpoint()
        # JSON deserialises int dict keys as strings — coerce for comparison
        assert {int(k): v for k, v in latest["per_island_best_fitness"].items()} == {0: 0.65, 1: 0.71, 2: 0.80}
        assert {int(k): v for k, v in latest["per_island_stagnation_counter"].items()} == {0: 3, 1: 1, 2: 0}
        assert len(latest["retired_so_far"]) == 1
        assert latest["retired_so_far"][0]["island_id"] == 2

    def test_saves_rng_state(self, clean_checkpoint_dir):
        from evolution.persistence import save_checkpoint, load_latest_checkpoint
        import random
        rng = random.Random(42)
        rng_state = rng.getstate()
        save_checkpoint(
            cycle_id="20260624_120000",
            gen_idx=7,
            wall_time_used=700.0,
            per_island_best_fitness={},
            per_island_stagnation_counter={},
            retired_so_far=[],
            rng_state=rng_state,
        )
        latest = load_latest_checkpoint()
        assert latest["rng_state"] is not None
        assert latest["rng_state"][0] == 3  # version tuple


class TestLoadLatestCheckpoint:
    def test_returns_none_when_no_checkpoint(self, clean_checkpoint_dir):
        from evolution.persistence import load_latest_checkpoint
        assert load_latest_checkpoint() is None

    def test_cycle_id_filter(self, clean_checkpoint_dir):
        from evolution.persistence import save_checkpoint, load_latest_checkpoint
        save_checkpoint(
            cycle_id="OLD_CYCLE",
            gen_idx=2,
            wall_time_used=200.0,
            per_island_best_fitness={},
            per_island_stagnation_counter={},
            retired_so_far=[],
        )
        # Different cycle_id → None
        assert load_latest_checkpoint(cycle_id="NEW_CYCLE") is None
        # Matching cycle_id → loaded
        assert load_latest_checkpoint(cycle_id="OLD_CYCLE") is not None

    def test_corrupt_latest_returns_none(self, clean_checkpoint_dir):
        from evolution.persistence import load_latest_checkpoint
        clean_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (clean_checkpoint_dir / "latest.json").write_text("{not valid json")
        assert load_latest_checkpoint() is None


class TestCheckpointAge:
    def test_age_is_zero_for_fresh_checkpoint(self, clean_checkpoint_dir):
        from evolution.persistence import (
            save_checkpoint,
            load_latest_checkpoint,
            checkpoint_age_seconds,
        )
        save_checkpoint(
            cycle_id="20260624_120000",
            gen_idx=1,
            wall_time_used=60.0,
            per_island_best_fitness={},
            per_island_stagnation_counter={},
            retired_so_far=[],
        )
        latest = load_latest_checkpoint()
        age = checkpoint_age_seconds(latest)
        # Should be very small (< 2 seconds for a fresh save)
        assert age < 2.0
        assert age >= 0.0

    def test_age_reflects_old_timestamp(self, clean_checkpoint_dir):
        from evolution.persistence import checkpoint_age_seconds
        # 1000s-old checkpoint
        payload = {"saved_at": time.time() - 1000.0, "cycle_id": "X"}
        age = checkpoint_age_seconds(payload)
        assert 999.0 <= age <= 1001.0


class TestListCheckpoints:
    def test_empty_when_no_checkpoints(self, clean_checkpoint_dir):
        from evolution.persistence import list_checkpoints
        assert list_checkpoints() == []

    def test_lists_checkpoints_newest_first(self, clean_checkpoint_dir):
        from evolution.persistence import list_checkpoints, save_checkpoint
        # Save three with different cycle_ids to force distinct filenames
        # (otherwise same-minute saves overwrite each other — by design)
        save_checkpoint("C1", 1, 100.0, {}, {}, [])
        save_checkpoint("C2", 2, 200.0, {}, {}, [])
        save_checkpoint("C3", 3, 300.0, {}, {}, [])
        listing = list_checkpoints()
        assert len(listing) == 3
        # Newest first by saved_at
        assert listing[0]["generation_index"] == 3
        assert listing[-1]["generation_index"] == 1

    def test_same_minute_saves_overwrite(self, clean_checkpoint_dir):
        """Saves within the same minute produce one file (cheap dedup)."""
        from evolution.persistence import list_checkpoints, save_checkpoint
        save_checkpoint("C", 1, 100.0, {}, {}, [])
        save_checkpoint("C", 2, 200.0, {}, {}, [])  # overwrites previous
        listing = list_checkpoints(cycle_id="C")
        # Only one file, but latest.json has gen 2
        assert len(listing) == 1
        assert listing[0]["generation_index"] == 2

    def test_filter_by_cycle_id(self, clean_checkpoint_dir):
        from evolution.persistence import list_checkpoints, save_checkpoint
        save_checkpoint("C1", 1, 100.0, {}, {}, [])
        save_checkpoint("C2", 1, 100.0, {}, {}, [])
        c1_only = list_checkpoints(cycle_id="C1")
        assert len(c1_only) == 1
        assert c1_only[0]["cycle_id"] == "C1"


class TestEvolutionConfigCheckpointFields:
    def test_defaults(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig()
        assert c.checkpoint_interval_minutes == 20
        assert c.force_retire_after_gens == 8
        assert c.force_retire_min_fitness == 0.70

    def test_can_override(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig(
            checkpoint_interval_minutes=10,
            force_retire_after_gens=5,
            force_retire_min_fitness=0.80,
        )
        assert c.checkpoint_interval_minutes == 10
        assert c.force_retire_after_gens == 5
        assert c.force_retire_min_fitness == 0.80

    def test_round_trip_through_dict(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig(checkpoint_interval_minutes=15)
        d = c.to_dict()
        assert d["checkpoint_interval_minutes"] == 15
        c2 = EvolutionConfig.from_dict(d)
        assert c2.checkpoint_interval_minutes == 15
