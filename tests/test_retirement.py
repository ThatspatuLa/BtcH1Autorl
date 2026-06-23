"""Tests for evolution.retirement (Stage 10 island retirement system).

Effective 2026-06-22 per Six's plan B extension: when an island's per-island
top fitness crosses retirement_threshold, the island is archived to
runs/retired_islands/ and the slot is re-seeded with a fresh bias.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from evolution.retirement import (
    BIAS_POOL,
    RetiredIslandRecord,
    RetirementPolicy,
    archive_island,
    check_for_retirements,
    list_retired_islands,
    pick_fresh_bias,
    retire_count,
    retire_summary,
)
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    SafetyGenome,
    SettingsOverrides,
    TpGenome,
    TpExitMethod,
)


# ---------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------

def _make_genome(
    genome_id: str = "G_test",
    grid_method: GridMethod = GridMethod.FIXED_PCT,
    grid_params: dict | None = None,
    allocation_method: AllocationMethod = AllocationMethod.EQUAL,
    tp_pct: float = 0.5,
    confirmations: list | None = None,
) -> CandidateGenome:
    """Build a minimal valid CandidateGenome for retirement tests.

    Default params match between calls — so two genomes with the same genome_id
    but different default params will be flagged as duplicates by dedup.
    Pass grid_params/grid_method/tp_pct to make a distinct genome.
    """
    if grid_params is None:
        grid_params = {"grid_pct": 0.5, "max_layers": 6}
    return CandidateGenome(
        genome_id=genome_id,
        dca_genome=DcaGenome(
            grid_method=grid_method,
            grid_params=grid_params,
            allocation_method=allocation_method,
            allocation_params={"base_notional": 100.0, "allocation_cap_pct": 0.10},
            combo_method="weighted_average",
            combo_params={},
            trigger_mode="price_only",
            confirmation_indicators=list(confirmations or []),
            indicator_params={},
            max_dca_layers=6,
        ),
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"tp_pct": tp_pct},
            sub_exits=[],
        ),
        safety_genome=SafetyGenome(),
        settings_overrides=SettingsOverrides(),
        lineage=LineageMetadata(),
    )


@pytest.fixture
def tmp_archive_dir(tmp_path):
    """Yield a temp dir for archive, clean up after."""
    archive_dir = tmp_path / "retired_islands"
    archive_dir.mkdir()
    yield str(archive_dir)
    # Cleanup
    if archive_dir.exists():
        shutil.rmtree(archive_dir, ignore_errors=True)


# ---------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------

class TestRetirementPolicy:
    def test_default_threshold_is_0_80(self):
        p = RetirementPolicy()
        assert p.threshold == 0.80

    def test_default_enabled(self):
        """RetirementPolicy defaults to enabled=True (it's a low-level policy
        object — the opt-in/opt-out is controlled by EvolutionConfig)."""
        p = RetirementPolicy()
        assert p.enabled is True

    def test_should_retire_below_threshold(self):
        p = RetirementPolicy(enabled=True, threshold=0.80)
        assert p.should_retire(0.79) is False
        assert p.should_retire(0.50) is False
        assert p.should_retire(0.0) is False

    def test_should_retire_at_threshold(self):
        p = RetirementPolicy(enabled=True, threshold=0.80)
        assert p.should_retire(0.80) is True

    def test_should_retire_above_threshold(self):
        p = RetirementPolicy(enabled=True, threshold=0.80)
        assert p.should_retire(0.85) is True
        assert p.should_retire(1.0) is True

    def test_disabled_policy_never_retires(self):
        p = RetirementPolicy(enabled=False, threshold=0.80)
        assert p.should_retire(0.95) is False
        assert p.should_retire(1.0) is False


# ---------------------------------------------------------------
# Bias pool tests
# ---------------------------------------------------------------

class TestBiasPool:
    def test_pool_has_at_least_16(self):
        # 8 original + 8 rotation = 16; we currently have 17 with the new ones
        assert len(BIAS_POOL) >= 16

    def test_all_biases_have_name(self):
        for b in BIAS_POOL:
            assert "name" in b
            assert isinstance(b["name"], str)

    def test_pick_fresh_bias_returns_dict(self):
        import random
        rng = random.Random(42)
        bias = pick_fresh_bias(rng)
        assert isinstance(bias, dict)
        assert "name" in bias

    def test_pick_fresh_bias_respects_exclusion(self):
        import random
        rng = random.Random(42)
        # Exclude everything in pool → should still return something
        all_names = [b["name"] for b in BIAS_POOL]
        bias = pick_fresh_bias(rng, exclude_recent=all_names)
        assert "name" in bias

    def test_pick_fresh_bias_excludes_partial(self):
        import random
        rng = random.Random(42)
        # Pick a name and exclude just that one — should get something else
        first = pick_fresh_bias(rng)
        exclude = [first["name"]]
        # Try 20 times — should never return the excluded name
        for _ in range(20):
            picked = pick_fresh_bias(rng, exclude_recent=exclude)
            assert picked["name"] != first["name"]


# ---------------------------------------------------------------
# archive_island() tests
# ---------------------------------------------------------------

class TestArchiveIsland:
    def test_archive_creates_directory_and_manifest(self, tmp_archive_dir):
        # Two DISTINCT genomes (different grid_pct) so dedup doesn't collapse them
        g1 = _make_genome("G_a1", grid_params={"grid_pct": 0.5, "max_layers": 6})
        g2 = _make_genome("G_a2", grid_params={"grid_pct": 0.8, "max_layers": 8})
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)

        rec = archive_island(
            policy=policy,
            cycle_id="20260622_152341",
            cycle_output_dir="runs/cycle_xyz",
            island_id=3,
            retired_at_gen=15,
            family_bias={"name": "trend"},
            per_island_top_fitness=0.82,
            elites=[(g1, 0.82), (g2, 0.78)],
            generations_evolved=15,
            per_island_history=[{"gen": 14, "best_fitness": 0.78}],
        )

        island_dir = Path(tmp_archive_dir) / "retired_20260622_152341_3_15"
        assert island_dir.exists()
        assert (island_dir / "manifest.json").exists()
        assert (island_dir / "top_3_elites.json").exists()
        assert (island_dir / "generation_history.json").exists()

        assert rec.island_id == 3
        assert rec.per_island_top_fitness == 0.82
        assert rec.n_elites_archived == 2
        assert "G_a1" in rec.top_3_elite_ids

    def test_archive_with_empty_history(self, tmp_archive_dir):
        g1 = _make_genome("G_a1")
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)

        rec = archive_island(
            policy=policy,
            cycle_id="20260622_X",
            cycle_output_dir="runs/cycle",
            island_id=1,
            retired_at_gen=5,
            family_bias={"name": "fixed_pct"},
            per_island_top_fitness=0.81,
            elites=[(g1, 0.81)],
            generations_evolved=5,
            per_island_history=None,
        )
        assert rec.n_elites_archived == 1
        # No history file should be written when per_island_history is None
        island_dir = Path(tmp_archive_dir) / "retired_20260622_X_1_5"
        assert not (island_dir / "generation_history.json").exists()

    def test_archive_serializable(self, tmp_archive_dir):
        g = _make_genome("G_z")
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        rec = archive_island(
            policy=policy,
            cycle_id="20260622_Z",
            cycle_output_dir="runs/cycle",
            island_id=8,
            retired_at_gen=20,
            family_bias={"name": "tight_dca", "max_dca_layers_cap": 8},
            per_island_top_fitness=0.85,
            elites=[(g, 0.85)],
            generations_evolved=20,
        )
        d = rec.to_dict()
        # Round-trip JSON
        s = json.dumps(d)
        d2 = json.loads(s)
        rec2 = RetiredIslandRecord.from_dict(d2)
        assert rec2.island_id == rec.island_id
        assert rec2.per_island_top_fitness == rec.per_island_top_fitness
        # Tuple-in-bias should be converted to list
        assert isinstance(rec2.family_bias.get("max_dca_layers_cap", None), (int, type(None)))


# ---------------------------------------------------------------
# check_for_retirements() tests
# ---------------------------------------------------------------

class _FakeGenRecord:
    """Minimal stand-in for GenerationRecord for retirement-check tests."""

    def __init__(
        self,
        per_island_best_fitness: dict[int, float],
        generation_index: int = 0,
        per_island_best_count: dict[int, int] | None = None,
    ):
        self.per_island_best_fitness = per_island_best_fitness
        self.generation_index = generation_index
        # Phase F: also accept per-island deployment-passing count.
        # Default to {island_id: 1} for each island in the fitness dict so
        # pre-Phase F tests still pass (assumed "at least one passed").
        self.per_island_best_count = per_island_best_count or {
            iid: 1 for iid in per_island_best_fitness
        }


class TestCheckForRetirements:
    def test_no_retirement_when_disabled(self, tmp_archive_dir):
        rec = _FakeGenRecord({1: 0.85, 2: 0.90})
        policy = RetirementPolicy(enabled=False, archive_dir=tmp_archive_dir)
        retired, assignments = check_for_retirements(
            policy=policy,
            cycle_id="x",
            cycle_output_dir="y",
            gen_record=rec,
            elites_by_island={},
            family_bias_by_island={},
        )
        assert retired == []
        assert assignments == {}

    def test_no_retirement_when_below_threshold(self, tmp_archive_dir):
        g = _make_genome("G_a")
        rec = _FakeGenRecord({1: 0.75, 2: 0.70})
        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)
        retired, assignments = check_for_retirements(
            policy=policy,
            cycle_id="x",
            cycle_output_dir="y",
            gen_record=rec,
            elites_by_island={1: [(g, 0.75)]},
            family_bias_by_island={1: {"name": "fixed_pct"}},
        )
        assert retired == []
        assert assignments == {}

    def test_retirement_when_at_threshold(self, tmp_archive_dir):
        g = _make_genome("G_a")
        rec = _FakeGenRecord({1: 0.80})
        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)
        retired, assignments = check_for_retirements(
            policy=policy,
            cycle_id="x",
            cycle_output_dir="y",
            gen_record=rec,
            elites_by_island={1: [(g, 0.80)]},
            family_bias_by_island={1: {"name": "fixed_pct"}},
        )
        assert len(retired) == 1
        assert retired[0].island_id == 1
        assert retired[0].per_island_top_fitness == 0.80
        # New bias should be assigned
        assert 1 in assignments
        assert "name" in assignments[1]

    def test_multiple_retirements_one_call(self, tmp_archive_dir):
        g1 = _make_genome("G_1")
        g2 = _make_genome("G_2")
        rec = _FakeGenRecord({1: 0.82, 3: 0.85, 5: 0.79, 7: 0.81})
        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)
        retired, assignments = check_for_retirements(
            policy=policy,
            cycle_id="x",
            cycle_output_dir="y",
            gen_record=rec,
            elites_by_island={
                1: [(g1, 0.82)],
                3: [(g2, 0.85)],
                5: [(_make_genome("G_5"), 0.79)],
                7: [(_make_genome("G_7"), 0.81)],
            },
            family_bias_by_island={
                1: {"name": "fixed_pct"},
                3: {"name": "atr"},
                5: {"name": "trend"},
                7: {"name": "tight_dca"},
            },
        )
        # Only 1, 3, 7 cross threshold (5 doesn't)
        assert len(retired) == 3
        retired_ids = sorted(r.island_id for r in retired)
        assert retired_ids == [1, 3, 7]
        assert sorted(assignments.keys()) == [1, 3, 7]

    def test_no_elites_means_no_retirement(self, tmp_archive_dir):
        rec = _FakeGenRecord({1: 0.85})  # crosses threshold
        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)
        retired, assignments = check_for_retirements(
            policy=policy,
            cycle_id="x",
            cycle_output_dir="y",
            gen_record=rec,
            elites_by_island={},   # empty
            family_bias_by_island={1: {"name": "fixed_pct"}},
        )
        # No elites → can't archive → skip
        assert retired == []

    def test_archive_dir_created(self, tmp_archive_dir):
        g = _make_genome("G_a")
        rec = _FakeGenRecord({1: 0.85})
        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)
        retired, _ = check_for_retirements(
            policy=policy,
            cycle_id="20260622_test",
            cycle_output_dir="runs/test",
            gen_record=rec,
            elites_by_island={1: [(g, 0.85)]},
            family_bias_by_island={1: {"name": "fixed_pct"}},
        )
        assert len(retired) == 1
        # Archive dir should now contain the retired island dir
        assert (Path(tmp_archive_dir) / "retired_20260622_test_1_0").exists()


# ---------------------------------------------------------------
# list_retired_islands() + retire_summary() tests
# ---------------------------------------------------------------

class TestListAndSummary:
    def test_list_empty(self, tmp_archive_dir):
        records = list_retired_islands(tmp_archive_dir)
        assert records == []

    def test_list_after_archive(self, tmp_archive_dir):
        g = _make_genome("G_x")
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        for i in range(3):
            archive_island(
                policy=policy,
                cycle_id=f"cycle_{i}",
                cycle_output_dir=f"runs/c{i}",
                island_id=i + 1,
                retired_at_gen=10 + i,
                family_bias={"name": f"bias_{i}"},
                per_island_top_fitness=0.80 + i * 0.01,
                elites=[(g, 0.80 + i * 0.01)],
                generations_evolved=10 + i,
            )
        records = list_retired_islands(tmp_archive_dir)
        assert len(records) == 3
        # Sorted by directory name (cycle_id)
        assert records[0].cycle_id == "cycle_0"

    def test_retire_count(self, tmp_archive_dir):
        g = _make_genome("G_x")
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        assert retire_count(tmp_archive_dir) == 0
        for i in range(2):
            archive_island(
                policy=policy,
                cycle_id=f"c_{i}",
                cycle_output_dir="r",
                island_id=i + 1,
                retired_at_gen=5,
                family_bias={"name": "x"},
                per_island_top_fitness=0.81,
                elites=[(g, 0.81)],
                generations_evolved=5,
            )
        assert retire_count(tmp_archive_dir) == 2

    def test_retire_summary(self, tmp_archive_dir):
        g = _make_genome("G_x")
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        archive_island(
            policy=policy,
            cycle_id="c1",
            cycle_output_dir="r",
            island_id=1,
            retired_at_gen=5,
            family_bias={"name": "fixed_pct"},
            per_island_top_fitness=0.81,
            elites=[(g, 0.81)],
            generations_evolved=5,
        )
        archive_island(
            policy=policy,
            cycle_id="c2",
            cycle_output_dir="r",
            island_id=2,
            retired_at_gen=10,
            family_bias={"name": "atr"},
            per_island_top_fitness=0.85,
            elites=[(g, 0.85)],
            generations_evolved=10,
        )
        summary = retire_summary(tmp_archive_dir)
        assert summary["n_retired"] == 2
        assert summary["max_top_fitness"] == 0.85
        assert summary["avg_top_fitness"] == pytest.approx(0.83)
        assert summary["by_family"]["fixed_pct"] == 1
        assert summary["by_family"]["atr"] == 1


# ---------------------------------------------------------------
# Integration: full archive → list → summary round-trip
# ---------------------------------------------------------------

class TestRetirementRoundTrip:
    def test_archive_list_summary_consistency(self, tmp_archive_dir):
        import random
        rng = random.Random(20260622)

        policy = RetirementPolicy(enabled=True, threshold=0.80, archive_dir=tmp_archive_dir)

        # Archive 5 islands
        g1 = _make_genome("G_1", grid_params={"grid_pct": 0.5, "max_layers": 6})
        g2 = _make_genome("G_2", grid_params={"grid_pct": 0.8, "max_layers": 8})
        for i in range(5):
            fitness = 0.80 + i * 0.01
            archive_island(
                policy=policy,
                cycle_id=f"cycle_{i}",
                cycle_output_dir=f"runs/c{i}",
                island_id=(i % 8) + 1,
                retired_at_gen=20,
                family_bias={"name": f"bias_{i}"},
                per_island_top_fitness=fitness,
                elites=[(g1, fitness), (g2, fitness - 0.01)],
                generations_evolved=20,
            )

        assert retire_count(tmp_archive_dir) == 5
        records = list_retired_islands(tmp_archive_dir)
        assert len(records) == 5

        # Summary
        summary = retire_summary(tmp_archive_dir)
        assert summary["n_retired"] == 5
        assert summary["max_top_fitness"] == pytest.approx(0.84)
        assert summary["avg_top_fitness"] == pytest.approx(0.82)

        # All elites preserved
        for rec in records:
            assert rec.n_elites_archived == 2


# ---------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------

class TestConfigIntegration:
    def test_config_has_retirement_fields(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig()
        assert hasattr(c, "retirement_enabled")
        assert hasattr(c, "retirement_threshold")
        assert hasattr(c, "retirement_archive_dir")
        assert hasattr(c, "max_retired_per_cycle")

    def test_config_retirement_defaults(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig()
        assert c.retirement_enabled is False  # off by default
        assert c.retirement_threshold == 0.80
        assert c.retirement_archive_dir == "runs/retired_islands"

    def test_config_can_enable_retirement(self):
        from evolution.config import EvolutionConfig
        c = EvolutionConfig(retirement_enabled=True, retirement_threshold=0.75)
        assert c.retirement_enabled is True
        assert c.retirement_threshold == 0.75
