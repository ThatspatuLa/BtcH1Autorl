"""Tests for per-island best genome persistence + independence (Fix 2026-06-25).

Pitfall #12 — "islands-converged-bug": All 8 islands were being seeded every
generation from the SAME global best_genome, causing the entire population to
collapse onto a single peak by gen ~10. The fix:

1. Each generation persists a per-island top-1 genome to
   `best_genomes/per_island_gen_<NNNN>_island_<II>.json`.
2. `_load_per_island_elites` loads each island's OWN previous-gen #1
   instead of the global #1.

This file covers:
- save/load round-trip for per-island best genomes
- Independent genomes per island (different dca_genome per file)
- Hydration back into CandidateGenome objects with correct grid methods
- Migration layering still works on top of per-island elites
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def clean_output_dir(tmp_path):
    """Fresh output_dir for each test."""
    out = tmp_path / "runs"
    out.mkdir(parents=True, exist_ok=True)
    yield out
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def tiny_df():
    """Tiny DataFrame for harness init (8 rows of BTC prices)."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=8, freq="1h"),
        "open": [40000.0] * 8,
        "high": [40100.0] * 8,
        "low": [39900.0] * 8,
        "close": [40050.0] * 8,
        "volume": [100.0] * 8,
    })


def _make_genome_dict(
    genome_id: str,
    grid_method: str = "fixed_pct",
    max_dca_layers: int = 3,
    bias_name: str | None = None,
) -> dict:
    """Build a minimal CandidateGenome dict for testing persistence."""
    return {
        "genome_id": genome_id,
        "dca_genome": {
            "grid_method": grid_method,
            "grid_params": {"grid_pct": 0.005},
            "allocation_method": "equal",
            "allocation_params": {},
            "max_dca_layers": max_dca_layers,
            "confirmation_indicators": [],
            "indicator_params": {},
        },
        "tp_genome": {
            "exit_method": "fixed",
            "exit_params": {"tp_pct": 0.005},
        },
        "settings_overrides": {},
        "lineage": {
            "parent_a_id": None,
            "parent_b_id": None,
            "generation_index": 1,
            "mutation_seed": 42,
            "mutation_ops": (
                [{"op": "island_assign", "island_id": 1}]
                if bias_name is None
                else [
                    {"op": "island_assign", "island_id": 1},
                    {"op": "family_bias", "name": bias_name},
                ]
            ),
            "created_at": 1700000000.0,
        },
    }


def _make_prev_gen(generation_index: int) -> "GenerationRecord":
    """Build a minimal GenerationRecord for use with _load_per_island_elites.

    Only generation_index matters for that method — it reads per_island_best
    files by that index. All other fields get safe defaults.
    """
    from evolution.persistence import GenerationRecord
    return GenerationRecord(
        generation_index=generation_index,
        started_at=0.0,
        ended_at=0.0,
        n_candidates=0,
        n_rejected=0,
        n_passed=0,
        n_deployment_passing=0,
        best_fitness=0.0,
        median_fitness=0.0,
        best_candidate_id="",
        best_genome_id="",
        wall_time_seconds_used=0.0,
        rejection_reasons={},
    )


class TestPerIslandPersistence:
    """Tests for save_per_island_best_genome + load_per_island_best_genome."""

    def test_save_writes_correct_filename(self, clean_output_dir):
        from evolution.persistence import save_per_island_best_genome
        genome_dict = _make_genome_dict("genome_G1_111111")
        save_per_island_best_genome(
            generation_index=3,
            island_id=2,
            genome_dict=genome_dict,
            output_dir=clean_output_dir,
        )
        expected = clean_output_dir / "best_genomes" / "per_island_gen_0003_island_02.json"
        assert expected.exists()

    def test_load_round_trip_preserves_grid_method(self, clean_output_dir):
        from evolution.persistence import (
            save_per_island_best_genome,
            load_per_island_best_genome,
        )
        genome_dict = _make_genome_dict("genome_G5_999999", grid_method="atr")
        save_per_island_best_genome(
            generation_index=5, island_id=3,
            genome_dict=genome_dict, output_dir=clean_output_dir,
        )
        loaded = load_per_island_best_genome(
            generation_index=5, island_id=3,
            output_dir=clean_output_dir,
        )
        assert loaded is not None
        assert loaded["dca_genome"]["grid_method"] == "atr"
        assert loaded["genome_id"] == "genome_G5_999999"

    def test_load_returns_none_for_missing_gen(self, clean_output_dir):
        from evolution.persistence import load_per_island_best_genome
        # No file written — should return None, not raise
        loaded = load_per_island_best_genome(
            generation_index=99, island_id=1,
            output_dir=clean_output_dir,
        )
        assert loaded is None

    def test_load_returns_none_for_missing_island(self, clean_output_dir):
        """If gen exists but not the requested island, return None."""
        from evolution.persistence import (
            save_per_island_best_genome,
            load_per_island_best_genome,
        )
        save_per_island_best_genome(
            generation_index=2, island_id=1,
            genome_dict=_make_genome_dict("genome_only_island_1"),
            output_dir=clean_output_dir,
        )
        loaded = load_per_island_best_genome(
            generation_index=2, island_id=5,
            output_dir=clean_output_dir,
        )
        assert loaded is None

    def test_independent_genomes_per_island(self, clean_output_dir):
        """Each island must have its OWN top genome — verify by saving
        different grid_methods per island and reading them back."""
        from evolution.persistence import (
            save_per_island_best_genome,
            load_per_island_best_genome,
        )
        grid_methods = ["fixed_pct", "atr", "volatility", "rsi_oversold",
                        "ma_distance", "trend_adjusted", "z_score", "drawdown_from_high"]
        for iid, gm in enumerate(grid_methods, start=1):
            save_per_island_best_genome(
                generation_index=1, island_id=iid,
                genome_dict=_make_genome_dict(f"genome_island_{iid}", grid_method=gm),
                output_dir=clean_output_dir,
            )
        # Verify each one round-trips independently
        for iid, gm in enumerate(grid_methods, start=1):
            loaded = load_per_island_best_genome(
                generation_index=1, island_id=iid,
                output_dir=clean_output_dir,
            )
            assert loaded is not None
            assert loaded["dca_genome"]["grid_method"] == gm, (
                f"Island {iid} should have grid_method={gm} but got "
                f"{loaded['dca_genome']['grid_method']}"
            )


class TestLoadPerIslandElitesHydration:
    """Verify the hydration logic in _load_per_island_elites works."""

    def test_hydrates_back_to_candidate_genome(self, clean_output_dir, tiny_df):
        """When a per-island best file exists, _load_per_island_elites must
        return a CandidateGenome with the correct dca_genome fields."""
        from evolution.persistence import save_per_island_best_genome
        from evolution.config import EvolutionConfig
        from evolution.harness import EvolutionHarness

        # Save 3 distinct island genomes
        for iid in (1, 2, 3):
            save_per_island_best_genome(
                generation_index=4, island_id=iid,
                genome_dict=_make_genome_dict(
                    f"genome_island_{iid}_v1",
                    grid_method=["fixed_pct", "atr", "volatility"][iid - 1],
                    max_dca_layers=iid + 1,  # 2, 3, 4
                ),
                output_dir=clean_output_dir,
            )

        # Build a minimal harness config pointing at our tmp output_dir
        cfg = EvolutionConfig(
            experiment_id="test_island_independence",
            output_dir=str(clean_output_dir),
            candidates_per_gen=24,
            max_generations=2,
            n_islands=3,
            island_mode=True,
        )
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        prev_gen = _make_prev_gen(generation_index=4)
        elites = harness._load_per_island_elites(prev_gen=prev_gen, gen_idx=5)
        # Each island should have exactly one elite (its own prior top)
        assert set(elites.keys()) == {1, 2, 3}
        assert len(elites[1]) == 1
        assert len(elites[2]) == 1
        assert len(elites[3]) == 1
        # Verify hydration: each island's elite has the right grid_method
        assert elites[1][0].dca_genome.grid_method == "fixed_pct"
        assert elites[2][0].dca_genome.grid_method == "atr"
        assert elites[3][0].dca_genome.grid_method == "volatility"
        # And the right max_dca_layers
        assert elites[1][0].dca_genome.max_dca_layers == 2
        assert elites[2][0].dca_genome.max_dca_layers == 3
        assert elites[3][0].dca_genome.max_dca_layers == 4

    def test_missing_files_return_empty_lists(self, clean_output_dir, tiny_df):
        """If no per-island files exist (e.g. first gen), return empty
        lists — caller handles fallback to spec seeding."""
        from evolution.config import EvolutionConfig
        from evolution.harness import EvolutionHarness
        from evolution.persistence import GenerationRecord
        cfg = EvolutionConfig(
            experiment_id="test_island_missing",
            output_dir=str(clean_output_dir),
            candidates_per_gen=24,
            max_generations=2,
            n_islands=4,
            island_mode=True,
        )
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        prev_gen = _make_prev_gen(generation_index=0)
        elites = harness._load_per_island_elites(prev_gen=prev_gen, gen_idx=1)
        # No files yet — all 4 islands return empty lists
        assert set(elites.keys()) == {1, 2, 3, 4}
        for iid in (1, 2, 3, 4):
            assert elites[iid] == []

    def test_migrants_still_layered_on_top(self, clean_output_dir, tiny_df):
        """Incoming migrants from last migration step must still be added
        on top of the per-island elite (preserves migration behavior)."""
        from evolution.persistence import save_per_island_best_genome
        from evolution.config import EvolutionConfig
        from evolution.harness import EvolutionHarness

        save_per_island_best_genome(
            generation_index=7, island_id=2,
            genome_dict=_make_genome_dict("genome_native_island_2"),
            output_dir=clean_output_dir,
        )

        cfg = EvolutionConfig(
            experiment_id="test_migrants_layer",
            output_dir=str(clean_output_dir),
            candidates_per_gen=24,
            max_generations=2,
            n_islands=3,
            island_mode=True,
        )
        harness = EvolutionHarness(config=cfg, df=tiny_df)

        # Simulate 1 migrant arriving for island 2 — use a CandidateGenome
        # with a fully-formed dca_genome so it survives hydration untouched.
        from genome.schema import CandidateGenome, DcaGenome, TpGenome
        migrant = CandidateGenome(
            genome_id="genome_migrant_from_1",
            dca_genome=DcaGenome(
                grid_method="atr",
                grid_params={},
                allocation_method="equal",
                allocation_params={},
                max_dca_layers=3,
                confirmation_indicators=[],
                indicator_params={},
            ),
            tp_genome=TpGenome(exit_method="fixed", exit_params={"tp_pct": 0.005}),
        )
        harness._incoming_migrants = {1: [], 2: [migrant], 3: []}

        prev_gen = _make_prev_gen(generation_index=7)
        elites = harness._load_per_island_elites(prev_gen=prev_gen, gen_idx=8)
        # Island 2 has its native elite + 1 migrant = 2 elites
        assert len(elites[2]) == 2
        assert elites[2][0].genome_id == "genome_native_island_2"
        assert elites[2][1].genome_id == "genome_migrant_from_1"
        # Islands 1 and 3 have no native + no migrant = empty
        assert len(elites[1]) == 0
        assert len(elites[3]) == 0
