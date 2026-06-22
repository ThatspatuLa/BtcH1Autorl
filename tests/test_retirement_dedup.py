"""Tests for retirement elite dedup (Bug #1 fix).

Bug: archive_island() sorted elites by fitness desc and took top-3 by raw
slice. When migration brought back the same elite lineage with different
genome_ids, the top-3 collapsed to 3x the same genome — wasting 2/3 of
archive material. Stage 12 TP evolution would have no diverse genomes to breed.

Fix: dedup_elites_by_signature() collapses entries that share an elite_signature
(grid method, all grid_params, allocation method+params, confirmations, TP params,
combo method, trigger mode) and keeps the highest-fitness copy.

This file: targeted tests proving (a) duplicate-param clones collapse to 1,
(b) distinct-param genomes are preserved, (c) the dedup is keyed on actual
params not genome_id, (d) signature distinguishes across all relevant params.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evolution.retirement import (
    RetirementPolicy,
    archive_island,
    dedup_elites_by_signature,
    elite_signature,
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
# Test fixtures — parameterized genome builder
# ---------------------------------------------------------------

def _mk(
    gid: str = "G",
    grid_method: GridMethod = GridMethod.FIXED_PCT,
    pct: float = 0.005,
    drawdown: float = 0.0,
    max_layers: int = 8,
    cooldown: int = 0,
    alloc: AllocationMethod = AllocationMethod.EQUAL,
    multiplier: float = 1.0,
    tp_pct: float = 0.005,
    confirmations: tuple = (),
    combo: str = "weighted_average",
    trigger: str = "price_only",
) -> CandidateGenome:
    return CandidateGenome(
        genome_id=gid,
        dca_genome=DcaGenome(
            grid_method=grid_method,
            grid_params={
                "pct": pct,
                "drawdown_pct": drawdown,
                "tp_pct": tp_pct,
                "max_layers": max_layers,
                "cooldown_candles": cooldown,
            },
            allocation_method=alloc,
            allocation_params={"multiplier": multiplier, "max_layer_size_pct": 5.0},
            combo_method=combo,
            combo_params={},
            trigger_mode=trigger,
            confirmation_indicators=list(confirmations),
            indicator_params={},
            max_dca_layers=max_layers,
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
    archive_dir = tmp_path / "retired_islands"
    archive_dir.mkdir()
    yield str(archive_dir)
    if archive_dir.exists():
        shutil.rmtree(archive_dir, ignore_errors=True)


# ---------------------------------------------------------------
# elite_signature unit tests
# ---------------------------------------------------------------

class TestEliteSignature:
    def test_identical_params_produce_same_signature(self):
        """Two genomes with same params but different genome_id MUST have the same signature."""
        g1 = _mk(gid="G_aaa", pct=0.005, tp_pct=0.005)
        g2 = _mk(gid="G_bbb", pct=0.005, tp_pct=0.005)
        assert elite_signature(g1) == elite_signature(g2)

    def test_different_pct_produces_different_signature(self):
        g1 = _mk(gid="G1", pct=0.005)
        g2 = _mk(gid="G2", pct=0.008)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_grid_method_produces_different_signature(self):
        g1 = _mk(gid="G1", grid_method=GridMethod.FIXED_PCT)
        g2 = _mk(gid="G2", grid_method=GridMethod.ATR)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_allocation_method_produces_different_signature(self):
        g1 = _mk(gid="G1", alloc=AllocationMethod.EQUAL)
        g2 = _mk(gid="G2", alloc=AllocationMethod.CONTROLLED_EXP)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_multiplier_produces_different_signature(self):
        g1 = _mk(gid="G1", multiplier=1.0)
        g2 = _mk(gid="G2", multiplier=1.5)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_tp_pct_produces_different_signature(self):
        g1 = _mk(gid="G1", tp_pct=0.003)
        g2 = _mk(gid="G2", tp_pct=0.005)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_max_layers_produces_different_signature(self):
        g1 = _mk(gid="G1", max_layers=8)
        g2 = _mk(gid="G2", max_layers=12)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_cooldown_produces_different_signature(self):
        g1 = _mk(gid="G1", cooldown=0)
        g2 = _mk(gid="G2", cooldown=5)
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_confirmations_produces_different_signature(self):
        g1 = _mk(gid="G1", confirmations=())
        g2 = _mk(gid="G2", confirmations=(ConfirmationIndicator.RSI_BELOW,))
        assert elite_signature(g1) != elite_signature(g2)

    def test_different_trigger_mode_produces_different_signature(self):
        g1 = _mk(gid="G1", trigger="price_only")
        g2 = _mk(gid="G2", trigger="indicator_confirmed")
        assert elite_signature(g1) != elite_signature(g2)


# ---------------------------------------------------------------
# dedup_elites_by_signature unit tests
# ---------------------------------------------------------------

class TestDedupElites:
    def test_collapse_clones_same_fitness(self):
        """Three clones (same params, different genome_ids, same fitness) → 1 entry."""
        clones = [
            (_mk(gid="G_a", pct=0.005), 0.82),
            (_mk(gid="G_b", pct=0.005), 0.82),
            (_mk(gid="G_c", pct=0.005), 0.82),
        ]
        result = dedup_elites_by_signature(clones)
        assert len(result) == 1
        assert result[0][1] == 0.82

    def test_collapse_clones_keeps_highest_fitness(self):
        """Same params, different fitnesses → keep the highest."""
        clones = [
            (_mk(gid="G_a", pct=0.005), 0.80),
            (_mk(gid="G_b", pct=0.005), 0.85),
            (_mk(gid="G_c", pct=0.005), 0.82),
        ]
        result = dedup_elites_by_signature(clones)
        assert len(result) == 1
        assert result[0][1] == 0.85
        assert result[0][0].genome_id == "G_b"

    def test_diverse_genomes_preserved(self):
        """Distinct params → all preserved, sorted by fitness desc."""
        elites = [
            (_mk(gid="G1", pct=0.005), 0.80),
            (_mk(gid="G2", pct=0.008), 0.85),
            (_mk(gid="G3", pct=0.012), 0.78),
        ]
        result = dedup_elites_by_signature(elites)
        assert len(result) == 3
        fits = [f for _, f in result]
        assert fits == sorted(fits, reverse=True)
        assert fits == [0.85, 0.80, 0.78]

    def test_mixed_clones_and_distinct(self):
        """Some clones collapse, distinct ones preserved."""
        elites = [
            (_mk(gid="G1", pct=0.005), 0.85),  # distinct
            (_mk(gid="G2a", pct=0.008), 0.80),  # clone pair
            (_mk(gid="G2b", pct=0.008), 0.82),  # clone of G2a, higher fit
            (_mk(gid="G3", pct=0.012), 0.78),  # distinct
        ]
        result = dedup_elites_by_signature(elites)
        assert len(result) == 3
        # G2b (higher fit) wins over G2a
        assert result[0][0].genome_id == "G1"
        assert result[1][0].genome_id == "G2b"
        assert result[2][0].genome_id == "G3"

    def test_empty_input(self):
        assert dedup_elites_by_signature([]) == []


# ---------------------------------------------------------------
# archive_island end-to-end with dedup
# ---------------------------------------------------------------

class TestArchiveWithDedup:
    def test_archive_top_3_dedup_collapses_clones(self, tmp_archive_dir):
        """The exact Bug #1 scenario: 3 clones with different genome_ids but
        same params. Before fix: archive had 3x same genome in top_3_elites.
        After fix: top_3 has 1 entry from those clones (others are below it).
        """
        clones = [
            (_mk(gid="G_clone_a", pct=0.009083, drawdown=0.078122, tp_pct=0.003269), 0.812009),
            (_mk(gid="G_clone_b", pct=0.009083, drawdown=0.078122, tp_pct=0.003269), 0.812009),
            (_mk(gid="G_clone_c", pct=0.009083, drawdown=0.078122, tp_pct=0.003269), 0.812009),
        ]
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        rec = archive_island(
            policy=policy,
            cycle_id="20260622_bug1",
            cycle_output_dir="runs/cycle_bug1",
            island_id=1,
            retired_at_gen=23,
            family_bias={"name": "fixed_pct"},
            per_island_top_fitness=0.812009,
            elites=clones,
            generations_evolved=24,
        )
        # 3 clones → dedup → 1 unique signature
        assert rec.n_elites_archived == 1
        assert len(rec.top_3_elite_ids) == 1
        assert rec.top_3_elite_ids[0] in {"G_clone_a", "G_clone_b", "G_clone_c"}

        # top_3_elites.json should also be 1 entry
        top3_path = Path(tmp_archive_dir) / "retired_20260622_bug1_1_23" / "top_3_elites.json"
        data = json.loads(top3_path.read_text())
        assert len(data) == 1

    def test_archive_diverse_top_3_preserved(self, tmp_archive_dir):
        """3 DISTINCT genomes (different params) → all 3 in top_3, no dedup."""
        elites = [
            (_mk(gid="G_a", pct=0.005, tp_pct=0.003), 0.85),
            (_mk(gid="G_b", pct=0.008, tp_pct=0.004), 0.83),
            (_mk(gid="G_c", pct=0.012, tp_pct=0.005), 0.81),
        ]
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        rec = archive_island(
            policy=policy,
            cycle_id="20260622_diverse",
            cycle_output_dir="runs/cycle_diverse",
            island_id=2,
            retired_at_gen=20,
            family_bias={"name": "atr"},
            per_island_top_fitness=0.85,
            elites=elites,
            generations_evolved=20,
        )
        assert rec.n_elites_archived == 3
        assert len(rec.top_3_elite_ids) == 3
        assert set(rec.top_3_elite_ids) == {"G_a", "G_b", "G_c"}

        # Verify top_3_elites.json has 3 distinct entries
        top3_path = Path(tmp_archive_dir) / "retired_20260622_diverse_2_20" / "top_3_elites.json"
        data = json.loads(top3_path.read_text())
        assert len(data) == 3
        pcts = sorted(e["genome"]["dca_genome"]["grid_params"]["pct"] for e in data)
        assert pcts == [0.005, 0.008, 0.012]

    def test_archive_dedup_higher_fitness_clone_wins(self, tmp_archive_dir):
        """When clones have different fitness, dedup keeps highest. Manifest
        should reflect the WINNING genome_id."""
        elites = [
            (_mk(gid="G_loser", pct=0.009), 0.80),
            (_mk(gid="G_winner", pct=0.009), 0.84),  # same params, higher fit
            (_mk(gid="G_mid", pct=0.009), 0.82),
        ]
        policy = RetirementPolicy(archive_dir=tmp_archive_dir)
        rec = archive_island(
            policy=policy,
            cycle_id="20260622_clonefit",
            cycle_output_dir="runs/cycle_clonefit",
            island_id=5,
            retired_at_gen=30,
            family_bias={"name": "tight_dca"},
            per_island_top_fitness=0.84,
            elites=elites,
            generations_evolved=30,
        )
        assert rec.n_elites_archived == 1
        assert rec.top_3_elite_ids == ["G_winner"]
        assert rec.top_3_elite_fitness == [0.84]
