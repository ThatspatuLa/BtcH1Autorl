"""Tests for evolution islands — specs, migration, population builder.

Effective 2026-06-22 per Six's Plan B (8 islands × 62 cands + 4 random).
"""
from __future__ import annotations

import random

import pytest

from evolution.islands import (
    ISLAND_SPECS,
    IslandSpec,
    IslandTracker,
    distribute_migrants,
    get_island_spec,
    get_island_specs,
    island_assignment_for_population,
    per_island_elites,
    select_migrants,
)
from evolution.population_builder import (
    build_island_population,
    get_island_id_for_genome,
)
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    DcaGenome,
    GridMethod,
    TpExitMethod,
    TpGenome,
)


def _make_genome(gn: str, grid: GridMethod = GridMethod.FIXED_PCT) -> CandidateGenome:
    return CandidateGenome(
        genome_id=gn,
        dca_genome=DcaGenome(
            grid_method=grid,
            grid_params={"pct": 0.01, "max_layers": 5, "tp_pct": 0.005, "cooldown_candles": 0},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=5,
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.005}),
    )


# ============================================================
# IslandSpec / ISLAND_SPECS
# ============================================================

def test_island_specs_count():
    """8 islands, total 496 candidates."""
    assert len(ISLAND_SPECS) == 8
    assert sum(s.n_candidates for s in ISLAND_SPECS) == 496


def test_island_specs_unique_ids():
    ids = [s.island_id for s in ISLAND_SPECS]
    assert len(set(ids)) == 8
    assert sorted(ids) == list(range(1, 9))


def test_island_specs_have_names():
    for spec in ISLAND_SPECS:
        assert spec.name
        assert spec.note


def test_get_island_spec_returns_correct():
    spec = get_island_spec(1)
    assert spec.island_id == 1
    assert spec.name == "fixed_pct"
    with pytest.raises(ValueError):
        get_island_spec(99)


def test_get_island_specs_returns_copy():
    """Mutating the returned list must not affect the module-level one."""
    specs = get_island_specs()
    specs.clear()
    assert len(ISLAND_SPECS) == 8


# ============================================================
# build_island_population
# ============================================================

def test_build_island_population_count():
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    assert len(pop) == 500


def test_build_island_population_grid_method_island1():
    """Island 1 is fixed_pct — all 62 of its candidates must use fixed_pct."""
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    fixed_pct = [g for g in pop[:62] if g.dca_genome.grid_method == GridMethod.FIXED_PCT]
    assert len(fixed_pct) == 62


def test_build_island_population_grid_method_island2():
    """Island 2 is atr — all 62 must use atr."""
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    island_2 = pop[62:124]
    atr = [g for g in island_2 if g.dca_genome.grid_method == GridMethod.ATR]
    assert len(atr) == 62


def test_build_island_population_tight_dca_cap():
    """Island 8 has max_dca_layers_cap=8 — all its cands must have max_layers <= 8."""
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    # Island 8 is the last 62 in the list
    island_8 = pop[496 - 62:496]
    for g in island_8:
        assert g.dca_genome.max_dca_layers <= 8, (
            f"Island 8 cand {g.genome_id} has max_layers={g.dca_genome.max_dca_layers}"
        )


def test_build_island_population_vola_adj_alloc():
    """Island 6 forces volatility_adjusted allocation."""
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    # Island 6: starts at idx 5*62=310, runs 62
    island_6 = pop[5 * 62 : 6 * 62]
    vola = [g for g in island_6 if g.dca_genome.allocation_method == AllocationMethod.VOLATILITY_ADJUSTED]
    assert len(vola) == 62


def test_build_island_population_tags_island_id():
    """Every candidate should have an island_assign lineage op."""
    rng = random.Random(42)
    pop = build_island_population(rng, 0, ISLAND_SPECS, gid_start=0, random_count=4)
    for c in pop:
        island_id = get_island_id_for_genome(c)
        assert 0 <= island_id <= 8


def test_get_island_id_for_genome_no_tag_returns_zero():
    """A genome with no island_assign tag should return 0."""
    g = _make_genome("test")
    assert get_island_id_for_genome(g) == 0


def test_get_island_id_for_genome_with_tag():
    """A genome with an island_assign tag returns that island."""
    g = _make_genome("test")
    g.lineage.mutation_ops = [{"op": "island_assign", "island_id": 5}]
    assert get_island_id_for_genome(g) == 5


# ============================================================
# Migration
# ============================================================

def test_select_migrants_returns_top_n():
    elites = [_make_genome(f"e_{i}") for i in range(10)]
    selected = select_migrants(1, elites, n_migrants=4, rng=random.Random(42))
    assert len(selected) == 4
    assert selected == elites[:4]


def test_select_migrants_fewer_than_n():
    elites = [_make_genome(f"e_{i}") for i in range(2)]
    selected = select_migrants(1, elites, n_migrants=4, rng=random.Random(42))
    assert len(selected) == 2


def test_select_migrants_empty():
    assert select_migrants(1, [], n_migrants=4) == []


def test_distribute_migrants_round_robin():
    """Island 1 should send migrants to islands 8 and 2 (its neighbors in 1..N)."""
    source_1 = [_make_genome(f"m1_{i}") for i in range(4)]
    received = distribute_migrants({1: source_1}, n_islands=8, rng=random.Random(42))
    # Island 1 neighbors are 8 and 2
    assert len(received[8]) + len(received[2]) == 4
    # All migrants should be tagged with from_island=1
    for m in received[8] + received[2]:
        assert m.lineage.mutation_ops[-1]["from_island"] == 1


def test_distribute_migrants_isolated_islands_get_nothing():
    """If only source 1 has migrants, only islands 2 and 8 get any."""
    source_1 = [_make_genome("m1") for _ in range(2)]
    received = distribute_migrants({1: source_1}, n_islands=8, rng=random.Random(42))
    for iid in [3, 4, 5, 6, 7]:
        assert received[iid] == []


# ============================================================
# island_assignment_for_population
# ============================================================

def test_island_assignment_for_population_500():
    cands = [_make_genome(f"c_{i:04d}") for i in range(500)]
    specs = get_island_specs()
    assign = island_assignment_for_population(cands, specs, random_count=4)
    from collections import Counter
    counts = Counter(assign.values())
    assert counts[0] == 4  # 4 random
    for iid in range(1, 9):
        assert counts[iid] == 62


# ============================================================
# IslandTracker (legacy helper — still exported)
# ============================================================

def test_island_tracker_register_and_get():
    t = IslandTracker()
    t.register("g1", 1)
    t.register("g2", 2)
    assert t.get_island_id("g1") == 1
    assert t.get_island_id("g2") == 2
    assert t.get_island_id("unknown") is None


def test_island_tracker_route():
    t = IslandTracker()
    cands = [_make_genome(f"c_{i}") for i in range(4)]
    assignment = {"c_0": 1, "c_1": 1, "c_2": 2, "c_3": 2}
    buckets = t.route(cands, assignment)
    assert len(buckets[1]) == 2
    assert len(buckets[2]) == 2
