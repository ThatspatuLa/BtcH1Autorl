"""Tests for the GLOBAL_MAX_DCA_LAYERS = 5 cap.

User directive 2026-06-23: "lets set up a max DCA layer to be 5 for ongoing crons"
The cap is a system-wide policy enforced at every place a candidate's
max_layers is set or read.

Single source of truth: evolution.operators.GLOBAL_MAX_DCA_LAYERS = 5

Defence-in-depth:
1. DCA_PARAM_RANGES["max_layers"] upper bound == GLOBAL_MAX_DCA_LAYERS
2. operators.random_dca_genome() never produces max_layers > GLOBAL_MAX_DCA_LAYERS
3. operators.mutate() clamps max_layers ≤ GLOBAL_MAX_DCA_LAYERS
4. operators.crossover() clamps max_layers ≤ GLOBAL_MAX_DCA_LAYERS
5. population_builder._make_candidate() clamps max_layers ≤ GLOBAL_MAX_DCA_LAYERS
6. extract_dca_params_from_genome() clamps max_layers ≤ GLOBAL_MAX_DCA_LAYERS
   (defense in depth — even if upstream misses it)
"""
import sys
sys.path.insert(0, '/home/spatula/Projects/BtcH1Autorl')

import random
import pytest

from evolution.operators import (
    GLOBAL_MAX_DCA_LAYERS,
    DCA_PARAM_RANGES,
    random_dca_genome,
    random_candidate_genome,
    mutate,
    crossover,
)
from evolution.population_builder import _make_candidate, _random_layers, build_exploit_population, build_explore_population
from genome.schema import (
    CandidateGenome, DcaGenome, TpGenome, TpExitMethod,
    GridMethod, AllocationMethod,
)


# ============================================================
# Source-of-truth tests
# ============================================================

def test_global_max_dca_layers_is_10():
    """User directive 2026-06-25: max DCA layer = 10 (was 5)."""
    assert GLOBAL_MAX_DCA_LAYERS == 10


def test_dca_param_ranges_max_layers_upper_bound_matches_cap():
    hi = DCA_PARAM_RANGES["max_layers"][1]
    assert hi <= GLOBAL_MAX_DCA_LAYERS, (
        f"DCA_PARAM_RANGES['max_layers'] upper bound {hi} > GLOBAL_MAX_DCA_LAYERS {GLOBAL_MAX_DCA_LAYERS}"
    )


def test_dca_param_ranges_max_layers_lower_bound_reasonable():
    lo = DCA_PARAM_RANGES["max_layers"][0]
    assert lo >= 1, f"max_layers lower bound {lo} should be >= 1"


# ============================================================
# Generation-path tests
# ============================================================

def test_random_dca_genome_respects_cap():
    """1000 random genomes: every max_layers ≤ GLOBAL_MAX_DCA_LAYERS."""
    rng = random.Random(42)
    for _ in range(1000):
        g = random_dca_genome(rng=rng)
        assert g.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS, g.max_dca_layers
        assert g.grid_params.get("max_layers", 0) <= GLOBAL_MAX_DCA_LAYERS


def test_random_candidate_genome_respects_cap():
    """1000 random candidate genomes: every max_layers ≤ GLOBAL_MAX_DCA_LAYERS."""
    rng = random.Random(42)
    for _ in range(1000):
        c = random_candidate_genome(rng=rng)
        assert c.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS


def test_mutate_clamps_max_layers_to_cap():
    """If a parent has max_layers > cap (e.g. legacy 18), mutate() clamps it."""
    # Build a parent with max_layers=18 (legacy)
    parent = random_candidate_genome(rng=random.Random(1))
    parent.dca_genome.grid_params["max_layers"] = 18
    parent.dca_genome.max_dca_layers = 18

    # Force mutate() to keep adjusting max_layers upward by setting high rate
    rng = random.Random(2)
    child = mutate(parent, rng=rng, mutation_rate=1.0)
    # Even after many random up-shifts, must be clamped
    assert child.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS
    assert child.dca_genome.grid_params["max_layers"] <= GLOBAL_MAX_DCA_LAYERS


def test_crossover_clamps_max_layers_to_cap():
    """crossover() with one parent at max_layers=18 must clamp."""
    pa = random_candidate_genome(rng=random.Random(1))
    pb = random_candidate_genome(rng=random.Random(2))
    pa.dca_genome.grid_params["max_layers"] = 18
    pa.dca_genome.max_dca_layers = 18
    pb.dca_genome.grid_params["max_layers"] = 18
    pb.dca_genome.max_dca_layers = 18

    child = crossover(pa, pb, rng=random.Random(3))
    assert child.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS
    assert child.dca_genome.grid_params["max_layers"] <= GLOBAL_MAX_DCA_LAYERS


# ============================================================
# Population-builder tests
# ============================================================

def test_population_builder_random_layers_respects_cap():
    """_random_layers never exceeds cap."""
    rng = random.Random(42)
    for _ in range(1000):
        assert _random_layers(rng) <= GLOBAL_MAX_DCA_LAYERS


def test_build_exploit_population_respects_cap():
    """build_exploit_population: every candidate has max_layers ≤ cap."""
    rng = random.Random(42)
    cands = build_exploit_population(rng, generation_index=0, gid_start=0, n=100)
    for c in cands:
        assert c.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS, c.dca_genome.max_dca_layers
        assert c.dca_genome.grid_params["max_layers"] <= GLOBAL_MAX_DCA_LAYERS


def test_build_explore_population_respects_cap():
    """build_explore_population: every candidate has max_layers ≤ cap."""
    rng = random.Random(42)
    cands = build_explore_population(rng, generation_index=0, gid_start=0, n=100)
    for c in cands:
        assert c.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS


# ============================================================
# Defense-in-depth — evaluator extractor
# ============================================================

def test_extract_dca_params_clamps_to_cap():
    """Even a malformed genome with max_layers=99 must be clamped by the extractor."""
    from dca_engine.tp_baseline import extract_dca_params_from_genome

    g = CandidateGenome(
        genome_id="test_evil",
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"pct": 0.005, "max_layers": 99, "tp_pct": 0.005},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=99,
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.005}),
    )
    params = extract_dca_params_from_genome(g)
    assert params["max_layers"] <= GLOBAL_MAX_DCA_LAYERS, params["max_layers"]
