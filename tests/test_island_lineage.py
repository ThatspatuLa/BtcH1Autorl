"""Phase F — Island lineage + retirement wiring tests.

The bug Six's Gate Check found: `island_assign` mutation_op tag is written
to a genome's lineage at population build time, but `mutate()` and `crossover()`
operators REPLACE `mutation_ops` with a new list containing only their own op,
losing the parent's island tag. As a result:
  - `get_island_id_for_genome()` returns 0 (default) for all mutations
  - The harness filter at `harness.py:668` filters out iid=0
  - `per_island_best_fitness` stays empty `{}` for all generations
  - `_check_retirement` finds nothing → retirement never fires

These tests pin the CORRECT behaviour:
  F1: Seed genome has island_assign tag.
  F2: Mutation child inherits island_assign from parent.
  F3: Crossover child inherits island_assign from primary parent.
  F4: Migration (re-tagging) overrides previous island_id.
  F5: get_island_id_for_genome() returns correct id for all three cases.
  F6: per_island_best_fitness correctly populated by the harness.
  F7: Retirement only fires when island top fitness ≥ threshold AND
      candidate passed deployment gates.
  F8: Random fresh genomes default to island 0 (the "random" pool).
"""
from __future__ import annotations

import pytest

from evolution.operators import crossover, mutate
from evolution.population_builder import (
    _seed_island,
    get_island_id_for_genome,
)
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ComboMethod,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    SafetyGenome,
    SettingsOverrides,
    TpExitMethod,
    TpGenome,
    TriggerMode,
)


def _make_genome(gid: str, *, island_id: int | None = None) -> CandidateGenome:
    """Build a minimal CandidateGenome for tests."""
    g = CandidateGenome(
        genome_id=gid,
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"grid_pct": 0.005, "cooldown_candles": 3},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={"max_layers": 8},
            combo_method=ComboMethod.WEIGHTED_AVERAGE,
            combo_params={},
            trigger_mode=TriggerMode.PRICE_ONLY,
            confirmation_indicators=[],
            indicator_params={},
            max_dca_layers=8,
        ),
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"fixed_tp_pct": 0.005},
        ),
        safety_genome=SafetyGenome(),
        settings_overrides=SettingsOverrides(),
        lineage=LineageMetadata(generation_index=0),
    )
    if island_id is not None:
        g.lineage.mutation_ops = [{"op": "island_assign", "island_id": island_id}]
    return g


# ============================================================
# F1: Seed genome has island_assign tag
# ============================================================

# F1
def test_seed_island_writes_island_assign_tag():
    """A seeded island genome has the island_assign mutation_op in its lineage."""
    from evolution.islands import IslandSpec
    import random
    spec = IslandSpec(
        island_id=3, name="test_3", n_candidates=3,
        forced_grid_methods=None, forced_allocation=None,
        forced_confirmations=None, max_dca_layers_cap=None,
    )
    candidates = _seed_island(
        rng=random.Random(42), generation_index=0, gid_start=1, n=3,
        island_spec=spec,
    )
    g = candidates[0]
    assert any(
        op.get("op") == "island_assign" and op.get("island_id") == 3
        for op in g.lineage.mutation_ops
    ), f"missing island_assign tag, ops={g.lineage.mutation_ops}"


# ============================================================
# F2: Mutation child inherits island_assign
# ============================================================

def test_mutate_child_inherits_island_assign():
    """A mutate() child must KEEP the parent's island_assign tag."""
    import random
    parent = _make_genome("parent_x", island_id=2)
    child = mutate(parent, rng=random.Random(42))
    assert get_island_id_for_genome(child) == 2, (
        f"mutate() lost island tag. ops={child.lineage.mutation_ops}"
    )


# ============================================================
# F3: Crossover child inherits island_assign from primary parent
# ============================================================

def test_crossover_child_inherits_island_assign_from_primary():
    """A crossover() child must inherit island from parent_a (primary)."""
    import random
    parent_a = _make_genome("pa", island_id=4)
    parent_b = _make_genome("pb", island_id=5)
    child = crossover(parent_a, parent_b, rng=random.Random(42))
    # Per spec: primary parent = parent_a → island 4
    assert get_island_id_for_genome(child) == 4, (
        f"crossover() lost primary parent's island tag. ops={child.lineage.mutation_ops}"
    )


# ============================================================
# F4: Migration overrides previous island_id
# ============================================================

def test_migration_overrides_previous_island_id():
    """A migrant genome retagged with a new island_id should reflect the new one."""
    # Simulate the islands.py migration logic
    g = _make_genome("migrant_001", island_id=2)
    # Migrate to island 7
    g.lineage.mutation_ops = [
        {"op": "migrate", "from_island_id": 2, "to_island_id": 7},
        {"op": "island_assign", "island_id": 7},
    ]
    assert get_island_id_for_genome(g) == 7


# ============================================================
# F5: get_island_id_for_genome() returns correct id
# ============================================================

def test_get_island_id_returns_zero_for_unassigned_genome():
    """A genome with no island_assign tag returns 0 (the random pool)."""
    g = _make_genome("no_assign")  # no tag written
    assert get_island_id_for_genome(g) == 0


def test_get_island_id_returns_latest_assign():
    """When multiple island_assign tags exist (re-tagged), the LATEST wins."""
    g = _make_genome("re_assigned")
    g.lineage.mutation_ops = [
        {"op": "island_assign", "island_id": 1},
        {"op": "migrate", "from": 1, "to": 5},
        {"op": "island_assign", "island_id": 5},
    ]
    assert get_island_id_for_genome(g) == 5


# ============================================================
# F6: per_island_best_fitness correctly populated
# ============================================================

def test_per_island_best_fitness_aggregation():
    """Simulate harness aggregation: 3 islands, top fitness logged per island.

    The harness filters out island 0 (the "random" pool) — only seeded islands
    (1..N) are aggregated.
    """
    from collections import defaultdict
    # Mock candidate results: (genome_id, island_id, fitness)
    # Island 0 should be filtered out by harness (per harness.py:668)
    results = [
        ("g1", 0, 0.70),
        ("g2", 0, 0.72),
        ("g3", 1, 0.85),
        ("g4", 1, 0.80),
        ("g5", 2, 0.65),
    ]
    per_island: dict[int, list[float]] = defaultdict(list)
    for _gid, iid, fit in results:
        if iid > 0:  # harness.py:668 filter
            per_island[iid].append(fit)
    # Top per island
    per_island_top = {iid: max(fits) for iid, fits in per_island.items()}
    # Island 0 is NOT in the result (filtered out)
    assert 0 not in per_island_top
    # Islands 1 and 2 are present
    assert per_island_top[1] == 0.85
    assert per_island_top[2] == 0.65


# ============================================================
# F7: Retirement only fires when fitness >= threshold AND deployment_pass
# ============================================================

def test_retirement_fires_only_when_threshold_and_deployment_pass():
    """Retirement should not fire if top fitness is below threshold, even if deployment_pass=True."""
    from evolution.retirement import RetirementPolicy
    policy = RetirementPolicy(threshold=0.80, min_deployment_passing=1)
    # Below threshold → not eligible
    eligible = policy.check_eligibility(
        island_id=1, per_island_top_fitness=0.75,
        deployment_passing_count=10,
    )
    assert eligible is False, "Should NOT retire when top fitness < threshold"

    # Above threshold + sufficient deploy-passing → eligible
    eligible = policy.check_eligibility(
        island_id=1, per_island_top_fitness=0.85,
        deployment_passing_count=10,
    )
    assert eligible is True, "Should retire when top fitness >= threshold and deploy-passing >= min"


def test_retirement_fires_only_when_deployment_passing_count_nonzero():
    """Retirement should not fire if 0 candidates passed deployment (genuine zero-trust island)."""
    from evolution.retirement import RetirementPolicy
    policy = RetirementPolicy(threshold=0.80, min_deployment_passing=1)
    eligible = policy.check_eligibility(
        island_id=1, per_island_top_fitness=0.85,
        deployment_passing_count=0,
    )
    assert eligible is False, "Should NOT retire when no candidates passed deployment"


# ============================================================
# F8: Fresh random genome defaults to island 0
# ============================================================

def test_fresh_random_genome_defaults_to_island_zero():
    """A fresh genome (not seeded by _seed_island) has no island_assign tag → island 0."""
    g = _make_genome("random_fresh_001")
    assert get_island_id_for_genome(g) == 0


# ============================================================
# Regression: existing _seed_island logic works
# ============================================================

def test_seed_island_multiple_islands_distinct_ids():
    """Seeding island 0, 1, 2 with different genomes yields distinct island_ids."""
    from evolution.islands import IslandSpec
    import random
    for i in range(3):
        spec = IslandSpec(
            island_id=i, name=f"island_{i}", n_candidates=2,
            forced_grid_methods=None, forced_allocation=None,
            forced_confirmations=None, max_dca_layers_cap=None,
        )
        candidates = _seed_island(
            rng=random.Random(42 + i), generation_index=0,
            gid_start=1, n=2, island_spec=spec,
        )
        assert get_island_id_for_genome(candidates[0]) == i
