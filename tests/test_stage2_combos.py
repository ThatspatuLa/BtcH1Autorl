"""Stage 2 combo tests — true per-layer zones (Option A).

Tests:
  - ComboSpec generation: 60 combos from 5 families × {pairs, triples} × 3 splits
  - 3-split layer counts: contiguous chunks summing to max_layers
  - Weighted-blend layer counts: exponential weighting, deepest zone gets most
  - Alternating layer counts: round-robin, all families get ~equal
  - Zone validation: contiguous, non-overlapping, cover 1..max_dca_layers
  - Zone mutation: zones are immutable across mutate()
  - Zone crossover: zones preserved across crossover()
  - OrderManager zone dispatch: picks correct method+params per layer index
  - extract_dca_params: emits zones when present
  - build_combo_population: produces candidates with shared zones
  - Queue runner dry-run: starts first combo
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dca_calc.grid_spacing import (
    compute_next_layer_price_zoned,
    select_active_zone,
)
from dca_engine.order_manager import OrderManager
from dca_engine.tp_baseline import extract_dca_params_from_genome
from evolution.combo_specs import (
    COMBO_DEFAULT_MAX_DCA_LAYERS,
    COMBO_SPLIT_STRATEGIES,
    ComboSpec,
    build_pairs,
    build_stage2_combos,
    build_triples,
    build_zones_for_combo,
    select_top_families_from_stage1,
)
from evolution.family_contracts import (
    clear_active_combo_contract,
    clear_active_family_contract,
    combo_contract_from_spec,
    set_active_combo_contract,
    set_active_family_contract,
)
from evolution.hyperopt_config import build_family_specs
from evolution.operators import crossover, mutate, random_candidate_genome
from evolution.population_builder import build_combo_population
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    DcaGenome,
    GridMethod,
    GridZoneSpec,
    TpExitMethod,
    TpGenome,
    validate_genome,
)
import scripts.minato_stage2_queue_runner as queue_runner


TOP5_NAMES = [
    "hybrid_volatility_drawdown_spacing",
    "drawdown_from_high_spacing",
    "hybrid_trend_drawdown_spacing",
    "hybrid_fixed_atr_spacing",
    "atr_spacing",
]


# ============================================================
# ComboSpec generation
# ============================================================


def test_stage2_generates_60_combos_from_top5() -> None:
    """5 families × {C(5,2)=10 pairs + C(5,3)=10 triples} × 3 splits = 60."""
    combos = build_stage2_combos(TOP5_NAMES)
    assert len(combos) == 60
    # All combos use the canonical 10-layer policy cap
    assert all(c.max_dca_layers == COMBO_DEFAULT_MAX_DCA_LAYERS for c in combos)
    # All combos are either pairs (10) or triples (10)
    n_pairs = sum(1 for c in combos if c.is_pair)
    n_triples = sum(1 for c in combos if c.is_triple)
    assert n_pairs == 30  # 10 pairs × 3 splits
    assert n_triples == 30  # 10 triples × 3 splits


def test_stage2_combos_cover_all_three_splits() -> None:
    combos = build_stage2_combos(TOP5_NAMES)
    splits_seen = {c.split_strategy for c in combos}
    assert splits_seen == set(COMBO_SPLIT_STRATEGIES)
    # Each split appears 20 times (10 pairs + 10 triples)
    for split in COMBO_SPLIT_STRATEGIES:
        assert sum(1 for c in combos if c.split_strategy == split) == 20


def test_select_top_families_picks_best_by_fitness() -> None:
    fake_results = [
        {"family": "atr_spacing", "best_fitness": 0.6406},
        {"family": "drawdown_from_high_spacing", "best_fitness": 0.6852},
        {"family": "hybrid_volatility_drawdown_spacing", "best_fitness": 0.7113},
        {"family": "hybrid_trend_drawdown_spacing", "best_fitness": 0.6817},
        {"family": "hybrid_fixed_atr_spacing", "best_fitness": 0.6408},
    ]
    top = select_top_families_from_stage1(fake_results, top_n=5)
    assert top[0] == "hybrid_volatility_drawdown_spacing"  # highest fitness
    assert set(top) == {r["family"] for r in fake_results}


# ============================================================
# Layer-split math
# ============================================================


def test_3_split_pair_divides_evenly() -> None:
    """2 families × 10 layers → [5, 5]"""
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing"], "3_split", max_dca_layers=10
    )
    assert [z.layer_count for z in zones] == [5, 5]
    # First zone starts at 1, second zone continues contiguously
    assert zones[0].layer_start == 1
    assert zones[1].layer_start == 6


def test_3_split_triple_with_remainder() -> None:
    """3 families × 10 layers → [3, 3, 4] (remainder absorbed by deepest zone)."""
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing", "volatility_spacing"],
        "3_split",
        max_dca_layers=10,
    )
    assert [z.layer_count for z in zones] == [3, 3, 4]
    # All zones contiguous and non-overlapping
    cursor = 1
    for z in zones:
        assert z.layer_start == cursor
        cursor += z.layer_count
    assert cursor - 1 == 10


def test_weighted_blend_pair_favors_deeper() -> None:
    """2 families × 10 layers weighted [0.33, 0.67] → [3, 7]."""
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing"], "weighted_blend", max_dca_layers=10
    )
    assert [z.layer_count for z in zones] == [3, 7]


def test_weighted_blend_triple_gives_most_to_deepest() -> None:
    """3 families × 10 layers weighted exponentially. Weights are [1, 2, 4]/7.

    With max_layers=10: raw counts ≈ [1.43, 2.86, 5.71]. Floor → [1, 2, 5],
    then 2 leftover added to deepest → [1, 2, 7]. The deepest zone strictly
    dominates — that's the "weighted" intuition: more layers go where the
    deeper-dip specialist matters most.
    """
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing", "volatility_spacing"],
        "weighted_blend",
        max_dca_layers=10,
    )
    counts = [z.layer_count for z in zones]
    # Deepest zone strictly largest
    assert counts[-1] > counts[-2] > counts[0]
    # Sums to exactly max_layers
    assert sum(counts) == 10
    # Shallowest zone is at most 2 layers (so the weight skew is real)
    assert counts[0] <= 2


def test_alternating_pair_balances_layers() -> None:
    """2 families × 10 layers alternating → [5, 5]."""
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing"], "alternating", max_dca_layers=10
    )
    assert [z.layer_count for z in zones] == [5, 5]


def test_alternating_triple_balances_layers() -> None:
    """3 families × 10 layers alternating → [4, 3, 3] (round-robin)."""
    zones = build_zones_for_combo(
        ["atr_spacing", "drawdown_from_high_spacing", "volatility_spacing"],
        "alternating",
        max_dca_layers=10,
    )
    assert [z.layer_count for z in zones] == [4, 3, 3]


# ============================================================
# Zone validation
# ============================================================


def test_all_60_combos_have_valid_zones() -> None:
    """Every combo's zones pass genome validation (contiguous, no overlap, full cover)."""
    combos = build_stage2_combos(TOP5_NAMES)
    for combo in combos:
        # Build a candidate using the combo's zones; it should validate.
        g = CandidateGenome(
            genome_id="test_" + combo.name,
            dca_genome=DcaGenome(
                grid_method=GridMethod.FIXED_PCT,
                grid_params={"pct": 0.015, "max_layers": combo.max_dca_layers, "tp_pct": 0.02},
                allocation_method=AllocationMethod.EQUAL,
                allocation_params={},
                max_dca_layers=combo.max_dca_layers,
                zones=list(combo.zones),
            ),
            tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
        )
        # Should not raise
        validate_genome(g)


def test_zones_with_gap_fail_validation() -> None:
    """Non-contiguous zones (gap at layer 5) should fail."""
    g = CandidateGenome(
        genome_id="gap_test",
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"pct": 0.015, "max_layers": 10, "tp_pct": 0.02},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=10,
            zones=[
                GridZoneSpec(layer_start=1, layer_count=3, grid_method=GridMethod.ATR, grid_params={}),
                # Gap: layer 4 missing
                GridZoneSpec(layer_start=5, layer_count=6, grid_method=GridMethod.VOLATILITY, grid_params={}),
            ],
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
    )
    with pytest.raises(Exception, match="contiguous"):
        validate_genome(g)


def test_zones_that_exceed_max_dca_layers_fail_validation() -> None:
    """Zones covering layers 1..12 but max_dca_layers=10 should fail."""
    g = CandidateGenome(
        genome_id="overflow_test",
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"pct": 0.015, "max_layers": 10, "tp_pct": 0.02},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=10,
            zones=[
                GridZoneSpec(layer_start=1, layer_count=5, grid_method=GridMethod.ATR, grid_params={}),
                GridZoneSpec(layer_start=6, layer_count=7, grid_method=GridMethod.VOLATILITY, grid_params={}),
            ],
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
    )
    with pytest.raises(Exception, match="must match"):
        validate_genome(g)


# ============================================================
# Zone-aware OrderManager dispatch
# ============================================================


def test_order_manager_picks_zone_by_layer_index() -> None:
    """OrderManager._active_zone returns the right zone for layer N."""
    zones = [
        GridZoneSpec(layer_start=1, layer_count=3, grid_method=GridMethod.ATR, grid_params={"atr_multiplier": 1.5}),
        GridZoneSpec(layer_start=4, layer_count=3, grid_method=GridMethod.DRAWDOWN_FROM_HIGH, grid_params={"drawdown_pct": 0.05}),
        GridZoneSpec(layer_start=7, layer_count=4, grid_method=GridMethod.VOLATILITY, grid_params={"base_pct": 0.02}),
    ]
    om = OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=10, zones=zones)
    z1 = om._active_zone(1)
    z2 = om._active_zone(2)
    z3 = om._active_zone(3)
    z4 = om._active_zone(4)
    z6 = om._active_zone(6)
    z7 = om._active_zone(7)
    z10 = om._active_zone(10)
    assert z1 is not None and z1.grid_method == GridMethod.ATR
    assert z2 is not None and z2.grid_method == GridMethod.ATR
    assert z3 is not None and z3.grid_method == GridMethod.ATR
    assert z4 is not None and z4.grid_method == GridMethod.DRAWDOWN_FROM_HIGH
    assert z6 is not None and z6.grid_method == GridMethod.DRAWDOWN_FROM_HIGH
    assert z7 is not None and z7.grid_method == GridMethod.VOLATILITY
    assert z10 is not None and z10.grid_method == GridMethod.VOLATILITY


def test_order_manager_without_zones_uses_flat_method() -> None:
    """Legacy behaviour: no zones → use flat grid_method/grid_params."""
    om = OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=5,
                      grid_method="atr", grid_params={"atr_multiplier": 2.0})
    assert om.zones is None
    zone = om._active_zone(3)
    assert zone is None


def test_select_active_zone_helper() -> None:
    zones = [
        GridZoneSpec(layer_start=1, layer_count=5, grid_method=GridMethod.FIXED_PCT, grid_params={}),
        GridZoneSpec(layer_start=6, layer_count=5, grid_method=GridMethod.ATR, grid_params={}),
    ]
    z1 = select_active_zone(zones, 1)
    z5 = select_active_zone(zones, 5)
    z6 = select_active_zone(zones, 6)
    z10 = select_active_zone(zones, 10)
    z99 = select_active_zone(zones, 99)
    assert z1 is not None and z1.grid_method == GridMethod.FIXED_PCT
    assert z5 is not None and z5.grid_method == GridMethod.FIXED_PCT
    assert z6 is not None and z6.grid_method == GridMethod.ATR
    assert z10 is not None and z10.grid_method == GridMethod.ATR
    # Out-of-bounds → falls back to last zone (defensive)
    assert z99 is not None and z99.grid_method == GridMethod.ATR


def test_compute_next_layer_price_zoned_dispatch() -> None:
    """Zoned dispatcher picks the right (method, params) per layer index."""
    from dca_calc.grid_spacing import GridContext
    zones = [
        GridZoneSpec(layer_start=1, layer_count=5, grid_method=GridMethod.FIXED_PCT,
                     grid_params={"pct": 0.02}),
        GridZoneSpec(layer_start=6, layer_count=5, grid_method=GridMethod.FIXED_PCT,
                     grid_params={"pct": 0.10}),
    ]
    ctx = GridContext(current_price=100.0, avg_entry=100.0, cycle_high=100.0,
                      layers_filled=0, n_layers_total=10)
    # Layer 1 uses pct=0.02 → trigger at 100 * (1 - 0.02) = 98
    p1 = compute_next_layer_price_zoned(zones, 1, ctx)
    assert p1 == pytest.approx(98.0, abs=1e-6)
    # Layer 6 uses pct=0.10 → trigger at 100 * (1 - 0.10) = 90
    p6 = compute_next_layer_price_zoned(zones, 6, ctx)
    assert p6 == pytest.approx(90.0, abs=1e-6)


# ============================================================
# Extract DCA params with zones
# ============================================================


def test_extract_dca_params_emits_zones_when_present() -> None:
    zones = [
        GridZoneSpec(layer_start=1, layer_count=5, grid_method=GridMethod.ATR,
                     grid_params={"atr_multiplier": 2.0}),
        GridZoneSpec(layer_start=6, layer_count=5, grid_method=GridMethod.VOLATILITY,
                     grid_params={"base_pct": 0.01}),
    ]
    g = CandidateGenome(
        genome_id="zones_test",
        dca_genome=DcaGenome(
            grid_method=GridMethod.ATR,
            grid_params={"pct": 0.015, "max_layers": 10, "tp_pct": 0.02},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=10,
            zones=zones,
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
    )
    params = extract_dca_params_from_genome(g)
    assert params["zones"] is not None
    assert len(params["zones"]) == 2
    assert params["zones"][0].grid_method == GridMethod.ATR
    assert params["zones"][1].grid_method == GridMethod.VOLATILITY


def test_extract_dca_params_zones_none_for_legacy_genomes() -> None:
    """Genomes without zones get zones=None (single-zone legacy)."""
    g = CandidateGenome(
        genome_id="legacy_test",
        dca_genome=DcaGenome(
            grid_method=GridMethod.ATR,
            grid_params={"pct": 0.015, "max_layers": 5, "tp_pct": 0.02},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=5,
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
    )
    params = extract_dca_params_from_genome(g)
    assert params["zones"] is None


# ============================================================
# Zone preservation across mutation/crossover
# ============================================================


def test_mutation_preserves_zones() -> None:
    """Zones are part of the combo contract and survive mutation."""
    zones = [
        GridZoneSpec(layer_start=1, layer_count=5, grid_method=GridMethod.ATR, grid_params={"atr_multiplier": 1.5}),
        GridZoneSpec(layer_start=6, layer_count=5, grid_method=GridMethod.VOLATILITY, grid_params={}),
    ]
    parent = random_candidate_genome(rng=random.Random(42), forced_grid_method=GridMethod.ATR,
                                     generation_index=0, zones=zones)
    assert parent.dca_genome.zones is not None
    child = mutate(parent, rng=random.Random(43), mutation_rate=1.0)
    assert child.dca_genome.zones is not None
    assert len(child.dca_genome.zones) == 2
    assert child.dca_genome.zones[0].grid_method == GridMethod.ATR
    assert child.dca_genome.zones[1].grid_method == GridMethod.VOLATILITY


def test_crossover_prefers_zoned_parent() -> None:
    """When one parent has zones and the other doesn't, child gets zones."""
    zoned = random_candidate_genome(rng=random.Random(1), forced_grid_method=GridMethod.ATR,
                                    generation_index=0, zones=[
        GridZoneSpec(1, 5, GridMethod.ATR, {}),
        GridZoneSpec(6, 5, GridMethod.VOLATILITY, {}),
    ])
    unzoned = random_candidate_genome(rng=random.Random(2), forced_grid_method=GridMethod.FIXED_PCT,
                                      generation_index=0)
    # Cross with unzoned as primary
    child = crossover(unzoned, zoned, rng=random.Random(3))
    assert child.dca_genome.zones is not None
    assert child.dca_genome.zones[0].grid_method == GridMethod.ATR


def test_crossover_preserves_zones_when_both_have_them() -> None:
    """When both parents have zones, child takes parent_a's zones (primary)."""
    zones_a = [GridZoneSpec(1, 10, GridMethod.ATR, {"atr_multiplier": 2.0})]
    zones_b = [GridZoneSpec(1, 10, GridMethod.VOLATILITY, {"base_pct": 0.01})]
    parent_a = random_candidate_genome(rng=random.Random(10), forced_grid_method=GridMethod.ATR,
                                       generation_index=0, zones=zones_a)
    parent_b = random_candidate_genome(rng=random.Random(11), forced_grid_method=GridMethod.VOLATILITY,
                                       generation_index=0, zones=zones_b)
    child = crossover(parent_a, parent_b, rng=random.Random(12))
    assert child.dca_genome.zones is not None
    # Primary (parent_a) wins
    assert child.dca_genome.zones[0].grid_method == GridMethod.ATR


# ============================================================
# Combo population builder
# ============================================================


def test_build_combo_population_produces_candidates_with_shared_zones() -> None:
    """All N candidates share the same zones; only allocation/cooldown vary."""
    zones = [
        GridZoneSpec(1, 5, GridMethod.ATR, {"atr_multiplier": 2.0}),
        GridZoneSpec(6, 5, GridMethod.VOLATILITY, {"base_pct": 0.01}),
    ]
    candidates = build_combo_population(rng=random.Random(99), generation_index=0,
                                       gid_start=0, zones=zones, n=10)
    assert len(candidates) == 10
    # All share the same zones (same method + same params)
    for c in candidates:
        assert c.dca_genome.zones is not None
        assert len(c.dca_genome.zones) == 2
        assert c.dca_genome.zones[0].grid_method == GridMethod.ATR
        assert c.dca_genome.zones[1].grid_method == GridMethod.VOLATILITY
        # max_dca_layers derived from zones (sum of layer_count)
        assert c.dca_genome.max_dca_layers == 10


def test_build_combo_population_rejects_empty_zones() -> None:
    """Empty zones list is invalid — combo population requires a contract."""
    with pytest.raises(ValueError, match="non-empty"):
        build_combo_population(rng=random.Random(0), generation_index=0, gid_start=0,
                               zones=[], n=10)


# ============================================================
# Combo contract wiring
# ============================================================


def test_combo_contract_from_spec_carries_all_fields() -> None:
    combos = build_stage2_combos(TOP5_NAMES)
    sample = combos[0]  # first 3-split pair
    contract = combo_contract_from_spec(sample)
    assert contract.name == sample.name
    assert contract.families == tuple(sample.families)
    assert contract.split_strategy == sample.split_strategy
    assert contract.max_dca_layers == sample.max_dca_layers
    assert len(contract.zones) == len(sample.zones)
    assert len(contract.family_grid_methods) == len(sample.families)


def test_active_combo_contract_round_trip() -> None:
    combos = build_stage2_combos(TOP5_NAMES)
    contract = combo_contract_from_spec(combos[0])
    set_active_combo_contract(contract)
    try:
        from evolution.family_contracts import active_combo_contract

        active = active_combo_contract()
        assert active is not None
        assert active.name == contract.name
    finally:
        clear_active_combo_contract()
    # Cleared
    from evolution.family_contracts import active_combo_contract

    assert active_combo_contract() is None


# ============================================================
# Stage 2 queue runner (dry-run)
# ============================================================


def test_stage2_queue_runner_dry_run_starts_first_combo(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "stage2_state.json"
    # Stub out the completion check so all 60 appear pending
    monkeypatch.setattr(queue_runner, "_combo_complete", lambda _name: False)

    result = queue_runner.run_queue_once(state_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["next_action"] == "would_start_combo"
    assert result["total_combos"] == 60
    assert result["completed_count"] == 0
    assert result["current"]["combo"] == build_stage2_combos(TOP5_NAMES)[0].name
    # Dry-run should NOT persist state
    assert not state_path.exists()


def test_stage2_queue_runner_status_when_all_complete(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "stage2_state.json"
    # All complete → next_action should be review_stage2_ranking
    monkeypatch.setattr(queue_runner, "_combo_complete", lambda _name: True)

    result = queue_runner.run_queue_once(state_path, dry_run=False)

    assert result["status"] == "complete"
    assert result["next_action"] == "review_stage2_ranking"
    assert result["completed_count"] == 60
    assert result["pending_count"] == 0


# ============================================================
# Backward compatibility — single-zone genomes still work
# ============================================================


def test_legacy_genome_without_zones_validates() -> None:
    """Genomes with zones=None validate cleanly (no regression for Stage 1)."""
    g = CandidateGenome(
        genome_id="legacy",
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"pct": 0.015, "max_layers": 5, "tp_pct": 0.02},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={},
            max_dca_layers=5,
            zones=None,  # explicit None = single-zone legacy
        ),
        tp_genome=TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.02}),
    )
    validate_genome(g)  # should not raise


def test_order_manager_legacy_signature_unchanged() -> None:
    """OrderManager constructor without zones argument still works (defaults to None)."""
    om = OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=5)
    assert om.zones is None
    assert om.grid_method == "fixed_pct"
    assert om.grid_params == {"pct": 0.015}