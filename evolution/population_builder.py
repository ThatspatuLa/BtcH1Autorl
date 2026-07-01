"""PopulationBuilder — constructs the 500-candidate GA population per the spec.

Split:
  250 = exploit current volhigh survivor family
  150 = explore new DCA families (RSI, MA-distance, Z-score, drawdown, ATR, trend)
   75 = hybrid crossover between known and new families
   25 = fresh random valid genomes

Total: 500 candidates per generation.

All generated genomes use only grid methods that OrderManager actually executes:
fixed_pct, atr, volatility, drawdown_from_high, ma_distance, rsi_oversold,
z_score, trend_adjusted.

All generated genomes use only confirmation indicators that are fully computed
and consumed: rsi_below, rsi_above, ma_above, ma_below, volatility_high,
volatility_low.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ComboMethod,
    ConfirmationIndicator,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    TpExitMethod,
    TpGenome,
    TriggerMode,
)
from evolution.operators import (
    ALL_CONFIRMATION_INDICATORS,
    ALL_GRID_METHODS,
    DCA_PARAM_RANGES,
    GRID_METHOD_DEFAULT_PARAMS,
    INDICATOR_DEFAULT_PARAMS,
    crossover,
    mutate,
    random_candidate_genome,
)

# ============================================================
# Family constraints — set externally before hyperopt runs.
# When active, all population generation respects these bounds.
# ============================================================

_family_constraints: dict[str, Any] = {}


def set_family_constraints(
    forced_grid_methods: tuple | None = None,
    forced_allocation: AllocationMethod | None = None,
    forced_confirmations: tuple | None = None,
    max_dca_layers_cap: int | None = None,
) -> None:
    """Set family DNA constraints for all subsequent population generation.

    Call once before starting an evolution run. Call clear_family_constraints()
    afterward to restore unconstrained generation.
    """
    global _family_constraints
    _family_constraints = {
        "forced_grid_methods": forced_grid_methods if forced_grid_methods else None,
        "forced_allocation": forced_allocation,
        "forced_confirmations": forced_confirmations if forced_confirmations is not None else None,
        "max_dca_layers_cap": max_dca_layers_cap,
    }


def clear_family_constraints() -> None:
    """Clear all family constraints (restore unconstrained generation)."""
    global _family_constraints
    _family_constraints = {}


def _get_constrained_grid_methods(pool: list[GridMethod]) -> list[GridMethod]:
    """Return grid methods filtered by family constraints (if active)."""
    forced = _family_constraints.get("forced_grid_methods")
    if forced:
        return [g for g in pool if g in forced]
    return pool


def _get_constrained_allocation() -> tuple[AllocationMethod, dict[str, float]] | None:
    """Return forced allocation if set, else None."""
    return None  # handled inline below


def _constrained_pick_allocation(rng: random.Random) -> tuple[AllocationMethod, dict[str, float]]:
    """Pick allocation method, respecting family constraints if active."""
    forced = _family_constraints.get("forced_allocation")
    if forced:
        return forced, _build_allocation_params(rng, forced)
    return _pick_allocation(rng)


def _constrained_pick_confirmations(
    rng: random.Random,
    required: list[ConfirmationIndicator] | None = None,
) -> tuple[list[ConfirmationIndicator], dict[str, dict[str, float]]]:
    """Pick confirmations, respecting family constraints if active."""
    forced = _family_constraints.get("forced_confirmations")
    if forced is not None:
        indicators = list(forced)
        params: dict[str, dict[str, float]] = {}
        for ind in indicators:
            if ind.value in INDICATOR_DEFAULT_PARAMS:
                params[ind.value] = dict(INDICATOR_DEFAULT_PARAMS[ind.value])
                for key in params[ind.value]:
                    current = params[ind.value][key]
                    params[ind.value][key] = current + rng.gauss(0, abs(current) * 0.1)
        return indicators, params
    return _pick_confirmations(rng, required=required)


def _constrained_random_layers(rng: random.Random) -> int:
    """Pick random max_layers, respecting family cap if active."""
    cap = _family_constraints.get("max_dca_layers_cap")
    if cap is not None:
        lo, _ = DCA_PARAM_RANGES["max_layers"]
        return rng.randint(int(lo), cap)
    return _random_layers(rng)


def _make_genome_id(generation_index: int, gid: int) -> str:
    return f"genome_G{generation_index}_{gid:06d}"


def _build_grid_params(
    rng: random.Random,
    grid_method: GridMethod,
    base_pct: float | None = None,
) -> dict[str, float]:
    """Build grid_params for the given grid method with random values in range."""
    defaults = dict(GRID_METHOD_DEFAULT_PARAMS.get(grid_method.value, {"pct": 0.015}))
    params: dict[str, float] = {}

    # pct is always present
    if base_pct is not None:
        params["pct"] = base_pct
    else:
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        params["pct"] = rng.uniform(lo, hi)

    # Method-specific params
    if grid_method == GridMethod.ATR:
        params["atr_multiplier"] = rng.uniform(1.0, 4.0)
    elif grid_method == GridMethod.VOLATILITY:
        params["base_pct"] = rng.uniform(0.005, 0.03)
        params["vol_scale_factor"] = rng.uniform(0.1, 1.0)
    elif grid_method == GridMethod.DRAWDOWN_FROM_HIGH:
        params["drawdown_pct"] = rng.uniform(0.02, 0.10)
    elif grid_method == GridMethod.MA_DISTANCE:
        params["ma_distance_pct"] = rng.uniform(0.01, 0.08)
    elif grid_method == GridMethod.RSI_OVERSOLD:
        params["rsi_threshold"] = rng.uniform(20.0, 40.0)
        params["oversold_depth_pct"] = rng.uniform(0.01, 0.05)
    elif grid_method == GridMethod.Z_SCORE:
        params["z_threshold"] = rng.uniform(1.0, 3.0)
        params["lookback_std"] = rng.uniform(0.01, 0.05)
    elif grid_method == GridMethod.TREND_ADJUSTED:
        params["base_pct"] = rng.uniform(0.005, 0.03)
        params["trend_multiplier"] = rng.uniform(0.1, 1.0)

    return params


def _pick_allocation(rng: random.Random) -> tuple[AllocationMethod, dict[str, float]]:
    """Pick a random allocation method and its params."""
    method = rng.choice([
        AllocationMethod.EQUAL,
        AllocationMethod.LINEAR_INCREASING,
        AllocationMethod.CONTROLLED_EXP,
        AllocationMethod.DRAWDOWN_ADJUSTED,
        AllocationMethod.VOLATILITY_ADJUSTED,
    ])
    params: dict[str, float] = {}
    if method == AllocationMethod.LINEAR_INCREASING:
        params["increment_pct"] = rng.uniform(0.05, 0.5)
    elif method == AllocationMethod.CONTROLLED_EXP:
        params["multiplier"] = rng.uniform(1.2, 3.0)
        params["max_layer_size_pct"] = rng.uniform(2.0, 10.0)
    elif method == AllocationMethod.DRAWDOWN_ADJUSTED:
        params["sensitivity"] = rng.uniform(1.0, 5.0)
        params["min_size_pct"] = rng.uniform(0.3, 1.0)
        params["max_size_pct"] = rng.uniform(2.0, 10.0)
    elif method == AllocationMethod.VOLATILITY_ADJUSTED:
        params["reference_vol"] = rng.uniform(0.01, 0.05)
        params["min_size_pct"] = rng.uniform(0.3, 1.0)
        params["max_size_pct"] = rng.uniform(2.0, 5.0)
    return method, params


def _pick_confirmations(
    rng: random.Random,
    required: list[ConfirmationIndicator] | None = None,
    max_indicators: int = 3,
) -> tuple[list[ConfirmationIndicator], dict[str, dict[str, float]]]:
    """Pick random confirmation indicators, optionally requiring some."""
    indicators: list[ConfirmationIndicator] = list(required or [])
    remaining = [i for i in ALL_CONFIRMATION_INDICATORS if i not in indicators]
    n_extra = rng.randint(0, min(max_indicators - len(indicators), len(remaining)))
    if n_extra > 0:
        indicators.extend(rng.sample(remaining, k=n_extra))
    # Build params
    params: dict[str, dict[str, float]] = {}
    for ind in indicators:
        if ind.value in INDICATOR_DEFAULT_PARAMS:
            params[ind.value] = dict(INDICATOR_DEFAULT_PARAMS[ind.value])
            # Tweak thresholds slightly
            for key in params[ind.value]:
                current = params[ind.value][key]
                params[ind.value][key] = current + rng.gauss(0, abs(current) * 0.1)
    return indicators, params


def _make_candidate(
    rng: random.Random,
    generation_index: int,
    gid: int,
    grid_method: GridMethod,
    grid_params: dict[str, float],
    tp_pct: float,
    max_layers: int,
    cooldown: int,
    allocation_method: AllocationMethod,
    allocation_params: dict[str, float],
    confirmation_indicators: list[ConfirmationIndicator],
    indicator_params: dict[str, dict[str, float]],
) -> CandidateGenome:
    """Build a CandidateGenome from explicit params."""
    grid_params_full = dict(grid_params)
    grid_params_full["max_layers"] = max_layers
    grid_params_full["tp_pct"] = tp_pct
    grid_params_full["cooldown_candles"] = cooldown

    dca = DcaGenome(
        grid_method=grid_method,
        grid_params=grid_params_full,
        allocation_method=allocation_method,
        allocation_params=allocation_params,
        combo_method=ComboMethod.WEIGHTED_AVERAGE,
        combo_params={},
        trigger_mode=TriggerMode.PRICE_ONLY,
        confirmation_indicators=confirmation_indicators,
        indicator_params=indicator_params,
        max_dca_layers=max_layers,
    )
    tp = TpGenome(
        exit_method=TpExitMethod.FIXED,
        exit_params={"tp_pct": tp_pct},
    )
    return CandidateGenome(
        genome_id=_make_genome_id(generation_index, gid),
        dca_genome=dca,
        tp_genome=tp,
        lineage=LineageMetadata(
            parent_a_id=None,
            parent_b_id=None,
            generation_index=generation_index,
            mutation_seed=rng.randint(0, 2**31 - 1),
        ),
    )


def _random_pct(rng: random.Random) -> float:
    lo, hi = DCA_PARAM_RANGES["grid_pct"]
    return rng.uniform(lo, hi)


def _random_tp(rng: random.Random) -> float:
    lo, hi = DCA_PARAM_RANGES["tp_pct"]
    return rng.uniform(lo, hi)


def _random_layers(rng: random.Random) -> int:
    """Random max_layers value within policy range (User directive 2026-06-23: max 5)."""
    lo, hi = DCA_PARAM_RANGES["max_layers"]
    return rng.randint(int(lo), int(hi))


def _random_cooldown(rng: random.Random) -> int:
    lo, hi = DCA_PARAM_RANGES["cooldown_candles"]
    return rng.randint(int(lo), int(hi))


# ============================================================
# Population builder
# ============================================================

def build_exploit_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    n: int,
) -> list[CandidateGenome]:
    """250 candidates: exploit the known volhigh survivor family.

    Variations around: volatility_high confirmation + tight grid + quick TP.
    Grid methods: mostly fixed_pct, some atr and volatility.
    Confirmations: always include volatility_high, optionally add rsi_below or ma_below.
    """
    candidates: list[CandidateGenome] = []
    gid = gid_start

    for i in range(n):
        # Grid method: 70% fixed_pct, 15% atr, 15% volatility
        # (constrained to family's forced grid methods if active)
        forced = _family_constraints.get("forced_grid_methods")
        if forced:
            gm = rng.choice(forced)
        else:
            r = rng.random()
            if r < 0.70:
                gm = GridMethod.FIXED_PCT
            elif r < 0.85:
                gm = GridMethod.ATR
            else:
                gm = GridMethod.VOLATILITY

        gp = _build_grid_params(rng, gm)
        tp = _random_tp(rng)
        ml = _constrained_random_layers(rng)
        cd = _random_cooldown(rng)
        am, ap = _constrained_pick_allocation(rng)

        # Confirmations: always volhigh + optional second
        required = [ConfirmationIndicator.VOLATILITY_HIGH]
        # 40% chance to add rsi_below, 20% ma_below, 40% just volhigh alone
        r2 = rng.random()
        if r2 < 0.40:
            required.append(ConfirmationIndicator.RSI_BELOW)
        elif r2 < 0.60:
            required.append(ConfirmationIndicator.MA_BELOW)
        inds, ip = _constrained_pick_confirmations(rng, required=required)

        # Vol threshold: 1.1–2.0 (spec range)
        if "volatility_high" in ip:
            ip["volatility_high"]["threshold"] = rng.uniform(1.1, 2.0)

        candidates.append(_make_candidate(
            rng, generation_index, gid, gm, gp, tp, ml, cd, am, ap, inds, ip,
        ))
        gid += 1

    return candidates


def build_explore_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    n: int,
) -> list[CandidateGenome]:
    """150 candidates: explore new DCA families.

    Grid methods: rsi_oversold, ma_distance, z_score, drawdown_from_high,
    trend_adjusted, atr (without volhigh confirmation).
    Confirmations: varied, NOT centered on volhigh.
    """
    candidates: list[CandidateGenome] = []
    gid = gid_start

    explore_methods = [
        GridMethod.RSI_OVERSOLD,
        GridMethod.MA_DISTANCE,
        GridMethod.Z_SCORE,
        GridMethod.DRAWDOWN_FROM_HIGH,
        GridMethod.TREND_ADJUSTED,
        GridMethod.ATR,
    ]
    # Apply family constraints: filter explore pool to forced methods
    explore_methods = _get_constrained_grid_methods(explore_methods)
    if not explore_methods:
        explore_methods = [GridMethod.FIXED_PCT]  # fallback if constraint eliminates all

    for i in range(n):
        gm = rng.choice(explore_methods)
        gp = _build_grid_params(rng, gm)
        tp = _random_tp(rng)
        ml = _constrained_random_layers(rng)
        cd = _random_cooldown(rng)
        am, ap = _constrained_pick_allocation(rng)

        # Confirmations: varied, no volhigh bias
        # 30% rsi_below, 20% ma_below, 20% rsi_above, 10% ma_above, 20% volhigh
        r = rng.random()
        required: list[ConfirmationIndicator] = []
        if r < 0.30:
            required = [ConfirmationIndicator.RSI_BELOW]
        elif r < 0.50:
            required = [ConfirmationIndicator.MA_BELOW]
        elif r < 0.70:
            required = [ConfirmationIndicator.RSI_ABOVE]
        elif r < 0.80:
            required = [ConfirmationIndicator.MA_ABOVE]
        else:
            required = [ConfirmationIndicator.VOLATILITY_HIGH]
        inds, ip = _constrained_pick_confirmations(rng, required=required)

        candidates.append(_make_candidate(
            rng, generation_index, gid, gm, gp, tp, ml, cd, am, ap, inds, ip,
        ))
        gid += 1

    return candidates


def build_hybrid_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    n: int,
    exploit_pool: list[CandidateGenome],
) -> list[CandidateGenome]:
    """75 candidates: hybrid crossover between volhigh family and new families.

    Take one parent from exploit_pool (volhigh family) and one from a newly
    generated explore-style genome, then crossover.
    """
    candidates: list[CandidateGenome] = []
    gid = gid_start

    explore_methods = [
        GridMethod.RSI_OVERSOLD,
        GridMethod.MA_DISTANCE,
        GridMethod.Z_SCORE,
        GridMethod.DRAWDOWN_FROM_HIGH,
        GridMethod.TREND_ADJUSTED,
    ]
    # Apply family constraints
    constrained_explore = _get_constrained_grid_methods(explore_methods)
    if not constrained_explore:
        constrained_explore = [GridMethod.FIXED_PCT]

    for i in range(n):
        # Parent A: from exploit pool (volhigh family)
        parent_a = rng.choice(exploit_pool)

        # Parent B: new explore-style genome
        gm = rng.choice(constrained_explore)
        gp = _build_grid_params(rng, gm)
        tp = _random_tp(rng)
        ml = _constrained_random_layers(rng)
        cd = _random_cooldown(rng)
        am, ap = _constrained_pick_allocation(rng)
        r = rng.random()
        required: list[ConfirmationIndicator] = []
        if r < 0.50:
            required = [ConfirmationIndicator.VOLATILITY_HIGH]
        elif r < 0.80:
            required = [ConfirmationIndicator.RSI_BELOW]
        else:
            required = [ConfirmationIndicator.MA_BELOW]
        inds, ip = _constrained_pick_confirmations(rng, required=required)
        parent_b = _make_candidate(
            rng, generation_index, gid + 10000, gm, gp, tp, ml, cd, am, ap, inds, ip,
        )

        child = crossover(parent_a, parent_b, rng=rng)
        # Override the child's genome_id to our gid scheme
        child.genome_id = _make_genome_id(generation_index, gid)
        child.lineage.generation_index = generation_index
        candidates.append(child)
        gid += 1

    return candidates


def build_random_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    n: int,
) -> list[CandidateGenome]:
    """25 candidates: fresh random valid genomes from the (possibly constrained) search space."""
    candidates: list[CandidateGenome] = []
    for i in range(n):
        gid = gid_start + i
        tp = _random_tp(rng)
        g = random_candidate_genome(
            rng=rng,
            genome_id=_make_genome_id(generation_index, gid),
            generation_index=generation_index,
            tp_pct=tp,
        )
        # Apply family constraints post-generation
        forced_gm = _family_constraints.get("forced_grid_methods")
        if forced_gm and g.dca_genome.grid_method not in forced_gm:
            # Re-generate grid params for a valid method
            new_gm = rng.choice(forced_gm)
            g.dca_genome.grid_method = new_gm
            g.dca_genome.grid_params = _build_grid_params(rng, new_gm)
        forced_alloc = _family_constraints.get("forced_allocation")
        if forced_alloc and g.dca_genome.allocation_method != forced_alloc:
            g.dca_genome.allocation_method = forced_alloc
            g.dca_genome.allocation_params = _build_allocation_params(rng, forced_alloc)
        forced_conf = _family_constraints.get("forced_confirmations")
        if forced_conf is not None:
            g.dca_genome.confirmation_indicators = list(forced_conf)
        cap = _family_constraints.get("max_dca_layers_cap")
        if cap is not None:
            g.dca_genome.max_dca_layers = min(g.dca_genome.max_dca_layers, cap)
            g.dca_genome.grid_params["max_layers"] = g.dca_genome.max_dca_layers
        # Ensure cooldown is set
        g.dca_genome.grid_params["cooldown_candles"] = _random_cooldown(rng)
        candidates.append(g)
    return candidates


def build_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int = 0,
) -> list[CandidateGenome]:
    """Build the full 500-candidate population per the spec.

    Returns exactly 500 candidates:
      250 exploit + 150 explore + 75 hybrid + 25 random
    """
    exploit = build_exploit_population(rng, generation_index, gid_start, 250)
    explore = build_explore_population(rng, generation_index, gid_start + 250, 150)
    hybrid = build_hybrid_population(rng, generation_index, gid_start + 400, 75, exploit)
    rand = build_random_population(rng, generation_index, gid_start + 475, 25)

    population = exploit + explore + hybrid + rand
    assert len(population) == 500, f"Expected 500, got {len(population)}"
    return population


# ============================================================
# Island-mode population (Plan B, effective 2026-06-22)
# ============================================================
#
# 8 islands × 62 candidates = 496, plus 4 random = 500 total.
# Each island forces a structural bias (grid method, allocation, layer cap)
# so it evolves a distinct niche. The harness routes candidates to islands
# by genome_id, then per-island elites + migration every 5 gens.

def _seed_island(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    n: int,
    island_spec,  # evolution.islands.IslandSpec — avoid circular import
    base_pct: float | None = None,
) -> list[CandidateGenome]:
    """Build n candidates biased by island_spec's forced fields."""
    from evolution.islands import IslandSpec  # local import to avoid circular

    candidates: list[CandidateGenome] = []
    gid = gid_start

    for _ in range(n):
        # Grid method
        if island_spec.forced_grid_methods is not None:
            gm = rng.choice(island_spec.forced_grid_methods)
        else:
            gm = rng.choice(ALL_GRID_METHODS)

        gp = _build_grid_params(rng, gm, base_pct=base_pct)

        # Max layers (apply cap if set)
        ml = _constrained_random_layers(rng)
        if island_spec.max_dca_layers_cap is not None:
            ml = min(ml, island_spec.max_dca_layers_cap)

        tp = _random_tp(rng)
        cd = _random_cooldown(rng)

        # Allocation method
        if island_spec.forced_allocation is not None:
            am = island_spec.forced_allocation
            ap = _build_allocation_params(rng, am)
        else:
            am, ap = _constrained_pick_allocation(rng)

        # Confirmations: if forced, use that; otherwise random
        if island_spec.forced_confirmations is not None:
            required = [island_spec.forced_confirmations[rng.randint(0, len(island_spec.forced_confirmations) - 1)]]
            inds, ip = _constrained_pick_confirmations(rng, required=required)
        else:
            # No forced confirmations — 40% no confirmations, 60% one random
            r = rng.random()
            if r < 0.40:
                inds, ip = [], {}
            else:
                inds, ip = _constrained_pick_confirmations(rng, required=None)

        c = _make_candidate(
            rng, generation_index, gid, gm, gp, tp, ml, cd, am, ap, inds, ip,
        )
        # Tag island_id into lineage.mutation_ops so it's accessible during eval
        c.lineage.mutation_ops = list(c.lineage.mutation_ops) + [{
            "op": "island_assign", "island_id": island_spec.island_id,
        }]
        candidates.append(c)
        gid += 1

    return candidates


def _build_allocation_params(
    rng: random.Random,
    method: AllocationMethod,
) -> dict[str, float]:
    """Build allocation params for the given method."""
    if method == AllocationMethod.EQUAL:
        return {}
    if method == AllocationMethod.LINEAR_INCREASING:
        return {"increment_pct": rng.uniform(0.05, 0.5)}
    if method == AllocationMethod.CONTROLLED_EXP:
        return {
            "multiplier": rng.uniform(1.2, 3.0),
            "max_layer_size_pct": rng.uniform(2.0, 10.0),
        }
    if method == AllocationMethod.DRAWDOWN_ADJUSTED:
        return {
            "sensitivity": rng.uniform(1.0, 5.0),
            "min_size_pct": rng.uniform(0.3, 1.0),
            "max_size_pct": rng.uniform(2.0, 10.0),
        }
    if method == AllocationMethod.VOLATILITY_ADJUSTED:
        return {
            "reference_vol": rng.uniform(0.01, 0.05),
            "min_size_pct": rng.uniform(0.3, 1.0),
            "max_size_pct": rng.uniform(2.0, 5.0),
        }
    return {}


def build_island_population(
    rng: random.Random,
    generation_index: int,
    island_specs,  # list[IslandSpec]
    gid_start: int = 0,
    random_count: int = 4,
) -> list[CandidateGenome]:
    """Build a 500-candidate population partitioned across islands.

    Each island gets its forced biases applied. Final `random_count`
    candidates are pure-random (assigned to island 0 by the harness).
    Each candidate is tagged with its island_id via lineage.mutation_ops
    so it's accessible during eval (no separate tracker needed).
    """
    candidates: list[CandidateGenome] = []
    gid = gid_start

    for spec in island_specs:
        candidates.extend(_seed_island(rng, generation_index, gid, spec.n_candidates, spec))
        gid += spec.n_candidates

    # Pure-random tail (island 0)
    for _ in range(random_count):
        from evolution.operators import random_candidate_genome as _rcg
        c = _rcg(rng=rng, genome_id=_make_genome_id(generation_index, gid), generation_index=generation_index)
        # Ensure cooldown is set
        c.dca_genome.grid_params["cooldown_candles"] = _random_cooldown(rng)
        c.lineage.mutation_ops = list(c.lineage.mutation_ops) + [{
            "op": "island_assign", "island_id": 0,
        }]
        candidates.append(c)
        gid += 1

    return candidates


def get_island_id_for_genome(genome: CandidateGenome) -> int:
    """Read island_id from a genome's lineage.mutation_ops.

    Returns 0 if no island_assign tag is found (defaults to "random" island).
    """
    for op in reversed(genome.lineage.mutation_ops or []):
        if isinstance(op, dict) and op.get("op") == "island_assign":
            return int(op.get("island_id", 0))
    return 0


# ============================================================
# Combo population (Stage 2)
# ============================================================


def build_combo_population(
    rng: random.Random,
    generation_index: int,
    gid_start: int,
    zones: list,
    n: int,
) -> list[CandidateGenome]:
    """Build n combo candidates all sharing the same per-layer zones.

    Each candidate has:
    - zones from the combo spec (immutable across mutation/crossover)
    - a grid_method equal to zones[0].grid_method (flat — OrderManager picks from zones)
    - random allocation_method + allocation_params (free to mutate)
    - random confirmation_indicators (free to mutate, capped at 3)
    - random cooldown_candles
    - max_dca_layers = sum of zone layer_count (from the combo contract)

    Allocation/depth/cooldown vary per candidate. Zones stay fixed — that's the
    whole point of a combo: the layer-to-method mapping is the contract, the
    candidate's job is to find the best params WITHIN that contract.
    """
    from evolution.operators import random_candidate_genome as _rcg

    if not zones:
        raise ValueError("build_combo_population requires non-empty zones list")

    # Sum layer_count across zones — must equal max_dca_layers.
    max_layers_total = sum(z.layer_count for z in zones)
    # Each candidate gets the same zones, same grid_method (zones[0] for the flat slot),
    # but allocation/cooldown/confirmation_indicators are randomised.
    primary_method = zones[0].grid_method
    primary_params = dict(zones[0].grid_params)
    primary_params["max_layers"] = max_layers_total
    primary_params["tp_pct"] = rng.uniform(*DCA_PARAM_RANGES["tp_pct"])

    candidates: list[CandidateGenome] = []
    for i in range(n):
        gid = gid_start + i
        cooldown = rng.randint(
            int(DCA_PARAM_RANGES["cooldown_candles"][0]),
            int(DCA_PARAM_RANGES["cooldown_candles"][1]),
        )
        primary_params["cooldown_candles"] = cooldown
        c = _rcg(
            rng=rng,
            genome_id=_make_genome_id(generation_index, gid),
            generation_index=generation_index,
            tp_pct=primary_params["tp_pct"],
            forced_grid_method=primary_method,
            zones=list(zones),  # copy so future candidates aren't aliased
        )
        # The flat grid_method slot is set to zones[0]'s method (legacy compat).
        # OrderManager uses zones when present, so this is for reporting only.
        c.dca_genome.grid_method = primary_method
        c.dca_genome.grid_params = dict(primary_params)
        c.dca_genome.max_dca_layers = max_layers_total
        candidates.append(c)
    return candidates
