"""Genetic operators — random genome generation, mutation, crossover.

Stage 10 varies the DCA genome: grid_method, grid_params, allocation_method,
confirmation_indicators, cooldown_candles. The TP genome stays fixed at the
Stage 9 baseline (tp_pct mutates as a controlled Stage 10 exit parameter).

Mutation strategy: per-parameter Gaussian noise with probability = mutation_rate.
For enum fields, mutation means re-rolling from the enum.

Crossover strategy: per-parameter uniform crossover. Each param has a 50%
chance of coming from parent A vs parent B.
"""
from __future__ import annotations

import random

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

# ============================================================
# Available confirmation indicators for random selection
# ============================================================

ALL_CONFIRMATION_INDICATORS = [
    ConfirmationIndicator.RSI_BELOW,
    ConfirmationIndicator.RSI_ABOVE,
    ConfirmationIndicator.MA_ABOVE,
    ConfirmationIndicator.MA_BELOW,
    ConfirmationIndicator.VOLATILITY_HIGH,
    ConfirmationIndicator.VOLATILITY_LOW,
]

# Default params for each indicator type
INDICATOR_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "rsi_below": {"threshold": 35.0},
    "rsi_above": {"threshold": 65.0},
    "volatility_high": {"threshold": 1.5},
    "volatility_low": {"threshold": 0.5},
}

# All wired grid methods that OrderManager actually executes
ALL_GRID_METHODS = [
    GridMethod.FIXED_PCT,
    GridMethod.ATR,
    GridMethod.VOLATILITY,
    GridMethod.DRAWDOWN_FROM_HIGH,
    GridMethod.MA_DISTANCE,
    GridMethod.RSI_OVERSOLD,
    GridMethod.Z_SCORE,
    GridMethod.TREND_ADJUSTED,
]

# Default grid_params for each grid method (used when generating random genomes)
GRID_METHOD_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "fixed_pct": {"pct": 0.015},
    "atr": {"pct": 0.015, "atr_multiplier": 2.0},
    "volatility": {"pct": 0.015, "base_pct": 0.01, "vol_scale_factor": 0.5},
    "drawdown_from_high": {"pct": 0.015, "drawdown_pct": 0.05},
    "ma_distance": {"pct": 0.015, "ma_distance_pct": 0.03},
    "rsi_oversold": {"pct": 0.015, "rsi_threshold": 30.0, "oversold_depth_pct": 0.02},
    "z_score": {"pct": 0.015, "z_threshold": 1.5, "lookback_std": 0.02},
    "trend_adjusted": {"pct": 0.015, "base_pct": 0.015, "trend_multiplier": 0.5},
}

# All allocation methods
ALL_ALLOCATION_METHODS = [
    AllocationMethod.EQUAL,
    AllocationMethod.LINEAR_INCREASING,
    AllocationMethod.CONTROLLED_EXP,
    AllocationMethod.DRAWDOWN_ADJUSTED,
    AllocationMethod.VOLATILITY_ADJUSTED,
]

# Default allocation_params for each allocation method
ALLOCATION_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "equal": {},
    "linear_increasing": {"increment_pct": 0.15},
    "controlled_exp": {"multiplier": 1.8, "max_layer_size_pct": 5.0},
    "drawdown_adjusted": {"sensitivity": 2.0, "min_size_pct": 0.5, "max_size_pct": 5.0},
    "volatility_adjusted": {"reference_vol": 0.02, "min_size_pct": 0.5, "max_size_pct": 3.0},
}

# ============================================================
# Parameter ranges (the search space) — Stage 10 widened
# ============================================================

DCA_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "grid_pct": (0.0025, 0.0125),    # 0.25% – 1.25%
    "max_layers": (6, 24),           # 6 – 24 layers
    "tp_pct": (0.002, 0.010),        # 0.2% – 1.0%
    "cooldown_candles": (0, 12),     # 0 – 12 candles
    "vol_threshold": (1.1, 2.0),     # volatility_high threshold
}

ALLOCATION_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    AllocationMethod.EQUAL: {},
    AllocationMethod.LINEAR_INCREASING: {"increment_pct": (0.05, 0.5)},
    AllocationMethod.CONTROLLED_EXP: {
        "multiplier": (1.2, 3.0),
        "max_layer_size_pct": (2.0, 10.0),
    },
    AllocationMethod.DRAWDOWN_ADJUSTED: {
        "sensitivity": (1.0, 5.0),
        "min_size_pct": (0.3, 1.0),
        "max_size_pct": (2.0, 10.0),
    },
    AllocationMethod.VOLATILITY_ADJUSTED: {
        "reference_vol": (0.01, 0.05),
        "min_size_pct": (0.3, 1.0),
        "max_size_pct": (2.0, 5.0),
    },
}


# ============================================================
# Random genome
# ============================================================

def _random_grid_params(rng: random.Random, grid_method: GridMethod) -> dict[str, float]:
    """Generate random grid_params for the given grid method."""
    defaults = dict(GRID_METHOD_DEFAULT_PARAMS.get(grid_method.value, {"pct": 0.015}))
    params = dict(defaults)
    # Mutate key params within reasonable ranges
    if "pct" in params:
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        params["pct"] = rng.uniform(lo, hi)
    if "atr_multiplier" in params:
        params["atr_multiplier"] = rng.uniform(1.0, 4.0)
    if "base_pct" in params:
        params["base_pct"] = rng.uniform(0.005, 0.03)
    if "vol_scale_factor" in params:
        params["vol_scale_factor"] = rng.uniform(0.1, 1.0)
    if "drawdown_pct" in params:
        params["drawdown_pct"] = rng.uniform(0.02, 0.10)
    if "ma_distance_pct" in params:
        params["ma_distance_pct"] = rng.uniform(0.01, 0.08)
    if "rsi_threshold" in params:
        params["rsi_threshold"] = rng.uniform(20.0, 40.0)
    if "oversold_depth_pct" in params:
        params["oversold_depth_pct"] = rng.uniform(0.01, 0.05)
    if "z_threshold" in params:
        params["z_threshold"] = rng.uniform(1.0, 3.0)
    if "lookback_std" in params:
        params["lookback_std"] = rng.uniform(0.01, 0.05)
    if "trend_multiplier" in params:
        params["trend_multiplier"] = rng.uniform(0.1, 1.0)
    return params


def _random_allocation_params(rng: random.Random, method: AllocationMethod) -> dict[str, float]:
    """Generate random allocation_params for the given allocation method."""
    ranges = ALLOCATION_PARAM_RANGES.get(method, {})
    params: dict[str, float] = {}
    for key, (lo, hi) in ranges.items():
        params[key] = rng.uniform(lo, hi)
    return params


def random_dca_genome(
    rng: random.Random | None = None,
    genome_id: str | None = None,
    generation_index: int = 0,
    tp_pct: float | None = None,
    forced_grid_method: GridMethod | None = None,
    forced_allocation_method: AllocationMethod | None = None,
) -> DcaGenome:
    """Generate a random DcaGenome within the Stage 10 search space.

    Grid method is randomly selected from ALL_GRID_METHODS (or forced).
    Allocation method is randomly selected from ALL_ALLOCATION_METHODS (or forced).
    Confirmation indicators: 0-3 from the 6 available types.
    Cooldown: 0-12 candles.
    """
    rng = rng or random.Random()

    # Grid method
    grid_method = forced_grid_method or rng.choice(ALL_GRID_METHODS)
    grid_params = _random_grid_params(rng, grid_method)

    # Max layers
    max_layers_lo, max_layers_hi = DCA_PARAM_RANGES["max_layers"]
    max_layers = rng.randint(int(max_layers_lo), int(max_layers_hi))
    grid_params["max_layers"] = max_layers

    # TP
    if tp_pct is None:
        tp_pct = rng.uniform(*DCA_PARAM_RANGES["tp_pct"])
    grid_params["tp_pct"] = tp_pct

    # Cooldown
    cd_lo, cd_hi = DCA_PARAM_RANGES["cooldown_candles"]
    cooldown = rng.randint(int(cd_lo), int(cd_hi))
    grid_params["cooldown_candles"] = cooldown

    # Allocation method
    alloc_method = forced_allocation_method or rng.choice(ALL_ALLOCATION_METHODS)
    alloc_params = _random_allocation_params(rng, alloc_method)

    # Confirmation indicators: 0-3, no duplicates
    n_indicators = rng.randint(0, 3)
    selected_indicators = rng.sample(ALL_CONFIRMATION_INDICATORS, k=n_indicators)
    indicator_params = {}
    for ind in selected_indicators:
        if ind.value in INDICATOR_DEFAULT_PARAMS:
            indicator_params[ind.value] = dict(INDICATOR_DEFAULT_PARAMS[ind.value])

    return DcaGenome(
        grid_method=grid_method,
        grid_params=grid_params,
        allocation_method=alloc_method,
        allocation_params=alloc_params,
        combo_method=ComboMethod.WEIGHTED_AVERAGE,
        combo_params={},
        trigger_mode=TriggerMode.PRICE_ONLY,
        confirmation_indicators=selected_indicators,
        indicator_params=indicator_params,
        max_dca_layers=max_layers,
    )


def random_candidate_genome(
    rng: random.Random | None = None,
    genome_id: str | None = None,
    generation_index: int = 0,
    tp_pct: float | None = None,
    forced_grid_method: GridMethod | None = None,
    forced_allocation_method: AllocationMethod | None = None,
) -> CandidateGenome:
    """Generate a full random CandidateGenome within the Stage 10 search space."""
    rng = rng or random.Random()
    dca = random_dca_genome(
        rng=rng,
        generation_index=generation_index,
        tp_pct=tp_pct,
        forced_grid_method=forced_grid_method,
        forced_allocation_method=forced_allocation_method,
    )
    gid = genome_id or f"genome_G{generation_index}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=gid,
        dca_genome=dca,
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"tp_pct": dca.grid_params["tp_pct"]},
        ),
        lineage=LineageMetadata(
            parent_a_id=None,
            parent_b_id=None,
            generation_index=generation_index,
            mutation_seed=rng.randint(0, 2**31 - 1),
        ),
    )


# ============================================================
# Mutation
# ============================================================

def mutate(
    parent: CandidateGenome,
    rng: random.Random | None = None,
    mutation_rate: float = 0.30,
    child_id: str | None = None,
) -> CandidateGenome:
    """Mutate a parent genome to produce a child.

    Mutates: grid_pct, max_layers, tp_pct, cooldown, grid_method (rare),
    allocation_method (rare), confirmation_indicators (add/remove),
    indicator_params (threshold tweaks).
    """
    rng = rng or random.Random()
    parent_dca = parent.dca_genome

    # Normalize grid_method to enum (may be string if loaded from JSON)
    raw_gm = parent_dca.grid_method
    if isinstance(raw_gm, str):
        try:
            current_grid_method = GridMethod(raw_gm)
        except ValueError:
            current_grid_method = GridMethod.FIXED_PCT
    else:
        current_grid_method = raw_gm

    # --- Allocation method (may be string if loaded from JSON) ---
    raw_am = parent_dca.allocation_method
    if isinstance(raw_am, str):
        try:
            current_alloc_method = AllocationMethod(raw_am)
        except ValueError:
            current_alloc_method = AllocationMethod.EQUAL
    else:
        current_alloc_method = raw_am

    # --- Grid method (rare structural mutation) ---
    new_grid_method = current_grid_method
    if rng.random() < mutation_rate * 0.15:  # 15% of mutation_rate chance
        new_grid_method = rng.choice(ALL_GRID_METHODS)

    # --- Grid params ---
    new_grid_params = dict(parent_dca.grid_params)

    # grid_pct
    current_pct = float(new_grid_params.get("pct", 0.015))
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        span = hi - lo
        current_pct = current_pct + rng.gauss(0, span * 0.20)
        current_pct = max(lo, min(hi, current_pct))
    new_grid_params["pct"] = current_pct

    # max_layers
    current_layers = int(new_grid_params.get("max_layers", parent_dca.max_dca_layers))
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["max_layers"]
        delta = rng.choice([-2, -1, 1, 2])
        current_layers = max(lo, min(hi, current_layers + delta))
    new_grid_params["max_layers"] = current_layers

    # tp_pct
    current_tp = float(new_grid_params.get("tp_pct", 0.02))
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["tp_pct"]
        span = hi - lo
        current_tp = current_tp + rng.gauss(0, span * 0.20)
        current_tp = max(lo, min(hi, current_tp))
    new_grid_params["tp_pct"] = current_tp

    # cooldown_candles
    current_cd = int(new_grid_params.get("cooldown_candles", 0))
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["cooldown_candles"]
        delta = rng.choice([-2, -1, 0, 1, 2])
        current_cd = max(lo, min(hi, current_cd + delta))
    new_grid_params["cooldown_candles"] = current_cd

    # Grid method specific params
    if new_grid_method.value in GRID_METHOD_DEFAULT_PARAMS:
        method_defaults = GRID_METHOD_DEFAULT_PARAMS[new_grid_method.value]
        for param_key in method_defaults:
            if param_key in ("pct", "max_layers", "tp_pct", "cooldown_candles"):
                continue  # already handled above
            if rng.random() < mutation_rate:
                current_val = float(new_grid_params.get(param_key, method_defaults[param_key]))
                # Gaussian perturbation: 20% of the default value as std
                std = abs(method_defaults[param_key]) * 0.20
                new_val = current_val + rng.gauss(0, std)
                new_grid_params[param_key] = new_val

    # --- Allocation method (rare structural mutation) ---
    new_alloc_method = current_alloc_method
    new_alloc_params = dict(parent_dca.allocation_params)
    if rng.random() < mutation_rate * 0.15:
        new_alloc_method = rng.choice(ALL_ALLOCATION_METHODS)
        new_alloc_params = _random_allocation_params(rng, new_alloc_method)
    else:
        # Tweak existing allocation params
        for key in new_alloc_params:
            if rng.random() < mutation_rate:
                lo_hi = ALLOCATION_PARAM_RANGES.get(new_alloc_method, {}).get(key)
                if lo_hi:
                    lo, hi = lo_hi
                    span = hi - lo
                    new_alloc_params[key] = max(lo, min(hi, new_alloc_params[key] + rng.gauss(0, span * 0.15)))

    # --- Confirmation indicators ---
    new_indicators = list(parent_dca.confirmation_indicators)
    new_ind_params = dict(parent_dca.indicator_params)
    if rng.random() < mutation_rate * 0.5:  # lower rate for structural changes
        if new_indicators and rng.random() < 0.5:
            # Remove one
            idx = rng.randint(0, len(new_indicators) - 1)
            removed = new_indicators.pop(idx)
            new_ind_params.pop(removed.value, None)
        else:
            # Add one (not already present)
            available = [i for i in ALL_CONFIRMATION_INDICATORS if i not in new_indicators]
            if available and len(new_indicators) < 3:
                added = rng.choice(available)
                new_indicators.append(added)
                if added.value in INDICATOR_DEFAULT_PARAMS:
                    new_ind_params[added.value] = dict(INDICATOR_DEFAULT_PARAMS[added.value])

    # Tweak indicator thresholds
    for ind_name in list(new_ind_params.keys()):
        if rng.random() < mutation_rate * 0.3:
            for param_key in new_ind_params[ind_name]:
                current_val = new_ind_params[ind_name][param_key]
                std = abs(current_val) * 0.10
                new_ind_params[ind_name][param_key] = current_val + rng.gauss(0, std)

    # Build child genome
    new_dca = DcaGenome(
        grid_method=new_grid_method,
        grid_params=new_grid_params,
        allocation_method=new_alloc_method,
        allocation_params=new_alloc_params,
        combo_method=parent_dca.combo_method,
        combo_params=dict(parent_dca.combo_params),
        trigger_mode=parent_dca.trigger_mode,
        confirmation_indicators=new_indicators,
        indicator_params=new_ind_params,
        max_dca_layers=int(current_layers),
    )
    new_tp = TpGenome(
        exit_method=parent.tp_genome.exit_method,
        exit_params={"tp_pct": current_tp},
    )
    cid = child_id or f"genome_G{parent.lineage.generation_index + 1}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=cid,
        dca_genome=new_dca,
        tp_genome=new_tp,
        lineage=LineageMetadata(
            parent_a_id=parent.genome_id,
            parent_b_id=None,
            generation_index=parent.lineage.generation_index + 1,
            mutation_seed=rng.randint(0, 2**31 - 1),
            mutation_ops=[{
                "op": "mutate",
                "parent_id": parent.genome_id,
            }],
        ),
    )


# ============================================================
# Crossover
# ============================================================

def crossover(
    parent_a: CandidateGenome,
    parent_b: CandidateGenome,
    rng: random.Random | None = None,
    child_id: str | None = None,
) -> CandidateGenome:
    """Uniform crossover: each param has 50% chance of coming from A or B.

    For enum fields (grid_method, allocation_method), pick from one parent.
    For grid_params, merge: pick each numeric param from one parent.
    For confirmation_indicators: union then randomly sample.
    """
    rng = rng or random.Random()

    a_dca = parent_a.dca_genome
    b_dca = parent_b.dca_genome

    # Normalize enum fields (may be strings if loaded from JSON)
    def _normalize_grid_method(raw):
        if isinstance(raw, str):
            try:
                return GridMethod(raw)
            except ValueError:
                return GridMethod.FIXED_PCT
        return raw

    def _normalize_alloc_method(raw):
        if isinstance(raw, str):
            try:
                return AllocationMethod(raw)
            except ValueError:
                return AllocationMethod.EQUAL
        return raw

    a_grid_method = _normalize_grid_method(a_dca.grid_method)
    b_grid_method = _normalize_grid_method(b_dca.grid_method)
    a_alloc_method = _normalize_alloc_method(a_dca.allocation_method)
    b_alloc_method = _normalize_alloc_method(b_dca.allocation_method)

    # Grid method: pick from one parent
    new_grid_method = a_grid_method if rng.random() < 0.5 else b_grid_method

    # Allocation method: pick from one parent
    new_alloc_method = a_alloc_method if rng.random() < 0.5 else b_alloc_method

    # Grid params: per-param crossover
    merged_grid: dict[str, float] = {}
    all_keys = set(a_dca.grid_params.keys()) | set(b_dca.grid_params.keys())
    for key in all_keys:
        a_val = a_dca.grid_params.get(key)
        b_val = b_dca.grid_params.get(key)
        if a_val is not None and b_val is not None:
            merged_grid[key] = float(a_val) if rng.random() < 0.5 else float(b_val)
        elif a_val is not None:
            merged_grid[key] = float(a_val)
        elif b_val is not None:
            merged_grid[key] = float(b_val)

    # Ensure required keys exist
    if "pct" not in merged_grid:
        merged_grid["pct"] = 0.015
    if "max_layers" not in merged_grid:
        merged_grid["max_layers"] = rng.randint(6, 24)
    if "tp_pct" not in merged_grid:
        merged_grid["tp_pct"] = rng.uniform(*DCA_PARAM_RANGES["tp_pct"])
    if "cooldown_candles" not in merged_grid:
        merged_grid["cooldown_candles"] = rng.randint(0, 12)

    # Allocation params: merge from the chosen parent's method
    if new_alloc_method == a_alloc_method:
        merged_alloc = dict(a_dca.allocation_params)
    else:
        merged_alloc = dict(b_dca.allocation_params)
    # If the other parent has params for this method, blend
    other_alloc = b_dca.allocation_params if new_alloc_method == a_alloc_method else a_dca.allocation_params
    for key in merged_alloc:
        if key in other_alloc and rng.random() < 0.5:
            merged_alloc[key] = float(other_alloc[key])

    # Confirmation indicators: union then sample
    combined_indicators = list(set(
        list(a_dca.confirmation_indicators) +
        list(b_dca.confirmation_indicators)
    ))
    n_keep = rng.randint(0, min(3, len(combined_indicators)))
    if n_keep > 0 and combined_indicators:
        new_indicators = rng.sample(combined_indicators, k=n_keep)
    else:
        new_indicators = []

    # Merge indicator_params
    merged_ind_params: dict[str, dict[str, float]] = {}
    for ind in new_indicators:
        a_params = a_dca.indicator_params.get(ind.value, {})
        b_params = b_dca.indicator_params.get(ind.value, {})
        if a_params and b_params:
            merged_ind_params[ind.value] = dict(a_params) if rng.random() < 0.5 else dict(b_params)
        elif a_params:
            merged_ind_params[ind.value] = dict(a_params)
        elif b_params:
            merged_ind_params[ind.value] = dict(b_params)
        elif ind.value in INDICATOR_DEFAULT_PARAMS:
            merged_ind_params[ind.value] = dict(INDICATOR_DEFAULT_PARAMS[ind.value])

    max_layers = int(merged_grid["max_layers"])
    tp_pct = float(merged_grid["tp_pct"])

    new_dca = DcaGenome(
        grid_method=new_grid_method,
        grid_params=merged_grid,
        allocation_method=new_alloc_method,
        allocation_params=merged_alloc,
        combo_method=a_dca.combo_method,
        combo_params=dict(a_dca.combo_params),
        trigger_mode=a_dca.trigger_mode,
        confirmation_indicators=new_indicators,
        indicator_params=merged_ind_params,
        max_dca_layers=max_layers,
    )
    new_tp_genome = TpGenome(
        exit_method=parent_a.tp_genome.exit_method,
        exit_params={"tp_pct": tp_pct},
    )
    cid = child_id or f"genome_G{max(parent_a.lineage.generation_index, parent_b.lineage.generation_index) + 1}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=cid,
        dca_genome=new_dca,
        tp_genome=new_tp_genome,
        lineage=LineageMetadata(
            parent_a_id=parent_a.genome_id,
            parent_b_id=parent_b.genome_id,
            generation_index=max(parent_a.lineage.generation_index, parent_b.lineage.generation_index) + 1,
            mutation_seed=rng.randint(0, 2**31 - 1),
            mutation_ops=[{
                "op": "crossover",
                "parent_a_id": parent_a.genome_id,
                "parent_b_id": parent_b.genome_id,
            }],
        ),
    )
