"""HyperoptConfig — Stage 10 family-budgeted hyperopt (3-phase).

Phase 1: Discovery sweep — 22 pure-axis families × 500 epochs each.
Phase 2: Deep optimisation — top-5 families × 5,000 epochs each.
Phase 3: Combo deep-dive — top-10 triples × (10 iterations × 500 epochs).

Locked: reward weights, market, timeframe, direction, shorting, safety, TP method.
Only DCA accumulation params + simple fixed-TP pct mutate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from genome.schema import (
    AllocationMethod,
    ConfirmationIndicator,
    GridMethod,
)


# ============================================================
# Phase budgets
# ============================================================

PHASE1_EPOCHS_PER_FAMILY: int = 500
PHASE2_EPOCHS_PER_FAMILY: int = 5000
PHASE3_EPOCHS_PER_ITERATION: int = 500
PHASE3_ITERATIONS_PER_COMBO: int = 10
PHASE3_TOP_N_COMBOS: int = 10  # top-10 triples from C(5,3)=10

# Selection
PHASE2_TOP_N_FAMILIES: int = 5  # advance top-5 from Phase 1

# No wall-time caps — run to completion
# No deployment-pass floor — rank by fitness only

# Candidate pool size (LOCKED, same as main evolution)
CANDIDATES_PER_GEN: int = 500
ELITE_COUNT: int = 20

# Smart mutation defaults for Phase 2+
PHASE2_MUTATION_RATE: float = 0.55
PHASE2_CROSSOVER_RATE: float = 0.35
PHASE2_RANDOM_INJECTION: int = 300

# Smart adjustment: tighten ranges to ±N% around elite median
SMART_ADJUST_TIGHTEN_FACTOR: float = 0.20


# ============================================================
# 22 Pure-Axis Families
# ============================================================

# Sentinel: "not set" vs "explicitly empty"
_UNSET = object()


@dataclass
class FamilySpec:
    """A single DCA family to test in Phase 1."""
    name: str
    forced_grid_methods: tuple[GridMethod, ...] = ()
    forced_allocation: AllocationMethod | None = None
    forced_confirmations: Any = _UNSET
    max_dca_layers_cap: int | None = None  # if set, overrides GLOBAL_MAX_DCA_LAYERS for this family
    description: str = ""

    @property
    def deterministic_seed(self) -> int:
        """Different but reproducible seed per family."""
        h = hashlib.sha256(self.name.encode()).hexdigest()
        return int(h[:8], 16)

    @property
    def group(self) -> str:
        if self.forced_allocation is not None and not self.forced_grid_methods:
            return "allocation"
        if self.forced_confirmations is not _UNSET:
            return "confirmation"
        if self.max_dca_layers_cap is not None and not self.forced_grid_methods:
            return "depth"
        return "grid"


def build_family_specs() -> list[FamilySpec]:
    """Build the 22 pure-axis family specifications."""
    return [
        # === Group A: Pure grid families (8) ===
        FamilySpec(
            name="pure_fixed_pct",
            forced_grid_methods=(GridMethod.FIXED_PCT,),
            description="Fixed percentage grid spacing only",
        ),
        FamilySpec(
            name="pure_atr",
            forced_grid_methods=(GridMethod.ATR,),
            description="ATR-based grid spacing only",
        ),
        FamilySpec(
            name="pure_volatility",
            forced_grid_methods=(GridMethod.VOLATILITY,),
            description="Volatility-based grid spacing only",
        ),
        FamilySpec(
            name="pure_drawdown",
            forced_grid_methods=(GridMethod.DRAWDOWN_FROM_HIGH,),
            description="Drawdown-from-high grid spacing only",
        ),
        FamilySpec(
            name="pure_ma_distance",
            forced_grid_methods=(GridMethod.MA_DISTANCE,),
            description="MA distance grid spacing only",
        ),
        FamilySpec(
            name="pure_rsi_oversold",
            forced_grid_methods=(GridMethod.RSI_OVERSOLD,),
            description="RSI oversold grid spacing only",
        ),
        FamilySpec(
            name="pure_z_score",
            forced_grid_methods=(GridMethod.Z_SCORE,),
            description="Z-score grid spacing only",
        ),
        FamilySpec(
            name="pure_trend_adjusted",
            forced_grid_methods=(GridMethod.TREND_ADJUSTED,),
            description="Trend-adjusted grid spacing only",
        ),
        # === Group B: Allocation-focused (5) ===
        FamilySpec(
            name="alloc_equal",
            forced_allocation=AllocationMethod.EQUAL,
            description="Equal allocation across DCA layers",
        ),
        FamilySpec(
            name="alloc_linear_inc",
            forced_allocation=AllocationMethod.LINEAR_INCREASING,
            description="Linear increasing allocation across DCA layers",
        ),
        FamilySpec(
            name="alloc_ctrl_exp",
            forced_allocation=AllocationMethod.CONTROLLED_EXP,
            description="Controlled exponential allocation across DCA layers",
        ),
        FamilySpec(
            name="alloc_dd_adj",
            forced_allocation=AllocationMethod.DRAWDOWN_ADJUSTED,
            description="Drawdown-adjusted allocation across DCA layers",
        ),
        FamilySpec(
            name="alloc_vol_adj",
            forced_allocation=AllocationMethod.VOLATILITY_ADJUSTED,
            description="Volatility-adjusted allocation across DCA layers",
        ),
        # === Group C: Confirmation-focused (6) ===
        FamilySpec(
            name="confirm_rsi",
            forced_confirmations=(ConfirmationIndicator.RSI_BELOW, ConfirmationIndicator.RSI_ABOVE),
            description="RSI-based entry confirmations",
        ),
        FamilySpec(
            name="confirm_ma",
            forced_confirmations=(ConfirmationIndicator.MA_BELOW, ConfirmationIndicator.MA_ABOVE),
            description="MA-based entry confirmations",
        ),
        FamilySpec(
            name="confirm_vol",
            forced_confirmations=(ConfirmationIndicator.VOLATILITY_HIGH, ConfirmationIndicator.VOLATILITY_LOW),
            description="Volatility-based entry confirmations",
        ),
        FamilySpec(
            name="confirm_none",
            forced_confirmations=(),
            description="No entry confirmations (price trigger only)",
        ),
        FamilySpec(
            name="confirm_rsi_ma",
            forced_confirmations=(ConfirmationIndicator.RSI_BELOW, ConfirmationIndicator.MA_ABOVE),
            description="Mixed RSI+MA entry confirmations",
        ),
        FamilySpec(
            name="confirm_vol_rsi",
            forced_confirmations=(ConfirmationIndicator.VOLATILITY_LOW, ConfirmationIndicator.RSI_BELOW),
            description="Mixed volatility+RSI entry confirmations",
        ),
        # === Group D: Depth-focused (3) ===
        FamilySpec(
            name="shallow_dca",
            max_dca_layers_cap=4,
            description="Shallow DCA: max 2-4 layers",
        ),
        FamilySpec(
            name="medium_dca",
            max_dca_layers_cap=7,
            description="Medium DCA: max 5-7 layers",
        ),
        FamilySpec(
            name="deep_dca",
            max_dca_layers_cap=10,
            description="Deep DCA: max 8-10 layers",
        ),
    ]


# ============================================================
# Phase 3 combo generation
# ============================================================

def build_triple_combos(top5_families: list[str]) -> list[dict[str, str]]:
    """Build C(5,3) = 10 triple combinations from top-5 families."""
    from itertools import combinations
    combos = []
    for trio in combinations(top5_families, 3):
        name = f"combo_{'_'.join(trio)}"
        combos.append({
            "name": name,
            "families": list(trio),
            "layer_split": {
                "layers_1_3": trio[0],
                "layers_4_6": trio[1],
                "layers_7_10": trio[2],
            },
        })
    return combos


# ============================================================
# Hyperopt run config (passed to runner)
# ============================================================

@dataclass
class HyperoptRunConfig:
    """Configuration for a single family/combo run."""
    phase: int  # 1, 2, or 3
    family_name: str
    output_dir: str
    max_generations: int
    candidates_per_gen: int = CANDIDATES_PER_GEN
    elite_count: int = ELITE_COUNT
    mutation_rate: float = 0.30
    crossover_rate: float = 0.50
    random_injection: int = 120
    stagnation_generations: int = 5
    all_rejected_generations: int = 3
    parallel_workers: int = 8
    base_seed: int = 42
    # Family DNA constraints
    forced_grid_methods: tuple[GridMethod, ...] = ()
    forced_allocation: AllocationMethod | None = None
    forced_confirmations: tuple[ConfirmationIndicator, ...] = ()
    max_dca_layers_cap: int | None = None
    # Phase 3 specific
    combo_families: list[str] = field(default_factory=list)
    layer_split: dict[str, str] = field(default_factory=dict)
    iteration: int = 0  # Phase 3 iteration number (1-10)
