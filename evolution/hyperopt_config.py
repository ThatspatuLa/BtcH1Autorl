"""HyperoptConfig — Stage 10 family-budgeted hyperopt (3-phase).

Phase 1: Discovery sweep — spacing families × 500 epochs each.
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
from evolution.family_contracts import FamilyMutationContract


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
# Stage 1 Spacing Families
# ============================================================

# Sentinel: "not set" vs "explicitly empty"
_UNSET = object()


@dataclass
class FamilySpec:
    """A single spacing family to test in Phase 1."""

    name: str
    forced_grid_methods: tuple[GridMethod, ...] = ()
    forced_allocation: AllocationMethod | None = None
    forced_confirmations: Any = _UNSET
    max_dca_layers_cap: int | None = None  # if set, overrides GLOBAL_MAX_DCA_LAYERS for this family
    description: str = ""
    family_kind: str = "spacing"

    @property
    def deterministic_seed(self) -> int:
        """Different but reproducible seed per family."""
        h = hashlib.sha256(self.name.encode()).hexdigest()
        return int(h[:8], 16)

    @property
    def group(self) -> str:
        return self.family_kind

    @property
    def mutation_contract(self) -> FamilyMutationContract:
        forced_confirmations = (
            tuple(self.forced_confirmations)
            if self.forced_confirmations is not _UNSET
            else None
        )
        return FamilyMutationContract(
            name=self.name,
            forced_grid_methods=tuple(self.forced_grid_methods),
            forced_confirmations=forced_confirmations,
            allow_grid_method_switch=len(self.forced_grid_methods) > 1,
            max_dca_layers_cap=self.max_dca_layers_cap,
            notes=self.description,
        )


def build_family_specs() -> list[FamilySpec]:
    """Build Stage 1 spacing-family specifications.

    Allocation, depth, and fixed TP mutate inside every family. They are not
    standalone Stage 1 families.
    """
    no_confirmations = ()
    return [
        # === Pure executable spacing families (8) ===
        FamilySpec(
            name="fixed_pct_spacing",
            forced_grid_methods=(GridMethod.FIXED_PCT,),
            forced_confirmations=no_confirmations,
            description="Fixed percentage grid spacing only",
        ),
        FamilySpec(
            name="atr_spacing",
            forced_grid_methods=(GridMethod.ATR,),
            forced_confirmations=no_confirmations,
            description="ATR-based grid spacing only",
        ),
        FamilySpec(
            name="volatility_spacing",
            forced_grid_methods=(GridMethod.VOLATILITY,),
            forced_confirmations=no_confirmations,
            description="Volatility-based grid spacing only",
        ),
        FamilySpec(
            name="drawdown_from_high_spacing",
            forced_grid_methods=(GridMethod.DRAWDOWN_FROM_HIGH,),
            forced_confirmations=no_confirmations,
            description="Drawdown-from-high grid spacing only",
        ),
        FamilySpec(
            name="ma_distance_spacing",
            forced_grid_methods=(GridMethod.MA_DISTANCE,),
            forced_confirmations=no_confirmations,
            description="MA distance grid spacing only",
        ),
        FamilySpec(
            name="rsi_oversold_spacing",
            forced_grid_methods=(GridMethod.RSI_OVERSOLD,),
            forced_confirmations=no_confirmations,
            description="RSI oversold grid spacing only",
        ),
        FamilySpec(
            name="z_score_spacing",
            forced_grid_methods=(GridMethod.Z_SCORE,),
            forced_confirmations=no_confirmations,
            description="Z-score grid spacing only",
        ),
        FamilySpec(
            name="trend_adjusted_spacing",
            forced_grid_methods=(GridMethod.TREND_ADJUSTED,),
            forced_confirmations=no_confirmations,
            description="Trend-adjusted grid spacing only",
        ),
        # === Executable hybrid spacing families (composed of existing methods) ===
        FamilySpec(
            name="hybrid_atr_drawdown_spacing",
            forced_grid_methods=(GridMethod.ATR, GridMethod.DRAWDOWN_FROM_HIGH),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="ATR and drawdown spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_rsi_zscore_spacing",
            forced_grid_methods=(GridMethod.RSI_OVERSOLD, GridMethod.Z_SCORE),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="RSI oversold and Z-score spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_ma_trend_spacing",
            forced_grid_methods=(GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="MA distance and trend-adjusted spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_volatility_atr_spacing",
            forced_grid_methods=(GridMethod.VOLATILITY, GridMethod.ATR),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="Volatility and ATR spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_volatility_drawdown_spacing",
            forced_grid_methods=(GridMethod.VOLATILITY, GridMethod.DRAWDOWN_FROM_HIGH),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="Volatility and drawdown spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_ma_volatility_spacing",
            forced_grid_methods=(GridMethod.MA_DISTANCE, GridMethod.VOLATILITY),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="MA distance and volatility spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_trend_drawdown_spacing",
            forced_grid_methods=(GridMethod.TREND_ADJUSTED, GridMethod.DRAWDOWN_FROM_HIGH),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="Trend-adjusted and drawdown spacing hybrid",
        ),
        FamilySpec(
            name="hybrid_fixed_atr_spacing",
            forced_grid_methods=(GridMethod.FIXED_PCT, GridMethod.ATR),
            forced_confirmations=no_confirmations,
            family_kind="hybrid_spacing",
            description="Fixed percentage and ATR spacing hybrid",
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
