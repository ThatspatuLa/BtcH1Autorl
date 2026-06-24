"""Family-specific reasoning hints — Six's domain knowledge per DCA family.

The smart mutator reads these hints to bias mutations toward patterns
that make sense for each family, instead of blindly perturbing all params.

Heuristics are explicit, not learned. They encode Six's understanding of:
- Which params matter most for each family
- Which params to dampen (low mutation variance)
- Which indicators complement each family
- Whether the family is regime-aware (needs different params for bull/bear/chop)
- What "next move" each family should consider based on current patterns

This is the "thought process" the user described — Six-style reasoning about
each DCA family's strengths/weaknesses, applied automatically by the mutator.

Update frequency: edit this file when Six learns something new about a family.
No retraining needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FamilyHint:
    """Per-family reasoning hints consumed by SmartMutator."""
    bias_name: str  # matches IslandSpec.name

    # Param-level guidance
    params_to_explore_more: list[str] = field(default_factory=list)
    """Params that historically matter for this family — boost mutation std."""

    params_to_dampen: list[str] = field(default_factory=list)
    """Params that are sensitive or already saturated — reduce mutation std."""

    params_to_freeze: list[str] = field(default_factory=list)
    """Params that should barely change (e.g., allocation_method on forced islands)."""

    # Indicator guidance
    indicator_suggestions: list[str] = field(default_factory=list)
    """Confirmation indicators that complement this family."""

    # Regime awareness
    regime_aware: bool = False
    """If True, mutator biases toward bear/chop survival when in downtrend."""

    # Allocation guidance
    preferred_allocation_methods: list[str] = field(default_factory=list)
    """If empty, no bias. Otherwise bias crossover toward these."""

    # Diagnostic reasoning
    diagnostic_questions: list[str] = field(default_factory=list)
    """Six-style questions the mutator could 'reason' about when stagnant."""

    # Mutation multipliers (per-param std boost/dampen)
    param_std_multipliers: dict[str, float] = field(default_factory=dict)
    """Override per-param Gaussian std (e.g., {"pct": 1.5} = 50% bigger steps)."""

    # Notes
    note: str = ""


# ============================================================
# Per-family hints — Six's domain knowledge
# ============================================================

FAMILY_HINTS: dict[str, FamilyHint] = {

    # ------------------------------------------------------------------
    # I1: fixed_pct — current ATB family, exploit deeper
    # ------------------------------------------------------------------
    "fixed_pct": FamilyHint(
        bias_name="fixed_pct",
        params_to_explore_more=["pct", "tp_pct", "max_layers"],
        params_to_dampen=["cooldown_candles"],  # pct strategies are time-insensitive
        params_to_freeze=["grid_method"],  # forced bias, don't change
        indicator_suggestions=[],  # fixed_pct works without confirmations
        regime_aware=False,  # regime-agnostic
        preferred_allocation_methods=["equal", "volatility_adjusted"],
        diagnostic_questions=[
            "Is our grid spacing too tight? (high trade count, low profit per trade)",
            "Are we DCA-ing into downtrends that don't recover? (worst DD months in bear regime)",
            "Should we tighten max_layers to reduce exposure during chop?",
        ],
        param_std_multipliers={
            "pct": 1.2,  # grid spacing matters most
            "tp_pct": 1.3,  # TP capture matters a lot
            "max_layers": 0.8,  # less variance — already saturated around 5-7
        },
        note="fixed_pct is the ATB baseline — exploit more, less exploration",
    ),

    # ------------------------------------------------------------------
    # I2: atr — second-best dynamic grid
    # ------------------------------------------------------------------
    "atr": FamilyHint(
        bias_name="atr",
        params_to_explore_more=["atr_multiplier", "atr_period"],
        params_to_dampen=["cooldown_candles"],
        params_to_freeze=["grid_method"],
        indicator_suggestions=["volatility_high"],  # ATR benefits from vol filter
        regime_aware=True,  # ATR widens in high vol
        preferred_allocation_methods=["equal", "volatility_adjusted"],
        diagnostic_questions=[
            "Is our ATR period too short (choppy signals) or too long (lagging)?",
            "Are we entering too late in trends because ATR is wide?",
            "Should we adjust atr_multiplier based on volatility regime?",
        ],
        param_std_multipliers={
            "atr_multiplier": 1.5,  # key driver
            "atr_period": 1.0,
            "pct": 0.7,  # less important when ATR drives spacing
        },
        note="ATR adapts to volatility — explore period/multiplier combinations",
    ),

    # ------------------------------------------------------------------
    # I3: volatility_or_dd — volatility-relative spacing
    # ------------------------------------------------------------------
    "volatility_or_dd": FamilyHint(
        bias_name="volatility_or_dd",
        params_to_explore_more=["volatility_lookback", "drawdown_pct", "z_threshold"],
        params_to_dampen=["max_layers"],  # vol strategies benefit from fewer layers
        params_to_freeze=[],  # this island has 2 grid methods — allow switching
        indicator_suggestions=["volatility_high"],
        regime_aware=True,  # vol strategies shine in high-vol regimes
        preferred_allocation_methods=["volatility_adjusted", "controlled_exp"],
        diagnostic_questions=[
            "How are we calculating volatility? (14-period default — try 7, 21, 30)",
            "Does drawdown_pct reflect actual BTC H1 swings? (try 4-8% range)",
            "How does volatility affect our DCA weight distribution?",
            "Are we capturing regime changes or reacting too late?",
        ],
        param_std_multipliers={
            "volatility_lookback": 1.4,
            "drawdown_pct": 1.3,
            "z_threshold": 1.2,
            "max_layers": 0.6,  # vol strategies need shallow DCA
        },
        note="Volatility-relative — explore lookback windows + DD thresholds. "
             "Use volatility_adjusted allocation to scale with regime.",
    ),

    # ------------------------------------------------------------------
    # I4: trend — trend-following DCA
    # ------------------------------------------------------------------
    "trend": FamilyHint(
        bias_name="trend",
        params_to_explore_more=["ma_distance", "trend_adjusted", "ma_period"],
        params_to_dampen=["cooldown_candles"],  # trend signals need space
        params_to_freeze=[],
        indicator_suggestions=["ma_above", "volatility_high"],  # trend + vol filter
        regime_aware=True,  # trend strategies die in chop
        preferred_allocation_methods=["volatility_adjusted", "equal"],
        diagnostic_questions=[
            "What type of trends should we consider? (short, medium, long term)",
            "Are our MA distances catching trends early enough?",
            "Can we add volatility_high filter to suppress signals in chop?",
            "How does BTC H1 trend structure affect our entry timing?",
        ],
        param_std_multipliers={
            "ma_distance": 1.3,
            "trend_adjusted": 1.4,
            "ma_period": 1.0,
        },
        note="Trend-following — strongly regime-aware. Add volatility filter to "
             "suppress signals during chop. Explore MA distance thresholds.",
    ),

    # ------------------------------------------------------------------
    # I5: oscillator — mean-reversion DCA
    # ------------------------------------------------------------------
    "oscillator": FamilyHint(
        bias_name="oscillator",
        params_to_explore_more=["rsi_period", "rsi_oversold", "z_threshold"],
        params_to_dampen=["max_layers", "tp_pct"],
        params_to_freeze=[],
        indicator_suggestions=["rsi_above", "rsi_below"],
        regime_aware=False,  # mean-reversion works in ranging markets
        preferred_allocation_methods=["equal"],
        diagnostic_questions=[
            "Is our RSI period too short (whipsaw) or too long (late signals)?",
            "Are we catching real reversals or just noise?",
            "Should we use z_score instead of RSI for some regimes?",
        ],
        param_std_multipliers={
            "rsi_period": 1.3,
            "rsi_oversold": 1.4,
            "z_threshold": 1.2,
            "max_layers": 0.7,
        },
        note="Mean-reversion — RSI oversold entries. Be careful in strong trends "
             "(will get run over). Dampen max_layers to limit exposure.",
    ),

    # ------------------------------------------------------------------
    # I6: vola_adj_alloc — any grid, forced volatility_adjusted allocation
    # ------------------------------------------------------------------
    "vola_adj_alloc": FamilyHint(
        bias_name="vola_adj_alloc",
        params_to_explore_more=["volatility_lookback", "volatility_weight"],
        params_to_dampen=[],
        params_to_freeze=["allocation_method"],  # forced
        indicator_suggestions=["volatility_high"],
        regime_aware=True,
        preferred_allocation_methods=["volatility_adjusted"],  # forced
        diagnostic_questions=[
            "How does our allocation weight scale with volatility?",
            "Are we allocating too much in low-vol (overconfidence) or too little (under-utilized)?",
            "Should we adjust the volatility_lookback to match our grid's time scale?",
        ],
        param_std_multipliers={
            "volatility_lookback": 1.3,
            "volatility_weight": 1.4,
        },
        note="Cross-grid exploration — any grid method with vol-adjusted allocation. "
             "Explore how allocation scales with volatility, not just grid spacing.",
    ),

    # ------------------------------------------------------------------
    # I7: ctrl_exp_alloc — any grid, forced controlled_exp allocation
    # ------------------------------------------------------------------
    "ctrl_exp_alloc": FamilyHint(
        bias_name="ctrl_exp_alloc",
        params_to_explore_more=["exponent", "base_weight"],
        params_to_dampen=[],
        params_to_freeze=["allocation_method"],  # forced
        indicator_suggestions=[],
        regime_aware=False,
        preferred_allocation_methods=["controlled_exp"],  # forced
        diagnostic_questions=[
            "How aggressive is our controlled_exp curve? (linear vs exponential)",
            "Does our exponent match the typical DCA depth needed for BTC H1?",
            "Are early entries getting too little weight (under-utilized) or too much (over-exposed)?",
        ],
        param_std_multipliers={
            "exponent": 1.5,  # key driver of curve shape
            "base_weight": 1.2,
        },
        note="Cross-grid exploration — controlled_exp allocation. "
             "Exponent controls how aggressively late entries are weighted.",
    ),

    # ------------------------------------------------------------------
    # I8: tight_dca — fast-exit / low-DCA strategies
    # ------------------------------------------------------------------
    "tight_dca": FamilyHint(
        bias_name="tight_dca",
        params_to_explore_more=["tp_pct", "max_layers"],
        params_to_dampen=["cooldown_candles"],  # tight DCA needs tight cycle
        params_to_freeze=[],
        indicator_suggestions=["volatility_low"],  # low-vol entries
        regime_aware=False,
        preferred_allocation_methods=["equal"],
        diagnostic_questions=[
            "Are we capping max_layers too low? (missing recovery opportunities)",
            "Is our TP aggressive enough? (tight DCA = fast cycle)",
            "Does volatility_low filter help us avoid chop entries?",
        ],
        param_std_multipliers={
            "tp_pct": 1.4,
            "max_layers": 1.0,  # can vary between 2-8
        },
        note="Fast-exit — low DCA depth, tight TP. Quick in-and-out, "
             "low exposure. Best for ranging markets.",
    ),
}


# ============================================================
# Helpers
# ============================================================

def get_family_hint(bias_name: str) -> FamilyHint | None:
    """Look up family hint by bias name (e.g., 'trend', 'volatility_or_dd')."""
    return FAMILY_HINTS.get(bias_name)


def get_hint_for_island(island_id: int) -> FamilyHint | None:
    """Look up family hint for an island by its ID."""
    from evolution.islands import get_island_spec
    try:
        spec = get_island_spec(island_id)
    except (ValueError, KeyError):
        return None
    return FAMILY_HINTS.get(spec.name)


def all_family_hints() -> dict[str, FamilyHint]:
    """Return all family hints (for testing / iteration)."""
    return dict(FAMILY_HINTS)
