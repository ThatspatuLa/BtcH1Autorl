"""Stage 9 — Simple Fixed TP Baseline.

A clean, minimal TP implementation that evolution can use while the DCA
logic evolves. Stage 10 (DCA Evolution) varies the DCA genome while
TP stays fixed per this baseline. Stage 14 (Joint Evolution) is when
the TP genome is allowed to vary.

The baseline closes a cycle when:
    current_price >= avg_entry × (1 + tp_pct)

That's it. No trailing, no partial, no hybrid. Just a simple take-profit
that the Stage 5 reward engine + Stage 6 monthly fitness will evaluate.

This module:
1. Defines the FIXED TP baseline (default tp_pct from config)
2. Wires a TpGenome (fixed) into the existing OrderManager
3. Provides a thin wrapper to run a full backtest with a CandidateGenome
   using this baseline (so Stage 10 has a single, simple call to make).

Stage 11 (TP Library) builds the richer TP primitives (trailing, hybrid,
etc.). Stage 12 (TP Evolution) varies the TP genome. Stage 14 (Joint
Evolution) varies both DCA + TP together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dca_engine.backtest import BacktestResult, backtest_candidate
from genome.schema import CandidateGenome, DcaGenome, TpExitMethod, TpGenome

# ============================================================
# Default TP baseline
# ============================================================

# Locked for v1. Single source of truth for "simple fixed TP".
# tp_pct = 0.02 means close when price >= avg × 1.02 (i.e. 2% above avg).
DEFAULT_TP_PCT: float = 0.02

# Per Kanban: max_tp_pct prevents runaway params. Not strictly needed
# for the baseline (it's hardcoded) but exposed for future use.
MAX_TP_PCT: float = 0.50

# The fixed tp_pct used for the baseline. Held in a single tuple so
# Stage 10 can override it via per-experiment config.
DEFAULT_TP_GENOME: TpGenome = TpGenome(
    exit_method=TpExitMethod.FIXED,
    exit_params={"tp_pct": DEFAULT_TP_PCT},
)


@dataclass
class FixedTPBaseline:
    """The single, well-tested TP baseline for Stage 10.

    Wraps a TpGenome with method=FIXED and a tp_pct. Provides:
    - to_order_manager_kwargs() → params to feed into OrderManager
    - description() → human-readable summary for logs
    """
    tp_pct: float = DEFAULT_TP_PCT

    def __post_init__(self) -> None:
        if self.tp_pct <= 0:
            raise ValueError(f"tp_pct must be > 0, got {self.tp_pct}")
        if self.tp_pct > MAX_TP_PCT:
            raise ValueError(f"tp_pct must be <= {MAX_TP_PCT}, got {self.tp_pct}")

    @classmethod
    def from_genome(cls, tp_genome: TpGenome) -> FixedTPBaseline:
        """Build a baseline from a TpGenome.

        Stage 9 only handles method=FIXED. Other methods (trailing, hybrid,
        etc.) are Stage 11/12 territory and should error if used here.
        """
        if tp_genome.exit_method != TpExitMethod.FIXED:
            raise ValueError(
                f"Stage 9 FixedTPBaseline only supports method=FIXED, "
                f"got {tp_genome.exit_method!r}. Use Stage 11+ for richer TPs."
            )
        tp_pct = float(tp_genome.exit_params.get("tp_pct", DEFAULT_TP_PCT))
        return cls(tp_pct=tp_pct)

    def to_order_manager_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to OrderManager.__init__."""
        return {"tp_pct": self.tp_pct}

    def to_tp_genome(self) -> TpGenome:
        """Convert back to a TpGenome (e.g. for reporting)."""
        return TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"tp_pct": self.tp_pct},
        )

    def description(self) -> str:
        return f"fixed_tp@{self.tp_pct:.2%}"


# ============================================================
# Backtest wrapper
# ============================================================

def backtest_with_fixed_tp(
    df: Any,
    candidate_id: str,
    genome_id: str,
    experiment_id: str,
    tp_genome: TpGenome | None = None,
    grid_pct: float = 0.015,
    max_layers: int = 3,
    initial_deposit: float = 10000.0,
    stake_amount: float = 100.0,
    fee_pct: float = 0.001,
    symbol: str = "BTC/USDT",
    confirmation_indicators: list[str] | None = None,
    indicator_params: dict[str, dict[str, float]] | None = None,
) -> BacktestResult:
    """Run a full backtest using the Stage 9 fixed TP baseline.

    This is the SINGLE entry point Stage 10 will call. It takes a
    candidate's DCA params (grid_pct, max_layers) and a TP genome
    (only FIXED supported here), and returns a BacktestResult that
    flows into Stage 5 reward + Stage 6 monthly fitness.

    Stage 10 evolves (grid_pct, max_layers, confirmation_indicators).
    Stage 14 will swap this for a fuller genome-driven path.

    Args:
        df: OHLCV dataframe (Stage 2 format: date, open, high, low, close, volume)
        candidate_id: ID of the candidate
        genome_id: ID of the genome
        experiment_id: ID of the experiment
        tp_genome: TP genome (default: DEFAULT_TP_GENOME)
        grid_pct: DCA layer spacing as % of avg_entry (default 1.5%)
        max_layers: max DCA layers per cycle (default 3)
        initial_deposit: starting equity (default 10000 USDT)
        stake_amount: USDT per layer (default 100)
        fee_pct: round-trip fee (default 0.1%)
        symbol: trading pair (default BTC/USDT)
        confirmation_indicators: list of indicator names to gate on
        indicator_params: dict of {indicator_name: {param: value}}

    Returns:
        BacktestResult with equity_curve + trades_df for Stage 5+6 to score.
    """
    baseline = FixedTPBaseline.from_genome(
        tp_genome if tp_genome is not None else DEFAULT_TP_GENOME
    )
    tp_kwargs = baseline.to_order_manager_kwargs()
    return backtest_candidate(
        df=df,
        candidate_id=candidate_id,
        genome_id=genome_id,
        experiment_id=experiment_id,
        grid_pct=grid_pct,
        max_layers=max_layers,
        initial_deposit=initial_deposit,
        stake_amount=stake_amount,
        fee_pct=fee_pct,
        symbol=symbol,
        confirmation_indicators=confirmation_indicators,
        indicator_params=indicator_params,
        **tp_kwargs,
    )


def extract_dca_params_from_genome(genome: CandidateGenome | DcaGenome) -> dict[str, Any]:
    """Extract the Stage-9-relevant params from a CandidateGenome or DcaGenome.

    Reads from dca_genome:
    - grid_params["pct"] → grid_pct
    - grid_params["max_layers"] → max_layers
    - grid_params["tp_pct"] → tp_pct
    - confirmation_indicators → list of indicator names
    - indicator_params → dict of {indicator_name: {param: value}}

    Stage 9 only supports fixed_pct. Other grid_methods fall back to a
    default of 0.015 because the OrderManager can't compute indicators.
    Stage 10 (the full wiring) is a separate stage.
    """
    dca = genome.dca_genome if isinstance(genome, CandidateGenome) else genome
    grid_pct = 0.015  # default fallback
    if dca.grid_method.value == "fixed_pct":
        grid_pct = float(dca.grid_params.get("pct", 0.015))
    max_layers = int(
        dca.grid_params.get("max_layers", dca.max_dca_layers)
    )
    tp_pct = float(dca.grid_params.get("tp_pct", 0.02))
    confirmation_indicators = [c.value for c in dca.confirmation_indicators]
    # Build indicator_params from genome if present, else use defaults
    indicator_params = dca.indicator_params if hasattr(dca, 'indicator_params') and dca.indicator_params else _default_indicator_params(confirmation_indicators)
    return {
        "grid_pct": grid_pct,
        "max_layers": max_layers,
        "tp_pct": tp_pct,
        "confirmation_indicators": confirmation_indicators,
        "indicator_params": indicator_params,
    }


def _default_indicator_params(indicators: list[str]) -> dict[str, dict[str, float]]:
    """Return sensible default params for each indicator type."""
    defaults: dict[str, dict[str, float]] = {}
    for ind in indicators:
        if ind == "rsi_below":
            defaults[ind] = {"threshold": 35.0}
        elif ind == "rsi_above":
            defaults[ind] = {"threshold": 65.0}
        elif ind == "volatility_high":
            defaults[ind] = {"threshold": 1.5}
        elif ind == "volatility_low":
            defaults[ind] = {"threshold": 0.5}
        # ma_above / ma_below need no extra params
    return defaults
