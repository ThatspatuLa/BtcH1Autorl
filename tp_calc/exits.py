"""TP / exit calculators — pure functions returning ExitDecisions.

Each function evaluates a single exit method. The decision includes:
- should_close: bool — whether to close now
- target_price: float — at what price the exit is "satisfied"
- close_fraction: float — what fraction of the position to close (0.0-1.0)
- reason: str — human-readable explanation
"""
from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Result type
# ============================================================

@dataclass
class ExitDecision:
    """Result of an exit-method evaluation."""
    should_close: bool
    target_price: float
    close_fraction: float      # 1.0 = close all, 0.5 = close half, etc.
    reason: str = ""

    @classmethod
    def hold(cls) -> ExitDecision:
        """Helper: don't close."""
        return cls(should_close=False, target_price=0.0, close_fraction=0.0, reason="hold")

    @classmethod
    def close_all(cls, target_price: float, reason: str) -> ExitDecision:
        """Helper: close 100% at target price."""
        return cls(should_close=True, target_price=target_price, close_fraction=1.0, reason=reason)

    @classmethod
    def close_partial(cls, target_price: float, fraction: float, reason: str) -> ExitDecision:
        """Helper: close a fraction of the position."""
        return cls(should_close=True, target_price=target_price, close_fraction=fraction, reason=reason)


# ============================================================
# Context
# ============================================================

@dataclass
class ExitContext:
    """All state needed to evaluate an exit method."""
    current_price: float
    avg_entry: float              # average entry of the open cycle
    cycle_high: float             # highest price seen in this cycle
    cycle_low: float              # lowest price seen in this cycle
    n_layers_filled: int          # DCA depth
    candles_in_position: int      # how long the cycle has been open
    unrealised_pnl_pct: float     # (current - avg) / avg
    # Indicators (precomputed, optional)
    atr: float | None = None
    volatility: float | None = None
    rsi_value: float | None = None
    ma_value: float | None = None
    volume_ratio: float | None = None     # current / avg volume
    # Reference
    reference_vol: float | None = None   # for vol-adjusted / exhaustion


# ============================================================
# Dispatcher
# ============================================================

def compute_exit_decision(
    exit_method: str,
    exit_params: dict[str, float],
    ctx: ExitContext,
) -> ExitDecision:
    """Top-level dispatcher — returns the exit decision."""
    if exit_method == "fixed":
        return exit_fixed(exit_params, ctx)
    if exit_method == "atr":
        return exit_atr(exit_params, ctx)
    if exit_method == "vol_adjusted":
        return exit_vol_adjusted(exit_params, ctx)
    if exit_method == "dca_depth_adjusted":
        return exit_dca_depth_adjusted(exit_params, ctx)
    if exit_method == "partial_ladder":
        return exit_partial_ladder(exit_params, ctx)
    if exit_method == "trailing":
        return exit_trailing(exit_params, ctx)
    if exit_method == "break_even":
        return exit_break_even(exit_params, ctx)
    if exit_method == "momentum_decay":
        return exit_momentum_decay(exit_params, ctx)
    if exit_method == "exhaustion":
        return exit_exhaustion(exit_params, ctx)
    if exit_method == "trend_hold":
        return exit_trend_hold(exit_params, ctx)
    if exit_method == "failed_continuation":
        return exit_failed_continuation(exit_params, ctx)
    if exit_method == "time_in_position":
        return exit_time_in_position(exit_params, ctx)
    if exit_method == "hybrid":
        return exit_hybrid(exit_params, ctx)
    raise ValueError(f"Unknown exit_method: {exit_method!r}")


# ============================================================
# Implementations
# ============================================================

def exit_fixed(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close at avg_entry × (1 + tp_pct).

    params: { "tp_pct": 0.02 } — 2% above avg
    """
    tp_pct = float(params.get("tp_pct", 0.02))
    target = ctx.avg_entry * (1.0 + tp_pct)
    if ctx.current_price >= target:
        return ExitDecision.close_all(target, f"fixed_tp_hit ({tp_pct:.2%})")
    return ExitDecision.hold()


def exit_atr(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close at avg_entry + N × ATR.

    params: { "atr_multiplier": 3.0 }
    """
    if ctx.atr is None or ctx.atr <= 0:
        return ExitDecision.hold()
    mult = float(params.get("atr_multiplier", 3.0))
    target = ctx.avg_entry + mult * ctx.atr
    if ctx.current_price >= target:
        return ExitDecision.close_all(target, f"atr_tp_hit ({mult:.1f}x ATR)")
    return ExitDecision.hold()


def exit_vol_adjusted(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close at avg × (1 + base_pct × vol_scale).

    Higher volatility → wider TP target.

    params: { "base_pct": 0.02, "vol_scale_factor": 0.5, "reference_vol": 0.02 }
    """
    if ctx.volatility is None or ctx.reference_vol is None:
        return ExitDecision.hold()
    base_pct = float(params.get("base_pct", 0.02))
    vol_scale = float(params.get("vol_scale_factor", 0.5))
    ref_vol = float(params.get("reference_vol", 0.02))
    # At ref vol → base_pct. At 2x ref vol → base_pct * (1 + 0.5) = 1.5x base_pct.
    effective_pct = base_pct * (1.0 + vol_scale * (ctx.volatility / ref_vol - 1.0))
    target = ctx.avg_entry * (1.0 + effective_pct)
    if ctx.current_price >= target:
        return ExitDecision.close_all(target, f"vol_tp_hit ({effective_pct:.2%})")
    return ExitDecision.hold()


def exit_dca_depth_adjusted(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """TP widens with DCA depth (more layers = bigger target).

    params: { "base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10 }
    effective_pct = base_pct + (n_layers - 1) * depth_multiplier
    Capped at max_pct.
    """
    base_pct = float(params.get("base_pct", 0.02))
    depth_mult = float(params.get("depth_multiplier", 0.005))
    max_pct = float(params.get("max_pct", 0.10))
    effective_pct = min(max_pct, base_pct + (ctx.n_layers_filled - 1) * depth_mult)
    effective_pct = max(effective_pct, 0.001)  # floor
    target = ctx.avg_entry * (1.0 + effective_pct)
    if ctx.current_price >= target:
        return ExitDecision.close_all(target, f"dca_depth_tp ({effective_pct:.2%}, layers={ctx.n_layers_filled})")
    return ExitDecision.hold()


def exit_partial_ladder(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close a fraction at first TP, rest at second TP.

    params: {
        "tp1_pct": 0.01, "tp1_fraction": 0.5,
        "tp2_pct": 0.03
    }
    At tp1 → close tp1_fraction. At tp2 → close remaining (1.0).
    """
    tp1_pct = float(params.get("tp1_pct", 0.01))
    tp1_fraction = float(params.get("tp1_fraction", 0.5))
    tp2_pct = float(params.get("tp2_pct", 0.03))
    target1 = ctx.avg_entry * (1.0 + tp1_pct)
    target2 = ctx.avg_entry * (1.0 + tp2_pct)
    if ctx.current_price >= target2:
        return ExitDecision.close_all(target2, f"ladder_tp2 ({tp2_pct:.2%})")
    if ctx.current_price >= target1:
        return ExitDecision.close_partial(target1, tp1_fraction, f"ladder_tp1 ({tp1_pct:.2%})")
    return ExitDecision.hold()


def exit_trailing(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Trailing stop: close if price drops X% from cycle high.

    params: { "trail_pct": 0.02, "activation_pct": 0.005 }
    The cycle_high only starts tracking once price >= avg × (1 + activation_pct).
    Before activation: hold. After activation: trail from cycle_high.
    """
    trail_pct = float(params.get("trail_pct", 0.02))
    activation_pct = float(params.get("activation_pct", 0.005))
    activation_price = ctx.avg_entry * (1.0 + activation_pct)
    if ctx.cycle_high <= activation_price:
        # Not yet activated — high hasn't reached activation price
        return ExitDecision.hold()
    trail_stop = ctx.cycle_high * (1.0 - trail_pct)
    if ctx.current_price <= trail_stop:
        return ExitDecision.close_all(trail_stop, f"trailing_stop ({trail_pct:.2%} from high {ctx.cycle_high:.2f})")
    return ExitDecision.hold()


def exit_break_even(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Once break-even, close at break-even + small buffer.

    params: { "buffer_pct": 0.002, "min_profit_pct": 0.005 }
    Only activates after price has been min_profit_pct above avg. Then
    tightens to avg × (1 + buffer_pct).
    """
    buffer_pct = float(params.get("buffer_pct", 0.002))
    min_profit = float(params.get("min_profit_pct", 0.005))
    target = ctx.avg_entry * (1.0 + buffer_pct)
    if ctx.cycle_high < ctx.avg_entry * (1.0 + min_profit):
        # Not yet activated
        return ExitDecision.hold()
    if ctx.current_price <= target:
        return ExitDecision.close_all(target, f"break_even_stop ({buffer_pct:.2%})")
    return ExitDecision.hold()


def exit_momentum_decay(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close when RSI decays (momentum exhaustion).

    params: { "rsi_peak": 70.0, "rsi_exit": 50.0 }
    Close when RSI was above rsi_peak and dropped below rsi_exit.
    """
    if ctx.rsi_value is None:
        return ExitDecision.hold()
    rsi_exit = float(params.get("rsi_exit", 50.0))
    if ctx.rsi_value < rsi_exit and ctx.unrealised_pnl_pct > 0:
        return ExitDecision.close_all(ctx.current_price, f"momentum_decay (RSI={ctx.rsi_value:.1f})")
    return ExitDecision.hold()


def exit_exhaustion(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close on volume/volatility exhaustion.

    params: { "vol_low_threshold": 0.5, "vol_spike_threshold": 2.0 }
    Close when volatility drops below threshold (exhaustion) AND
    unrealised PnL is positive. OR when vol spikes (climax).
    """
    if ctx.volatility is None or ctx.reference_vol is None:
        return ExitDecision.hold()
    vol_low = float(params.get("vol_low_threshold", 0.5))
    vol_spike = float(params.get("vol_spike_threshold", 2.0))
    if ctx.unrealised_pnl_pct <= 0:
        return ExitDecision.hold()
    if ctx.volatility < ctx.reference_vol * vol_low:
        return ExitDecision.close_all(ctx.current_price, f"exhaustion (vol={ctx.volatility:.4f} < {vol_low}x ref)")
    if ctx.volatility > ctx.reference_vol * vol_spike:
        return ExitDecision.close_all(ctx.current_price, f"climax (vol={ctx.volatility:.4f} > {vol_spike}x ref)")
    return ExitDecision.hold()


def exit_trend_hold(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Hold while trend intact, close if trend reverses.

    params: { "ma_distance_pct": 0.02, "min_profit_pct": 0.005 }
    Close when price closes below MA × (1 - distance) AND we have some profit.
    """
    if ctx.ma_value is None:
        return ExitDecision.hold()
    distance = float(params.get("ma_distance_pct", 0.02))
    min_profit = float(params.get("min_profit_pct", 0.005))
    if ctx.unrealised_pnl_pct < min_profit:
        return ExitDecision.hold()
    trend_line = ctx.ma_value * (1.0 - distance)
    if ctx.current_price < trend_line:
        return ExitDecision.close_all(ctx.current_price, f"trend_reverse (price < MA × {1-distance:.2f})")
    return ExitDecision.hold()


def exit_failed_continuation(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close if a breakout attempt fails.

    params: { "min_pnl_pct": 0.01, "reversal_pct": 0.005 }
    Close when price was at min_pnl_pct, then dropped reversal_pct from there.
    """
    min_pnl = float(params.get("min_pnl_pct", 0.01))
    reversal = float(params.get("reversal_pct", 0.005))
    # cycle_high should reflect the post-entry peak
    if ctx.cycle_high <= 0 or ctx.avg_entry <= 0:
        return ExitDecision.hold()
    peak_pnl = (ctx.cycle_high - ctx.avg_entry) / ctx.avg_entry
    if peak_pnl < min_pnl:
        return ExitDecision.hold()
    reversal_level = ctx.cycle_high * (1.0 - reversal)
    if ctx.current_price <= reversal_level:
        return ExitDecision.close_all(reversal_level, f"failed_continuation (peak +{peak_pnl:.2%}, dropped {reversal:.2%})")
    return ExitDecision.hold()


def exit_time_in_position(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Close after N candles in position.

    params: { "max_candles": 100, "min_profit_pct": 0.0 }
    If min_profit_pct > 0, only close if profitable.
    """
    max_candles = int(params.get("max_candles", 100))
    min_profit = float(params.get("min_profit_pct", 0.0))
    if ctx.candles_in_position < max_candles:
        return ExitDecision.hold()
    if ctx.unrealised_pnl_pct < min_profit:
        return ExitDecision.hold()
    return ExitDecision.close_all(ctx.current_price, f"time_exit ({ctx.candles_in_position} >= {max_candles})")


def exit_hybrid(params: dict[str, float], ctx: ExitContext) -> ExitDecision:
    """Combine multiple exit methods. OR logic: any method triggers close.

    params convention: each sub-method's parameters are prefixed with the
    method name. e.g. for "fixed" + "trailing":
        {
            "methods": "fixed,trailing",
            "fixed_tp_pct": 0.02,           # passed to exit_fixed as "tp_pct"
            "trailing_trail_pct": 0.015,    # passed to exit_trailing as "trail_pct"
            "trailing_activation_pct": 0.005,  # passed to exit_trailing as "activation_pct"
        }
    The prefix is stripped before passing to the sub-method.

    OR logic: first sub-method to trigger wins.
    """
    methods_str = str(params.get("methods", "fixed"))
    method_names = [m.strip() for m in methods_str.split(",") if m.strip()]
    for method_name in method_names:
        sub_params = _strip_method_prefix(method_name, params)
        try:
            sub_decision = compute_exit_decision(method_name, sub_params, ctx)
        except ValueError:
            continue
        if sub_decision.should_close:
            sub_decision.reason = f"hybrid:{method_name}:{sub_decision.reason}"
            return sub_decision
    return ExitDecision.hold()


def _strip_method_prefix(method_name: str, full_params: dict[str, float]) -> dict[str, float]:
    """Strip the method_name_ prefix from each key in full_params.

    e.g. {"trailing_trail_pct": 0.01} → {"trail_pct": 0.01} for method "trailing".
    Keys that don't start with the prefix are dropped (they belong to other methods).
    """
    prefix = f"{method_name}_"
    sub = {}
    for k, v in full_params.items():
        if k.startswith(prefix):
            sub[k[len(prefix):]] = v
    return sub
