"""Grid spacing calculators — pure functions returning trigger prices.

Each function takes a context (current price, avg entry, indicators) and
returns the price at which the next DCA layer should fire.

Conventions:
- Return price BELOW current price (DCA buys dips)
- Return None if conditions don't make sense (e.g. no indicators available)
- Functions are stateless — caller manages cycle state
"""
from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Context
# ============================================================

@dataclass
class GridContext:
    """All state needed to compute the next layer price.

    Carried by the OrderManager. Most fields are optional — only what's
    needed by the selected grid_method.
    """
    current_price: float
    avg_entry: float             # average entry of current cycle
    cycle_high: float            # highest price seen in this cycle
    layers_filled: int           # how many layers have been added (0 = initial)
    n_layers_total: int          # total layers planned (from genome)
    # Indicators (precomputed, optional)
    atr: float | None = None
    volatility: float | None = None       # realised vol (e.g. std of returns)
    ma_value: float | None = None
    rsi_value: float | None = None
    z_score: float | None = None
    trend_strength: float | None = None   # positive = uptrend, negative = downtrend
    # Reference candle for high/low
    reference_high: float | None = None  # cycle high (alternative name)


# ============================================================
# Dispatcher
# ============================================================

def compute_next_layer_price(
    grid_method: str,
    grid_params: dict[str, float],
    ctx: GridContext,
) -> float | None:
    """Top-level dispatcher — returns the next layer trigger price.

    grid_method: one of "fixed_pct", "atr", "volatility",
                 "drawdown_from_high", "ma_distance", "rsi_oversold",
                 "z_score", "trend_adjusted"

    Returns the price at which the next layer should fire, or None if
    the method cannot compute a price (missing indicator etc.).
    """
    if grid_method == "fixed_pct":
        return grid_fixed_pct(grid_params, ctx)
    if grid_method == "atr":
        return grid_atr(grid_params, ctx)
    if grid_method == "volatility":
        return grid_volatility(grid_params, ctx)
    if grid_method == "drawdown_from_high":
        return grid_drawdown_from_high(grid_params, ctx)
    if grid_method == "ma_distance":
        return grid_ma_distance(grid_params, ctx)
    if grid_method == "rsi_oversold":
        return grid_rsi_oversold(grid_params, ctx)
    if grid_method == "z_score":
        return grid_z_score(grid_params, ctx)
    if grid_method == "trend_adjusted":
        return grid_trend_adjusted(grid_params, ctx)
    raise ValueError(f"Unknown grid_method: {grid_method!r}")


# ============================================================
# Implementations
# ============================================================

def grid_fixed_pct(params: dict[str, float], ctx: GridContext) -> float | None:
    """Simple: each layer is X% below the previous.

    params: { "pct": 0.015 } — 1.5% below avg_entry
    Layer 1 (initial) uses avg_entry. Layer 2 = avg_entry * (1 - pct).
    Layer 3 = layer_2_trigger (one more pct below) — but we always
    return the trigger for the NEXT layer relative to current avg.
    """
    pct = float(params.get("pct", 0.015))
    if pct <= 0:
        raise ValueError(f"pct must be > 0, got {pct}")
    # The next layer fires when price drops pct below avg_entry
    return ctx.avg_entry * (1.0 - pct)


def grid_atr(params: dict[str, float], ctx: GridContext) -> float | None:
    """ATR-based: each layer is N × ATR below the previous.

    params: { "atr_multiplier": 2.0 } — 2 × ATR
    """
    if ctx.atr is None or ctx.atr <= 0:
        return None
    mult = float(params.get("atr_multiplier", 2.0))
    if mult <= 0:
        raise ValueError(f"atr_multiplier must be > 0, got {mult}")
    return ctx.avg_entry - mult * ctx.atr


def grid_volatility(params: dict[str, float], ctx: GridContext) -> float | None:
    """Volatility-adjusted: grid spacing = base_pct × volatility_scale.

    params: { "base_pct": 0.01, "vol_scale_factor": 0.5 }
    Higher volatility → wider grid (fewer triggers, larger moves).
    """
    if ctx.volatility is None or ctx.volatility <= 0:
        return None
    base_pct = float(params.get("base_pct", 0.01))
    vol_scale = float(params.get("vol_scale_factor", 0.5))
    # If vol = 0.02 (2%), scale factor 0.5 → pct = 0.01 * (1 + 0.5*0.02/0.02) = 0.015
    # Reference vol: 2% per candle (0.02)
    reference_vol = 0.02
    effective_pct = base_pct * (1.0 + vol_scale * (ctx.volatility / reference_vol - 1.0))
    effective_pct = max(0.001, effective_pct)  # floor
    return ctx.avg_entry * (1.0 - effective_pct)


def grid_drawdown_from_high(params: dict[str, float], ctx: GridContext) -> float | None:
    """Layer fires at X% drawdown from the cycle high (not avg entry).

    params: { "drawdown_pct": 0.05 } — 5% drawdown from cycle high
    """
    pct = float(params.get("drawdown_pct", 0.05))
    if pct <= 0:
        raise ValueError(f"drawdown_pct must be > 0, got {pct}")
    ref_high = ctx.reference_high if ctx.reference_high is not None else ctx.cycle_high
    if ref_high <= 0:
        return None
    return ref_high * (1.0 - pct)


def grid_ma_distance(params: dict[str, float], ctx: GridContext) -> float | None:
    """Layer fires when price is X% below the moving average.

    params: { "ma_distance_pct": 0.03 } — 3% below MA
    """
    if ctx.ma_value is None or ctx.ma_value <= 0:
        return None
    pct = float(params.get("ma_distance_pct", 0.03))
    if pct <= 0:
        raise ValueError(f"ma_distance_pct must be > 0, got {pct}")
    return ctx.ma_value * (1.0 - pct)


def grid_rsi_oversold(params: dict[str, float], ctx: GridContext) -> float | None:
    """Layer fires when RSI is below threshold.

    The price target is approximated as: avg_entry * (1 - oversold_depth_pct).
    Real implementation needs historical RSI to know the actual price at
    oversold; this is an approximation.

    params: { "rsi_threshold": 30.0, "oversold_depth_pct": 0.02 }
    """
    if ctx.rsi_value is None:
        return None
    threshold = float(params.get("rsi_threshold", 30.0))
    depth_pct = float(params.get("oversold_depth_pct", 0.02))
    if ctx.rsi_value >= threshold:
        # Not yet oversold — return a price target well below current
        return ctx.current_price * (1.0 - depth_pct * 5.0)
    return ctx.avg_entry * (1.0 - depth_pct)


def grid_z_score(params: dict[str, float], ctx: GridContext) -> float | None:
    """Layer fires at N standard deviations below the rolling mean.

    params: { "z_threshold": 1.5, "lookback_std": 0.02 }
    Returns a price target. Without rolling mean, uses avg_entry as proxy.
    """
    if ctx.z_score is None:
        return None
    z_thresh = float(params.get("z_threshold", 1.5))
    lookback_std = float(params.get("lookback_std", 0.02))
    if ctx.z_score < z_thresh:
        # Already oversold enough
        return ctx.avg_entry * (1.0 - lookback_std * z_thresh)
    # Approximate target based on current price
    return ctx.current_price * (1.0 - lookback_std * z_thresh)


def grid_trend_adjusted(params: dict[str, float], ctx: GridContext) -> float | None:
    """Grid spacing widens/narrows based on trend strength.

    params: { "base_pct": 0.015, "trend_multiplier": 0.5 }
    In downtrend (trend_strength < 0): pct = base * (1 + |trend| * mult) — wider
    In uptrend: pct = base * (1 - trend * mult) — tighter (buy dips faster)
    """
    base_pct = float(params.get("base_pct", 0.015))
    trend_mult = float(params.get("trend_multiplier", 0.5))
    trend = ctx.trend_strength or 0.0
    if trend < 0:
        # Downtrend: wider grid
        effective_pct = base_pct * (1.0 + abs(trend) * trend_mult)
    else:
        # Uptrend: tighter grid
        effective_pct = base_pct * max(0.1, 1.0 - trend * trend_mult)
    return ctx.avg_entry * (1.0 - effective_pct)
