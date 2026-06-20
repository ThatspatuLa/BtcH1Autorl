"""Stage 8 — DCA Calculation Library.

Pure-function building blocks for DCA order placement. Stage 10 (DCA
Evolution) wires these into the OrderManager; Stage 8 is just the math.

Three sections, all functions take only their inputs and return outputs:

1. Grid Spacing — at what price should the next DCA layer fire?
   - fixed_pct: price drops by X% from avg entry
   - atr: price drops by N × ATR
   - volatility: price drops by volatility-adjusted %
   - drawdown_from_high: price drops by X% from cycle high
   - ma_distance: price drops by N% below moving average
   - rsi_oversold: price hits oversold RSI
   - z_score: price drops by N standard deviations
   - trend_adjusted: grid widens when in downtrend

2. Allocation — how much capital to commit to each layer?
   - equal: same size every layer
   - linear_increasing: size grows linearly with layer index
   - controlled_exp: size grows by controlled exponential (martingale-like)
   - drawdown_adjusted: size grows with current drawdown
   - volatility_adjusted: size inversely with volatility

3. Confirmation — additional gates before firing a layer.
   - rsi_below / rsi_above
   - ma_above / ma_below
   - volatility_high / volatility_low
"""
from __future__ import annotations

from .allocation import (
    allocation_controlled_exp,
    allocation_drawdown_adjusted,
    allocation_equal,
    allocation_linear_increasing,
    allocation_volatility_adjusted,
    compute_layer_allocation,
    compute_total_position_size,
)
from .confirmation import (
    check_all_confirmations,
    check_confirmation,
    confirm_ma_above,
    confirm_ma_below,
    confirm_rsi_above,
    confirm_rsi_below,
    confirm_volatility_high,
    confirm_volatility_low,
)
from .grid_spacing import (
    compute_next_layer_price,
    grid_atr,
    grid_drawdown_from_high,
    grid_fixed_pct,
    grid_ma_distance,
    grid_rsi_oversold,
    grid_trend_adjusted,
    grid_volatility,
    grid_z_score,
)

__all__ = [
    "allocation_controlled_exp",
    "allocation_drawdown_adjusted",
    "allocation_equal",
    "allocation_linear_increasing",
    "allocation_volatility_adjusted",
    "check_all_confirmations",
    "check_confirmation",
    "compute_layer_allocation",
    "compute_next_layer_price",
    "compute_total_position_size",
    "confirm_ma_above",
    "confirm_ma_below",
    "confirm_rsi_above",
    "confirm_rsi_below",
    "confirm_volatility_high",
    "confirm_volatility_low",
    "grid_atr",
    "grid_drawdown_from_high",
    "grid_fixed_pct",
    "grid_ma_distance",
    "grid_rsi_oversold",
    "grid_trend_adjusted",
    "grid_volatility",
    "grid_z_score",
]
