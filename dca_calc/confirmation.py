"""Confirmation checks — pure functions returning True/False.

Each function checks whether an additional gate is satisfied before a
DCA layer is allowed to fire. The OrderManager combines all enabled
confirmations with AND logic (all must pass).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfirmationContext:
    """State for confirmation checks."""
    rsi_value: float | None = None
    ma_value: float | None = None
    current_price: float | None = None
    volatility: float | None = None
    reference_vol: float | None = None  # baseline for vol_high/low


# ============================================================
# Dispatcher
# ============================================================

def check_confirmation(
    indicator: str,
    params: dict[str, float],
    ctx: ConfirmationContext,
) -> bool:
    """Top-level dispatcher — returns True if the confirmation passes."""
    if indicator == "rsi_below":
        return confirm_rsi_below(params, ctx)
    if indicator == "rsi_above":
        return confirm_rsi_above(params, ctx)
    if indicator == "ma_above":
        return confirm_ma_above(params, ctx)
    if indicator == "ma_below":
        return confirm_ma_below(params, ctx)
    if indicator == "volatility_high":
        return confirm_volatility_high(params, ctx)
    if indicator == "volatility_low":
        return confirm_volatility_low(params, ctx)
    raise ValueError(f"Unknown confirmation indicator: {indicator!r}")


def check_all_confirmations(
    indicators: list[str],
    params_map: dict[str, dict[str, float]],
    ctx: ConfirmationContext,
) -> tuple[bool, list[str]]:
    """Run all confirmations; return (all_passed, failed_indicators).

    AND logic: ALL must pass. Returns the list of indicators that failed
    for debugging.
    """
    failed = []
    for ind in indicators:
        params = params_map.get(ind, {})
        if not check_confirmation(ind, params, ctx):
            failed.append(ind)
    return (len(failed) == 0, failed)


# ============================================================
# Implementations
# ============================================================

def confirm_rsi_below(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """RSI is below threshold (oversold)."""
    if ctx.rsi_value is None:
        return False  # missing indicator → fail
    threshold = float(params.get("threshold", 30.0))
    return ctx.rsi_value < threshold


def confirm_rsi_above(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """RSI is above threshold (overbought — used for some exit signals)."""
    if ctx.rsi_value is None:
        return False
    threshold = float(params.get("threshold", 70.0))
    return ctx.rsi_value > threshold


def confirm_ma_above(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """Price is above moving average."""
    if ctx.ma_value is None or ctx.current_price is None:
        return False
    return ctx.current_price > ctx.ma_value


def confirm_ma_below(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """Price is below moving average."""
    if ctx.ma_value is None or ctx.current_price is None:
        return False
    return ctx.current_price < ctx.ma_value


def confirm_volatility_high(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """Volatility is high (above threshold × reference)."""
    if ctx.volatility is None or ctx.reference_vol is None:
        return False
    threshold = float(params.get("threshold", 1.5))
    return ctx.volatility > ctx.reference_vol * threshold


def confirm_volatility_low(params: dict[str, float], ctx: ConfirmationContext) -> bool:
    """Volatility is low (below threshold × reference)."""
    if ctx.volatility is None or ctx.reference_vol is None:
        return False
    threshold = float(params.get("threshold", 0.5))
    return ctx.volatility < ctx.reference_vol * threshold
