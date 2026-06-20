"""Indicator precomputer — computes RSI, MA, ATR, volatility once per backtest.

Runs on the full OHLCV dataframe before the state machine loop so that
the OrderManager's decide() can read indicator values in O(1) per candle
without re-computing rolling windows.

All functions use only pandas (no ta-lib dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class IndicatorSnapshot:
    """Per-candle indicator values passed to OrderManager.decide()."""
    rsi: Optional[float] = None
    ma: Optional[float] = None
    atr: Optional[float] = None
    volatility: Optional[float] = None  # rolling std of returns
    reference_vol: Optional[float] = None  # longer-period vol baseline


@dataclass
class IndicatorFrame:
    """Pre-computed indicator dataframe aligned to the OHLCV index.

    Use .snapshot_at(idx) to get the IndicatorSnapshot for candle idx.
    """
    rsi: pd.Series
    ma: pd.Series
    volatility: pd.Series
    reference_vol: pd.Series
    _has_rsi: bool = False
    _has_ma: bool = False
    _has_vol: bool = False

    def snapshot_at(self, idx: int) -> IndicatorSnapshot:
        rsi_val = None
        ma_val = None
        vol_val = None
        ref_vol_val = None
        if self._has_rsi:
            v = self.rsi.iloc[idx] if idx < len(self.rsi) else None
            rsi_val = float(v) if v is not None and not pd.isna(v) else None
        if self._has_ma:
            v = self.ma.iloc[idx] if idx < len(self.ma) else None
            ma_val = float(v) if v is not None and not pd.isna(v) else None
        if self._has_vol:
            v = self.volatility.iloc[idx] if idx < len(self.volatility) else None
            vol_val = float(v) if v is not None and not pd.isna(v) else None
            v2 = self.reference_vol.iloc[idx] if idx < len(self.reference_vol) else None
            ref_vol_val = float(v2) if v2 is not None and not pd.isna(v2) else None
        return IndicatorSnapshot(
            rsi=rsi_val,
            ma=ma_val,
            volatility=vol_val,
            reference_vol=ref_vol_val,
        )


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    ma_period: int = 200,
    vol_period: int = 20,
    ref_vol_period: int = 100,
) -> IndicatorFrame:
    """Pre-compute all indicators on the full dataframe.

    Args:
        df: OHLCV dataframe with columns (date, open, high, low, close, volume)
        rsi_period: RSI lookback (default 14)
        ma_period: Moving average lookback (default 200)
        vol_period: Short vol lookback (default 20)
        ref_vol_period: Long vol baseline lookback (default 100)

    Returns:
        IndicatorFrame with aligned series.
    """
    close = df["close"].astype(float)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    has_rsi = True

    # Simple Moving Average
    ma = close.rolling(window=ma_period, min_periods=ma_period).mean()
    has_ma = True

    # Volatility: rolling std of log returns
    log_ret = close.pct_change().apply(lambda x: float(x) if pd.notna(x) else 0.0)
    # Use a more numerically stable approach
    returns = close.pct_change()
    volatility = returns.rolling(window=vol_period, min_periods=vol_period).std()
    reference_vol = returns.rolling(window=ref_vol_period, min_periods=ref_vol_period).std()
    has_vol = True

    return IndicatorFrame(
        rsi=rsi,
        ma=ma,
        volatility=volatility,
        reference_vol=reference_vol,
        _has_rsi=has_rsi,
        _has_ma=has_ma,
        _has_vol=has_vol,
    )
