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
    z_score: Optional[float] = None  # price z-score over lookback window
    trend_strength: Optional[float] = None  # positive=uptrend, negative=downtrend


@dataclass
class IndicatorFrame:
    """Pre-computed indicator dataframe aligned to the OHLCV index.

    Use .snapshot_at(idx) to get the IndicatorSnapshot for candle idx.
    """
    rsi: pd.Series
    ma: pd.Series
    volatility: pd.Series
    reference_vol: pd.Series
    atr: pd.Series
    z_score: pd.Series
    trend_strength: pd.Series
    _has_rsi: bool = False
    _has_ma: bool = False
    _has_vol: bool = False
    _has_atr: bool = False
    _has_z_score: bool = False
    _has_trend: bool = False

    def snapshot_at(self, idx: int) -> IndicatorSnapshot:
        rsi_val = None
        ma_val = None
        vol_val = None
        ref_vol_val = None
        atr_val = None
        z_val = None
        trend_val = None
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
        if self._has_atr:
            v = self.atr.iloc[idx] if idx < len(self.atr) else None
            atr_val = float(v) if v is not None and not pd.isna(v) else None
        if self._has_z_score:
            v = self.z_score.iloc[idx] if idx < len(self.z_score) else None
            z_val = float(v) if v is not None and not pd.isna(v) else None
        if self._has_trend:
            v = self.trend_strength.iloc[idx] if idx < len(self.trend_strength) else None
            trend_val = float(v) if v is not None and not pd.isna(v) else None
        return IndicatorSnapshot(
            rsi=rsi_val,
            ma=ma_val,
            volatility=vol_val,
            reference_vol=ref_vol_val,
            atr=atr_val,
            z_score=z_val,
            trend_strength=trend_val,
        )


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    ma_period: int = 200,
    vol_period: int = 20,
    ref_vol_period: int = 100,
    atr_period: int = 14,
    z_score_lookback: int = 50,
    trend_lookback: int = 20,
) -> IndicatorFrame:
    """Pre-compute all indicators on the full dataframe.

    Args:
        df: OHLCV dataframe with columns (date, open, high, low, close, volume)
        rsi_period: RSI lookback (default 14)
        ma_period: Moving average lookback (default 200)
        vol_period: Short vol lookback (default 20)
        ref_vol_period: Long vol baseline lookback (default 100)
        atr_period: ATR lookback (default 14)
        z_score_lookback: Z-score rolling window (default 50)
        trend_lookback: Trend strength lookback (default 20)

    Returns:
        IndicatorFrame with aligned series.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

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
    returns = close.pct_change()
    volatility = returns.rolling(window=vol_period, min_periods=vol_period).std()
    reference_vol = returns.rolling(window=ref_vol_period, min_periods=ref_vol_period).std()
    has_vol = True

    # ATR (Average True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period, min_periods=atr_period).mean()
    has_atr = True

    # Z-score: (close - rolling_mean) / rolling_std over lookback window
    rolling_mean = close.rolling(window=z_score_lookback, min_periods=z_score_lookback).mean()
    rolling_std = close.rolling(window=z_score_lookback, min_periods=z_score_lookback).std()
    z_score = (close - rolling_mean) / rolling_std.replace(0, float("nan"))
    has_z_score = True

    # Trend strength: slope of linear regression over lookback window
    # Normalised by price to be scale-independent
    # Positive = uptrend, negative = downtrend
    def _trend_slope(s: pd.Series) -> float:
        n = len(s)
        if n < 2:
            return 0.0
        x = pd.Series(range(n), dtype=float)
        x_mean = x.mean()
        y_mean = s.mean()
        num = ((x - x_mean) * (s - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        if den == 0:
            return 0.0
        return num / den

    # Compute as rolling apply — use percentage slope (slope / mean price)
    if len(close) >= trend_lookback:
        raw_slope = close.rolling(window=trend_lookback, min_periods=trend_lookback).apply(
            _trend_slope, raw=True
        )
        # Normalise: slope as % of mean price over the window
        mean_price = close.rolling(window=trend_lookback, min_periods=trend_lookback).mean()
        trend_strength = raw_slope / mean_price.replace(0, float("nan"))
    else:
        trend_strength = pd.Series(0.0, index=close.index)
    has_trend = True

    return IndicatorFrame(
        rsi=rsi,
        ma=ma,
        volatility=volatility,
        reference_vol=reference_vol,
        atr=atr,
        z_score=z_score,
        trend_strength=trend_strength,
        _has_rsi=has_rsi,
        _has_ma=has_ma,
        _has_vol=has_vol,
        _has_atr=has_atr,
        _has_z_score=has_z_score,
        _has_trend=has_trend,
    )
