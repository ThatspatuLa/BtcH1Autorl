"""Phase A — Recovery sub-metrics for Discovery Fitness v2.

Four pure functions feed `recovery_score`:
    drawdown_recovery_speed        (weight 0.40)
    post_loss_month_bounce_rate    (weight 0.30)
    equity_high_reclaim_rate       (weight 0.20)
    cycle_recovery_health          (weight 0.10)

Plus `compute_recovery_score` aggregator (weighted sum).

All inputs/outputs in [0, 1] range. Edge cases (empty curve, no losers, etc.)
return documented neutral values.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "RECOVERY_WEIGHTS",
    "compute_cycle_recovery_health",
    "compute_drawdown_recovery_speed",
    "compute_equity_high_reclaim_rate",
    "compute_post_loss_month_bounce_rate",
    "compute_recovery_score",
]


# LOCKED weights for the four recovery sub-metrics (sum must = 1.0)
RECOVERY_WEIGHTS: dict[str, float] = {
    "drawdown_recovery_speed": 0.40,
    "post_loss_month_bounce_rate": 0.30,
    "equity_high_reclaim_rate": 0.20,
    "cycle_recovery_health": 0.10,
}

assert abs(sum(RECOVERY_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"RECOVERY_WEIGHTS must sum to 1.0; got {sum(RECOVERY_WEIGHTS.values())}"
)


# ============================================================
# Pure recovery sub-metric functions
# ============================================================

def _find_drawdown_events(equity: np.ndarray) -> list[tuple[int, int, int, bool]]:
    """Find all drawdown events in an equity array.

    Returns a list of (peak_idx, trough_idx, recovery_idx, recovered) tuples.
    For unrecovered events, recovery_idx = len(equity) - 1.

    A drawdown event is a sequence: peak → trough → recovery (or end-of-curve).
    After recovery (or unrecovered end), the peak resets to the recovery point.
    Single-candle "fake" events where the trough is just 1 candle are ignored
    (not meaningful drawdowns).
    """
    n = len(equity)
    if n < 2:
        return []

    events: list[tuple[int, int, int, bool]] = []

    # Find every peak in the curve by running max
    # A "true" peak is a position where equity == running_max AND it's followed
    # by a dip below it (otherwise it's just end-of-curve, not a peak with a DD).
    i = 0
    while i < n - 1:
        # Skip if not at running max
        running_max = float(np.max(equity[: i + 1]))
        if equity[i] < running_max - 1e-12:
            i += 1
            continue

        # We're at a peak (or flat at running max). Check if next candle drops.
        peak_idx = i
        peak_val = float(equity[i])
        if equity[i + 1] >= peak_val - 1e-12:
            # No drop → not a DD event. Continue.
            i += 1
            continue

        # We have a DD start. Find trough + recovery.
        trough_idx = i + 1
        recovered = False
        recovery_idx = n - 1
        j = i + 1
        while j < n:
            if equity[j] < equity[trough_idx]:
                trough_idx = j
            if equity[j] >= peak_val - 1e-12:
                recovery_idx = j
                recovered = True
                break
            j += 1

        # Filter out single-candle "fake" events (trough == peak_idx+1 only).
        # If the trough was a single candle AND it recovered within a few candles
        # after, treat it as noise. If it didn't recover, it's a real drawdown.
        # Simple rule: keep event if trough > peak+1 OR if unrecovered.
        is_single_candle_drop = trough_idx == peak_idx + 1
        if not is_single_candle_drop or not recovered:
            events.append((peak_idx, trough_idx, recovery_idx, recovered))

        # Advance past this event
        if recovered:
            i = recovery_idx + 1
        else:
            break  # unreached DDs at end of curve → done

    return events


def compute_drawdown_recovery_speed(equity_curve: pd.Series) -> float:
    """For each drawdown event in the equity curve, measure the recovery speed.

    A drawdown event is peak → trough → recovery (or end-of-curve).
    Recovery speed for one event = 1 - (recovery_time / dd_event_duration).
        - recovery_time = candles from trough to recovery
        - dd_event_duration = candles from peak to recovery
    If the drawdown never recovers, that event scores 0.0.
    Final score = mean across all events (or 1.0 if no events at all).

    Args:
        equity_curve: pd.Series of equity values over time.

    Returns:
        float in [0, 1]. Higher = faster recovery.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 1.0  # No curve to draw down → trivially "perfect recovery"

    eq = np.asarray(equity_curve.values, dtype=float)
    events = _find_drawdown_events(eq)
    if not events:
        return 1.0  # Monotonic up — no drawdowns

    speeds: list[float] = []
    for peak_idx, trough_idx, recovery_idx, recovered in events:
        if not recovered:
            speeds.append(0.0)  # Phase A0 Bug 3 fix: never recovered → 0
            continue
        dd_duration = recovery_idx - peak_idx
        recovery_time = recovery_idx - trough_idx
        if dd_duration <= 0:
            speeds.append(1.0)
        else:
            speed = 1.0 - (recovery_time / dd_duration)
            speeds.append(max(0.0, min(1.0, speed)))

    return float(np.mean(speeds))


def compute_post_loss_month_bounce_rate(
    monthly_scores: Iterable[bool],
    look_ahead: int = 3,
) -> float:
    """For each month with net_profit_pct <= 0, count whether any of the next
    `look_ahead` months was profitable.

    Args:
        monthly_scores: iterable of booleans. True = profitable month, False = losing month.
        look_ahead: how many months forward to look for recovery.

    Returns:
        float in [0, 1]. Higher = more bounces. 1.0 if no losses (neutral).
    """
    scores = list(monthly_scores)
    if not scores:
        return 1.0

    loss_indices = [i for i, s in enumerate(scores) if not s]
    if not loss_indices:
        return 1.0  # No losses → neutral

    n = len(scores)
    bounces = 0
    for li in loss_indices:
        window_end = min(li + look_ahead + 1, n)
        window = scores[li + 1 : window_end]
        if any(window):
            bounces += 1

    return bounces / len(loss_indices)


def compute_equity_high_reclaim_rate(equity_curve: pd.Series) -> float:
    """Fraction of historical peaks that were reclaimed by end-of-curve.

    A "peak" is a NEW high in the running-max sequence. We then count what
    fraction of those peaks are >= the final equity value (i.e. the curve
    is at or above that peak at the end).

    Returns:
        float in [0, 1]. 1.0 = curve ends at or above every prior peak.
        0.0 = curve ends below every peak.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 1.0

    eq = np.asarray(equity_curve.values, dtype=float)
    running_max = np.maximum.accumulate(eq)
    final_eq = float(eq[-1])

    # Collect unique peaks (running max steps up to new highs)
    peaks: list[float] = []
    last_seen = -np.inf
    for v in running_max:
        if v > last_seen:
            peaks.append(float(v))
            last_seen = v

    if not peaks:
        return 1.0

    # A peak is "reclaimed" if final equity is >= that peak value.
    # Since the peak was reached at some point in the curve, having
    # final_eq >= peak means we never lost it permanently.
    reclaimed = sum(1 for p in peaks if final_eq >= p - 1e-9)
    return reclaimed / len(peaks)


def compute_cycle_recovery_health(
    trades_df: pd.DataFrame,
    recovery_window_days: int = 30,
) -> float:
    """Fraction of DCA cycles that closed profitably.

    Used as a proxy for "recovery from cycle loss".

    Edge cases:
        - Empty trades → 0.5 (neutral, avoids biasing)
        - All profitable → 1.0
        - All losing → 0.0
    """
    if trades_df is None or trades_df.empty:
        return 0.5  # Neutral: don't bias the score either way for missing data

    if "pnl" not in trades_df.columns:
        return 0.5

    pnl = trades_df["pnl"]
    if len(pnl) == 0:
        return 0.5

    profitable = (pnl > 0).sum()
    return float(profitable / len(pnl))


def compute_recovery_score(
    drawdown_recovery_speed: float,
    post_loss_month_bounce_rate: float,
    equity_high_reclaim_rate: float,
    cycle_recovery_health: float,
) -> float:
    """Weighted sum of the 4 recovery sub-metrics using RECOVERY_WEIGHTS.

    Returns:
        float in [0, 1].
    """
    parts = {
        "drawdown_recovery_speed": drawdown_recovery_speed,
        "post_loss_month_bounce_rate": post_loss_month_bounce_rate,
        "equity_high_reclaim_rate": equity_high_reclaim_rate,
        "cycle_recovery_health": cycle_recovery_health,
    }
    raw = sum(RECOVERY_WEIGHTS[k] * float(v) for k, v in parts.items())
    return float(max(0.0, min(1.0, raw)))
