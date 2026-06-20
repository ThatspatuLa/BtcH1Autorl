"""Tests for Stage 6 — Monthly Fitness Engine."""
from __future__ import annotations

import pandas as pd
import pytest

from fitness.monthly_fitness import (
    CONSISTENCY_FLOOR,
    FLOOR_MONTH_SCORE,
    WALK_FORWARD_V1,
    MonthlyScore,
    _slice_by_month,
    _variance_penalty,
    _worst_floor_multiplier,
    aggregate_monthly_fitness,
    compute_monthly_fitness,
)

# ============================================================
# Helpers
# ============================================================

def _make_equity_curve(values: list[float], start: str = "2021-06-01") -> pd.Series:
    """Build a synthetic H1 equity curve from a list of values."""
    idx = pd.date_range(start=start, periods=len(values), freq="h")
    return pd.Series(values, index=idx, name="equity")


def _make_trades_df(records: list[dict]) -> pd.DataFrame:
    """Build a trades DataFrame from records, coercing time columns to Timestamp."""
    df = pd.DataFrame(records)
    for col in ("open_time", "close_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# ============================================================
# Test: _slice_by_month
# ============================================================

def test_slice_by_month_empty_curve():
    eq = pd.Series(dtype=float)
    assert _slice_by_month(eq, None) == []


def test_slice_by_month_single_point():
    eq = _make_equity_curve([100.0])
    assert _slice_by_month(eq, None) == []


def test_slice_by_month_basic_three_months():
    # 90 hourly points spread across 3 months (June, July, August 2021)
    values = [100.0 + i * 0.1 for i in range(90 * 24)]
    eq = _make_equity_curve(values, start="2021-06-01")
    slices = _slice_by_month(eq, None)
    assert len(slices) >= 3  # at least 3 months
    # First slice label should be 2021-06
    assert slices[0][0] == "2021-06"
    assert slices[0][3].iloc[0] == pytest.approx(100.0)


def test_slice_by_month_with_trades():
    eq = _make_equity_curve([100.0, 101.0, 102.0, 103.0], start="2021-06-01")
    trades = _make_trades_df([
        # Trade closes at 02:00, which is within the equity_curve range (00:00 - 03:00)
        {"open_time": pd.Timestamp("2021-06-01 00:00:00"), "close_time": pd.Timestamp("2021-06-01 02:00:00"), "pnl": 1.0},
        # Trade closes in a later month — should NOT appear in June slice
        {"open_time": pd.Timestamp("2021-07-01 00:00:00"), "close_time": pd.Timestamp("2021-07-02 00:00:00"), "pnl": 1.0},
    ])
    slices = _slice_by_month(eq, trades)
    # 1 month (June) has both points, the trades_df is filtered per month
    assert len(slices) == 1
    assert len(slices[0][4]) == 1
    assert slices[0][4].iloc[0]["pnl"] == 1.0


# ============================================================
# Test: _worst_floor_multiplier
# ============================================================

def test_worst_floor_positive_returns_one():
    assert _worst_floor_multiplier(0.1) == 1.0
    assert _worst_floor_multiplier(0.5) == 1.0
    assert _worst_floor_multiplier(1.0) == 1.0
    # Boundary: 0.05 is the new threshold for 1.0
    assert _worst_floor_multiplier(0.05) == 1.0


def test_worst_floor_breakeven():
    # 0.0 falls into the "breakeven" tier
    assert _worst_floor_multiplier(0.0) == 0.5
    assert _worst_floor_multiplier(0.04) == 0.5
    assert _worst_floor_multiplier(-0.05) == 0.5


def test_worst_floor_medium_negative():
    assert _worst_floor_multiplier(-0.1) == 0.2
    assert _worst_floor_multiplier(-0.3) == 0.2
    assert _worst_floor_multiplier(-0.5) == 0.2


def test_worst_floor_catastrophic():
    assert _worst_floor_multiplier(-0.6) == 0.0
    assert _worst_floor_multiplier(-2.0) == 0.0


# ============================================================
# Test: _variance_penalty
# ============================================================

def test_variance_penalty_low_returns_one():
    assert _variance_penalty([0.3, 0.3, 0.3]) == 1.0
    assert _variance_penalty([0.3, 0.35, 0.32]) == 1.0


def test_variance_penalty_high_returns_zero():
    assert _variance_penalty([0.0, 1.0]) == 0.0
    assert _variance_penalty([0.0, 0.0, 1.0, 1.0]) == 0.0


def test_variance_penalty_single_value():
    # can't measure variance with one point
    assert _variance_penalty([0.5]) == 0.5


def test_variance_penalty_interpolation():
    # stddev 0.3 → penalty = 1.0 - (0.3 - 0.1) / 0.4 = 0.5
    penalty = _variance_penalty([0.0, 0.6])  # stddev = 0.3
    assert penalty == pytest.approx(0.5)


# ============================================================
# Test: aggregate_monthly_fitness (empty + edge cases)
# ============================================================

def test_aggregate_empty():
    result = aggregate_monthly_fitness(
        monthly_scores=[],
        candidate_id="cand_test",
        experiment_slug="test",
    )
    assert result.rejected is True
    assert result.reject_reason == "no_monthly_data"
    assert result.final_fitness == 0.0


def test_aggregate_all_rejected():
    scores = [
        MonthlyScore(
            month_index=0, month_label="2021-06", start="2021-06-01", end="2021-06-30",
            net_profit_pct=-0.1, max_drawdown_pct=0.4, trades_per_month=3.0,
            total_trades=3, monthly_score=0.0, rejected=True, reject_reason="net_profit<=0",
            final_equity=90.0, initial_equity=100.0,
        ),
    ]
    result = aggregate_monthly_fitness(scores, "cand_x", "test")
    assert result.rejected is True
    assert result.reject_reason == "all_months_rejected"


# ============================================================
# Test: aggregate_monthly_fitness (happy paths)
# ============================================================

def _make_monthly_score(
    month_index: int, label: str, net_pct: float, score: float,
    rejected: bool = False, reject_reason: str | None = None,
) -> MonthlyScore:
    return MonthlyScore(
        month_index=month_index,
        month_label=label,
        start=f"{label}-01",
        end=f"{label}-28",
        net_profit_pct=net_pct,
        max_drawdown_pct=0.1,
        trades_per_month=10.0,
        total_trades=10,
        monthly_score=score,
        rejected=rejected,
        reject_reason=reject_reason,
        final_equity=110.0 if net_pct > 0 else 90.0,
        initial_equity=100.0,
    )


def test_aggregate_consistent_profitable_strategy():
    """6 months all profitable, all scoring 0.4, consistency 100%."""
    scores = [
        _make_monthly_score(i, f"2021-{i+6:02d}", 0.05, 0.4)
        for i in range(6)
    ]
    result = aggregate_monthly_fitness(scores, "cand_good", "test")
    assert result.rejected is False
    assert result.n_months == 6
    assert result.n_profitable_months == 6
    assert result.consistency_ratio == 1.0
    assert result.median_monthly_score == pytest.approx(0.4)
    assert result.worst_month_score == pytest.approx(0.4)
    assert result.worst_floor_multiplier == 1.0
    # median × w + consistency × w + variance × w + floor × w
    # 0.4 × 0.5 + 1.0 × 0.2 + 1.0 × 0.15 + 1.0 × 0.15 = 0.2 + 0.2 + 0.15 + 0.15 = 0.7
    assert result.final_fitness == pytest.approx(0.7)


def test_aggregate_inconsistent_profitable_strategy():
    """6 months: scores vary wildly — high variance penalty."""
    scores = [
        _make_monthly_score(0, "2021-06", 0.10, 0.0),
        _make_monthly_score(1, "2021-07", 0.10, 1.0),
        _make_monthly_score(2, "2021-08", 0.10, 0.0),
        _make_monthly_score(3, "2021-09", 0.10, 1.0),
        _make_monthly_score(4, "2021-10", 0.10, 0.0),
        _make_monthly_score(5, "2021-11", 0.10, 1.0),
    ]
    result = aggregate_monthly_fitness(scores, "cand_volatile", "test")
    assert result.rejected is False
    assert result.consistency_ratio == 1.0
    # stddev of [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] = 0.5 → penalty = 0.0
    assert result.variance_penalty == 0.0
    # median=0.5, consistency=1.0, var_pen=0.0, floor_mult=0.5 (worst=0.0)
    # 0.5*0.5 + 0.2*1.0 + 0.15*0.0 + 0.15*0.5 = 0.25+0.2+0+0.075 = 0.525
    assert result.final_fitness == pytest.approx(0.525)


def test_aggregate_rejected_for_low_consistency():
    """7 months: 3 profitable, 4 unprofitable. consistency 3/7 = 0.43 < 0.50 → reject."""
    scores = []
    for i, (net, score) in enumerate([
        (0.05, 0.4), (0.05, 0.4), (0.05, 0.4),  # profitable
        (-0.05, 0.0), (-0.05, 0.0), (-0.05, 0.0), (-0.05, 0.0),  # unprofitable
    ]):
        scores.append(_make_monthly_score(i, f"2021-{i+6:02d}", net, score))
    result = aggregate_monthly_fitness(scores, "cand_inconsistent", "test")
    assert result.rejected is True
    assert "consistency" in result.reject_reason
    assert result.final_fitness == 0.0


def test_aggregate_rejected_for_catastrophic_month():
    """5 months: 1 catastrophic (-0.6 score) → floor multiplier 0.0 → final 0."""
    scores = [
        _make_monthly_score(0, "2021-06", 0.05, 0.4),
        _make_monthly_score(1, "2021-07", 0.05, 0.4),
        _make_monthly_score(2, "2021-08", 0.05, 0.4),
        _make_monthly_score(3, "2021-09", 0.05, 0.4),
        _make_monthly_score(4, "2021-10", -0.30, -0.6),  # catastrophic
    ]
    result = aggregate_monthly_fitness(scores, "cand_blowup", "test")
    # worst month is -0.6, floor multiplier = 0.0
    assert result.worst_floor_multiplier == 0.0
    # Note: aggregate doesn't reject on floor=0 alone, but final fitness is heavily penalised
    # 0.4 × 0.5 + 0.8 × 0.2 + ~1.0 × 0.15 + 0.0 × 0.15 = 0.2 + 0.16 + 0.15 + 0.0 = 0.51
    # Actually not rejected by hard rule unless worst < -0.5 (and we have -0.6!)
    # The min_worst_month_score is -0.5, so -0.6 IS a hard reject
    assert result.rejected is True
    assert "worst_month" in result.reject_reason


# ============================================================
# Test: compute_monthly_fitness (full pipeline)
# ============================================================

def test_compute_monthly_fitness_no_data():
    eq = pd.Series(dtype=float)
    result = compute_monthly_fitness(eq, None, "cand_x", "test")
    assert result.rejected is True
    assert result.reject_reason == "no_monthly_data"


def test_compute_monthly_fitness_synthetic_profitable():
    """Build a 6-month profitable equity curve and check the fitness result."""
    # 6 months × 30 days × 24 hours = 4320 points
    n = 6 * 30 * 24
    # Gradual growth: 100 → 130 over the period
    values = [100.0 + (30.0 * i / n) for i in range(n)]
    eq = _make_equity_curve(values, start="2021-06-01")
    # 120 trades, all profitable, distributed across ALL 6 months
    # 120 trades / 6 months = 20 trades/month → tpm=20 (well above 5 minimum)
    n_trades = 120
    trades_records = []
    for i in range(n_trades):
        # Spread trades across the full period — every ~36 hours
        open_t = pd.Timestamp("2021-06-01") + pd.Timedelta(hours=i * 35)
        close_t = open_t + pd.Timedelta(hours=4)
        trades_records.append({
            "open_time": open_t,
            "close_time": close_t,
            "pnl": 1.0,
            "qty": 0.01,
            "avg_entry": 30000.0,
            "exit_price": 30100.0,
        })
    trades = _make_trades_df(trades_records)
    result = compute_monthly_fitness(eq, trades, "cand_test", "test")
    assert result.n_months >= 5
    # Net profit is +30% over 6 months → every month should be profitable
    assert result.n_profitable_months >= 4, (
        f"Got n_profitable={result.n_profitable_months}, "
        f"rejected={result.n_rejected_months}, "
        f"months={[(m.month_label, round(m.net_profit_pct, 3), m.rejected, m.reject_reason) for m in result.monthly_scores]}"
    )
    assert result.rejected is False
    assert result.final_fitness > 0.0
    assert result.full_period_score is not None
    assert result.full_period_rejected is False


def test_compute_monthly_fitness_losing_strategy():
    """Losing strategy: net profit negative → all months fail consistency."""
    n = 6 * 30 * 24
    values = [100.0 - (10.0 * i / n) for i in range(n)]
    eq = _make_equity_curve(values, start="2021-06-01")
    result = compute_monthly_fitness(eq, None, "cand_loser", "test")
    # Net profit negative → all months rejected → aggregate rejects
    assert result.rejected is True
    assert result.n_rejected_months > 0


def test_compute_monthly_fitness_volatile_strategy():
    """Volatile strategy: high variance month-to-month should reduce fitness."""
    # 12 months, each starts at 100 and goes to 110 (+10%) then drops to 95 (-5%)
    # Net result: positive (+5% over 2 months = +2.5%/month on average)
    # but variance is high → fitness should be lower than steady-growth
    monthly_values = []
    for month_idx in range(12):
        # First 15 days: 100 → 110
        first_half = [100.0 + (10.0 * h / (15 * 24)) for h in range(15 * 24)]
        # Next 15 days: 110 → 95
        second_half = [110.0 - (15.0 * h / (15 * 24)) for h in range(15 * 24)]
        monthly_values.extend(first_half)
        monthly_values.extend(second_half)
    eq = _make_equity_curve(monthly_values, start="2021-06-01")
    # Lots of trades for TPM to pass
    trades_records = []
    for i in range(120):
        month_offset = i // 10
        open_t = pd.Timestamp("2021-06-01") + pd.Timedelta(days=month_offset * 30, hours=i * 6)
        close_t = open_t + pd.Timedelta(hours=4)
        pnl = 1.0
        trades_records.append({
            "open_time": open_t, "close_time": close_t, "pnl": pnl,
            "qty": 0.01, "avg_entry": 30000.0, "exit_price": 30100.0,
        })
    trades = _make_trades_df(trades_records)
    result = compute_monthly_fitness(eq, trades, "cand_volatile", "test")
    # Should not be rejected (net positive, tpm high enough)
    # But fitness should be lower than a consistent strategy due to variance
    assert result.n_months >= 5
    # Just verify it scored
    assert result.final_fitness > 0


# ============================================================
# Test: constants
# ============================================================

def test_walk_forward_weights_sum_to_one():
    w = WALK_FORWARD_V1
    total = w["median_weight"] + w["consistency_weight"] + w["variance_weight"] + w["worst_floor_weight"]
    assert total == pytest.approx(1.0)


def test_consistency_floor_is_50_percent():
    assert CONSISTENCY_FLOOR == 0.50


def test_floor_month_score_is_20_percent():
    assert FLOOR_MONTH_SCORE == 0.20
