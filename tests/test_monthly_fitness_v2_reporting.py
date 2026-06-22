"""Phase C/D — Tests for v2 reporting fields and hard-reject preservation.

Verifies:
- C1: MonthlyScore gains per-month recovery subscores (currently empty list,
  reserved for future per-month reporting).
- C2: MonthlyFitnessResult has new fields populated.
- C3.5: Hard rejects (worst_month < -0.50, median < 0.10) preserve their
  rejection in v2 and zero all v2 fields.
- D: All new fields surface in MonthlyFitnessResult.to_dict() for reports.
"""
from __future__ import annotations

import pytest

from fitness.monthly_fitness import (
    MonthlyFitnessResult,
    aggregate_monthly_fitness,
)


def _make_monthly_score(idx: int, label: str, net_profit_pct: float, monthly_score: float):
    """Build a MonthlyScore-shaped dict (we use the real dataclass via monthly_fitness)."""
    from fitness.monthly_fitness import MonthlyScore
    return MonthlyScore(
        month_index=idx,
        month_label=label,
        start="2021-06-01T00:00:00",
        end="2021-07-01T00:00:00",
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=0.1,
        trades_per_month=5.0,
        total_trades=5,
        monthly_score=monthly_score,
        rejected=False,
        reject_reason=None,
        final_equity=11000.0,
        initial_equity=10000.0,
    )


# ============================================================
# C1: Per-month recovery subscores
# ============================================================

def test_monthly_score_has_recovery_subscores_field():
    """MonthlyScore gains recovery_subscores dict (defaults to empty)."""
    from fitness.monthly_fitness import MonthlyScore
    ms = MonthlyScore(
        month_index=0, month_label="2021-06",
        start="2021-06-01T00:00:00", end="2021-07-01T00:00:00",
        net_profit_pct=0.05, max_drawdown_pct=0.1,
        trades_per_month=5.0, total_trades=5,
        monthly_score=0.4, rejected=False, reject_reason=None,
        final_equity=11000.0, initial_equity=10000.0,
    )
    assert hasattr(ms, "recovery_subscores")
    assert isinstance(ms.recovery_subscores, dict)


# ============================================================
# C2: MonthlyFitnessResult has new fields populated
# ============================================================

def test_monthly_fitness_result_has_v2_fields():
    """MonthlyFitnessResult gains full_period_base_score, recovery_score, etc."""
    # Build a minimal result and verify all new fields exist
    result = MonthlyFitnessResult(
        candidate_id="test", experiment_slug="test",
        monthly_scores=[], n_months=0, n_profitable_months=0, n_rejected_months=0,
        consistency_ratio=0.0, median_monthly_score=0.0, worst_month_score=0.0,
        stddev_monthly_score=0.0, variance_penalty=0.0, worst_floor_multiplier=0.0,
        base_aggregate_fitness=0.0, discovery_fitness=0.0, consistency_multiplier=0.0,
        full_period_base_score=0.0, recovery_score=0.0, stability_score=0.0,
        concentration_score=0.0,
        recovery_breakdown={}, per_month_recovery=[],
        deployment_fitness=0.0, deployment_pass=False,
        failed_deployment_gates=[], closest_to_passing_score=0.0,
        final_fitness=0.0, rejected=True, reject_reason="test",
        full_period_score=None, full_period_rejected=False,
    )
    for attr in ["full_period_base_score", "recovery_score", "stability_score",
                 "concentration_score", "recovery_breakdown", "per_month_recovery"]:
        assert hasattr(result, attr), f"MonthlyFitnessResult missing {attr}"


def test_aggregate_populates_v2_fields():
    """aggregate_monthly_fitness populates new v2 fields when given valid data."""
    scores = [
        _make_monthly_score(i, f"2021-{i+6:02d}", 0.05, 0.5)
        for i in range(6)
    ]
    result = aggregate_monthly_fitness(
        scores, "cand_v2", "test",
        full_period_score=0.6, full_period_rejected=False,
    )
    # New fields must be populated
    assert result.full_period_base_score == pytest.approx(0.6)
    assert result.recovery_score > 0.0
    assert result.stability_score == 1.0  # zero stddev
    assert result.concentration_score > 0.0
    assert "drawdown_recovery_speed" in result.recovery_breakdown
    assert len(result.recovery_breakdown) == 4


def test_aggregate_v2_to_dict_includes_all_fields():
    """to_dict() includes all new v2 fields for downstream reporting."""
    scores = [_make_monthly_score(0, "2021-06", 0.05, 0.5)]
    result = aggregate_monthly_fitness(scores, "cand_v2", "test")
    d = result.to_dict()
    for field in ["full_period_base_score", "recovery_score", "stability_score",
                  "concentration_score", "recovery_breakdown", "per_month_recovery",
                  "discovery_fitness", "consistency_ratio", "deployment_fitness"]:
        assert field in d, f"to_dict() missing {field}"


# ============================================================
# C3.5: Hard reject preservation
# ============================================================

def test_hard_reject_worst_month_zeroes_all_v2_fields():
    """worst_month_score < -0.50 → discovery_fitness=0, all v2 fields=0."""
    scores = [
        _make_monthly_score(0, "2021-06", 0.10, 0.5),    # profitable
        _make_monthly_score(1, "2021-07", -0.10, -0.6),   # catastrophic loss
        _make_monthly_score(2, "2021-08", 0.10, 0.5),
    ]
    result = aggregate_monthly_fitness(scores, "cand_rejected", "test")
    assert result.rejected is True
    assert result.reject_reason == "worst_month<-0.50"
    assert result.discovery_fitness == 0.0
    assert result.full_period_base_score == 0.0
    assert result.recovery_score == 0.0
    assert result.stability_score == 0.0
    assert result.concentration_score == 0.0
    assert all(v == 0.0 for v in result.recovery_breakdown.values())


def test_hard_reject_median_month_zeroes_all_v2_fields():
    """median_monthly_score < 0.10 → discovery_fitness=0, all v2 fields=0."""
    scores = [
        _make_monthly_score(i, f"2021-{i+6:02d}", 0.0, 0.05)  # all just above catastrophic
        for i in range(6)
    ]
    # All scores 0.05, median = 0.05 < 0.10
    result = aggregate_monthly_fitness(scores, "cand_median_rej", "test")
    assert result.rejected is True
    assert result.reject_reason == "median<0.10"
    assert result.discovery_fitness == 0.0
    assert result.full_period_base_score == 0.0


def test_no_data_rejected_zeroes_all_v2_fields():
    """No monthly data → rejected=True, all v2 fields=0."""
    result = aggregate_monthly_fitness([], "cand_empty", "test")
    assert result.rejected is True
    assert result.discovery_fitness == 0.0
    assert result.full_period_base_score == 0.0
    assert result.recovery_score == 0.0


# ============================================================
# Realistic data integration
# ============================================================

def test_aggregate_with_synthetic_equity_curve_populates_recovery():
    """When equity_curve is passed, recovery sub-metrics compute real values."""
    import pandas as pd
    scores = [_make_monthly_score(i, f"2021-{i+6:02d}", 0.05, 0.5) for i in range(6)]
    # Synthetic equity: rises steadily → no DD → recovery_speed = 1.0
    idx = pd.date_range("2021-06-01", periods=100, freq="D")
    eq = pd.Series([100.0 + i * 1.0 for i in range(100)], index=idx)
    result = aggregate_monthly_fitness(
        scores, "cand_eq", "test",
        equity_curve=eq,
    )
    # No DD → drawdown_recovery_speed should be 1.0
    assert result.recovery_breakdown["drawdown_recovery_speed"] == pytest.approx(1.0)
    # All months profitable → consistency_ratio = 1.0 → bounce_rate = 1.0
    assert result.recovery_breakdown["post_loss_month_bounce_rate"] == pytest.approx(1.0)
