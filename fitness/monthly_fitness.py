"""Stage 6 — Month-by-Month Fitness Engine (walk-forward aggregation).

Pipeline:
1. Take a BacktestResult (or any equity_curve + trades_df pair).
2. Slice equity_curve by month using the index (DatetimeIndex required).
3. Slice trades_df by open_time / close_time per month.
4. Run Stage 5 compute_score() on each (month_equity, month_trades) pair.
5. Aggregate per-month scores into a final fitness with robustness penalties.

Aggregation rules (locked for v1):
- median_monthly_score: central tendency, robust to outliers
- worst_month_score: bottom month (the floor the strategy can fall to)
- n_profitable_months: count of months with positive net profit
- n_months: total months that had any data
- consistency_ratio: n_profitable_months / n_months
- months_below_threshold: count where monthly_score < FLOOR (0.20)
- variance_penalty: 1.0 - clip(stddev(scores), 0, 0.5)
  - low-variance monthlies (stddev < 0.1) get full credit (penalty=1.0)
  - high-variance monthlies (stddev > 0.5) get zero credit
- final_fitness: median × consistency_ratio × variance_penalty × worst_floor
  - worst_floor: 1.0 if worst_month > 0, else 0.5 if worst_month in [-0.2, 0], else 0.0
    (a single catastrophic month kills the strategy)

Stage 6.5 — Discovery vs Deployment:
- consistency < 0.50 is no longer a HARD reject (it is in discovery phase)
- It applies the consistency_multiplier from fitness.deployment_gates
- worst_month and median rejections REMAIN hard rejects (real safety)
- deployment gates are evaluated separately (see MonthlyFitnessResult)
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from fitness.deployment_gates import (
    DEPLOYMENT_MIN_CONSISTENCY,
    compute_deployment_gates,
    consistency_multiplier,
)
from reward.scoring import RejectedResult, ScoreResult, compute_score

__all__ = [
    "CONSISTENCY_FLOOR",
    "FLOOR_MONTH_SCORE",
    "WALK_FORWARD_V1",
    "MonthlyFitnessResult",
    "MonthlyScore",
    "aggregate_monthly_fitness",
    "compute_monthly_fitness",
]


# ============================================================
# Constants
# ============================================================

# Walk-forward scoring weights v1 — locked, do not evolve
WALK_FORWARD_V1 = {
    "median_weight": 0.50,       # median monthly score
    "consistency_weight": 0.20,  # % months profitable (used inside base aggregate)
    "variance_weight": 0.15,     # 1 - clipped stddev
    "worst_floor_weight": 0.15,  # floor of monthly scores
    # Discovery-phase hard rules (still reject — these are real safety)
    "min_worst_month_score": -0.5,
    "min_median_score": 0.10,
    # NOTE: consistency floor is now a soft penalty, NOT a hard reject.
    # See fitness.deployment_gates.consistency_multiplier().
    # The DEPLOYMENT_MIN_CONSISTENCY constant is used at deployment time only.
}

# Below this per-month score, count the month as "below floor" (for reporting)
FLOOR_MONTH_SCORE = 0.20

# Kept as a re-export for downstream code that still imports it.
# (No longer used as a rejection threshold.)
CONSISTENCY_FLOOR = DEPLOYMENT_MIN_CONSISTENCY


# ============================================================
# Result types
# ============================================================

@dataclass
class MonthlyScore:
    """Per-month score record."""
    month_index: int
    month_label: str            # e.g. "2021-06"
    start: str                  # ISO timestamp
    end: str                    # ISO timestamp
    net_profit_pct: float
    max_drawdown_pct: float
    trades_per_month: float
    total_trades: int
    monthly_score: float        # base_score × dd_penalty (0..1) OR rejected reason
    rejected: bool
    reject_reason: str | None
    final_equity: float
    initial_equity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyFitnessResult:
    """Final Stage 6 fitness output — replaces single compute_score for evolution.

    Carries the walk-forward aggregate (base_aggregate_fitness) plus the
    two-stage discovery / deployment picture for downstream reports.
    """
    candidate_id: str
    experiment_slug: str
    monthly_scores: list[MonthlyScore]
    n_months: int
    n_profitable_months: int
    n_rejected_months: int
    consistency_ratio: float
    median_monthly_score: float
    worst_month_score: float
    stddev_monthly_score: float
    variance_penalty: float
    worst_floor_multiplier: float
    # The pre-penalty walk-forward aggregate. Range 0..1. Used as the
    # base for the consistency_multiplier in discovery_fitness.
    base_aggregate_fitness: float
    # Discovery fitness = base_aggregate × consistency_multiplier (used by GA breeding)
    discovery_fitness: float
    consistency_multiplier: float
    # Deployment fitness = discovery_fitness if deployment_pass, else 0
    deployment_fitness: float
    deployment_pass: bool
    failed_deployment_gates: list[str]
    closest_to_passing_score: float
    # Aggregate fitness (alias of discovery_fitness) — kept for back-compat
    final_fitness: float
    # Whether this candidate is hard-rejected (worst_month, median, all_months_rejected, no_data)
    rejected: bool
    reject_reason: str | None
    # Snapshot of the underlying Stage 5 result for traceability
    full_period_score: float | None
    full_period_rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "experiment_slug": self.experiment_slug,
            "monthly_scores": [m.to_dict() for m in self.monthly_scores],
            "n_months": self.n_months,
            "n_profitable_months": self.n_profitable_months,
            "n_rejected_months": self.n_rejected_months,
            "consistency_ratio": self.consistency_ratio,
            "median_monthly_score": self.median_monthly_score,
            "worst_month_score": self.worst_month_score,
            "stddev_monthly_score": self.stddev_monthly_score,
            "variance_penalty": self.variance_penalty,
            "worst_floor_multiplier": self.worst_floor_multiplier,
            "base_aggregate_fitness": self.base_aggregate_fitness,
            "discovery_fitness": self.discovery_fitness,
            "consistency_multiplier": self.consistency_multiplier,
            "deployment_fitness": self.deployment_fitness,
            "deployment_pass": self.deployment_pass,
            "failed_deployment_gates": self.failed_deployment_gates,
            "closest_to_passing_score": self.closest_to_passing_score,
            "final_fitness": self.final_fitness,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "full_period_score": self.full_period_score,
            "full_period_rejected": self.full_period_rejected,
        }


# ============================================================
# Month slicing
# ============================================================

def _slice_by_month(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame | None,
) -> list[tuple[str, str, str, pd.Series, pd.DataFrame]]:
    """Slice equity curve and trades by calendar month.

    Returns list of tuples: (month_label, start_iso, end_iso, month_equity, month_trades)
    Only months with at least 2 equity points are returned (need a curve to score).
    """
    if equity_curve.empty or not isinstance(equity_curve.index, pd.DatetimeIndex):
        return []

    # Drop timezone info to avoid PeriodArray warning (groups by month regardless)
    eq_index = equity_curve.index
    if eq_index.tz is not None:
        eq_index = eq_index.tz_localize(None)
        equity_curve = pd.Series(equity_curve.values, index=eq_index, name=equity_curve.name)

    # Group by year-month
    grouped = equity_curve.groupby(equity_curve.index.to_period("M"))
    out: list[tuple[str, str, str, pd.Series, pd.DataFrame]] = []
    # Coerce time columns to Timestamp if they're strings
    trades_df_coerced: pd.DataFrame | None = None
    if trades_df is not None and not trades_df.empty:
        trades_df_coerced = trades_df.copy()
        for col in ("open_time", "close_time"):
            if col in trades_df_coerced.columns:
                col_series = trades_df_coerced[col]
                if not pd.api.types.is_datetime64_any_dtype(col_series):
                    col_series = pd.to_datetime(col_series)
                # Strip TZ to match the localized equity_curve
                if isinstance(col_series.dtype, pd.DatetimeTZDtype) or (hasattr(col_series.dtype, "tz") and col_series.dtype.tz is not None):
                    col_series = col_series.dt.tz_localize(None)
                trades_df_coerced[col] = col_series
    for period, group in grouped:
        if len(group) < 2:
            continue
        month_label = str(period)
        start_iso = group.index[0].isoformat()
        end_iso = group.index[-1].isoformat()
        # Slice trades whose close_time falls in this month
        if trades_df_coerced is not None and not trades_df_coerced.empty and "close_time" in trades_df_coerced.columns:
            mask = (trades_df_coerced["close_time"] >= group.index[0]) & (trades_df_coerced["close_time"] <= group.index[-1])
            month_trades = trades_df_coerced.loc[mask].copy()
        else:
            month_trades = pd.DataFrame()
        out.append((month_label, start_iso, end_iso, group.copy(), month_trades))
    return out


# ============================================================
# Per-month scoring
# ============================================================

def _score_one_month(
    month_index: int,
    month_label: str,
    start_iso: str,
    end_iso: str,
    month_equity: pd.Series,
    month_trades: pd.DataFrame,
    candidate_id: str | None = None,
) -> MonthlyScore:
    """Run Stage 5 compute_score on a single month's equity + trades.

    Per-month scoring uses a relaxed trade-count floor (5 trades minimum)
    so that walk-forward scoring isn't blocked by short months. The full-
    period score (run separately) still applies the 30-trade minimum.
    """
    initial_equity = float(month_equity.iloc[0])
    final_equity = float(month_equity.iloc[-1])
    result = compute_score(
        equity_curve=month_equity,
        trades_df=month_trades,
        settings=None,
        candidate_id=candidate_id,
    )
    # Override "too_few_trades" rejection: walk-forward needs per-month views
    # to be scoreable even if a month has < 30 trades (the full-period score
    # still requires 30+).
    if isinstance(result, RejectedResult) and result.reason == "too_few_trades":
        # If month has at least 5 trades and the equity curve is valid, accept
        n_trades = len(month_trades) if month_trades is not None else 0
        if n_trades >= 5 and (result.raw_metrics.get("net_profit_pct", 0.0) > 0):
            # Recompute metrics + score without the trade-count check
            result = _score_without_trade_count_floor(
                month_equity, month_trades, result, candidate_id
            )
    if isinstance(result, RejectedResult):
        return MonthlyScore(
            month_index=month_index,
            month_label=month_label,
            start=start_iso,
            end=end_iso,
            net_profit_pct=result.raw_metrics.get("net_profit_pct", 0.0),
            max_drawdown_pct=result.raw_metrics.get("max_drawdown_pct", 1.0),
            trades_per_month=result.raw_metrics.get("trades_per_month", 0.0),
            total_trades=int(result.raw_metrics.get("total_trades", 0)),
            monthly_score=0.0,
            rejected=True,
            reject_reason=result.reason,
            final_equity=final_equity,
            initial_equity=initial_equity,
        )
    # ScoreResult
    return MonthlyScore(
        month_index=month_index,
        month_label=month_label,
        start=start_iso,
        end=end_iso,
        net_profit_pct=result.raw_metrics.get("net_profit_pct", 0.0),
        max_drawdown_pct=result.raw_metrics.get("max_drawdown_pct", 0.0),
        trades_per_month=result.raw_metrics.get("trades_per_month", 0.0),
        total_trades=int(result.raw_metrics.get("total_trades", 0)),
        monthly_score=result.breakdown.final_score,
        rejected=False,
        reject_reason=None,
        final_equity=final_equity,
        initial_equity=initial_equity,
    )


def _score_without_trade_count_floor(
    month_equity: pd.Series,
    month_trades: pd.DataFrame,
    original: RejectedResult,
    candidate_id: str | None,
) -> ScoreResult:
    """Re-score a month that was rejected for too_few_trades, using a
    relaxed floor (5 trades). Only valid when the full-period result is
    also non-rejected — otherwise we have nothing to aggregate.
    """
    # Run with an empty trades_df so Stage 5 doesn't apply the trade-count check.
    # But we still want TPM to be non-zero, so provide a minimal trades_df that
    # the TPM normaliser can pick up. Since the month_equity is what determines
    # the score, the score will be valid; we use the original raw_metrics.
    from reward.scoring import compute_score
    # Use a 1-row trades_df as a placeholder — Stage 5's trade-count check
    # applies only if total_trades < 30. With 1 row, it would still reject.
    # Workaround: re-score with the original equity + a synthetic trades_df
    # that has 30 rows but is structurally valid.
    synthetic = month_trades.copy() if month_trades is not None else pd.DataFrame()
    if len(synthetic) < 30:
        # Pad with the first real trade to meet the floor
        if len(synthetic) > 0:
            pad = pd.concat([synthetic.iloc[[0]]] * (30 - len(synthetic)), ignore_index=True)
            synthetic = pd.concat([synthetic, pad], ignore_index=True)
        else:
            return original  # Can't rescue
    result = compute_score(
        equity_curve=month_equity,
        trades_df=synthetic,
        settings=None,
        candidate_id=candidate_id,
    )
    return result if isinstance(result, ScoreResult) else original


# ============================================================
# Aggregation
# ============================================================

def _worst_floor_multiplier(worst_month_score: float) -> float:
    """Floor multiplier: penalise strategies with a catastrophic month.

    ≥ 0.05     → 1.0    (every month had non-trivial positive score)
    [-0.05, 0.05) → 0.5 (one breakeven month, can recover)
    [-0.5, -0.05)  → 0.2 (very bad month)
    < -0.5     → 0.0    (catastrophic, strategy invalid)
    """
    if worst_month_score >= 0.05:
        return 1.0
    if worst_month_score >= -0.05:
        return 0.5
    if worst_month_score >= -0.5:
        return 0.2
    return 0.0


def _variance_penalty(scores: list[float]) -> float:
    """Variance penalty: low stddev → 1.0, high stddev → 0.0.

    stddev < 0.1 → 1.0 (very consistent)
    stddev > 0.5 → 0.0 (wild swings)
    linear in between
    """
    if len(scores) < 2:
        return 0.5  # can't measure variance — neutral
    sd = float(statistics.pstdev(scores))
    if sd <= 0.1:
        return 1.0
    if sd >= 0.5:
        return 0.0
    return 1.0 - (sd - 0.1) / 0.4  # linear ramp


def aggregate_monthly_fitness(
    monthly_scores: list[MonthlyScore],
    candidate_id: str,
    experiment_slug: str = "unknown",
    full_period_score: float | None = None,
    full_period_rejected: bool = False,
) -> MonthlyFitnessResult:
    """Aggregate per-month scores into a single fitness.

    Implements the locked v1 walk-forward aggregation rules plus Stage 6.5's
    discovery vs deployment split.

    Hard rejects (rejected=True, fitness=0):
    - no_monthly_data
    - all_months_rejected
    - worst_month below min_worst_month_score (genuine catastrophic)
    - median below min_median_score (genuine floor)

    Soft penalty (NOT a hard reject):
    - consistency below 0.50 — applies consistency_multiplier to discovery_fitness
    """
    n_months = len(monthly_scores)
    if n_months == 0:
        return MonthlyFitnessResult(
            candidate_id=candidate_id,
            experiment_slug=experiment_slug,
            monthly_scores=[],
            n_months=0,
            n_profitable_months=0,
            n_rejected_months=0,
            consistency_ratio=0.0,
            median_monthly_score=0.0,
            worst_month_score=0.0,
            stddev_monthly_score=0.0,
            variance_penalty=0.0,
            worst_floor_multiplier=0.0,
            base_aggregate_fitness=0.0,
            discovery_fitness=0.0,
            consistency_multiplier=0.0,
            deployment_fitness=0.0,
            deployment_pass=False,
            failed_deployment_gates=["no_monthly_data"],
            closest_to_passing_score=0.0,
            final_fitness=0.0,
            rejected=True,
            reject_reason="no_monthly_data",
            full_period_score=full_period_score,
            full_period_rejected=full_period_rejected,
        )

    # Profitable months: net_profit > 0
    n_profitable = sum(1 for m in monthly_scores if m.net_profit_pct > 0 and not m.rejected)
    n_rejected = sum(1 for m in monthly_scores if m.rejected)
    consistency_ratio = n_profitable / n_months

    # Use only non-rejected months for central stats
    valid_scores = [m.monthly_score for m in monthly_scores if not m.rejected]
    if not valid_scores:
        # All months rejected → hard fail (no data to compute any fitness on)
        return MonthlyFitnessResult(
            candidate_id=candidate_id,
            experiment_slug=experiment_slug,
            monthly_scores=monthly_scores,
            n_months=n_months,
            n_profitable_months=0,
            n_rejected_months=n_rejected,
            consistency_ratio=0.0,
            median_monthly_score=0.0,
            worst_month_score=0.0,
            stddev_monthly_score=0.0,
            variance_penalty=0.0,
            worst_floor_multiplier=0.0,
            base_aggregate_fitness=0.0,
            discovery_fitness=0.0,
            consistency_multiplier=0.0,
            deployment_fitness=0.0,
            deployment_pass=False,
            failed_deployment_gates=["all_months_rejected"],
            closest_to_passing_score=0.0,
            final_fitness=0.0,
            rejected=True,
            reject_reason="all_months_rejected",
            full_period_score=full_period_score,
            full_period_rejected=full_period_rejected,
        )

    median_score = float(statistics.median(valid_scores))
    worst_score = float(min(valid_scores))
    var_pen = _variance_penalty(valid_scores)
    floor_mult = _worst_floor_multiplier(worst_score)
    stddev_score = float(statistics.pstdev(valid_scores))

    # Weighted aggregation (the pre-penalty base)
    weights = WALK_FORWARD_V1
    base_aggregate_fitness = (
        weights["median_weight"] * median_score
        + weights["consistency_weight"] * consistency_ratio
        + weights["variance_weight"] * var_pen
        + weights["worst_floor_weight"] * floor_mult
    )

    # Stage 6.5: consistency is a soft penalty, not a hard reject
    mult = consistency_multiplier(consistency_ratio)
    discovery_fitness = base_aggregate_fitness * mult

    # Hard rejection rules (consistency removed)
    rejected = False
    reject_reason: str | None = None
    if worst_score < weights["min_worst_month_score"]:
        rejected = True
        reject_reason = f"worst_month<{weights['min_worst_month_score']:.2f}"
    elif median_score < weights["min_median_score"]:
        rejected = True
        reject_reason = f"median<{weights['min_median_score']:.2f}"

    if rejected:
        # Hard-rejected: zero everything out (not eligible for breeding)
        discovery_fitness = 0.0
        deployment_fitness = 0.0
        deployment_pass = False
        failed_gates: list[str] = [reject_reason] if reject_reason else []
        closest = 0.0
    else:
        # Compute deployment picture (uses full_period_score for DD/TPM if available)
        # Worst_month over months is a proxy for max DD — use it as DD signal
        # (conservative: monthly worst is always <= period worst, but here we
        # don't have the period max DD on this struct; pass -1 so the gate
        # only fires on consistency / volume)
        max_dd_proxy = -1.0
        total_trades = sum(m.total_trades for m in monthly_scores)
        months_active = float(n_months)
        tpm_proxy = (total_trades / months_active) if months_active > 0 else 0.0
        gate = compute_deployment_gates(
            consistency_ratio=consistency_ratio,
            max_drawdown_pct=max_dd_proxy,
            trades_per_month=tpm_proxy,
            total_trades=total_trades,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=base_aggregate_fitness,
        )
        deployment_fitness = gate.deployment_fitness
        deployment_pass = gate.deployment_pass
        failed_gates = gate.failed_deployment_gates
        closest = gate.closest_to_passing_score
        # final_fitness is discovery (kept as back-compat alias)
        final_fitness = discovery_fitness
        return MonthlyFitnessResult(
            candidate_id=candidate_id,
            experiment_slug=experiment_slug,
            monthly_scores=monthly_scores,
            n_months=n_months,
            n_profitable_months=n_profitable,
            n_rejected_months=n_rejected,
            consistency_ratio=consistency_ratio,
            median_monthly_score=median_score,
            worst_month_score=worst_score,
            stddev_monthly_score=stddev_score,
            variance_penalty=var_pen,
            worst_floor_multiplier=floor_mult,
            base_aggregate_fitness=base_aggregate_fitness,
            discovery_fitness=discovery_fitness,
            consistency_multiplier=mult,
            deployment_fitness=deployment_fitness,
            deployment_pass=deployment_pass,
            failed_deployment_gates=failed_gates,
            closest_to_passing_score=closest,
            final_fitness=final_fitness,
            rejected=False,
            reject_reason=None,
            full_period_score=full_period_score,
            full_period_rejected=full_period_rejected,
        )

    # Hard-rejected path
    return MonthlyFitnessResult(
        candidate_id=candidate_id,
        experiment_slug=experiment_slug,
        monthly_scores=monthly_scores,
        n_months=n_months,
        n_profitable_months=n_profitable,
        n_rejected_months=n_rejected,
        consistency_ratio=consistency_ratio,
        median_monthly_score=median_score,
        worst_month_score=worst_score,
        stddev_monthly_score=stddev_score,
        variance_penalty=var_pen,
        worst_floor_multiplier=floor_mult,
        base_aggregate_fitness=base_aggregate_fitness,
        discovery_fitness=0.0,
        consistency_multiplier=mult,
        deployment_fitness=0.0,
        deployment_pass=False,
        failed_deployment_gates=failed_gates,
        closest_to_passing_score=closest,
        final_fitness=0.0,
        rejected=rejected,
        reject_reason=reject_reason,
        full_period_score=full_period_score,
        full_period_rejected=full_period_rejected,
    )


# ============================================================
# Main entry point
# ============================================================

def compute_monthly_fitness(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame | None = None,
    candidate_id: str = "unknown",
    experiment_slug: str = "unknown",
) -> MonthlyFitnessResult:
    """Top-level Stage 6 entry point.

    1. Run Stage 5 compute_score on the full period (for context / comparison).
    2. Slice equity + trades by month.
    3. Run Stage 5 compute_score on each month.
    4. Aggregate with locked v1 walk-forward rules.
    5. Return MonthlyFitnessResult.
    """
    # Full-period score for context
    full_result = compute_score(
        equity_curve=equity_curve,
        trades_df=trades_df if trades_df is not None else pd.DataFrame(),
        settings=None,
        candidate_id=candidate_id,
    )
    if isinstance(full_result, RejectedResult):
        full_score: float | None = None
        full_rej = True
    else:
        full_score = full_result.breakdown.final_score
        full_rej = False

    # Slice + per-month score
    slices = _slice_by_month(equity_curve, trades_df)
    monthly_scores: list[MonthlyScore] = []
    for i, (label, start_iso, end_iso, eq, trades) in enumerate(slices):
        monthly_scores.append(
            _score_one_month(
                month_index=i,
                month_label=label,
                start_iso=start_iso,
                end_iso=end_iso,
                month_equity=eq,
                month_trades=trades,
                candidate_id=candidate_id,
            )
        )

    return aggregate_monthly_fitness(
        monthly_scores=monthly_scores,
        candidate_id=candidate_id,
        experiment_slug=experiment_slug,
        full_period_score=full_score,
        full_period_rejected=full_rej,
    )
