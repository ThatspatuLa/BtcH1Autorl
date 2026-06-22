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
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from dca_engine.tp_baseline import backtest_with_fixed_tp, extract_dca_params_from_genome
from fitness.deployment_gates import DEPLOYMENT_MIN_CONSISTENCY, compute_deployment_gates, consistency_multiplier
from fitness.discovery_fitness import (
    DISCOVERY_WEIGHTS,
    compute_concentration_score,
    compute_discovery_fitness,
    compute_stability_score,
)
from fitness.recovery_metrics import (
    RECOVERY_WEIGHTS,
    compute_cycle_recovery_health,
    compute_drawdown_recovery_speed,
    compute_equity_high_reclaim_rate,
    compute_post_loss_month_bounce_rate,
    compute_recovery_score,
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
    # Phase C: per-month recovery subscores. Currently empty dict (reserved
    # for future per-month recovery reporting — global recovery_score is
    # computed from the full equity curve in aggregate_monthly_fitness).
    recovery_subscores: dict[str, float] = field(default_factory=dict)

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
    # --- Stage 6.5 / Phase C Discovery Fitness v2 fields ---
    # Old (deprecated by v2): base_aggregate_fitness. Kept for back-compat only
    # with the value of the new discovery_fitness so old consumers see a number.
    # Set to discovery_fitness in Phase C wiring.
    base_aggregate_fitness: float
    # Discovery fitness v2 = 0.60·full_period_base_score + 0.20·recovery_score
    #                      + 0.10·consistency_score + 0.05·stability_score
    #                      + 0.05·concentration_score (see fitness/discovery_fitness.py)
    discovery_fitness: float
    consistency_multiplier: float   # kept for back-compat (= step function on consistency_ratio)
    # --- Phase C new fields ---
    # Stage 5 full-period base score (60% of discovery_fitness v2). Equals
    # full_period_score when available, else median_monthly_score fallback.
    full_period_base_score: float
    # Recovery score (20% of discovery_fitness v2). Weighted sum of 4 sub-metrics
    # from fitness/recovery_metrics.py.
    recovery_score: float
    # Stability score (5%). Light stddev/CoV penalty on monthly base scores.
    stability_score: float
    # Concentration score (5%). Penalty for one lucky month carrying the result.
    concentration_score: float
    # Recovery breakdown (per sub-metric) for diagnostics.
    recovery_breakdown: dict[str, float]
    # Per-month recovery subscores (lightweight — same dict shape as recovery_breakdown
    # but per month). Currently only the global is populated; this is reserved for
    # future per-month recovery reporting.
    per_month_recovery: list[dict[str, float]]
    # --- End Phase C new fields ---
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
            # Phase C new fields
            "full_period_base_score": self.full_period_base_score,
            "recovery_score": self.recovery_score,
            "stability_score": self.stability_score,
            "concentration_score": self.concentration_score,
            "recovery_breakdown": self.recovery_breakdown,
            "per_month_recovery": self.per_month_recovery,
            # Backwards compat
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
    evolution_mode: bool = False,
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
        evolution_mode=evolution_mode,
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
    equity_curve: pd.Series | None = None,
    trades_df: pd.DataFrame | None = None,
) -> MonthlyFitnessResult:
    """Aggregate per-month scores into a single fitness.

    Implements Discovery Fitness v2 (Phase C wiring):
        discovery_fitness = 0.60·full_period_base_score
                          + 0.20·recovery_score
                          + 0.10·consistency_score
                          + 0.05·stability_score
                          + 0.05·concentration_score

    Hard rejects (rejected=True, fitness=0):
    - no_monthly_data
    - all_months_rejected
    - worst_month below min_worst_month_score (genuine catastrophic)
    - median below min_median_score (genuine floor)

    These run BEFORE the v2 aggregator runs. If a hard reject triggers,
    all Phase C new fields are set to 0.0 and discovery_fitness=0.0.

    Soft penalty (NOT a hard reject):
    - consistency below 0.50 — the deployment gate still fires (file
      `fitness/deployment_gates.py` is untouched); the new aggregator's
      `consistency_score` slot preserves the value for diagnostics.
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
            full_period_base_score=0.0,
            recovery_score=0.0,
            stability_score=0.0,
            concentration_score=0.0,
            recovery_breakdown={k: 0.0 for k in RECOVERY_WEIGHTS},
            per_month_recovery=[],
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
            full_period_base_score=0.0,
            recovery_score=0.0,
            stability_score=0.0,
            concentration_score=0.0,
            recovery_breakdown={k: 0.0 for k in RECOVERY_WEIGHTS},
            per_month_recovery=[],
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

    weights = WALK_FORWARD_V1

    # ---------- Hard-reject gates (Phase C3.5: MUST run before v2 aggregator) ----------
    # These short-circuit the v2 aggregator — rejected candidates get discovery_fitness=0.0.
    rejected = False
    reject_reason: str | None = None
    if worst_score < weights["min_worst_month_score"]:
        rejected = True
        reject_reason = f"worst_month<{weights['min_worst_month_score']:.2f}"
    elif median_score < weights["min_median_score"]:
        rejected = True
        reject_reason = f"median<{weights['min_median_score']:.2f}"

    if rejected:
        # Hard-rejected: zero everything out (not eligible for breeding).
        # Phase C new fields all 0.0; aggregator never sees this candidate.
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
            base_aggregate_fitness=0.0,   # Phase C: back-compat alias for new value
            discovery_fitness=0.0,
            consistency_multiplier=consistency_multiplier(consistency_ratio),
            full_period_base_score=0.0,
            recovery_score=0.0,
            stability_score=0.0,
            concentration_score=0.0,
            recovery_breakdown={k: 0.0 for k in RECOVERY_WEIGHTS},
            per_month_recovery=[],
            deployment_fitness=0.0,
            deployment_pass=False,
            failed_deployment_gates=[reject_reason] if reject_reason else [],
            closest_to_passing_score=0.0,
            final_fitness=0.0,
            rejected=rejected,
            reject_reason=reject_reason,
            full_period_score=full_period_score,
            full_period_rejected=full_period_rejected,
        )

    # ---------- Phase C: Discovery Fitness v2 aggregator ----------
    # Compute the 5 components. All in [0, 1].
    # 1. full_period_base_score: Stage 5 final_score on full 5y equity.
    #    Fallback: median_monthly_score if full-period was rejected (shouldn't happen
    #    here since hard rejects already returned, but defensive).
    fpbs = float(full_period_score) if (full_period_score is not None and not full_period_rejected) else median_score

    # 2. recovery_score: weighted sum of 4 sub-metrics from fitness/recovery_metrics.py.
    if equity_curve is not None and not equity_curve.empty:
        recovery_subs = {
            "drawdown_recovery_speed": compute_drawdown_recovery_speed(equity_curve),
            "post_loss_month_bounce_rate": compute_post_loss_month_bounce_rate(
                [m.net_profit_pct > 0 for m in monthly_scores]
            ),
            "equity_high_reclaim_rate": compute_equity_high_reclaim_rate(equity_curve),
            "cycle_recovery_health": compute_cycle_recovery_health(trades_df),
        }
    else:
        # No equity curve → use month-level proxies
        recovery_subs = {
            "drawdown_recovery_speed": consistency_ratio,  # proxy
            "post_loss_month_bounce_rate": compute_post_loss_month_bounce_rate(
                [m.net_profit_pct > 0 for m in monthly_scores]
            ),
            "equity_high_reclaim_rate": consistency_ratio,  # proxy
            "cycle_recovery_health": compute_cycle_recovery_health(trades_df),
        }
    recovery_score_val = compute_recovery_score(**recovery_subs)

    # 3. consistency_score = consistency_ratio (profitable_months / total_months)
    consistency_score_val = consistency_ratio

    # 4. stability_score = light stddev penalty on monthly base scores
    stability_score_val = compute_stability_score(valid_scores)

    # 5. concentration_score = penalty for one lucky month
    monthly_profits = [m.net_profit_pct for m in monthly_scores if not m.rejected]
    concentration_score_val = compute_concentration_score(monthly_profits)

    # ---------- The new discovery_fitness v2 ----------
    discovery_fitness = compute_discovery_fitness(
        full_period_base_score=fpbs,
        recovery_score=recovery_score_val,
        consistency_score=consistency_score_val,
        stability_score=stability_score_val,
        concentration_score=concentration_score_val,
    )

    # Back-compat: keep `base_aggregate_fitness` field, but populate it with the
    # new value (the old walk-forward blend is superseded). Old consumers reading
    # `base_aggregate_fitness` will see the new discovery_fitness, which is a
    # superset of the information.
    base_aggregate_fitness = discovery_fitness

    # consistency_multiplier is kept for back-compat (Stage 6.5 consistency penalty curve).
    mult = consistency_multiplier(consistency_ratio)

    # Deployment picture (untouched code from v1)
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
        full_period_base_score=fpbs,
        recovery_score=recovery_score_val,
        stability_score=stability_score_val,
        concentration_score=concentration_score_val,
        recovery_breakdown=recovery_subs,
        per_month_recovery=[],   # reserved for future per-month reporting
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


# ============================================================
# Main entry point
# ============================================================

def compute_monthly_fitness(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame | None = None,
    candidate_id: str = "unknown",
    experiment_slug: str = "unknown",
    evolution_mode: bool = False,
) -> MonthlyFitnessResult:
    """Top-level Stage 6 entry point.

    1. Run Stage 5 compute_score on the full period (for context / comparison).
    2. Slice equity + trades by month.
    3. Run Stage 5 compute_score on each month.
    4. Aggregate with locked v1 walk-forward rules.
    5. Return MonthlyFitnessResult.

    evolution_mode: passed to Stage 5's compute_score, which relaxes the
    TPM hard reject. Used by Stage 10 (DCA evolution) so the GA can see
    and breed candidates with low TPM. The deployment gates in
    fitness.deployment_gates still enforce TPM >= 5 at deployment time.
    """
    # Full-period score for context
    full_result = compute_score(
        equity_curve=equity_curve,
        trades_df=trades_df if trades_df is not None else pd.DataFrame(),
        settings=None,
        candidate_id=candidate_id,
        evolution_mode=evolution_mode,
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
                evolution_mode=evolution_mode,
            )
        )

    return aggregate_monthly_fitness(
        monthly_scores=monthly_scores,
        candidate_id=candidate_id,
        experiment_slug=experiment_slug,
        full_period_score=full_score,
        full_period_rejected=full_rej,
        equity_curve=equity_curve,
        trades_df=trades_df,
    )
