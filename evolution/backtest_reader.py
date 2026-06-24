"""Backtest reader — extracts patterns from recent evaluation results.

Reads the actual EvaluationResult objects from recent gens and produces
actionable insights the SmartMutator can use.

Outputs (per island, per gen):
- Regime performance (bull/bear/chop monthly scores)
- Drawdown hotspots (worst 5 months by DD%)
- Recovery speed (mean trades to recover from DD)
- Trade frequency (avg trades/month)
- Monthly consistency (stddev of monthly scores)
- Profit distribution (top vs bottom months)

Used by SmartMutator to:
- Detect regime shifts (e.g., "I3 is doing worse in chop — boost vol filter")
- Identify DD hotspots (e.g., "worst DD in 2022 bear — tighten max_layers")
- Adapt to recovery speed (e.g., "recovery slow — try shallower DCA")

This is Component 3 of the smart mutation system (Six's directive 2026-06-25).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from evolution.evaluator import EvaluationResult
from fitness.monthly_fitness import MonthlyScore


# ============================================================
# Regime classification (simple heuristic)
# ============================================================

def classify_regime(net_profit_pct: float) -> str:
    """Classify a month as bull / bear / chop based on net profit %.

    Heuristic:
    - bull: net_profit_pct > 5% (strong uptrend)
    - bear: net_profit_pct < -2% (clear downtrend)
    - chop: otherwise (ranging, low-vol)

    Can be tuned based on observed BTC H1 patterns.
    """
    if net_profit_pct > 5.0:
        return "bull"
    elif net_profit_pct < -2.0:
        return "bear"
    return "chop"


@dataclass
class RegimePerformance:
    """How this island performs across market regimes."""
    bull_avg_score: float = 0.0
    bear_avg_score: float = 0.0
    chop_avg_score: float = 0.0
    bull_n_months: int = 0
    bear_n_months: int = 0
    chop_n_months: int = 0
    # Best/worst regime for this island
    best_regime: str = "unknown"
    worst_regime: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DrawdownHotspots:
    """Where this island experiences worst drawdowns."""
    worst_5_months: list[dict[str, Any]] = field(default_factory=list)
    avg_worst_dd_pct: float = 0.0
    max_dd_pct: float = 0.0
    dd_concentration: float = 0.0
    """Fraction of total DD events in worst 3 months (0-1). High = a few months
    cause most of the damage."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryMetrics:
    """How quickly this island recovers from drawdowns."""
    avg_recovery_speed: float = 0.0  # 0-1 (higher = faster)
    avg_equity_high_reclaim_rate: float = 0.0
    avg_post_loss_bounce_rate: float = 0.0
    avg_cycle_recovery_health: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeFrequency:
    """How often this island trades."""
    avg_trades_per_month: float = 0.0
    median_trades_per_month: float = 0.0
    std_trades_per_month: float = 0.0
    total_trades_recent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IslandBacktestSummary:
    """All backtest patterns for one island, one gen."""
    island_id: int
    gen_index: int
    regime: RegimePerformance = field(default_factory=RegimePerformance)
    dd: DrawdownHotspots = field(default_factory=DrawdownHotspots)
    recovery: RecoveryMetrics = field(default_factory=RecoveryMetrics)
    frequency: TradeFrequency = field(default_factory=TradeFrequency)
    # Profit distribution (top vs bottom months)
    top_month_score: float = 0.0
    bottom_month_score: float = 0.0
    monthly_consistency: float = 0.0  # 0-1
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": self.island_id,
            "gen_index": self.gen_index,
            "regime": self.regime.to_dict(),
            "dd": self.dd.to_dict(),
            "recovery": self.recovery.to_dict(),
            "frequency": self.frequency.to_dict(),
            "top_month_score": self.top_month_score,
            "bottom_month_score": self.bottom_month_score,
            "monthly_consistency": self.monthly_consistency,
            "sample_size": self.sample_size,
        }


# ============================================================
# Helper — asdict for dataclasses that contain other dataclasses
# ============================================================

def _asdict(obj):
    """Recursive asdict that handles nested dataclasses without third-party libs."""
    from dataclasses import fields, is_dataclass
    if not is_dataclass(obj):
        return obj
    out = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, list):
            out[f.name] = [_asdict(x) for x in v]
        elif is_dataclass(v):
            out[f.name] = _asdict(v)
        else:
            out[f.name] = v
    return out


# Patch asdict to use recursive version
asdict = _asdict


# ============================================================
# Main reader
# ============================================================

class BacktestReader:
    """Reads evaluation results and produces IslandBacktestSummary."""

    # Regime classification thresholds (can be tuned)
    BULL_THRESHOLD = 5.0  # net_profit_pct
    BEAR_THRESHOLD = -2.0

    def __init__(self, bull_threshold: float = BULL_THRESHOLD, bear_threshold: float = BEAR_THRESHOLD):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def analyze(
        self,
        island_id: int,
        gen_index: int,
        results: list[EvaluationResult],
        top_n: int = 20,
    ) -> IslandBacktestSummary:
        """Analyze the top-N eval results for an island and produce summary.

        Args:
            island_id: which island
            gen_index: current gen
            results: all EvaluationResults for this gen (will filter + sort)
            top_n: how many top results to analyze (default 20)

        Returns:
            IslandBacktestSummary with regime / DD / recovery / frequency data
        """
        # Pick top-N by discovery_fitness
        sorted_results = sorted(results, key=lambda r: r.discovery_fitness, reverse=True)[:top_n]

        summary = IslandBacktestSummary(
            island_id=island_id,
            gen_index=gen_index,
            sample_size=len(sorted_results),
        )

        if not sorted_results:
            return summary

        # Recovery (use breakdown from top result) — do this first since
        # it doesn't need monthly_scores (it reads recovery_breakdown directly).
        summary.recovery = self._compute_recovery_metrics(sorted_results)

        # Aggregate monthly scores across top-N
        all_monthly: list[MonthlyScore] = []
        for r in sorted_results:
            for ms in r.monthly_fitness.monthly_scores:
                all_monthly.append(ms)

        if not all_monthly:
            return summary

        # Regime performance
        summary.regime = self._compute_regime_performance(all_monthly)

        # DD hotspots
        summary.dd = self._compute_dd_hotspots(all_monthly)

        # Trade frequency
        summary.frequency = self._compute_trade_frequency(all_monthly)

        # Monthly consistency + distribution
        non_rejected = [ms.monthly_score for ms in all_monthly if not ms.rejected]
        if non_rejected:
            summary.top_month_score = max(non_rejected)
            summary.bottom_month_score = min(non_rejected)
            mean_score = sum(non_rejected) / len(non_rejected)
            var = sum((s - mean_score) ** 2 for s in non_rejected) / max(len(non_rejected) - 1, 1)
            std = math.sqrt(var)
            # Coefficient of variation → consistency (low CoV = high consistency)
            summary.monthly_consistency = max(0.0, 1.0 - (std / max(mean_score, 1e-6)))

        return summary

    def _compute_regime_performance(self, monthly: list[MonthlyScore]) -> RegimePerformance:
        rp = RegimePerformance()
        regime_scores: dict[str, list[float]] = defaultdict(list)
        for ms in monthly:
            if ms.rejected:
                continue
            regime = classify_regime(ms.net_profit_pct)
            regime_scores[regime].append(ms.monthly_score)

        if regime_scores.get("bull"):
            rp.bull_avg_score = sum(regime_scores["bull"]) / len(regime_scores["bull"])
            rp.bull_n_months = len(regime_scores["bull"])
        if regime_scores.get("bear"):
            rp.bear_avg_score = sum(regime_scores["bear"]) / len(regime_scores["bear"])
            rp.bear_n_months = len(regime_scores["bear"])
        if regime_scores.get("chop"):
            rp.chop_avg_score = sum(regime_scores["chop"]) / len(regime_scores["chop"])
            rp.chop_n_months = len(regime_scores["chop"])

        # Best/worst regime
        regime_avgs = {
            r: getattr(rp, f"{r}_avg_score")
            for r in ("bull", "bear", "chop")
            if getattr(rp, f"{r}_n_months") > 0
        }
        if regime_avgs:
            rp.best_regime = max(regime_avgs, key=regime_avgs.get)  # type: ignore[arg-type]
            rp.worst_regime = min(regime_avgs, key=regime_avgs.get)  # type: ignore[arg-type]

        return rp

    def _compute_dd_hotspots(self, monthly: list[MonthlyScore]) -> DrawdownHotspots:
        dh = DrawdownHotspots()
        non_rejected = [ms for ms in monthly if not ms.rejected]
        if not non_rejected:
            return dh

        # Sort by DD descending
        by_dd = sorted(non_rejected, key=lambda ms: ms.max_drawdown_pct, reverse=True)
        dh.worst_5_months = [
            {
                "month": ms.month_label,
                "dd_pct": ms.max_drawdown_pct,
                "net_profit_pct": ms.net_profit_pct,
                "score": ms.monthly_score,
            }
            for ms in by_dd[:5]
        ]
        dh.max_dd_pct = by_dd[0].max_drawdown_pct
        # Average of worst 20% of months
        n_worst = max(1, len(non_rejected) // 5)
        dh.avg_worst_dd_pct = sum(
            ms.max_drawdown_pct for ms in by_dd[:n_worst]
        ) / n_worst

        # DD concentration — fraction of total DD in worst 3 months
        total_dd = sum(ms.max_drawdown_pct for ms in non_rejected)
        if total_dd > 0:
            top3_dd = sum(ms.max_drawdown_pct for ms in by_dd[:3])
            dh.dd_concentration = top3_dd / total_dd

        return dh

    def _compute_recovery_metrics(self, results: list[EvaluationResult]) -> RecoveryMetrics:
        rm = RecoveryMetrics()
        if not results:
            return rm

        speeds: list[float] = []
        reclaim_rates: list[float] = []
        bounce_rates: list[float] = []
        cycle_healths: list[float] = []

        for r in results:
            rb = r.monthly_fitness.recovery_breakdown or {}
            if "drawdown_recovery_speed" in rb:
                speeds.append(rb["drawdown_recovery_speed"])
            if "equity_high_reclaim_rate" in rb:
                reclaim_rates.append(rb["equity_high_reclaim_rate"])
            if "post_loss_month_bounce_rate" in rb:
                bounce_rates.append(rb["post_loss_month_bounce_rate"])
            if "cycle_recovery_health" in rb:
                cycle_healths.append(rb["cycle_recovery_health"])

        if speeds:
            rm.avg_recovery_speed = sum(speeds) / len(speeds)
        if reclaim_rates:
            rm.avg_equity_high_reclaim_rate = sum(reclaim_rates) / len(reclaim_rates)
        if bounce_rates:
            rm.avg_post_loss_bounce_rate = sum(bounce_rates) / len(bounce_rates)
        if cycle_healths:
            rm.avg_cycle_recovery_health = sum(cycle_healths) / len(cycle_healths)

        return rm

    def _compute_trade_frequency(self, monthly: list[MonthlyScore]) -> TradeFrequency:
        tf = TradeFrequency()
        non_rejected = [ms for ms in monthly if not ms.rejected]
        if not non_rejected:
            return tf

        tpm_values = [ms.trades_per_month for ms in non_rejected]
        tf.avg_trades_per_month = sum(tpm_values) / len(tpm_values)
        tf.median_trades_per_month = sorted(tpm_values)[len(tpm_values) // 2]
        mean = tf.avg_trades_per_month
        var = sum((v - mean) ** 2 for v in tpm_values) / max(len(tpm_values) - 1, 1)
        tf.std_trades_per_month = math.sqrt(var)
        tf.total_trades_recent = sum(ms.total_trades for ms in non_rejected)

        return tf
