"""CandidateEvaluator — runs one candidate through backtest + fitness.

The single function Stage 10 calls per candidate. Wraps:
1. Stage 9 backtest_with_fixed_tp
2. Stage 5 reward.compute_score
3. Stage 6 fitness.compute_monthly_fitness
4. Tracks timing, rejection reasons
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from dca_engine.tp_baseline import backtest_with_fixed_tp, extract_dca_params_from_genome
from fitness.monthly_fitness import MonthlyFitnessResult, compute_monthly_fitness
from genome.schema import CandidateGenome
from reward.scoring import RejectedResult, ScoreResult, compute_score


@dataclass
class EvaluationResult:
    """Result of evaluating one candidate. The shape the harness consumes."""
    candidate_id: str
    genome_id: str
    # Discovery fitness v2 = 0.60·full_period_base_score + 0.20·recovery_score
    #                      + 0.10·consistency_score + 0.05·stability_score
    #                      + 0.05·concentration_score. Used for breeding.
    # Range 0..1. NOT deployment-approved.
    discovery_fitness: float
    # Deployment fitness: == discovery_fitness if every deployment gate passes,
    # else 0. Used for reporting and final acceptance.
    deployment_fitness: float
    deployment_pass: bool
    failed_deployment_gates: list[str]
    closest_to_passing_score: float
    consistency_ratio: float
    consistency_multiplier: float
    # --- Phase D: v2 component scores (surfaced for leaderboards + reports) ---
    full_period_base_score: float
    recovery_score: float
    stability_score: float
    concentration_score: float
    recovery_breakdown: dict[str, float]
    # --- End Phase D ---
    # Whether this candidate is hard-rejected (safety). Soft-failing candidates
    # (consistency<0.50, low discovery_fitness) have rejected=False and a
    # non-zero discovery_fitness.
    rejected: bool
    reject_reason: str | None
    elapsed_seconds: float
    monthly_fitness: MonthlyFitnessResult
    score_breakdown: dict[str, Any] | None
    raw_metrics: dict[str, Any]
    n_cycles_closed: int
    final_equity: float
    max_dd_pct: float
    rejection_source: str | None = None
    error: str | None = None

    @property
    def fitness(self) -> float:
        """Back-compat alias — what the GA used to sort by. Now == discovery_fitness."""
        return self.discovery_fitness

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "genome_id": self.genome_id,
            "discovery_fitness": self.discovery_fitness,
            "deployment_fitness": self.deployment_fitness,
            "deployment_pass": self.deployment_pass,
            "failed_deployment_gates": self.failed_deployment_gates,
            "closest_to_passing_score": self.closest_to_passing_score,
            "consistency_ratio": self.consistency_ratio,
            "consistency_multiplier": self.consistency_multiplier,
            # Phase D: v2 component scores
            "full_period_base_score": self.full_period_base_score,
            "recovery_score": self.recovery_score,
            "stability_score": self.stability_score,
            "concentration_score": self.concentration_score,
            "recovery_breakdown": self.recovery_breakdown,
            # Back-compat
            "fitness": self.fitness,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "rejection_source": self.rejection_source,
            "elapsed_seconds": self.elapsed_seconds,
            "n_cycles_closed": self.n_cycles_closed,
            "final_equity": self.final_equity,
            "max_dd_pct": self.max_dd_pct,
            "raw_metrics": self.raw_metrics,
            "score_breakdown": self.score_breakdown,
            "error": self.error,
        }


class CandidateEvaluator:
    """Evaluates a single candidate. Stateless, reusable, thread-safe."""

    def __init__(self, df: pd.DataFrame, experiment_slug: str = "exp_default"):
        self.df = df
        self.experiment_slug = experiment_slug

    def evaluate(self, genome: CandidateGenome, candidate_id: str) -> EvaluationResult:
        """Run backtest + reward + monthly fitness for one genome.

        Returns EvaluationResult. NEVER raises — errors are caught and
        stored in the result so the harness can keep running.
        """
        t0 = time.time()
        try:
            return _evaluate_one(self.df, genome, candidate_id, self.experiment_slug, t0)
        except Exception as e:
            return EvaluationResult(
                candidate_id=candidate_id,
                genome_id=genome.genome_id,
                discovery_fitness=0.0,
                deployment_fitness=0.0,
                deployment_pass=False,
                failed_deployment_gates=["evaluation_error"],
                closest_to_passing_score=0.0,
                consistency_ratio=0.0,
                consistency_multiplier=0.0,
                full_period_base_score=0.0,
                recovery_score=0.0,
                stability_score=0.0,
                concentration_score=0.0,
                recovery_breakdown={},
                rejected=True,
                reject_reason="evaluation_error",
                rejection_source="evaluator",
                elapsed_seconds=time.time() - t0,
                monthly_fitness=_empty_monthly_fitness(candidate_id, self.experiment_slug),
                score_breakdown=None,
                raw_metrics={},
                n_cycles_closed=0,
                final_equity=0.0,
                max_dd_pct=1.0,
                error=str(e),
            )


def _evaluate_one(
    df: pd.DataFrame,
    genome: CandidateGenome,
    candidate_id: str,
    experiment_slug: str,
    t0: float,
) -> EvaluationResult:
    """Module-level evaluation function — picklable for ProcessPoolExecutor.

    All the actual evaluation logic lives here so it can be called from
    worker processes.
    """
    # Extract DCA params from genome
    dca_params = extract_dca_params_from_genome(genome)

    # Stage 9: run backtest with fixed TP
    bt = backtest_with_fixed_tp(
        df=df,
        candidate_id=candidate_id,
        genome_id=genome.genome_id,
        experiment_id=experiment_slug,
        tp_genome=genome.tp_genome,
        grid_pct=dca_params["grid_pct"],
        max_layers=dca_params["max_layers"],
        confirmation_indicators=dca_params.get("confirmation_indicators", []),
        indicator_params=dca_params.get("indicator_params", {}),
        cooldown_candles=dca_params.get("cooldown_candles", 0),
        grid_method=dca_params.get("grid_method", "fixed_pct"),
        grid_params=dca_params.get("grid_params"),
    )

    # Stage 6 + 6.5: monthly fitness + deployment gates
    fitness = compute_monthly_fitness(
        equity_curve=bt.equity_curve,
        trades_df=bt.trades_df,
        candidate_id=candidate_id,
        experiment_slug=experiment_slug,
        evolution_mode=True,
    )

    if fitness.rejected:
        return EvaluationResult(
            candidate_id=candidate_id,
            genome_id=genome.genome_id,
            discovery_fitness=fitness.discovery_fitness,
            deployment_fitness=fitness.deployment_fitness,
            deployment_pass=fitness.deployment_pass,
            failed_deployment_gates=fitness.failed_deployment_gates,
            closest_to_passing_score=fitness.closest_to_passing_score,
            consistency_ratio=fitness.consistency_ratio,
            consistency_multiplier=fitness.consistency_multiplier,
            full_period_base_score=fitness.full_period_base_score,
            recovery_score=fitness.recovery_score,
            stability_score=fitness.stability_score,
            concentration_score=fitness.concentration_score,
            recovery_breakdown=dict(fitness.recovery_breakdown),
            rejected=True,
            reject_reason=fitness.reject_reason or "unknown",
            rejection_source="monthly_fitness",
            elapsed_seconds=time.time() - t0,
            monthly_fitness=fitness,
            score_breakdown=None,
            raw_metrics={},
            n_cycles_closed=bt.n_cycles_closed,
            final_equity=bt.final_equity,
            max_dd_pct=_extract_max_dd(bt, fitness),
        )

    score_result = compute_score(
        equity_curve=bt.equity_curve,
        trades_df=bt.trades_df,
        settings=None,
        candidate_id=candidate_id,
        evolution_mode=True,
    )
    if isinstance(score_result, RejectedResult):
        return EvaluationResult(
            candidate_id=candidate_id,
            genome_id=genome.genome_id,
            discovery_fitness=fitness.discovery_fitness,
            deployment_fitness=fitness.deployment_fitness,
            deployment_pass=fitness.deployment_pass,
            failed_deployment_gates=fitness.failed_deployment_gates,
            closest_to_passing_score=fitness.closest_to_passing_score,
            consistency_ratio=fitness.consistency_ratio,
            consistency_multiplier=fitness.consistency_multiplier,
            full_period_base_score=fitness.full_period_base_score,
            recovery_score=fitness.recovery_score,
            stability_score=fitness.stability_score,
            concentration_score=fitness.concentration_score,
            recovery_breakdown=dict(fitness.recovery_breakdown),
            rejected=True,
            reject_reason=score_result.reason,
            rejection_source="score",
            elapsed_seconds=time.time() - t0,
            monthly_fitness=fitness,
            score_breakdown=None,
            raw_metrics=dict(score_result.raw_metrics),
            n_cycles_closed=bt.n_cycles_closed,
            final_equity=bt.final_equity,
            max_dd_pct=_extract_max_dd(bt, fitness),
        )

    return EvaluationResult(
        candidate_id=candidate_id,
        genome_id=genome.genome_id,
        discovery_fitness=fitness.discovery_fitness,
        deployment_fitness=fitness.deployment_fitness,
        deployment_pass=fitness.deployment_pass,
        failed_deployment_gates=fitness.failed_deployment_gates,
        closest_to_passing_score=fitness.closest_to_passing_score,
        consistency_ratio=fitness.consistency_ratio,
        consistency_multiplier=fitness.consistency_multiplier,
        full_period_base_score=fitness.full_period_base_score,
        recovery_score=fitness.recovery_score,
        stability_score=fitness.stability_score,
        concentration_score=fitness.concentration_score,
        recovery_breakdown=dict(fitness.recovery_breakdown),
        rejected=False,
        reject_reason=None,
        rejection_source=None,
        elapsed_seconds=time.time() - t0,
        monthly_fitness=fitness,
        score_breakdown=score_result.breakdown.to_dict() if isinstance(score_result, ScoreResult) else None,
        raw_metrics=score_result.raw_metrics if isinstance(score_result, ScoreResult) else {},
        n_cycles_closed=bt.n_cycles_closed,
        final_equity=bt.final_equity,
        max_dd_pct=_extract_max_dd(bt, fitness),
    )


def _extract_max_dd(bt: Any, fitness: MonthlyFitnessResult) -> float:
    """Best-effort max DD extraction from a backtest result."""
    try:
        if hasattr(bt, "peak_equity") and bt.peak_equity > 0 and hasattr(bt, "trough_equity"):
            return (bt.peak_equity - bt.trough_equity) / bt.peak_equity
    except Exception:
        pass
    return 0.0


def _empty_monthly_fitness(candidate_id: str, slug: str) -> MonthlyFitnessResult:
    """Empty MonthlyFitnessResult for error cases."""
    from fitness.monthly_fitness import MonthlyFitnessResult
    return MonthlyFitnessResult(
        candidate_id=candidate_id,
        experiment_slug=slug,
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
        failed_deployment_gates=["evaluation_error"],
        closest_to_passing_score=0.0,
        final_fitness=0.0,
        rejected=True,
        reject_reason="no_data",
        full_period_score=None,
        full_period_rejected=True,
    )
