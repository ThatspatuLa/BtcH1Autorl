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
    # Discovery fitness = base_aggregate × consistency_multiplier. Used for
    # breeding selection by the GA. Range 0..1. NOT deployment-approved.
    discovery_fitness: float
    # Deployment fitness: == discovery_fitness if every deployment gate passes,
    # else 0. Used for reporting and final acceptance.
    deployment_fitness: float
    deployment_pass: bool
    failed_deployment_gates: list[str]
    closest_to_passing_score: float
    consistency_ratio: float
    consistency_multiplier: float
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
            "fitness": self.fitness,  # back-compat
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
            return self._evaluate_safe(genome, candidate_id, t0)
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

    def _evaluate_safe(
        self,
        genome: CandidateGenome,
        candidate_id: str,
        t0: float,
    ) -> EvaluationResult:
        # Extract DCA params from genome
        dca_params = extract_dca_params_from_genome(genome)

        # Stage 9: run backtest with fixed TP
        # The tp_pct is read from tp_genome.exit_params["tp_pct"] by the
        # Stage 9 baseline. Our operators (random/mutate/crossover) keep
        # the dca_genome and tp_genome in sync.
        # Confirmation indicators are passed through from the genome.
        bt = backtest_with_fixed_tp(
            df=self.df,
            candidate_id=candidate_id,
            genome_id=genome.genome_id,
            experiment_id=self.experiment_slug,
            tp_genome=genome.tp_genome,
            grid_pct=dca_params["grid_pct"],
            max_layers=dca_params["max_layers"],
            confirmation_indicators=dca_params.get("confirmation_indicators", []),
            indicator_params=dca_params.get("indicator_params", {}),
            cooldown_candles=dca_params.get("cooldown_candles", 0),
        )

        # Stage 6 + 6.5: monthly fitness + deployment gates
        # evolution_mode=True relaxes the TPM hard reject so the GA can
        # see and breed low-TPM candidates. The deployment gates in
        # fitness.deployment_gates still enforce TPM >= 5 at deployment
        # time.
        fitness = compute_monthly_fitness(
            equity_curve=bt.equity_curve,
            trades_df=bt.trades_df,
            candidate_id=candidate_id,
            experiment_slug=self.experiment_slug,
            evolution_mode=True,
        )

        # If monthly fitness hard-rejected, that's the final word
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

        # Otherwise: also run Stage 5 directly to get the score breakdown
        # evolution_mode=True (TPM hard reject relaxed) — see comment above.
        score_result = compute_score(
            equity_curve=bt.equity_curve,
            trades_df=bt.trades_df,
            settings=None,
            candidate_id=candidate_id,
            evolution_mode=True,
        )
        if isinstance(score_result, RejectedResult):
            # Stage 5 rejected something Stage 6 didn't (edge case)
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
