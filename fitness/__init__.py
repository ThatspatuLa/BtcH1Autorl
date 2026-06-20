"""Stage 6 — Month-by-Month Fitness Engine + Stage 6.5 Deployment Gates.

Walk-forward fitness: score each month independently, then aggregate with
robust statistics that punish both poor months AND variance across months.

Stage 6.5 adds the discovery vs deployment split:
- discovery_fitness: soft penalty on consistency (used by GA breeding)
- deployment_fitness: hard gate, only if every deployment rule passes
  (used for final acceptance — never weakened)
"""
from __future__ import annotations

from .deployment_gates import (
    CONSISTENCY_PENALTY_TABLE,
    DEPLOYMENT_MAX_DD_PCT,
    DEPLOYMENT_MIN_CONSISTENCY,
    DEPLOYMENT_MIN_TOTAL_TRADES,
    DEPLOYMENT_MIN_TRADES_PER_MONTH,
    DeploymentGateResult,
    compute_deployment_gates,
    consistency_multiplier,
)
from .monthly_fitness import (
    MonthlyFitnessResult,
    MonthlyScore,
    aggregate_monthly_fitness,
    compute_monthly_fitness,
)

__all__ = [
    "CONSISTENCY_PENALTY_TABLE",
    "DEPLOYMENT_MAX_DD_PCT",
    "DEPLOYMENT_MIN_CONSISTENCY",
    "DEPLOYMENT_MIN_TOTAL_TRADES",
    "DEPLOYMENT_MIN_TRADES_PER_MONTH",
    "DeploymentGateResult",
    "MonthlyFitnessResult",
    "MonthlyScore",
    "aggregate_monthly_fitness",
    "compute_deployment_gates",
    "compute_monthly_fitness",
    "consistency_multiplier",
]
