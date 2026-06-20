"""Stage 6 — Month-by-Month Fitness Engine.

Walk-forward fitness: score each month independently, then aggregate with
robust statistics that punish both poor months AND variance across months.

Why per-month? Because a strategy that returns 80% over 5 years but loses
30% in months 18-22 is NOT robust. Evolution needs to see per-month variance
or it will overfit to a few good months. The locked v1 reward engine scores
a single equity curve; Stage 6 wraps it in a per-month loop.
"""
from __future__ import annotations

from .monthly_fitness import (
    MonthlyFitnessResult,
    MonthlyScore,
    aggregate_monthly_fitness,
    compute_monthly_fitness,
)

__all__ = [
    "MonthlyFitnessResult",
    "MonthlyScore",
    "aggregate_monthly_fitness",
    "compute_monthly_fitness",
]
