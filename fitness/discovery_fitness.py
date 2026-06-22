"""Phase B — Discovery Fitness v2 aggregator.

Discovery Fitness v2 formula (locked, Six's spec):
    discovery_fitness = 0.60·full_period_base_score
                      + 0.20·recovery_score
                      + 0.10·consistency_score
                      + 0.05·stability_score
                      + 0.05·concentration_score

Inputs are 5 normalised scores in [0, 1]. Output is also in [0, 1].

- `full_period_base_score` = Stage 5 compute_score().breakdown.final_score on
  the full 5y equity curve. The law, consumed unchanged.
- `recovery_score` = weighted sum of 4 sub-metrics from recovery_metrics.py.
- `consistency_score` = profitable_months / total_months (the consistency_ratio).
- `stability_score` = 1 - clipped(stddev(monthly_base_scores) / 0.3). Light penalty.
- `concentration_score` = penalty for one lucky month carrying the result
  (top_month_share ≤ 0.30 → 1.0, ≥ 0.70 → 0.0; linear in between).

The hard rejects (worst_month < -0.50, median < 0.10, deployment gates) are
enforced BEFORE this aggregator runs in `aggregate_monthly_fitness`. This
module NEVER sees rejected candidates — it only computes the soft score.
"""
from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "DISCOVERY_WEIGHTS",
    "compute_concentration_score",
    "compute_discovery_fitness",
    "compute_stability_score",
]


# LOCKED weights for the 5 discovery_fitness components (sum must = 1.0)
DISCOVERY_WEIGHTS: dict[str, float] = {
    "full_period_base_score": 0.60,
    "recovery_score": 0.20,
    "consistency_score": 0.10,
    "stability_score": 0.05,
    "concentration_score": 0.05,
}

assert abs(sum(DISCOVERY_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"DISCOVERY_WEIGHTS must sum to 1.0; got {sum(DISCOVERY_WEIGHTS.values())}"
)


# ============================================================
# Helpers (B2)
# ============================================================

def compute_stability_score(monthly_base_scores: Sequence[float]) -> float:
    """Light stddev/CoV-based stability score.

    stability_score = 1 - clipped(stddev(scores) / 0.3)
        stddev = 0      → 1.0 (perfectly stable)
        stddev >= 0.3   → 0.0 (very volatile)

    Empty list → 1.0 (neutral; avoid biasing missing data).

    Note: this is a LIGHT penalty because the Sharpe component in Stage 5
    already rewards smoothness. We don't want double-counting.
    """
    if not monthly_base_scores:
        return 1.0
    if len(monthly_base_scores) < 2:
        return 1.0  # one value has no variance

    n = len(monthly_base_scores)
    mean = sum(monthly_base_scores) / n
    var = sum((s - mean) ** 2 for s in monthly_base_scores) / n  # population variance
    stddev = var ** 0.5

    raw = 1.0 - stddev / 0.3
    return float(max(0.0, min(1.0, raw)))


def compute_concentration_score(monthly_profits: Sequence[float]) -> float:
    """Penalty for one lucky month carrying the result.

    share = top_month_profit / sum_positive_profit
        share <= 0.30 → 1.0 (well-distributed, no penalty)
        share >= 0.70 → 0.0 (heavily concentrated, full penalty)
        linear in between

    Edge cases:
        - No positive profits → 1.0 (no concentration possible)
        - Empty list → 1.0
    """
    if not monthly_profits:
        return 1.0

    positives = [p for p in monthly_profits if p > 0]
    if not positives:
        return 1.0

    total_positive = sum(positives)
    if total_positive <= 0:
        return 1.0

    top_month = max(positives)
    share = top_month / total_positive

    if share <= 0.30:
        return 1.0
    if share >= 0.70:
        return 0.0
    # Linear interpolation: share=0.3 → 1.0; share=0.7 → 0.0
    raw = 1.0 - (share - 0.30) / 0.40
    return float(max(0.0, min(1.0, raw)))


# ============================================================
# Main aggregator (B1)
# ============================================================

def compute_discovery_fitness(
    full_period_base_score: float,
    recovery_score: float,
    consistency_score: float,
    stability_score: float,
    concentration_score: float,
) -> float:
    """Discovery Fitness v2 — weighted sum of 5 component scores.

    Returns:
        float in [0, 1].
    """
    components = {
        "full_period_base_score": full_period_base_score,
        "recovery_score": recovery_score,
        "consistency_score": consistency_score,
        "stability_score": stability_score,
        "concentration_score": concentration_score,
    }
    raw = sum(DISCOVERY_WEIGHTS[k] * float(v) for k, v in components.items())
    return float(max(0.0, min(1.0, raw)))
