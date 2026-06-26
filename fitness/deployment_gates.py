"""Stage 6.5 — Discovery vs Deployment Fitness.

Two-stage fitness for the evolution loop.

Discovery fitness (used by the GA for breeding):
    base_aggregate_fitness × consistency_multiplier
    Where consistency_multiplier is:
        >= 0.50 → 1.00
        [0.40, 0.50) → 0.85
        [0.30, 0.40) → 0.65
        [0.20, 0.30) → 0.40
        < 0.20  → 0.15

Deployment fitness (used for final acceptance only):
    discovery_fitness, BUT only if every deployment gate passes.
    Gates (hard rejects that ALWAYS block deployment, even if discovery high):
    - consistency_ratio < 0.50           (locked v1: must profit in 50%+ of months)
    - invalid equity curve
    - margin failure
    - DCA completion failure
    - max drawdown > 35%

The key design property: consistency below 0.50 is a SOFT penalty during
evolution (so the GA can still see and breed "almost-passing" candidates)
but a HARD gate at deployment time. Final selection never weakens.

closest_to_passing_score: a 0..1 score that says "how close was this
candidate to deployment-passing?". Used for diagnostic reports so we
can see which near-miss genomes deserve more mutation. 0.0 = trivially
failing, 1.0 = deployment passing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Discovery-phase consistency penalty curve (LOCKED, do not evolve)
CONSISTENCY_PENALTY_TABLE: list[tuple[float, float]] = [
    # (consistency_threshold_inclusive_lower_bound, multiplier)
    (0.50, 1.00),
    (0.40, 0.85),
    (0.30, 0.65),
    (0.20, 0.40),
    (-1.00, 0.15),  # anything below 0.20 → 0.15
]

# Deployment-time hard gates (LOCKED, do not weaken)
DEPLOYMENT_MIN_CONSISTENCY: float = 0.50
DEPLOYMENT_MAX_DD_PCT: float = 0.35
DEPLOYMENT_MIN_TRADES_PER_MONTH: float = 5.0
DEPLOYMENT_MIN_TOTAL_TRADES: int = 300


@dataclass
class DeploymentGateResult:
    """The full deployment fitness picture for a single candidate."""
    # Raw inputs
    consistency_ratio: float
    max_drawdown_pct: float
    trades_per_month: float
    total_trades: int
    has_invalid_equity: bool
    has_margin_failure: bool
    has_dca_completion_failure: bool

    # Consistency penalty curve
    consistency_multiplier: float

    # Two fitnesses
    base_aggregate_fitness: float      # the walk-forward aggregate (0..1, before consistency penalty)
    discovery_fitness: float           # base × multiplier (used by GA breeding)
    deployment_fitness: float          # == discovery_fitness if all gates pass, else 0.0

    # Deployment gate outcomes
    deployment_pass: bool
    failed_deployment_gates: list[str]

    # Diagnostic for the report
    closest_to_passing_score: float    # 0..1, how close to deployment-passing

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency_ratio": self.consistency_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trades_per_month": self.trades_per_month,
            "total_trades": self.total_trades,
            "has_invalid_equity": self.has_invalid_equity,
            "has_margin_failure": self.has_margin_failure,
            "has_dca_completion_failure": self.has_dca_completion_failure,
            "consistency_multiplier": self.consistency_multiplier,
            "base_aggregate_fitness": self.base_aggregate_fitness,
            "discovery_fitness": self.discovery_fitness,
            "deployment_fitness": self.deployment_fitness,
            "deployment_pass": self.deployment_pass,
            "failed_deployment_gates": self.failed_deployment_gates,
            "closest_to_passing_score": self.closest_to_passing_score,
        }


# ============================================================
# Pure functions
# ============================================================

def consistency_multiplier(consistency_ratio: float) -> float:
    """Apply the discovery-phase consistency penalty curve.

    >= 0.50 → 1.00
    [0.40, 0.50) → 0.85
    [0.30, 0.40) → 0.65
    [0.20, 0.30) → 0.40
    < 0.20  → 0.15
    """
    # The table is sorted high-to-low, return the first match
    for threshold, mult in CONSISTENCY_PENALTY_TABLE:
        if consistency_ratio >= threshold:
            return mult
    return CONSISTENCY_PENALTY_TABLE[-1][1]


def compute_deployment_gates(
    consistency_ratio: float,
    max_drawdown_pct: float,
    trades_per_month: float,
    total_trades: int,
    has_invalid_equity: bool,
    has_margin_failure: bool,
    has_dca_completion_failure: bool,
    base_aggregate_fitness: float,
) -> DeploymentGateResult:
    """Compute the full two-stage fitness picture for one candidate.

    Inputs that are unknown can be passed as -1.0 / -1 / False; gates that
    require real data will treat negatives as "not yet measured" and skip
    (so a candidate with no trades isn't double-penalised by both a TPM
    fail AND a no-trades pass).
    """
    mult = consistency_multiplier(consistency_ratio)
    discovery = base_aggregate_fitness * mult

    failed: list[str] = []

    # Consistency gate (locked v1)
    if consistency_ratio < DEPLOYMENT_MIN_CONSISTENCY:
        failed.append(f"consistency<{DEPLOYMENT_MIN_CONSISTENCY:.2f}")

    # Safety gates — these are ALWAYS hard rejects, even during evolution
    if has_invalid_equity:
        failed.append("invalid_equity")
    if has_margin_failure:
        failed.append("margin_failure")
    if has_dca_completion_failure:
        failed.append("dca_completion_failure")

    # Drawdown gate (locked v1)
    if max_drawdown_pct < 0.0:
        # not measured yet — skip, don't penalise
        pass
    elif max_drawdown_pct > DEPLOYMENT_MAX_DD_PCT:
        failed.append(f"drawdown>{DEPLOYMENT_MAX_DD_PCT:.0%}")

    # Volume gates — only if the candidate actually has trades
    if total_trades > 0:
        if trades_per_month < DEPLOYMENT_MIN_TRADES_PER_MONTH:
            failed.append(f"tpm<{DEPLOYMENT_MIN_TRADES_PER_MONTH:.0f}")
        if total_trades < DEPLOYMENT_MIN_TOTAL_TRADES:
            failed.append(f"total_trades<{DEPLOYMENT_MIN_TOTAL_TRADES}")

    deployment_pass = len(failed) == 0
    deployment_fitness = discovery if deployment_pass else 0.0

    # closest_to_passing_score: 0..1, monotonic in how many gates pass
    # and how close consistency is to the floor
    # Two parts:
    #   1. consistency headroom: 1.0 if consistency >= 0.50, else consistency/0.50
    #   2. safety headroom: fraction of safety gates that pass
    consistency_headroom = (
        1.0 if consistency_ratio >= DEPLOYMENT_MIN_CONSISTENCY
        else consistency_ratio / DEPLOYMENT_MIN_CONSISTENCY
    )
    safety_gates = ["invalid_equity", "margin_failure", "dca_completion_failure"]
    n_safety_passed = sum(
        1 for g in safety_gates
        if g not in [f.split(">")[0].split("<")[0] for f in failed]
    )
    safety_headroom = n_safety_passed / len(safety_gates)

    # DD headroom (only if measured)
    if max_drawdown_pct < 0.0:
        dd_headroom = 1.0
    elif max_drawdown_pct > DEPLOYMENT_MAX_DD_PCT:
        dd_headroom = max(0.0, 1.0 - (max_drawdown_pct - DEPLOYMENT_MAX_DD_PCT))
    else:
        dd_headroom = 1.0

    # TPM / total_trades headroom (only if measured)
    if total_trades <= 0:
        volume_headroom = 0.0
    else:
        tpm_h = min(1.0, trades_per_month / DEPLOYMENT_MIN_TRADES_PER_MONTH) if trades_per_month > 0 else 0.0
        tt_h = min(1.0, total_trades / DEPLOYMENT_MIN_TOTAL_TRADES)
        volume_headroom = (tpm_h + tt_h) / 2.0

    closest = (
        0.4 * consistency_headroom
        + 0.2 * safety_headroom
        + 0.2 * dd_headroom
        + 0.2 * volume_headroom
    )

    return DeploymentGateResult(
        consistency_ratio=consistency_ratio,
        max_drawdown_pct=max_drawdown_pct,
        trades_per_month=trades_per_month,
        total_trades=total_trades,
        has_invalid_equity=has_invalid_equity,
        has_margin_failure=has_margin_failure,
        has_dca_completion_failure=has_dca_completion_failure,
        consistency_multiplier=mult,
        base_aggregate_fitness=base_aggregate_fitness,
        discovery_fitness=discovery,
        deployment_fitness=deployment_fitness,
        deployment_pass=deployment_pass,
        failed_deployment_gates=failed,
        closest_to_passing_score=closest,
    )
