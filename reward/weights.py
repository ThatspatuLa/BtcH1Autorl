"""Locked reward weights — DO NOT evolve in v1.

These constants are the canonical scoring weights per Kanban Stage 5. They are imported
into scoring.py and into tests. Mutating them is a contract break — would invalidate
all previous backtest comparisons.
"""
from __future__ import annotations

# LOCKED for v1 — sum MUST equal 1.00
WEIGHT_PROFIT: float = 0.55
WEIGHT_DD_QUALITY: float = 0.15
WEIGHT_SHARPE: float = 0.10
WEIGHT_PROFIT_FACTOR: float = 0.10
WEIGHT_TPM: float = 0.10

REWARD_WEIGHTS: dict[str, float] = {
    "profit": WEIGHT_PROFIT,
    "dd_quality": WEIGHT_DD_QUALITY,
    "sharpe": WEIGHT_SHARPE,
    "profit_factor": WEIGHT_PROFIT_FACTOR,
    "tpm": WEIGHT_TPM,
}

# Sanity check on import — fail loudly if anyone tries to mutate
assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"REWARD_WEIGHTS must sum to 1.0; got {sum(REWARD_WEIGHTS.values())}"
)

# ============================================================
# Hard rejection thresholds (LOCKED)
# ============================================================

MAX_DD_PCT: float = 0.35          # DD > 35% → reject
MIN_TPM: float = 5.0              # TPM < 5 → reject
MIN_TOTAL_TRADES: int = 30        # too few trades → reject (configurable)

# Drawdown penalty tiers (LOCKED)
# DD in [0.25, 0.30) → score × 0.85
# DD in [0.30, 0.35) → score × 0.50
# DD >= 0.35 → reject (above)
DD_PENALTY_TIERS: list[tuple[float, float]] = [
    (0.25, 0.85),
    (0.30, 0.50),
]
