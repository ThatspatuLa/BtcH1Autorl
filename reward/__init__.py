"""BTC H1 AutoRL — reward subpackage.

Stage 5 deliverable: Locked reward / scoring engine.

LOCKED weights (do not evolve in v1):
- 0.55 profit
- 0.15 drawdown quality
- 0.10 Sharpe ratio
- 0.10 profit factor
- 0.10 trades per month

Hard rejection (REJECT, not score):
- net_profit <= 0
- drawdown > 35%
- trades_per_month < 5
- invalid initial_deposit or final_equity (NaN/inf)
- too_few_trades (configurable, default 30)

Drawdown penalty tiers:
- 25-30% DD: score × 0.85
- 30-35% DD: score × 0.50
- >35% DD: reject (caught by hard rule)

Phase 1: synthetic-data unit tests (no Stage 3 dependency).
Phase 2 (gating Stage 10): real integration with Stage 3 backtest output.
"""
from reward.scoring import (
    DD_PENALTY_TIERS,
    MAX_DD_PCT,
    MIN_TOTAL_TRADES,
    MIN_TPM,
    # Constants
    REWARD_WEIGHTS,
    ComponentScore,
    RejectedResult,
    ScoreBreakdown,
    # Score result types
    ScoreResult,
    compute_dd_quality_normalizer,
    compute_pf_normalizer,
    compute_profit_normalizer,
    # Public API
    compute_score,
    compute_sharpe_normalizer,
    compute_tpm_normalizer,
)

__all__ = [
    "DD_PENALTY_TIERS",
    "MAX_DD_PCT",
    "MIN_TOTAL_TRADES",
    "MIN_TPM",
    "REWARD_WEIGHTS",
    "ComponentScore",
    "RejectedResult",
    "ScoreBreakdown",
    "ScoreResult",
    "compute_dd_quality_normalizer",
    "compute_pf_normalizer",
    "compute_profit_normalizer",
    "compute_score",
    "compute_sharpe_normalizer",
    "compute_tpm_normalizer",
]
