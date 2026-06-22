"""Scoring engine — compute_score(equity_curve, trades_df, settings) → ScoreResult or RejectedResult.

Inputs:
- equity_curve: pd.Series indexed by timestamp, values = account equity in stake currency
- trades_df: pd.DataFrame with columns [open_time, close_time, entry_price, exit_price, qty, pnl, fee, ...]
- settings: Settings object (for fee_pct validation, experiment_slug for context)

Output:
- ScoreResult: full per-component breakdown + final score
- RejectedResult: reason code if hard-reject triggered

Hard reject rules (per Kanban):
- net_profit_pct <= 0
- max_drawdown > 35%
- trades_per_month < 5
- invalid initial_deposit / final_equity (NaN/inf)
- too_few_trades (< 30 default)

Drawdown penalty tiers:
- 25-30% DD: score × 0.85
- 30-35% DD: score × 0.50
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from reward.weights import (
    DD_PENALTY_TIERS,
    MAX_DD_PCT,
    MIN_TOTAL_TRADES,
    MIN_TPM,
    REWARD_WEIGHTS,
)

__all__ = [
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


# ============================================================
# Result types
# ============================================================

@dataclass
class ComponentScore:
    raw_value: float
    normalised: float  # in [0, 1]
    weight: float
    contribution: float  # normalised × weight

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    profit: ComponentScore
    dd_quality: ComponentScore
    sharpe: ComponentScore
    profit_factor: ComponentScore
    tpm: ComponentScore
    dd_penalty_multiplier: float
    base_score: float  # sum of contributions, before DD penalty
    final_score: float  # base_score × dd_penalty_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "profit": self.profit.to_dict(),
            "dd_quality": self.dd_quality.to_dict(),
            "sharpe": self.sharpe.to_dict(),
            "profit_factor": self.profit_factor.to_dict(),
            "tpm": self.tpm.to_dict(),
            "dd_penalty_multiplier": self.dd_penalty_multiplier,
            "base_score": self.base_score,
            "final_score": self.final_score,
        }


@dataclass
class ScoreResult:
    experiment_slug: str
    candidate_id: str | None
    breakdown: ScoreBreakdown
    raw_metrics: dict[str, float]
    total_trades: int
    months_active: float
    exit_reason: str = "scored"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_slug": self.experiment_slug,
            "candidate_id": self.candidate_id,
            "breakdown": self.breakdown.to_dict(),
            "raw_metrics": self.raw_metrics,
            "total_trades": self.total_trades,
            "months_active": self.months_active,
            "exit_reason": self.exit_reason,
        }


@dataclass
class RejectedResult:
    experiment_slug: str
    candidate_id: str | None
    reason: str  # e.g. "net_profit<=0", "drawdown>35%", "tpm<5", "invalid_data", "too_few_trades"
    raw_metrics: dict[str, float] = field(default_factory=dict)
    total_trades: int = 0
    months_active: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Hard rejection checks
# ============================================================

def _check_hard_rejects(
    net_profit_pct: float,
    max_dd_pct: float,
    trades_per_month: float,
    initial_deposit: float,
    final_equity: float,
    total_trades: int,
    evolution_mode: bool = False,
) -> str | None:
    """Returns reason string if hard reject, else None.

    Hard rejects that ALWAYS block (even in evolution):
    - invalid initial_deposit / final_equity (NaN/inf)
    - net_profit_pct <= 0
    - max_drawdown > 35%
    - total_trades < 30

    Softened in evolution_mode (still tracked by the TPM component weight
    in REWARD_WEIGHTS, just not a hard reject):
    - trades_per_month < 5  → only enforced when not evolution_mode

    This is the Stage 5 counterpart of the Stage 6.5 discovery/deployment
    split: during evolution, the GA can see and breed candidates with low
    TPM (they'll get a low tpm_component_score in the final score anyway).
    At deployment time, the deployment_gates check enforces TPM >= 5.
    """
    if not _is_finite(initial_deposit) or not _is_finite(final_equity):
        return "invalid_data"
    if not _is_finite(net_profit_pct) or not _is_finite(max_dd_pct):
        return "invalid_data"
    if net_profit_pct <= 0:
        return "net_profit<=0"
    if max_dd_pct > MAX_DD_PCT:
        return "drawdown>35%"
    # TPM hard reject — relaxed in evolution mode
    if not evolution_mode and trades_per_month < MIN_TPM:
        return "tpm<5"
    if total_trades < MIN_TOTAL_TRADES:
        return "too_few_trades"
    return None


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


# ============================================================
# Raw metric extraction
# ============================================================

def _compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, float, float, bool]:
    """Return (max_drawdown_pct, recovery_time_candles, dd_event_duration_candles, recovered).

    max_drawdown_pct: peak-to-trough decline as positive fraction.
    recovery_time_candles: candles from peak to recovery (or len-1 if never recovered).
    dd_event_duration_candles: length of the drawdown event itself (peak → trough → recovery,
        or peak → end-of-curve if never recovered). NOT the entire curve length — that was the
        Stage 5 measurement bug fixed in Phase A0.
    recovered: True if the peak was reclaimed by end-of-curve.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0, 0.0, 0.0, True
    running_max = equity_curve.cummax()
    drawdown = (running_max - equity_curve) / running_max
    max_dd = float(drawdown.max())
    if max_dd <= 0:
        return 0.0, 0.0, 0.0, True
    # Position-based (not index-based) so we work with both DatetimeIndex and RangeIndex
    trough_pos = int(drawdown.values.argmax())
    peak_pos = int(running_max.iloc[:trough_pos + 1].values.argmax())
    recovery_target = float(equity_curve.iloc[peak_pos])
    after = equity_curve.iloc[trough_pos:]
    if (after >= recovery_target).any():
        recovery_pos = int((after >= recovery_target).values.argmax())
        recovery_time = float(recovery_pos)
        # Event duration = from peak → past trough → recovery
        dd_event_duration = float(recovery_pos + (trough_pos - peak_pos))
        recovered = True
    else:
        # Never recovered: recovery_time = candles remaining in curve (semantic kept for back-compat)
        # dd_event_duration = from peak → end of curve
        recovery_time = float(len(equity_curve) - 1)
        dd_event_duration = float(len(equity_curve) - 1 - peak_pos)
        recovered = False
    return max_dd, recovery_time, dd_event_duration, recovered


def _compute_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
) -> dict[str, float]:
    """Extract raw metrics from equity curve + trades DataFrame."""
    if equity_curve.empty:
        return {
            "initial_deposit": 0.0,
            "final_equity": 0.0,
            "net_profit_pct": 0.0,
            "max_drawdown_pct": 1.0,  # worst-case default
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "trades_per_month": 0.0,
            "total_trades": 0,
            "months_active": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "recovery_time_candles": 0.0,
            "dd_duration_candles": 0.0,
        }

    initial = float(equity_curve.iloc[0])
    final = float(equity_curve.iloc[-1])
    net_pct = (final - initial) / initial if initial > 0 else 0.0
    max_dd, recovery_time, dd_event_duration, recovered = _compute_max_drawdown(equity_curve)

    # Trades metrics
    if trades_df is None or trades_df.empty:
        total_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        tpm = 0.0
        months = 0.0
    else:
        total_trades = len(trades_df)
        pnl_col = trades_df.get("pnl", trades_df.get("profit", None))
        if pnl_col is None:
            gross_profit = 0.0
            gross_loss = 0.0
        else:
            gross_profit = float(pnl_col[pnl_col > 0].sum()) if (pnl_col > 0).any() else 0.0
            gross_loss = float(-pnl_col[pnl_col < 0].sum()) if (pnl_col < 0).any() else 0.0
        # Months active from equity_curve index range
        if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve) >= 2:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            months = max(1.0, days / 30.4375)
        elif len(equity_curve) >= 2:
            months = max(1.0, len(equity_curve) / (24 * 30))  # H1 ≈ 24 candles/day
        else:
            months = 1.0
        tpm = total_trades / months if months > 0 else 0.0

    # Sharpe (annualised, assume H1 → 24 candles/day × 365 = 8760 candles/year)
    if len(equity_curve) >= 2:
        rets = equity_curve.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float((rets.mean() / rets.std()) * np.sqrt(8760))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    return {
        "initial_deposit": initial,
        "final_equity": final,
        "net_profit_pct": net_pct,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "trades_per_month": tpm,
        "total_trades": total_trades,
        "months_active": months,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "recovery_time_candles": recovery_time,
        "dd_duration_candles": dd_event_duration,  # event duration, not len(curve) — Phase A0 fix
        "recovered_drawdown": recovered,            # Phase A0: surface for downstream consumers
        "unrecovered_drawdown": not recovered,      # Phase A0: convenience flag
    }


# ============================================================
# Normalizers — each maps raw metric → [0, 1]
# ============================================================

def compute_profit_normalizer(net_profit_pct: float) -> float:
    """Sigmoid scaling centred at 0% profit (k=1.5).

    Reference points (verified by Phase A0.5 pinning tests):
        -50%   → 0.32
         0%   → 0.50 (centre of sigmoid)
        +50%   → 0.68
        +100%  → 0.82
        +300%  → 0.99

    Below 0% is penalised (returns a value < 0.5). The hard reject at
    `net_profit_pct <= 0` is enforced separately by `_check_hard_rejects`,
    not by this normaliser.
    """
    # Sigmoid centred at 0% (k=1.5): -50% → 0.32, 0% → 0.50, +50% → 0.68,
    # +100% → 0.82, +300% → 0.99.
    return float(1.0 / (1.0 + math.exp(-net_profit_pct * 1.5)))


def compute_dd_quality_normalizer(
    max_dd_pct: float,
    recovery_time_candles: float,
    dd_duration_candles: float,
    unrecovered: bool = False,
) -> float:
    """Higher = better. Lower DD + faster recovery = higher score.

    Phase A0 fix (Bug 1 + Bug 3):
    - Bug 1: output is clamped to [0, 1] (defensive — negative max_dd no longer escapes).
    - Bug 3: when `unrecovered=True` (drawdown never reclaimed), recovery_ratio is forced to 0.0
      instead of ≈1.0 from the buggy len(curve) denominator.

    Formula (unchanged for normal cases):
        dd_score       = max(0, 1 - max_dd/0.35)
        recovery_ratio = 0 if unrecovered else max(0, 1 - recovery_time/dd_duration)
        result         = clamp(0.7*dd_score + 0.3*recovery_ratio, 0, 1)
    """
    dd_score = max(0.0, 1.0 - max_dd_pct / 0.35)  # 0 DD → 1.0, 35% DD → 0.0
    if unrecovered:
        recovery_ratio = 0.0  # Bug 3 fix: never recovered → no recovery credit
    elif dd_duration_candles <= 0:
        recovery_ratio = 1.0  # edge case: caller passed 0 → assume instant recovery
    else:
        recovery_ratio = max(0.0, 1.0 - recovery_time_candles / dd_duration_candles)
    raw = 0.7 * dd_score + 0.3 * recovery_ratio
    # Bug 1 fix: defensive clamp to [0, 1]
    return float(max(0.0, min(1.0, raw)))


def compute_sharpe_normalizer(sharpe: float) -> float:
    """Saturating curve: Sharpe 0 → ~0.5, Sharpe 1 → ~0.73, Sharpe 2 → ~0.88."""
    return float(1.0 / (1.0 + math.exp(-sharpe)))


def compute_pf_normalizer(profit_factor: float) -> float:
    """PF 1.0 (breakeven) → 0.5; PF 2.0 → ~0.73; PF 3.0 → ~0.88. Clamped at 0."""
    if profit_factor <= 0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-(profit_factor - 1.0))))


def compute_tpm_normalizer(tpm: float) -> float:
    """Sigmoid centred at TPM=5 (k=1/8).

    Reference points (verified by Phase A0.5 pinning tests):
        TPM  0   → 0.35
        TPM  5   → 0.50 (centre of sigmoid)
        TPM 10   → 0.65
        TPM 20   → 0.87
        TPM 40+  → 0.99 (saturated)

    The TPM<5 hard reject is enforced separately by `_check_hard_rejects`
    (when `evolution_mode=False`) and `DEPLOYMENT_MIN_TRADES_PER_MONTH`
    (when deploying). This normaliser only SCORES TPM — it does not gate.
    """
    # Sigmoid centred at TPM=5 (k=1/8): 0 → 0.35, 5 → 0.50, 10 → 0.65,
    # 20 → 0.87, 40+ → 0.99.
    return float(1.0 / (1.0 + math.exp(-(tpm - 5.0) / 8.0)))


# ============================================================
# Score computation
# ============================================================

def compute_score(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame | None,
    settings: Any | None = None,
    candidate_id: str | None = None,
    evolution_mode: bool = False,
) -> ScoreResult | RejectedResult:
    """Main entry point. Returns ScoreResult if scored, RejectedResult if hard-rejected.

    settings: optional Settings object (used for context like experiment_slug).
              Not required for scoring — just for traceability.

    evolution_mode: if True, the TPM hard reject is relaxed. Use during GA
    evolution so the algorithm can see and breed candidates with low TPM
    (the TPM component weight in REWARD_WEIGHTS still penalises them).
    At deployment time, keep evolution_mode=False so the standard
    is preserved.
    """
    metrics = _compute_metrics(equity_curve, trades_df if trades_df is not None else pd.DataFrame())

    slug = getattr(settings, "experiment_slug", "unknown") if settings else "synthetic"

    # Hard rejection check
    reject_reason = _check_hard_rejects(
        net_profit_pct=metrics["net_profit_pct"],
        max_dd_pct=metrics["max_drawdown_pct"],
        trades_per_month=metrics["trades_per_month"],
        initial_deposit=metrics["initial_deposit"],
        final_equity=metrics["final_equity"],
        total_trades=metrics["total_trades"],
        evolution_mode=evolution_mode,
    )
    if reject_reason is not None:
        return RejectedResult(
            experiment_slug=slug,
            candidate_id=candidate_id,
            reason=reject_reason,
            raw_metrics=metrics,
            total_trades=int(metrics["total_trades"]),
            months_active=metrics["months_active"],
        )

    # Score all 5 components
    profit = ComponentScore(
        raw_value=metrics["net_profit_pct"],
        normalised=compute_profit_normalizer(metrics["net_profit_pct"]),
        weight=REWARD_WEIGHTS["profit"],
        contribution=0.0,  # filled below
    )
    dd_q = ComponentScore(
        raw_value=metrics["max_drawdown_pct"],
        normalised=compute_dd_quality_normalizer(
            metrics["max_drawdown_pct"],
            metrics["recovery_time_candles"],
            metrics["dd_duration_candles"],
            unrecovered=bool(metrics.get("unrecovered_drawdown", False)),  # Phase A0 Bug 3
        ),
        weight=REWARD_WEIGHTS["dd_quality"],
        contribution=0.0,
    )
    sharpe = ComponentScore(
        raw_value=metrics["sharpe"],
        normalised=compute_sharpe_normalizer(metrics["sharpe"]),
        weight=REWARD_WEIGHTS["sharpe"],
        contribution=0.0,
    )
    pf = ComponentScore(
        raw_value=metrics["profit_factor"],
        normalised=compute_pf_normalizer(metrics["profit_factor"]),
        weight=REWARD_WEIGHTS["profit_factor"],
        contribution=0.0,
    )
    tpm = ComponentScore(
        raw_value=metrics["trades_per_month"],
        normalised=compute_tpm_normalizer(metrics["trades_per_month"]),
        weight=REWARD_WEIGHTS["tpm"],
        contribution=0.0,
    )

    # Contributions = normalised × weight
    profit.contribution = profit.normalised * profit.weight
    dd_q.contribution = dd_q.normalised * dd_q.weight
    sharpe.contribution = sharpe.normalised * sharpe.weight
    pf.contribution = pf.normalised * pf.weight
    tpm.contribution = tpm.normalised * tpm.weight

    base = profit.contribution + dd_q.contribution + sharpe.contribution + pf.contribution + tpm.contribution

    # DD penalty tier (only applies if 25% <= DD < 35% — already-passed hard reject)
    dd_penalty = 1.0
    for threshold, multiplier in DD_PENALTY_TIERS:
        if metrics["max_drawdown_pct"] >= threshold:
            dd_penalty = multiplier

    final = base * dd_penalty

    breakdown = ScoreBreakdown(
        profit=profit,
        dd_quality=dd_q,
        sharpe=sharpe,
        profit_factor=pf,
        tpm=tpm,
        dd_penalty_multiplier=dd_penalty,
        base_score=base,
        final_score=final,
    )

    return ScoreResult(
        experiment_slug=slug,
        candidate_id=candidate_id,
        breakdown=breakdown,
        raw_metrics=metrics,
        total_trades=int(metrics["total_trades"]),
        months_active=metrics["months_active"],
    )
