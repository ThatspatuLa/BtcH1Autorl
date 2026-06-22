"""Stage 5 acceptance tests — locked reward / scoring engine.

PHASE 1 tests (synthetic data, no Stage 3 dependency):
- LOCKED weights sum to 1.0 with correct breakdown
- All 5 normalizers produce values in [0, 1]
- Hard rejects trigger on each rejection rule (net<=0, DD>35%, TPM<5, invalid_data, too_few_trades)
- DD penalty tiers apply at correct thresholds (25-30% ×0.85, 30-35% ×0.50)
- Per-component breakdown JSON output matches locked weights
- Deterministic: same inputs → same output
- Empty / edge-case inputs handled (0 trades, single trade, all losses)

PHASE 2 tests will follow after Stage 3 produces real backtest output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reward import (
    DD_PENALTY_TIERS,
    MAX_DD_PCT,
    MIN_TOTAL_TRADES,
    MIN_TPM,
    REWARD_WEIGHTS,
    RejectedResult,
    ScoreResult,
    compute_dd_quality_normalizer,
    compute_pf_normalizer,
    compute_profit_normalizer,
    compute_score,
    compute_sharpe_normalizer,
    compute_tpm_normalizer,
)

pytestmark = pytest.mark.stage5


# ============================================================
# Helper: synthetic equity curves + trades
# ============================================================

def _steady_up_equity(n_candles: int = 24 * 365 * 5, start: float = 10000.0) -> pd.Series:
    """Steady upward equity, 50%/year."""
    rets = np.full(n_candles, 0.0002)  # ~0.02%/candle → ~50%/year compounded
    idx = pd.date_range("2021-06-20", periods=n_candles, freq="1h")
    eq = start * (1 + rets).cumprod()
    eq_series = pd.Series(eq, index=idx, name="equity")
    eq_series.iloc[0] = start
    return eq_series


def _steady_up_trades(n: int = 200) -> pd.DataFrame:
    """n profitable trades."""
    rng = np.random.RandomState(42)
    times = pd.date_range("2021-06-20", periods=n, freq="D")
    return pd.DataFrame({
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=4),
        "pnl": rng.uniform(50, 200, n),
        "qty": rng.uniform(0.01, 0.1, n),
        "avg_entry": rng.uniform(30000, 60000, n),
        "exit_price": rng.uniform(30000, 60000, n),
        "cycle_id": [f"c_{i}" for i in range(n)],
        "symbol": ["BTC/USDT"] * n,
        "n_layers": rng.randint(1, 5, n),
        "close_reason": ["tp"] * n,
    })


def _losing_equity(n_candles: int = 24 * 365, start: float = 10000.0) -> pd.Series:
    """Steady declining equity → hard reject for net_profit <= 0."""
    rets = np.full(n_candles, -0.0002)
    idx = pd.date_range("2021-06-20", periods=n_candles, freq="1h")
    eq = start * (1 + rets).cumprod()
    eq_series = pd.Series(eq, index=idx)
    eq_series.iloc[0] = start
    return eq_series


def _high_dd_equity(start: float = 10000.0, dd_pct: float = 0.50) -> pd.Series:
    """Equity that drops > 35% from peak → hard reject."""
    n = 24 * 200
    idx = pd.date_range("2021-06-20", periods=n, freq="1h")
    eq = np.ones(n) * start
    # Drop to (1 - dd_pct) at midpoint, then stay flat (or slight recovery)
    eq[n // 2:] = start * (1 - dd_pct)
    return pd.Series(eq, index=idx)


def _few_trades(n: int = 3) -> pd.DataFrame:
    """Only n trades → too_few_trades hard reject."""
    times = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=4),
        "pnl": np.full(n, 100.0),
    })


# ============================================================
# LOCKED weights tests
# ============================================================

def test_reward_weights_are_locked():
    assert REWARD_WEIGHTS == {
        "profit": 0.55,
        "dd_quality": 0.15,
        "sharpe": 0.10,
        "profit_factor": 0.10,
        "tpm": 0.10,
    }
    assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9


def test_dd_penalty_tiers_locked():
    assert DD_PENALTY_TIERS == [(0.25, 0.85), (0.30, 0.50)]


def test_hard_reject_thresholds_locked():
    assert MAX_DD_PCT == 0.35
    assert MIN_TPM == 5.0
    assert MIN_TOTAL_TRADES == 30


# ============================================================
# Normalizer tests
# ============================================================

@pytest.mark.parametrize("raw,expected_min,expected_max", [
    (-0.5, 0.15, 0.40),    # -50% loss → low
    (0.0, 0.40, 0.60),     # 0% profit → near sigmoid centre
    (0.5, 0.55, 0.75),     # +50% profit
    (1.0, 0.75, 0.85),     # +100%
    (2.0, 0.90, 0.97),     # +200%
    (5.0, 0.99, 1.00),     # +500% saturating
])
def test_profit_normalizer(raw, expected_min, expected_max):
    n = compute_profit_normalizer(raw)
    assert expected_min <= n <= expected_max, f"profit_norm({raw})={n} not in [{expected_min}, {expected_max}]"


@pytest.mark.parametrize("dd,rec_time,total,expected_min", [
    (0.00, 0.0, 1000, 0.95),   # no DD → near 1.0
    (0.10, 100, 1000, 0.65),   # small DD, fast recovery
    (0.25, 500, 1000, 0.30),   # mid DD, slow recovery
    (0.35, 1000, 1000, 0.0),   # max-DD threshold, never recovered
])
def test_dd_quality_normalizer(dd, rec_time, total, expected_min):
    n = compute_dd_quality_normalizer(dd, rec_time, total)
    assert expected_min - 0.10 <= n <= 1.0


def test_sharpe_normalizer_in_unit_range():
    for s in [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]:
        n = compute_sharpe_normalizer(s)
        assert 0.0 <= n <= 1.0


def test_pf_normalizer_in_unit_range():
    for pf in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]:
        n = compute_pf_normalizer(pf)
        assert 0.0 <= n <= 1.0


def test_pf_normalizer_breakeven_is_half():
    """PF = 1.0 (breakeven) → ~0.5 by sigmoid design."""
    n = compute_pf_normalizer(1.0)
    assert 0.45 <= n <= 0.55


def test_tpm_normalizer_in_unit_range():
    for tpm in [0.0, 5.0, 10.0, 20.0, 40.0, 100.0]:
        n = compute_tpm_normalizer(tpm)
        assert 0.0 <= n <= 1.0


# ============================================================
# Hard rejection tests
# ============================================================

def test_net_profit_zero_rejected():
    """net_profit <= 0 → reject."""
    eq = _losing_equity(n_candles=24 * 200)
    trades = _steady_up_trades(n=100)
    result = compute_score(eq, trades, candidate_id="losing-test")
    assert isinstance(result, RejectedResult)
    assert result.reason == "net_profit<=0"


def test_high_dd_rejected():
    """DD > 35% → reject."""
    # Equity: start 10000, drop to 6000 (-40%) at midpoint, then recover to 12000 (>start)
    n = 24 * 200
    idx = pd.date_range("2021-06-20", periods=n, freq="1h")
    eq = np.concatenate([
        np.linspace(10000, 10000, n // 4),
        np.linspace(10000, 6000, n // 4),  # -40% drop
        np.linspace(6000, 12000, n // 2),  # recovery to +20%
    ])
    eq_series = pd.Series(eq, index=idx)
    eq_series.iloc[0] = 10000.0
    trades = _steady_up_trades(n=100)
    result = compute_score(eq_series, trades, candidate_id="high-dd-test")
    assert isinstance(result, RejectedResult)
    assert result.reason == "drawdown>35%"


def test_too_few_trades_rejected():
    """< 30 total trades → reject."""
    eq = _steady_up_equity(n_candles=24 * 30)
    trades = _few_trades(n=10)
    result = compute_score(eq, trades, candidate_id="few-trades-test")
    assert isinstance(result, RejectedResult)
    assert result.reason == "too_few_trades"


def test_invalid_data_rejected_nan():
    """NaN in final_equity → reject (invalid_data, NOT net_profit<=0).

    Set NaN at final candle so net_profit_pct becomes NaN, exercising the finite-check first.
    """
    idx = pd.date_range("2021-06-20", periods=100, freq="1h")
    eq = pd.Series(np.linspace(10000, 11000, 100), index=idx)
    eq.iloc[-1] = float("nan")  # final equity NaN
    trades = _steady_up_trades(n=50)
    result = compute_score(eq, trades, candidate_id="nan-test")
    assert isinstance(result, RejectedResult)
    assert result.reason == "invalid_data"


def test_invalid_data_rejected_inf():
    """inf in equity → reject (invalid_data)."""
    idx = pd.date_range("2021-06-20", periods=100, freq="1h")
    eq = pd.Series(np.linspace(10000, 11000, 100), index=idx)
    eq.iloc[-1] = float("inf")  # final equity inf
    trades = _steady_up_trades(n=50)
    result = compute_score(eq, trades, candidate_id="inf-test")
    assert isinstance(result, RejectedResult)
    assert result.reason == "invalid_data"


def test_empty_trades_rejected():
    """No trades at all → too_few_trades."""
    eq = _steady_up_equity(n_candles=24 * 30)
    result = compute_score(eq, pd.DataFrame(), candidate_id="empty-trades-test")
    assert isinstance(result, RejectedResult)
    # Either too_few_trades or tpm<5 depending on metrics path
    assert result.reason in ("too_few_trades", "tpm<5")


# ============================================================
# Successful scoring + DD penalty tiers
# ============================================================

def test_evolution_mode_relaxes_tpm_reject():
    """evolution_mode=True should NOT hard-reject on TPM<5.

    Same candidate, two calls: one with evolution_mode=False (deployment
    standard) → rejected with tpm<5; one with evolution_mode=True →
    scored (with a low tpm component, but the algorithm sees the data)."""
    from reward.scoring import compute_score
    # Build a slow but profitable candidate: 30 trades, 200 days → TPM 4.5
    np.random.seed(42)
    eq = _steady_up_equity(n_candles=24 * 200)  # 200 days of H1
    trades_low = _steady_up_trades(n=30)
    # Deployment standard: should reject with tpm<5 OR too_few_trades
    dep_result = compute_score(eq, trades_low, candidate_id="tpm-deploy")
    assert isinstance(dep_result, RejectedResult)
    # In evolution mode: should score (or at least not reject for TPM)
    evo_result = compute_score(eq, trades_low, candidate_id="tpm-evo", evolution_mode=True)
    if isinstance(evo_result, RejectedResult):
        # If rejected, the reason must NOT be tpm<5
        assert evo_result.reason != "tpm<5", \
            f"evolution_mode should relax tpm<5, got: {evo_result.reason}"


def test_evolution_mode_preserves_safety_rejects():
    """evolution_mode=True should still hard-reject on safety issues.

    - net_profit<=0
    - drawdown>35%
    - invalid_data
    """
    from reward.scoring import compute_score
    # Losing equity → should still reject with net_profit<=0
    eq = _losing_equity(n_candles=24 * 200)
    trades = _steady_up_trades(n=100)
    result = compute_score(eq, trades, candidate_id="losing-evo", evolution_mode=True)
    assert isinstance(result, RejectedResult)
    assert result.reason == "net_profit<=0"


def test_good_candidate_scores_successfully():
    """Steady upward equity + many trades → ScoreResult with high score."""
    np.random.seed(42)
    eq = _steady_up_equity()
    trades = _steady_up_trades(n=600)
    result = compute_score(eq, trades, candidate_id="good-test")
    assert isinstance(result, ScoreResult)
    assert result.exit_reason == "scored"
    assert 0.0 <= result.breakdown.final_score <= 1.0
    # All 5 components present
    assert result.breakdown.profit.contribution > 0
    assert result.breakdown.dd_quality.contribution > 0
    assert result.breakdown.sharpe.contribution > 0
    assert result.breakdown.profit_factor.contribution > 0
    assert result.breakdown.tpm.contribution > 0


def test_dd_penalty_tier_25_to_30():
    """DD 25-30% → score × 0.85."""
    # Start 10000, peak at 13000, drop to 9500 (-27%), recover to 14000
    n = 24 * 365
    idx = pd.date_range("2021-06-20", periods=n, freq="1h")
    eq = np.concatenate([
        np.linspace(10000, 13000, n // 2),
        np.linspace(13000, 9500, n // 4),  # -27% from peak
        np.linspace(9500, 14000, n - 3 * n // 4),
    ])
    eq_series = pd.Series(eq, index=idx)
    trades = _steady_up_trades(n=600)
    result = compute_score(eq_series, trades, candidate_id="dd-25-test")
    assert isinstance(result, ScoreResult)
    assert abs(result.breakdown.dd_penalty_multiplier - 0.85) < 1e-9


def test_dd_penalty_tier_30_to_35():
    """DD 30-35% → score × 0.50."""
    # Start 10000, peak at 13000, drop to 8800 (-32.3%), recover to 14000
    n = 24 * 365
    idx = pd.date_range("2021-06-20", periods=n, freq="1h")
    eq = np.concatenate([
        np.linspace(10000, 13000, n // 2),
        np.linspace(13000, 8800, n // 4),  # -32.3% from peak (in 30-35% tier)
        np.linspace(8800, 14000, n - 3 * n // 4),
    ])
    eq_series = pd.Series(eq, index=idx)
    trades = _steady_up_trades(n=600)
    result = compute_score(eq_series, trades, candidate_id="dd-30-test")
    assert isinstance(result, ScoreResult)
    assert abs(result.breakdown.dd_penalty_multiplier - 0.50) < 1e-9


def test_score_breakdown_sum_to_weighted():
    """Sum of all component contributions = base_score."""
    np.random.seed(42)
    eq = _steady_up_equity()
    trades = _steady_up_trades(n=600)
    result = compute_score(eq, trades, candidate_id="sum-test")
    assert isinstance(result, ScoreResult)
    expected = (
        result.breakdown.profit.contribution
        + result.breakdown.dd_quality.contribution
        + result.breakdown.sharpe.contribution
        + result.breakdown.profit_factor.contribution
        + result.breakdown.tpm.contribution
    )
    assert abs(result.breakdown.base_score - expected) < 1e-9


def test_score_breakdown_to_dict_serialisable():
    """ScoreResult.to_dict() must produce JSON-serialisable output."""
    import json
    np.random.seed(42)
    eq = _steady_up_equity()
    trades = _steady_up_trades(n=600)
    result = compute_score(eq, trades, candidate_id="serialise-test")
    assert isinstance(result, ScoreResult)
    d = result.to_dict()
    j = json.dumps(d, default=str)
    parsed = json.loads(j)
    assert "breakdown" in parsed
    assert "raw_metrics" in parsed
    assert parsed["breakdown"]["final_score"] == result.breakdown.final_score


# ============================================================
# Determinism
# ============================================================

def test_score_is_deterministic():
    """Same inputs must produce same output across runs."""
    np.random.seed(42)
    eq = _steady_up_equity()
    # 600 trades over 60 months = 10 tpm (well above 5 minimum)
    trades = _steady_up_trades(n=600)
    r1 = compute_score(eq, trades, candidate_id="det-test")
    r2 = compute_score(eq, trades, candidate_id="det-test")
    assert isinstance(r1, ScoreResult) and isinstance(r2, ScoreResult)
    assert r1.breakdown.final_score == r2.breakdown.final_score
    assert r1.to_dict() == r2.to_dict()


# ============================================================
# PHASE A0 — Stage 5 bug-fix tests (Six's Gate Check finding)
# ============================================================
# Bug 1: dd_quality output not clamped to [0, 1]
# Bug 2: dd_duration_candles = len(equity_curve) instead of event duration
# Bug 3: unrecovered drawdowns get weak penalty
# All three feed into Stage 5 base_score via compute_dd_quality_normalizer.

from reward.scoring import _compute_max_drawdown, _compute_metrics


def test_dd_quality_normalizer_clamps_negative_max_dd():
    """Bug 1: dd_quality must be in [0, 1]. A negative max_dd_pct should not exceed 1.0."""
    # With negative max_dd (impossible in practice but defensive), the inner dd_score would
    # compute > 1.0 — outer clamp must bring it back. Output is clamped to 1.0.
    out = compute_dd_quality_normalizer(
        max_dd_pct=-0.05,
        recovery_time_candles=50.0,
        dd_duration_candles=200.0,
    )
    assert 0.0 <= out <= 1.0, f"dd_quality must be clamped to [0,1], got {out}"
    # In-range case: realistic DD should still hit a sensible value
    out2 = compute_dd_quality_normalizer(0.10, 50.0, 200.0)
    assert 0.0 <= out2 <= 1.0
    assert abs(out2 - (0.7 * (1.0 - 0.10 / 0.35) + 0.3 * 0.75)) < 1e-6  # 0.7250


def test_dd_quality_normalizer_unrecovered_uses_zero_recovery_ratio():
    """Bug 3: unrecovered drawdown must set recovery_ratio = 0, not ≈1.0."""
    # 10% DD with dd_duration=100 and recovery_time=999 (never recovered)
    out = compute_dd_quality_normalizer(
        max_dd_pct=0.10,
        recovery_time_candles=999.0,
        dd_duration_candles=100.0,
    )
    # dd_score = max(0, 1 - 0.10/0.35) = 0.7143
    # recovery_ratio = 0 (unrecovered → 0)
    # out = 0.7 * 0.7143 + 0.3 * 0 = 0.5000
    expected = 0.7 * (1.0 - 0.10 / 0.35) + 0.3 * 0.0
    assert abs(out - expected) < 1e-6, f"expected {expected:.6f}, got {out:.6f}"


def test_dd_quality_normalizer_recovered_uses_event_duration():
    """Bug 2: dd_duration must reflect the drawdown event, not len(curve)."""
    # 20% DD over an event lasting 80 candles, recovered in 40 candles within the event
    out = compute_dd_quality_normalizer(
        max_dd_pct=0.20,
        recovery_time_candles=40.0,
        dd_duration_candles=80.0,   # event duration, not len(curve)
    )
    # dd_score = max(0, 1 - 0.20/0.35) = 0.4286
    # recovery_ratio = max(0, 1 - 40/80) = 0.5
    # out = 0.7 * 0.4286 + 0.3 * 0.5 = 0.3000 + 0.1500 = 0.4500
    expected = 0.7 * (1.0 - 0.20 / 0.35) + 0.3 * 0.5
    assert abs(out - expected) < 1e-6, f"expected {expected:.6f}, got {out:.6f}"


def test_compute_max_drawdown_returns_event_duration_and_recovered_flag():
    """Bug 2/3: _compute_max_drawdown must return event_duration + recovered flag."""
    # Synthetic 200-candle curve: flat 100 → drop to 90 over 50 candles → stays at 90
    # dd_event_duration ≈ 100 candles (peak at 0 to end), recovered=False
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    eq = pd.Series([100.0] * 100 + [90.0] * 100, index=idx)
    result = _compute_max_drawdown(eq)
    # Bug 2 fix: result must be a 4-tuple including (max_dd, recovery_time, dd_event_duration, recovered)
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}-tuple: {result}"
    max_dd, recovery_time, dd_event_duration, recovered = result
    assert max_dd == pytest.approx(0.10, abs=1e-6)
    # Bug 3: never recovered → recovered=False, recovery_time = len(curve) - 1
    assert recovered is False
    # Bug 2: dd_event_duration is from peak_pos to end (or to recovery)
    # peak at idx 0, end at idx 199 → dd_event_duration = 199 - 0 = 199
    assert dd_event_duration > 0


def test_compute_metrics_exposes_recovered_drawdown_flag():
    """Bug 3: metrics dict must expose recovered_drawdown / unrecovered_drawdown."""
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    eq = pd.Series([100.0] * 50 + [85.0] * 50, index=idx)  # 15% DD, never recovers
    metrics = _compute_metrics(eq, None)
    assert "recovered_drawdown" in metrics, "metrics must expose recovered_drawdown flag"
    assert "unrecovered_drawdown" in metrics, "metrics must expose unrecovered_drawdown flag"
    assert metrics["recovered_drawdown"] is False
    assert metrics["unrecovered_drawdown"] is True


def test_compute_metrics_recovered_curve_flags_true():
    """Bug 3 control: a curve that recovers must flag recovered_drawdown=True."""
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    eq = pd.Series([100.0] * 50 + [85.0] * 30 + [100.0] * 20, index=idx)  # 15% DD, recovers at idx 80
    metrics = _compute_metrics(eq, None)
    assert metrics["recovered_drawdown"] is True
    assert metrics["unrecovered_drawdown"] is False


# ============================================================
# PHASE A0.5 — Sigmoid pinning tests
# ============================================================
# Pins actual intended behaviour of profit + TPM normalisers. Prevents silent
# regressions if the sigmoid is ever "fixed" to match the (wrong) docstrings.

def test_profit_normalizer_known_values():
    """Pin actual sigmoid output. Centred at 0% (k=1.5)."""
    assert abs(compute_profit_normalizer(-0.50) - 0.3208) < 1e-3
    assert abs(compute_profit_normalizer(0.00) - 0.5000) < 1e-3
    assert abs(compute_profit_normalizer(0.50) - 0.6792) < 1e-3
    assert abs(compute_profit_normalizer(1.00) - 0.8176) < 1e-3
    assert abs(compute_profit_normalizer(3.00) - 0.9890) < 1e-3


def test_tpm_normalizer_known_values():
    """Pin actual sigmoid output. Centred at TPM=5 (k=1/8)."""
    assert abs(compute_tpm_normalizer(0) - 0.3486) < 1e-3
    assert abs(compute_tpm_normalizer(5) - 0.5000) < 1e-3
    assert abs(compute_tpm_normalizer(10) - 0.6514) < 1e-3
    assert abs(compute_tpm_normalizer(20) - 0.8670) < 1e-3
    assert abs(compute_tpm_normalizer(40) - 0.9876) < 1e-3


def test_sharpe_normalizer_known_values():
    """Pin actual sigmoid output. Centred at Sharpe=0."""
    # Sharper: at 0 → 0.5; at 1 → 0.7311; at 2 → 0.8808
    assert abs(compute_sharpe_normalizer(0.0) - 0.5000) < 1e-3
    assert abs(compute_sharpe_normalizer(1.0) - 0.7311) < 1e-3
    assert abs(compute_sharpe_normalizer(2.0) - 0.8808) < 1e-3


def test_pf_normalizer_known_values():
    """Pin actual sigmoid output. Centred at PF=1."""
    # PF=1.0 → 0.5; PF=2.0 → 0.7311; PF=3.0 → 0.8808; PF<=0 → 0
    assert abs(compute_pf_normalizer(1.0) - 0.5000) < 1e-3
    assert abs(compute_pf_normalizer(2.0) - 0.7311) < 1e-3
    assert abs(compute_pf_normalizer(3.0) - 0.8808) < 1e-3
    assert compute_pf_normalizer(0.0) == 0.0
    assert compute_pf_normalizer(-1.0) == 0.0
