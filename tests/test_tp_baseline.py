"""Tests for Stage 9 — Simple Fixed TP Baseline."""
from __future__ import annotations

import pandas as pd
import pytest

from dca_engine.tp_baseline import (
    DEFAULT_TP_GENOME,
    DEFAULT_TP_PCT,
    MAX_TP_PCT,
    FixedTPBaseline,
    backtest_with_fixed_tp,
    extract_dca_params_from_genome,
)
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    DcaGenome,
    GridMethod,
    TpExitMethod,
    TpGenome,
)

# ============================================================
# Test: FixedTPBaseline construction
# ============================================================

def test_baseline_default():
    b = FixedTPBaseline()
    assert b.tp_pct == DEFAULT_TP_PCT


def test_baseline_custom_tp():
    b = FixedTPBaseline(tp_pct=0.05)
    assert b.tp_pct == 0.05


def test_baseline_rejects_zero_tp():
    with pytest.raises(ValueError):
        FixedTPBaseline(tp_pct=0.0)


def test_baseline_rejects_negative_tp():
    with pytest.raises(ValueError):
        FixedTPBaseline(tp_pct=-0.01)


def test_baseline_rejects_excessive_tp():
    with pytest.raises(ValueError, match="must be <="):
        FixedTPBaseline(tp_pct=0.99)


# ============================================================
# Test: from_genome
# ============================================================

def test_baseline_from_default_genome():
    b = FixedTPBaseline.from_genome(DEFAULT_TP_GENOME)
    assert b.tp_pct == DEFAULT_TP_PCT


def test_baseline_from_custom_genome():
    g = TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.04})
    b = FixedTPBaseline.from_genome(g)
    assert b.tp_pct == 0.04


def test_baseline_from_genome_missing_tp_pct():
    """If tp_pct not in exit_params, fall back to default."""
    g = TpGenome(exit_method=TpExitMethod.FIXED, exit_params={})
    b = FixedTPBaseline.from_genome(g)
    assert b.tp_pct == DEFAULT_TP_PCT


def test_baseline_rejects_non_fixed_method():
    """Stage 9 baseline only supports method=FIXED."""
    g = TpGenome(exit_method=TpExitMethod.TRAILING, exit_params={})
    with pytest.raises(ValueError, match="only supports method=FIXED"):
        FixedTPBaseline.from_genome(g)


# ============================================================
# Test: to_order_manager_kwargs
# ============================================================

def test_to_order_manager_kwargs():
    b = FixedTPBaseline(tp_pct=0.03)
    kwargs = b.to_order_manager_kwargs()
    assert kwargs == {"tp_pct": 0.03}


# ============================================================
# Test: to_tp_genome round-trip
# ============================================================

def test_to_tp_genome_round_trip():
    b = FixedTPBaseline(tp_pct=0.025)
    g = b.to_tp_genome()
    assert g.exit_method == TpExitMethod.FIXED
    assert g.exit_params["tp_pct"] == 0.025


# ============================================================
# Test: description
# ============================================================

def test_description_default():
    b = FixedTPBaseline()
    assert b.description() == "fixed_tp@2.00%"


def test_description_custom():
    b = FixedTPBaseline(tp_pct=0.05)
    assert b.description() == "fixed_tp@5.00%"


# ============================================================
# Test: backtest_with_fixed_tp (smoke)
# ============================================================
def _make_synthetic_ohlcv(n: int = 200, start_price: float = 100.0) -> pd.DataFrame:
    """Build a tiny OHLCV dataframe for smoke testing."""
    idx = pd.date_range("2021-06-01", periods=n, freq="h")
    # Simple uptrend then downtrend
    prices = [start_price + (i * 0.1) for i in range(n // 2)]
    prices += [start_price + (n // 2) * 0.1 - (i * 0.1) for i in range(n - n // 2)]
    df = pd.DataFrame({
        "date": idx,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    })
    return df


def test_backtest_with_fixed_tp_smoke():
    df = _make_synthetic_ohlcv(200)
    result = backtest_with_fixed_tp(
        df=df,
        candidate_id="cand_test",
        genome_id="genome_test",
        experiment_id="exp_test",
    )
    assert result.candidate_id == "cand_test"
    assert result.genome_id == "genome_test"
    assert result.experiment_id == "exp_test"
    assert len(result.equity_curve) > 0
    # backtest_meta should reflect the fixed TP
    assert result.backtest_meta["tp_pct"] == DEFAULT_TP_PCT
    assert result.backtest_meta["stage"] == 3  # stage 3 placeholder
    assert result.backtest_meta["is_placeholder_sizing"] is True


def test_backtest_with_custom_tp():
    df = _make_synthetic_ohlcv(200)
    g = TpGenome(exit_method=TpExitMethod.FIXED, exit_params={"tp_pct": 0.05})
    result = backtest_with_fixed_tp(
        df=df,
        candidate_id="cand_custom",
        genome_id="genome_custom",
        experiment_id="exp_test",
        tp_genome=g,
    )
    assert result.backtest_meta["tp_pct"] == 0.05


def test_backtest_with_default_genome():
    df = _make_synthetic_ohlcv(200)
    result = backtest_with_fixed_tp(
        df=df,
        candidate_id="cand_default",
        genome_id="genome_default",
        experiment_id="exp_test",
        # no tp_genome → uses DEFAULT_TP_GENOME
    )
    assert result.backtest_meta["tp_pct"] == DEFAULT_TP_PCT


# ============================================================
# Test: extract_dca_params_from_genome
# ============================================================

def _make_dca_genome(grid_method=GridMethod.FIXED_PCT, grid_pct=0.02, max_layers=3) -> DcaGenome:
    return DcaGenome(
        grid_method=grid_method,
        grid_params={"pct": grid_pct} if grid_method == GridMethod.FIXED_PCT else {},
        allocation_method=AllocationMethod.EQUAL,
        allocation_params={},
        max_dca_layers=max_layers,
    )


def test_extract_dca_params_fixed_pct():
    g = _make_dca_genome(grid_pct=0.025, max_layers=5)
    params = extract_dca_params_from_genome(g)
    assert params["grid_pct"] == 0.025
    assert params["max_layers"] == 5


def test_extract_dca_params_default_pct():
    """If grid_params missing pct, use default 0.015."""
    g = DcaGenome(
        grid_method=GridMethod.FIXED_PCT,
        grid_params={},
        allocation_method=AllocationMethod.EQUAL,
        allocation_params={},
        max_dca_layers=4,
    )
    params = extract_dca_params_from_genome(g)
    assert params["grid_pct"] == 0.015  # default fallback
    assert params["max_layers"] == 4


def test_extract_dca_params_non_fixed_method():
    """Non-fixed methods fall back to 0.015 (Stage 10 will handle)."""
    g = _make_dca_genome(grid_method=GridMethod.ATR, max_layers=4)
    params = extract_dca_params_from_genome(g)
    assert params["grid_pct"] == 0.015
    assert params["max_layers"] == 4


# ============================================================
# Test: integration with full CandidateGenome
# ============================================================

def test_extract_from_full_candidate_genome():
    cand = CandidateGenome(
        genome_id="genome_full",
        dca_genome=_make_dca_genome(grid_pct=0.018, max_layers=4),
        tp_genome=DEFAULT_TP_GENOME,
    )
    params = extract_dca_params_from_genome(cand)
    assert params["grid_pct"] == 0.018
    assert params["max_layers"] == 4


def test_constants_locked():
    """Sanity check the v1 locked defaults."""
    assert DEFAULT_TP_PCT == 0.02
    assert MAX_TP_PCT == 0.50
    assert DEFAULT_TP_GENOME.exit_method == TpExitMethod.FIXED
    assert DEFAULT_TP_GENOME.exit_params["tp_pct"] == 0.02
