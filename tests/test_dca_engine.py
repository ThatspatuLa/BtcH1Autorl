"""Stage 3 acceptance tests — DCA engine skeleton.

Verifies:
- Position lifecycle: layers, qty, cost, average entry, P&L
- OrderManager decisions: open, add_layer, close, no-action
- CycleLifecycle state transitions + P&L on close
- ExposureTracker snapshot
- State machine iterates candles, opens/closes cycles
- BacktestRunner produces BacktestResult with Stage 5-compatible shape
- End-to-end: real BTC 5y data → backtest → Stage 5 score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dca_engine import (
    BacktestResult,
    CycleLifecycle,
    CycleState,
    ExposureTracker,
    OrderAction,
    OrderManager,
    Position,
    PositionTracker,
    backtest_candidate,
    run_state_machine,
)
from reward import RejectedResult, ScoreResult, compute_score

pytestmark = pytest.mark.stage3


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """50 candles of synthetic price action with at least one dip and recovery."""
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Start at 50000, dip to 48000 by candle 20, recover to 52000 by candle 49
    prices = np.concatenate([
        np.linspace(50000, 50000, 5),  # initial flat
        np.linspace(50000, 48000, 15),  # dip
        np.linspace(48000, 52000, 30),  # recovery
    ])
    df = pd.DataFrame({
        "date": idx,
        "open": prices,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": 100.0,
    })
    return df


@pytest.fixture
def rising_ohlcv() -> pd.DataFrame:
    """30 candles, monotonically rising — no DCA layers triggered, just one cycle + TP."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    prices = np.linspace(50000, 51000, n)
    return pd.DataFrame({
        "date": idx, "open": prices, "high": prices + 10, "low": prices - 10,
        "close": prices, "volume": 100.0,
    })


@pytest.fixture
def default_order_manager() -> OrderManager:
    return OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=3)


# ============================================================
# Position / PositionTracker tests
# ============================================================

def test_position_layer_qty_and_cost():
    pos = Position(cycle_id="c1", symbol="BTC/USDT")
    pos.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    pos.layers.append({"price": 49000.0, "qty": 0.2, "fee": 9.8})
    assert pos.total_qty == pytest.approx(0.3)
    assert pos.total_cost == pytest.approx(5000.0 + 9800.0 + 5.0 + 9.8)
    # Avg entry = (50000×0.1 + 49000×0.2) / 0.3 = (5000+9800)/0.3 = 49333.33
    assert pos.average_entry == pytest.approx(49333.333, rel=1e-3)


def test_position_realised_and_unrealised_pnl():
    pos = Position(cycle_id="c1", symbol="BTC/USDT")
    pos.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    assert pos.unrealised_pnl(51000.0) == pytest.approx(100.0 - 5.0)
    # Realised if closed at 51000 with no exit fee: qty*price - total_cost
    assert pos.realised_pnl(51000.0, exit_fee=0.0) == pytest.approx(100.0 - 5.0)


def test_position_tracker_open_and_close():
    tracker = PositionTracker()
    pos = Position(cycle_id="c1", symbol="BTC/USDT")
    pos.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    tracker.open_position(pos)
    assert tracker.count() == 1
    assert tracker.has_position("c1")
    closed = tracker.close_position("c1")
    assert closed.cycle_id == "c1"
    assert tracker.count() == 0


def test_position_tracker_add_layer_to_existing():
    tracker = PositionTracker()
    pos = Position(cycle_id="c1", symbol="BTC/USDT")
    pos.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    tracker.open_position(pos)
    tracker.add_layer("c1", price=49000.0, qty=0.2, fee=9.8)
    p = tracker.get_position("c1")
    assert p is not None
    assert len(p.layers) == 2


def test_position_tracker_rejects_duplicate_open():
    tracker = PositionTracker()
    pos1 = Position(cycle_id="c1", symbol="BTC/USDT")
    pos1.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    tracker.open_position(pos1)
    pos2 = Position(cycle_id="c1", symbol="BTC/USDT")
    with pytest.raises(ValueError, match="already open"):
        tracker.open_position(pos2)


# ============================================================
# OrderManager tests
# ============================================================

def test_order_manager_open_decision(default_order_manager: OrderManager):
    d = default_order_manager.decide(
        cycle_id="c1", current_price=50000.0, current_time="t",
        position_layers=0, average_entry=0.0, has_open_position=False,
        stake_amount=100.0,
    )
    assert d.action == OrderAction.OPEN_CYCLE
    assert d.price == 50000.0
    assert d.qty == pytest.approx(0.002)


def test_order_manager_close_on_tp(default_order_manager: OrderManager):
    d = default_order_manager.decide(
        cycle_id="c1", current_price=51000.0, current_time="t",
        position_layers=1, average_entry=50000.0, has_open_position=True,
        stake_amount=100.0,
    )
    assert d.action == OrderAction.CLOSE_CYCLE
    assert "tp" in d.reason


def test_order_manager_no_action_when_no_trigger(default_order_manager: OrderManager):
    d = default_order_manager.decide(
        cycle_id="c1", current_price=50100.0, current_time="t",  # 0.2% above entry
        position_layers=1, average_entry=50000.0, has_open_position=True,
        stake_amount=100.0,
    )
    assert d.action == OrderAction.NONE


def test_order_manager_add_layer_when_price_drops(default_order_manager: OrderManager):
    # With 1 layer filled at 50000, layer 2 target = 50000 * (1 - 0.015*1) = 49250
    d = default_order_manager.decide(
        cycle_id="c1", current_price=49000.0, current_time="t",
        position_layers=1, average_entry=50000.0, has_open_position=True,
        stake_amount=100.0,
    )
    assert d.action == OrderAction.ADD_LAYER
    assert d.reason == "layer_2_triggered"


def test_order_manager_max_layers_respected(default_order_manager: OrderManager):
    d = default_order_manager.decide(
        cycle_id="c1", current_price=40000.0, current_time="t",
        position_layers=3, average_entry=50000.0, has_open_position=True,
        stake_amount=100.0,
    )
    assert d.action == OrderAction.NONE
    assert d.reason == "max_layers_reached"


def test_order_manager_rejects_bad_params():
    with pytest.raises(ValueError, match="grid_pct"):
        OrderManager(grid_pct=0, tp_pct=0.02, max_layers=3)
    with pytest.raises(ValueError, match="tp_pct"):
        OrderManager(grid_pct=0.015, tp_pct=0, max_layers=3)
    with pytest.raises(ValueError, match="max_layers"):
        OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=0)


# ============================================================
# CycleLifecycle tests
# ============================================================

def test_cycle_lifecycle_open_add_close():
    cl = CycleLifecycle(cycle_id="c1", symbol="BTC/USDT")
    assert cl.cycle.state == CycleState.PENDING
    cl.open(price=50000.0, qty=0.1, fee=5.0, opened_at="2024-01-01T00:00:00")
    assert cl.cycle.state == CycleState.ACTIVE
    cl.add_layer(price=49000.0, qty=0.2, fee=9.8)
    assert len(cl.cycle.position.layers) == 2
    pnl = cl.close(exit_price=50500.0, exit_fee=10.0, closed_at="2024-01-02T00:00:00", reason="tp")
    # qty*exit - total_cost - exit_fee = 0.3 * 50500 - (5000+9800+5+9.8) - 10 = 15150 - 14824.8 = 325.2
    assert pnl == pytest.approx(325.2, rel=1e-3)
    assert cl.cycle.state == CycleState.CLOSED
    assert cl.cycle.close_reason == "tp"


def test_cycle_lifecycle_rejects_invalid_state():
    cl = CycleLifecycle(cycle_id="c1")
    with pytest.raises(ValueError, match="Cannot add layer"):
        cl.add_layer(price=49000.0, qty=0.1, fee=0.0)  # not yet open
    cl.open(price=50000.0, qty=0.1, fee=5.0, opened_at="t")
    cl.close(exit_price=50500.0, exit_fee=0.0, closed_at="t2", reason="tp")
    with pytest.raises(ValueError, match="Cannot close"):
        cl.close(exit_price=50500.0, exit_fee=0.0, closed_at="t3", reason="tp")


def test_cycle_to_trade_record():
    cl = CycleLifecycle(cycle_id="c1", symbol="BTC/USDT")
    cl.open(price=50000.0, qty=0.1, fee=5.0, opened_at="2024-01-01T00:00:00")
    cl.add_layer(price=49000.0, qty=0.2, fee=9.8)
    cl.close(exit_price=50500.0, exit_fee=10.0, closed_at="2024-01-02T00:00:00", reason="tp")
    rec = cl.to_trade_record()
    assert rec["cycle_id"] == "c1"
    assert rec["symbol"] == "BTC/USDT"
    assert rec["n_layers"] == 2
    assert rec["close_reason"] == "tp"
    assert rec["pnl"] == pytest.approx(325.2, rel=1e-3)


# ============================================================
# ExposureTracker tests
# ============================================================

def test_exposure_tracker_initial_snapshot():
    et = ExposureTracker(initial_deposit=10000.0)
    tracker = PositionTracker()
    snap = et.snapshot(tracker, current_price=50000.0)
    assert snap.n_active_cycles == 0
    assert snap.gross_exposure == 0.0
    assert snap.account_equity == 10000.0
    assert snap.free_margin == 10000.0


def test_exposure_tracker_with_position():
    et = ExposureTracker(initial_deposit=10000.0)
    tracker = PositionTracker()
    pos = Position(cycle_id="c1", symbol="BTC/USDT")
    pos.layers.append({"price": 50000.0, "qty": 0.1, "fee": 5.0})
    tracker.open_position(pos)
    snap = et.snapshot(tracker, current_price=51000.0)
    assert snap.n_active_cycles == 1
    # gross_exposure = 50000 * 0.1 = 5000
    assert snap.gross_exposure == pytest.approx(5000.0)
    # unrealised = 0.1 * 51000 - (5000 + 5) = 5100 - 5005 = 95
    assert snap.unrealised_pnl == pytest.approx(95.0)
    # account_equity = 10000 + 95 = 10095
    assert snap.account_equity == pytest.approx(10095.0)


# ============================================================
# State machine tests
# ============================================================

def test_state_machine_runs_on_synthetic_data(small_ohlcv: pd.DataFrame, default_order_manager: OrderManager):
    result = run_state_machine(small_ohlcv, default_order_manager, initial_deposit=10000.0, stake_amount=100.0)
    assert len(result.equity_curve) == len(small_ohlcv)
    assert result.n_cycles_opened >= 1
    assert result.n_cycles_closed >= 1
    assert result.peak_equity >= result.trough_equity


def test_state_machine_rising_ohlcv_closes_via_tp(rising_ohlcv: pd.DataFrame):
    """Steadily rising prices → cycle opens early, hits TP, closes."""
    om = OrderManager(grid_pct=0.02, tp_pct=0.005, max_layers=3)  # 0.5% TP for fast trigger
    result = run_state_machine(rising_ohlcv, om, initial_deposit=10000.0, stake_amount=100.0)
    assert result.n_cycles_opened >= 1
    # At least one cycle should have closed via TP (rising prices)
    tp_closes = [t for t in result.trades if "tp" in t["close_reason"]]
    assert len(tp_closes) >= 1


def test_state_machine_rejects_empty():
    om = OrderManager(grid_pct=0.015, tp_pct=0.02, max_layers=3)
    with pytest.raises(ValueError, match="Empty"):
        run_state_machine(pd.DataFrame(), om)


# ============================================================
# BacktestRunner tests
# ============================================================

def test_backtest_candidate_returns_stage5_compatible(small_ohlcv: pd.DataFrame):
    result = backtest_candidate(
        df=small_ohlcv,
        candidate_id="test_001",
        genome_id="00000000-0000-4000-8000-000000000001",
        experiment_id="20240101_000000_gen0_test",
        grid_pct=0.015,
        tp_pct=0.02,
        max_layers=3,
        initial_deposit=10000.0,
        stake_amount=100.0,
    )
    assert isinstance(result, BacktestResult)
    # Stage 5 accepts equity_curve (pd.Series) + trades_df (pd.DataFrame with pnl column)
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.trades_df, pd.DataFrame)
    if len(result.trades_df) > 0:
        assert "pnl" in result.trades_df.columns
        assert "open_time" in result.trades_df.columns


def test_backtest_meta_indicates_placeholder_sizing(small_ohlcv: pd.DataFrame):
    result = backtest_candidate(
        df=small_ohlcv,
        candidate_id="test_002",
        genome_id="00000000-0000-4000-8000-000000000002",
        experiment_id="20240101_000000_gen0_test",
    )
    assert result.backtest_meta["is_placeholder_sizing"] is True
    assert result.backtest_meta["stage"] == 3


def test_backtest_to_dict_serialisable(small_ohlcv: pd.DataFrame):
    import json
    result = backtest_candidate(
        df=small_ohlcv,
        candidate_id="test_003",
        genome_id="00000000-0000-4000-8000-000000000003",
        experiment_id="20240101_000000_gen0_test",
    )
    d = result.to_dict()
    j = json.dumps(d, default=str)
    parsed = json.loads(j)
    assert parsed["candidate_id"] == "test_003"
    assert parsed["n_cycles_opened"] >= 1


# ============================================================
# End-to-end: real 5y data → Stage 5
# ============================================================

def test_real_5y_data_flows_to_stage5():
    """Real BTC H1 5y data through backtest → Stage 5 reward engine (no exception)."""
    from pathlib import Path
    feather_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "btc_h1_5y.feather"
    if not feather_path.exists():
        pytest.skip("Real 5y data not present (run data/pipeline/loader.py to generate)")
    df = pd.read_feather(feather_path)
    # Use a 1-year slice for speed (~8800 candles)
    df_1y = df.iloc[-24 * 365:].reset_index(drop=True)
    result = backtest_candidate(
        df=df_1y,
        candidate_id="real_1y_test",
        genome_id="00000000-0000-4000-8000-000000000099",
        experiment_id="20250101_000000_gen0_real1y",
        grid_pct=0.015,
        tp_pct=0.02,
        max_layers=3,
    )
    # Feed to Stage 5
    score = compute_score(result.equity_curve, result.trades_df, candidate_id="real_1y_test")
    # Either scored or rejected — both are valid
    assert isinstance(score, (ScoreResult, RejectedResult))
    if isinstance(score, ScoreResult):
        assert 0.0 <= score.breakdown.final_score <= 1.0
    else:
        assert score.reason in ("net_profit<=0", "drawdown>35%", "tpm<5", "invalid_data", "too_few_trades")


def test_full_5y_backtest_runs_under_5_seconds():
    """5y × 44k candles must complete in <5s for Stage 10 evolution feasibility."""
    from pathlib import Path
    feather_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "btc_h1_5y.feather"
    if not feather_path.exists():
        pytest.skip("Real 5y data not present")
    import time
    df = pd.read_feather(feather_path)
    t0 = time.time()
    result = backtest_candidate(
        df=df,
        candidate_id="perf_test",
        genome_id="00000000-0000-4000-8000-0000000000ff",
        experiment_id="20210601_000000_gen0_perf",
    )
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Backtest took {elapsed:.1f}s, must be < 5s for evolution feasibility"
    assert result.n_cycles_opened >= 1
