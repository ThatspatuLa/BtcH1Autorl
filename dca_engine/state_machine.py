"""State machine — orchestrates backtest loop candle-by-candle.

Stage 3 placeholder flow:
1. For each candle: order_manager.decide() → execute decision
2. Track open positions, equity, exposure
3. When cycle closes, record trade, update equity

Stage 10 (DCA evolution) calls run_state_machine() in the inner loop of evolution.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from dca_engine.cycle_lifecycle import CycleLifecycle
from dca_engine.exposure_tracker import ExposureTracker
from dca_engine.order_manager import OrderAction, OrderManager
from dca_engine.position_tracker import PositionTracker


@dataclass
class StateMachineResult:
    """Result of running the state machine through all candles."""
    equity_curve: pd.Series
    trades: list[dict]
    n_cycles_opened: int
    n_cycles_closed: int
    final_equity: float
    peak_equity: float
    trough_equity: float
    gross_exposure: list[float] = field(default_factory=list)
    free_margin: list[float] = field(default_factory=list)


def _next_cycle_id() -> str:
    return f"c_{uuid.uuid4().hex[:8]}"


def _fee_for_notional(notional: float, fee_pct: float) -> float:
    return abs(notional) * fee_pct


def run_state_machine(
    df: pd.DataFrame,
    order_manager: OrderManager,
    initial_deposit: float = 10000.0,
    stake_amount: float = 100.0,
    fee_pct: float = 0.001,  # 0.1% round-trip
) -> StateMachineResult:
    """Iterate over the candles, running the state machine.

    Args:
        df: OHLCV dataframe (Stage 2 format: date, open, high, low, close, volume)
        order_manager: configured OrderManager
        initial_deposit: starting account equity
        stake_amount: USDT per layer / per cycle
        fee_pct: round-trip fee as fraction of notional

    Returns:
        StateMachineResult with equity_curve, trades, exposure snapshots
    """
    if df.empty:
        raise ValueError("Empty dataframe")

    position_tracker = PositionTracker()
    exposure = ExposureTracker(initial_deposit=initial_deposit)
    closed_trades: list[dict] = []
    active_cycles: dict[str, CycleLifecycle] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    gross_exposure_series: list[float] = []
    free_margin_series: list[float] = []
    n_opened = 0
    n_closed = 0
    peak_equity = initial_deposit
    trough_equity = initial_deposit

    for _idx, row in df.iterrows():
        current_price = float(row["close"])
        current_time = pd.Timestamp(row["date"]).isoformat()

        # Snapshot exposure at start of candle
        snap = exposure.snapshot(position_tracker, current_price)
        gross_exposure_series.append(snap.gross_exposure)
        free_margin_series.append(snap.free_margin)

        # 1) If we have an active cycle, check CLOSE first (TP priority)
        for cid in list(active_cycles.keys()):
            cycle = active_cycles[cid]
            pos = position_tracker.get_position(cid)
            if pos is None:
                continue
            decision = order_manager.decide(
                cycle_id=cid,
                current_price=current_price,
                current_time=current_time,
                position_layers=len(pos.layers),
                average_entry=pos.average_entry,
                has_open_position=True,
                stake_amount=stake_amount,
            )
            if decision.action == OrderAction.CLOSE_CYCLE:
                # Close at current_price
                exit_fee = _fee_for_notional(pos.total_qty * current_price, fee_pct)
                pnl = cycle.close(
                    exit_price=current_price,
                    exit_fee=exit_fee,
                    closed_at=current_time,
                    reason=decision.reason,
                )
                position_tracker.close_position(cid)
                # Realise P&L into account equity
                exposure.update_equity(exposure._current_equity + pnl)
                closed_trades.append(cycle.to_trade_record())
                del active_cycles[cid]
                n_closed += 1

        # 2) Decide on new OPEN or ADD_LAYER (only one decision per candle in Stage 3)
        # Try to open a new cycle if no active ones, OR add a layer to existing
        decision = None
        if not active_cycles:
            # Try OPEN
            decision = order_manager.decide(
                cycle_id=_next_cycle_id(),
                current_price=current_price,
                current_time=current_time,
                position_layers=0,
                average_entry=0.0,
                has_open_position=False,
                stake_amount=stake_amount,
            )
        else:
            # Use first active cycle
            cid = next(iter(active_cycles.keys()))
            pos = position_tracker.get_position(cid)
            if pos is not None and len(pos.layers) < order_manager.max_layers:
                decision = order_manager.decide(
                    cycle_id=cid,
                    current_price=current_price,
                    current_time=current_time,
                    position_layers=len(pos.layers),
                    average_entry=pos.average_entry,
                    has_open_position=True,
                    stake_amount=stake_amount,
                )

        if decision is None or decision.action == OrderAction.NONE:
            pass  # no action this candle
        elif decision.action == OrderAction.OPEN_CYCLE:
            cycle_id = decision.cycle_id or _next_cycle_id()
            cycle = CycleLifecycle(cycle_id=cycle_id, symbol=order_manager.symbol)
            entry_fee = _fee_for_notional(decision.price * decision.qty, fee_pct)
            cycle.open(
                price=decision.price,
                qty=decision.qty,
                fee=entry_fee,
                opened_at=current_time,
            )
            position_tracker.open_position(cycle.cycle.position)
            active_cycles[cycle_id] = cycle
            n_opened += 1
        elif decision.action == OrderAction.ADD_LAYER:
            cid = decision.cycle_id
            if cid in active_cycles:
                layer_fee = _fee_for_notional(decision.price * decision.qty, fee_pct)
                active_cycles[cid].add_layer(price=decision.price, qty=decision.qty, fee=layer_fee)
                position_tracker.add_layer(cid, decision.price, decision.qty, layer_fee)

        # 3) Mark-to-market equity curve
        mtm = exposure.snapshot(position_tracker, current_price)
        equity_points.append((row["date"], mtm.account_equity))
        peak_equity = max(peak_equity, mtm.account_equity)
        trough_equity = min(trough_equity, mtm.account_equity)

    # Build equity curve
    if equity_points:
        times, eqs = zip(*equity_points, strict=True)
        equity_curve = pd.Series(eqs, index=pd.DatetimeIndex(times), name="equity")
    else:
        equity_curve = pd.Series(dtype=float)

    # Force-close any remaining open cycles at the last close price (mark-to-market exit)
    if position_tracker.count() > 0:
        last_close = float(df["close"].iloc[-1])
        last_time = pd.Timestamp(df["date"].iloc[-1]).isoformat()
        for cid in list(active_cycles.keys()):
            cycle = active_cycles[cid]
            pos = position_tracker.get_position(cid)
            if pos is None:
                continue
            exit_fee = _fee_for_notional(pos.total_qty * last_close, fee_pct)
            pnl = cycle.close(exit_price=last_close, exit_fee=exit_fee, closed_at=last_time, reason="backtest_end")
            position_tracker.close_position(cid)
            exposure.update_equity(exposure._current_equity + pnl)
            closed_trades.append(cycle.to_trade_record())
            del active_cycles[cid]
            n_closed += 1

    return StateMachineResult(
        equity_curve=equity_curve,
        trades=closed_trades,
        n_cycles_opened=n_opened,
        n_cycles_closed=n_closed,
        final_equity=exposure._current_equity,
        peak_equity=peak_equity,
        trough_equity=trough_equity,
        gross_exposure=gross_exposure_series,
        free_margin=free_margin_series,
    )
