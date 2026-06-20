"""BTC H1 AutoRL — DCA engine subpackage.

Stage 3 deliverable: DCA engine skeleton + backtest runner.

Components:
- position_tracker: tracks open positions (qty, cost basis, layers, current mark-to-market value)
- order_manager:    decides when to fire DCA layers / close on TP, given genome + price
- cycle_lifecycle:  start, fill layers, close, with per-cycle P&L tracking
- exposure_tracker: margin used across active cycles, free margin
- state_machine:    orchestrates backtest loop candle-by-candle
- backtest:         backtest_candidate() — one candidate through 5y data, returns BacktestResult

Stage 3 uses PLACEHOLDER sizing: each layer = (current_price x fixed_pct_spacing), and
TP = fixed_pct_above_average_entry. Real grid/allocation/combo logic comes in Stage 8.

Integration with Stage 5: BacktestResult has the same shape as the synthetic equity_curve +
trades_df that Stage 5 compute_score() accepts.
"""
from dca_engine.backtest import BacktestResult, backtest_candidate
from dca_engine.cycle_lifecycle import Cycle, CycleLifecycle, CycleState
from dca_engine.exposure_tracker import ExposureTracker
from dca_engine.order_manager import OrderAction, OrderDecision, OrderManager
from dca_engine.position_tracker import Position, PositionTracker
from dca_engine.state_machine import StateMachineResult, run_state_machine

__all__ = [
    "BacktestResult",
    "Cycle",
    "CycleLifecycle",
    "CycleState",
    "ExposureTracker",
    "OrderAction",
    "OrderDecision",
    "OrderManager",
    "Position",
    "PositionTracker",
    "StateMachineResult",
    "backtest_candidate",
    "run_state_machine",
]
