"""Cycle lifecycle — start, fill layers, close, with per-cycle P&L tracking.

Stage 3 placeholder:
- CycleState: PENDING → ACTIVE → CLOSED
- Each cycle tracks layers, total cost, P&L
- close_cycle() returns a closed Trade record for the trades_df output

Stage 8 will plug in real layer-sizing logic (allocation method).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from dca_engine.position_tracker import Position


class CycleState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"


@dataclass
class Cycle:
    cycle_id: str
    symbol: str
    state: CycleState = CycleState.PENDING
    opened_at: str = ""
    closed_at: str = ""
    close_reason: str = ""
    pnl: float = 0.0  # realised P&L after close
    position: Position = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = Position(cycle_id=self.cycle_id, symbol=self.symbol)


class CycleLifecycle:
    """Manages the lifecycle of a single cycle.

    Stage 3: simple state transitions + per-cycle P&L computation.
    """

    def __init__(self, cycle_id: str, symbol: str = "BTC/USDT") -> None:
        self.cycle = Cycle(cycle_id=cycle_id, symbol=symbol)

    def open(self, price: float, qty: float, fee: float, opened_at: str) -> None:
        """Open the cycle with the initial layer."""
        if self.cycle.state != CycleState.PENDING:
            raise ValueError(f"Cannot open cycle in state {self.cycle.state}")
        self.cycle.opened_at = opened_at
        self.cycle.position.opened_at = opened_at
        self.cycle.position.layers.append({"price": price, "qty": qty, "fee": fee})
        self.cycle.state = CycleState.ACTIVE

    def add_layer(self, price: float, qty: float, fee: float) -> None:
        """Add a DCA layer to the active cycle."""
        if self.cycle.state != CycleState.ACTIVE:
            raise ValueError(f"Cannot add layer in state {self.cycle.state}")
        self.cycle.position.layers.append({"price": price, "qty": qty, "fee": fee})

    def close(self, exit_price: float, exit_fee: float, closed_at: str, reason: str) -> float:
        """Close the cycle, return realised P&L."""
        if self.cycle.state != CycleState.ACTIVE:
            raise ValueError(f"Cannot close cycle in state {self.cycle.state}")
        pnl = self.cycle.position.realised_pnl(exit_price, exit_fee)
        self.cycle.pnl = pnl
        self.cycle.closed_at = closed_at
        self.cycle.close_reason = reason
        self.cycle.position.closed_at = closed_at
        self.cycle.state = CycleState.CLOSED
        return pnl

    def to_trade_record(self) -> dict:
        """Convert to a trade dict matching Stage 5 trades_df schema."""
        pos = self.cycle.position
        return {
            "open_time": self.cycle.opened_at,
            "close_time": self.cycle.closed_at,
            "cycle_id": self.cycle.cycle_id,
            "symbol": pos.symbol,
            "qty": pos.total_qty,
            "avg_entry": pos.average_entry,
            "exit_price": pos.layers[-1]["price"] if pos.layers else 0.0,  # not exact; real exit lives elsewhere
            "pnl": self.cycle.pnl,
            "close_reason": self.cycle.close_reason,
            "n_layers": len(pos.layers),
        }
