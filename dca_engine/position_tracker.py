"""Position tracker — tracks open positions, cost basis, layers, current mark-to-market value.

Stage 3 placeholder: each layer adds qty at a price. Average entry = sum(qty × price) / sum(qty).
Stage 8 will replace placeholder sizing with real grid/allocation/combo logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    """A single position in one DCA cycle."""
    cycle_id: str
    symbol: str  # e.g. "BTC/USDT"
    layers: list[dict[str, float]] = field(default_factory=list)  # each = {price, qty, fee}
    opened_at: str = ""
    closed_at: str = ""

    @property
    def total_qty(self) -> float:
        return sum(layer["qty"] for layer in self.layers)

    @property
    def total_cost(self) -> float:
        return sum(layer["price"] * layer["qty"] + layer["fee"] for layer in self.layers)

    @property
    def average_entry(self) -> float:
        qty = self.total_qty
        if qty <= 0:
            return 0.0
        return sum(layer["price"] * layer["qty"] for layer in self.layers) / qty

    def unrealised_pnl(self, current_price: float) -> float:
        """Unrealised P&L in stake currency at current_price (mark-to-market)."""
        return self.total_qty * current_price - self.total_cost

    def realised_pnl(self, exit_price: float, exit_fee: float) -> float:
        """Realised P&L if closed at exit_price (after exit_fee)."""
        return self.total_qty * exit_price - self.total_cost - exit_fee

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "symbol": self.symbol,
            "layers": list(self.layers),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "total_qty": self.total_qty,
            "total_cost": self.total_cost,
            "average_entry": self.average_entry,
        }


class PositionTracker:
    """Tracks all open positions across cycles.

    Stage 3 is a simple in-memory tracker: add_position, get_open_positions,
    close_position. Stage 7 (safety engine) will add margin/free-margin queries.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def open_position(self, position: Position) -> None:
        """Register a new open position."""
        if position.cycle_id in self._positions:
            raise ValueError(f"Position for cycle {position.cycle_id} already open")
        self._positions[position.cycle_id] = position

    def add_layer(self, cycle_id: str, price: float, qty: float, fee: float) -> None:
        """Add a new DCA layer to an existing position."""
        pos = self._positions.get(cycle_id)
        if pos is None:
            raise KeyError(f"No open position for cycle {cycle_id}")
        pos.layers.append({"price": price, "qty": qty, "fee": fee})

    def close_position(self, cycle_id: str) -> Position:
        """Remove and return a closed position."""
        pos = self._positions.pop(cycle_id, None)
        if pos is None:
            raise KeyError(f"No open position for cycle {cycle_id}")
        return pos

    def get_position(self, cycle_id: str) -> Position | None:
        return self._positions.get(cycle_id)

    def has_position(self, cycle_id: str) -> bool:
        return cycle_id in self._positions

    def open_position_ids(self) -> list[str]:
        return list(self._positions.keys())

    def count(self) -> int:
        return len(self._positions)

    def total_unrealised_pnl(self, current_price: float) -> float:
        return sum(p.unrealised_pnl(current_price) for p in self._positions.values())

    def reset(self) -> None:
        self._positions.clear()
