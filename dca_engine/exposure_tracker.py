"""Exposure tracker — margin used across active cycles, free margin.

Stage 3 placeholder: simple sum of (avg_entry × qty) for open positions.
Stage 7 (safety engine) will replace this with real margin_model.py logic (CFD funding,
maintenance margin, buffer_pct configurable).
"""
from __future__ import annotations

from dataclasses import dataclass

from dca_engine.position_tracker import PositionTracker


@dataclass
class ExposureSnapshot:
    """Snapshot of margin usage at a point in time."""
    n_active_cycles: int
    gross_exposure: float  # sum of (avg_entry × total_qty) for open positions
    unrealised_pnl: float
    free_margin: float
    account_equity: float


class ExposureTracker:
    """Track margin usage across all active cycles.

    Stage 3 placeholder: Stage 7 swaps in real margin_model with funding + buffer.
    """

    def __init__(self, initial_deposit: float = 10000.0) -> None:
        self.initial_deposit = initial_deposit
        self._current_equity = initial_deposit

    def snapshot(
        self,
        position_tracker: PositionTracker,
        current_price: float,
    ) -> ExposureSnapshot:
        """Compute exposure snapshot for the current candle."""
        positions = [position_tracker.get_position(cid) for cid in position_tracker.open_position_ids()]
        positions = [p for p in positions if p is not None]
        gross = sum(p.average_entry * p.total_qty for p in positions)
        unrealised = sum(p.unrealised_pnl(current_price) for p in positions)
        return ExposureSnapshot(
            n_active_cycles=len(positions),
            gross_exposure=gross,
            unrealised_pnl=unrealised,
            free_margin=self._current_equity + unrealised - gross,
            account_equity=self._current_equity + unrealised,
        )

    def update_equity(self, new_equity: float) -> None:
        """Update current equity (after cycle close P&L)."""
        self._current_equity = new_equity

    def reset(self) -> None:
        self._current_equity = self.initial_deposit
