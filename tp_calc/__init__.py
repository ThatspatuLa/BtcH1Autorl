"""Stage 11 — TP/Exit Calculation Library.

Pure-function building blocks for TP / exit decision logic. Stage 12 (TP
Evolution) wires these into the OrderManager; Stage 11 is just the math.

Each function evaluates a single exit method and returns an ExitDecision
indicating whether to close, at what price, and for what fraction of the
position.

Supports the full TpExitMethod enum:
- fixed, atr, vol_adjusted, dca_depth_adjusted, partial_ladder
- trailing, break_even, momentum_decay, exhaustion, trend_hold
- failed_continuation, time_in_position, hybrid
"""
from __future__ import annotations

from .exits import (
    ExitContext,
    ExitDecision,
    compute_exit_decision,
    exit_atr,
    exit_break_even,
    exit_dca_depth_adjusted,
    exit_exhaustion,
    exit_failed_continuation,
    exit_fixed,
    exit_hybrid,
    exit_momentum_decay,
    exit_partial_ladder,
    exit_time_in_position,
    exit_trailing,
    exit_trend_hold,
    exit_vol_adjusted,
)

__all__ = [
    "ExitContext",
    "ExitDecision",
    "compute_exit_decision",
    "exit_atr",
    "exit_break_even",
    "exit_dca_depth_adjusted",
    "exit_exhaustion",
    "exit_failed_continuation",
    "exit_fixed",
    "exit_hybrid",
    "exit_momentum_decay",
    "exit_partial_ladder",
    "exit_time_in_position",
    "exit_trailing",
    "exit_trend_hold",
    "exit_vol_adjusted",
]
