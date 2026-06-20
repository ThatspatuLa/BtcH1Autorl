"""Stage 7 — DCA Completion Safety Engine.

Runtime guards that wrap the DCA engine and prevent it from making unsafe
decisions during a backtest. Three independent guards, each can be toggled
in the genome's safety_genome section:

1. MarginGuard      — never let margin_in_use exceed free_margin
2. DrawdownGuard    — halt new cycles if equity drawdown exceeds threshold
3. DCACompletionGuard — stop adding layers if buffer_pct of free capital is exhausted

Plus an OverlapPolicy — when an existing cycle is being added to, can we
also open new cycles? Per locked decision: yes, ONLY if existing cycle is
break-even or better AND all active cycles can be funded under stress.

Per Kanban: buffer_pct is configurable per experiment. Default 20%, not hardcoded.
"""
from __future__ import annotations

from .guards import (
    DCASafetyDecision,
    SafetyContext,
    SafetyVerdict,
    check_dca_completion,
    check_drawdown,
    check_margin,
    check_overlap,
)
from .safety_engine import SafetyEngine, SafetyThresholds, evaluate_safety

__all__ = [
    "DCASafetyDecision",
    "SafetyContext",
    "SafetyEngine",
    "SafetyThresholds",
    "SafetyVerdict",
    "check_dca_completion",
    "check_drawdown",
    "check_margin",
    "check_overlap",
    "evaluate_safety",
]
