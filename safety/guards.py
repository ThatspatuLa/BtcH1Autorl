"""Safety guards — independent checks the Safety Engine composes.

Each guard is a pure function: (context) -> SafetyVerdict. Composable.
The SafetyEngine in safety_engine.py calls them in order and short-circuits
on the first HARD_BLOCK.

Hard semantics:
- HARD_BLOCK: order denied, strategy halts (drawdown breach, margin breach)
- SOFT_BLOCK: order denied, but strategy continues (DCA completion, overlap)
- ALLOW: order proceeds
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DCASafetyDecision(StrEnum):
    """The runtime decision the SafetyEngine returns."""
    ALLOW = "allow"               # order proceeds
    SOFT_BLOCK = "soft_block"     # order denied this candle, retry next
    HARD_BLOCK = "hard_block"     # order denied permanently for this run


@dataclass
class SafetyVerdict:
    """Result of a single guard check."""
    decision: DCASafetyDecision
    guard_name: str
    reason: str
    details: dict[str, Any] | None = None

    @property
    def is_allowed(self) -> bool:
        return self.decision == DCASafetyDecision.ALLOW

    @property
    def is_hard_block(self) -> bool:
        return self.decision == DCASafetyDecision.HARD_BLOCK


@dataclass
class SafetyContext:
    """All state the guards need to make a decision.

    Built by the DCA engine at each candle. Snapshot of:
    - Account state: equity, free_margin, margin_in_use, peak_equity
    - Active cycle state: count, current_pnl, total_unrealised_pnl
    - Per-layer decision: what the order manager WANTS to do
    - Thresholds: from the genome's safety_genome section
    """
    # Account state
    current_equity: float
    free_margin: float
    margin_in_use: float
    peak_equity: float
    starting_equity: float
    # Active cycle state
    n_active_cycles: int
    active_cycle_unrealised_pnl: float   # sum across all active cycles
    proposed_order_size: float           # qty * price the order manager wants
    proposed_order_type: str             # "OPEN_CYCLE", "ADD_LAYER", "CLOSE_CYCLE"
    # Stress test state
    stress_test_decline: float = 0.0     # e.g. 0.77 for BTC 2022 (peak-to-trough)
    # Thresholds (from safety_genome)
    max_dd_pct: float = 0.35             # hard halt at this drawdown
    buffer_pct: float = 0.20             # default 20% free-capital buffer
    allow_overlap_when_break_even: bool = True
    # Latency
    candle_index: int = 0


# ============================================================
# Individual guards
# ============================================================

def check_margin(ctx: SafetyContext) -> SafetyVerdict:
    """Margin guard: never let margin_in_use exceed free_margin.

    Computes projected margin after the order executes. If projected >
    free_margin, the order is denied.
    """
    if ctx.proposed_order_type == "CLOSE_CYCLE":
        # Closing reduces exposure, never blocked by margin
        return SafetyVerdict(DCASafetyDecision.ALLOW, "margin", "close_cycle_no_margin_check")

    if ctx.proposed_order_size <= 0:
        return SafetyVerdict(DCASafetyDecision.ALLOW, "margin", "zero_size")

    projected_margin = ctx.margin_in_use + ctx.proposed_order_size
    if projected_margin > ctx.free_margin:
        return SafetyVerdict(
            DCASafetyDecision.SOFT_BLOCK,
            "margin",
            f"projected_margin={projected_margin:.2f} > free_margin={ctx.free_margin:.2f}",
            {"projected_margin": projected_margin, "free_margin": ctx.free_margin},
        )
    return SafetyVerdict(DCASafetyDecision.ALLOW, "margin", "margin_ok")


def check_drawdown(ctx: SafetyContext) -> SafetyVerdict:
    """Drawdown guard: halt new cycles if equity drawdown exceeds threshold.

    Only applies to OPEN_CYCLE / ADD_LAYER (not CLOSE_CYCLE — we always
    want to be able to close losing positions).

    HARD_BLOCK when drawdown exceeds max_dd_pct (e.g. 0.35 = 35%).
    """
    if ctx.proposed_order_type == "CLOSE_CYCLE":
        return SafetyVerdict(DCASafetyDecision.ALLOW, "drawdown", "close_cycle_no_dd_check")

    if ctx.peak_equity <= 0:
        return SafetyVerdict(DCASafetyDecision.ALLOW, "drawdown", "no_peak_yet")

    current_dd = (ctx.peak_equity - ctx.current_equity) / ctx.peak_equity
    if current_dd > ctx.max_dd_pct:
        return SafetyVerdict(
            DCASafetyDecision.HARD_BLOCK,
            "drawdown",
            f"drawdown={current_dd:.2%} > max_dd_pct={ctx.max_dd_pct:.2%}",
            {"current_dd": current_dd, "max_dd_pct": ctx.max_dd_pct},
        )
    return SafetyVerdict(DCASafetyDecision.ALLOW, "drawdown", "dd_within_bounds")


def check_dca_completion(ctx: SafetyContext) -> SafetyVerdict:
    """DCA completion guard: stop adding layers when buffer is exhausted.

    The buffer is a fraction of free_margin. Once free_margin drops below
    (1 - buffer_pct) of starting_equity, ADD_LAYER is blocked.

    OPEN_CYCLE is allowed as long as free_margin > 0 (each new cycle uses
    a small base_stake, not the full buffer).
    """
    if ctx.proposed_order_type == "CLOSE_CYCLE":
        return SafetyVerdict(DCASafetyDecision.ALLOW, "dca_completion", "close_cycle_no_buffer_check")

    if ctx.proposed_order_type == "OPEN_CYCLE":
        # Only block if essentially no capital left
        if ctx.free_margin <= 0:
            return SafetyVerdict(
                DCASafetyDecision.SOFT_BLOCK,
                "dca_completion",
                f"free_margin={ctx.free_margin:.2f} <= 0",
            )
        return SafetyVerdict(DCASafetyDecision.ALLOW, "dca_completion", "open_cycle_buffer_ok")

    # ADD_LAYER: stricter check — buffer must be intact
    buffer_floor = ctx.starting_equity * (1.0 - ctx.buffer_pct)
    if ctx.current_equity < buffer_floor:
        return SafetyVerdict(
            DCASafetyDecision.SOFT_BLOCK,
            "dca_completion",
            f"equity={ctx.current_equity:.2f} < buffer_floor={buffer_floor:.2f}",
            {"current_equity": ctx.current_equity, "buffer_floor": buffer_floor},
        )
    return SafetyVerdict(DCASafetyDecision.ALLOW, "dca_completion", "buffer_intact")


def check_overlap(ctx: SafetyContext) -> SafetyVerdict:
    """Overlap guard: only allow new cycle when existing is break-even+ AND
    all active cycles can be funded under the stress test.

    Per locked decision: overlap allowed only when existing cycle is break-
    even or better AND all active cycles can be funded under stress.

    Stress = worst peak-to-trough decline in dataset. If active cycle
    unrealised P&L + stress_decline * margin_in_use > free_margin, we
    can't fund all active cycles under stress → no new cycle.
    """
    if ctx.proposed_order_type != "OPEN_CYCLE":
        return SafetyVerdict(DCASafetyDecision.ALLOW, "overlap", "not_opening_new_cycle")

    if ctx.n_active_cycles == 0:
        return SafetyVerdict(DCASafetyDecision.ALLOW, "overlap", "no_active_cycles")

    if not ctx.allow_overlap_when_break_even and ctx.n_active_cycles > 0:
        return SafetyVerdict(
            DCASafetyDecision.SOFT_BLOCK,
            "overlap",
            "overlap_disabled_in_genome",
        )

    # Existing cycle must be break-even or better
    if ctx.active_cycle_unrealised_pnl < 0:
        return SafetyVerdict(
            DCASafetyDecision.SOFT_BLOCK,
            "overlap",
            f"existing_cycle_unrealised_pnl={ctx.active_cycle_unrealised_pnl:.2f} < 0 (not break-even)",
        )

    # All active cycles can be funded under stress
    # Stress impact: if BTC drops by stress_test_decline, active margin
    # requirements could increase (margin call risk). Conservative: assume
    # margin_in_use grows by stress_test_decline fraction.
    stressed_margin = ctx.margin_in_use * (1.0 + ctx.stress_test_decline)
    if stressed_margin > ctx.free_margin:
        return SafetyVerdict(
            DCASafetyDecision.SOFT_BLOCK,
            "overlap",
            f"stressed_margin={stressed_margin:.2f} > free_margin={ctx.free_margin:.2f}",
            {"stressed_margin": stressed_margin, "free_margin": ctx.free_margin},
        )
    return SafetyVerdict(DCASafetyDecision.ALLOW, "overlap", "overlap_safe")
