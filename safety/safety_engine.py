"""SafetyEngine — composes the four guards into a single decision."""
from __future__ import annotations

from dataclasses import dataclass

from .guards import (
    DCASafetyDecision,
    SafetyContext,
    SafetyVerdict,
    check_dca_completion,
    check_drawdown,
    check_margin,
    check_overlap,
)


@dataclass
class SafetyThresholds:
    """Configurable thresholds from the genome's safety_genome section.

    These are the default values from the locked Kanban. Per-experiment
    overrides come from configs/experiments/<slug>.json. The buffer_pct
    is configurable per Kanban rule (NOT hardcoded).
    """
    max_dd_pct: float = 0.35
    buffer_pct: float = 0.20           # default fallback per Kanban
    allow_overlap_when_break_even: bool = True
    stress_test_decline: float = 0.772  # BTC 2021-11 to 2022-11 = -77.2%


class SafetyEngine:
    """The single entry point the DCA engine uses to gate every order.

    Runs all 4 guards in order: margin → drawdown → dca_completion → overlap.
    Returns the first non-ALLOW verdict (short-circuit). If all allow, returns ALLOW.
    """
    def __init__(self, thresholds: SafetyThresholds | None = None):
        self.thresholds = thresholds or SafetyThresholds()

    def evaluate(self, ctx: SafetyContext) -> SafetyVerdict:
        # Merge thresholds into context
        merged = SafetyContext(
            current_equity=ctx.current_equity,
            free_margin=ctx.free_margin,
            margin_in_use=ctx.margin_in_use,
            peak_equity=ctx.peak_equity,
            starting_equity=ctx.starting_equity,
            n_active_cycles=ctx.n_active_cycles,
            active_cycle_unrealised_pnl=ctx.active_cycle_unrealised_pnl,
            proposed_order_size=ctx.proposed_order_size,
            proposed_order_type=ctx.proposed_order_type,
            stress_test_decline=ctx.stress_test_decline or self.thresholds.stress_test_decline,
            max_dd_pct=ctx.max_dd_pct or self.thresholds.max_dd_pct,
            buffer_pct=ctx.buffer_pct or self.thresholds.buffer_pct,
            allow_overlap_when_break_even=ctx.allow_overlap_when_break_even,
            candle_index=ctx.candle_index,
        )

        # Order matters: HARD_BLOCK first (drawdown), then SOFT_BLOCK (margin, dca, overlap)
        for guard_fn in (check_drawdown, check_margin, check_dca_completion, check_overlap):
            verdict = guard_fn(merged)
            if not verdict.is_allowed:
                return verdict
        return SafetyVerdict(
            DCASafetyDecision.ALLOW,
            "all",
            "all_guards_passed",
        )


def evaluate_safety(
    current_equity: float,
    free_margin: float,
    margin_in_use: float,
    peak_equity: float,
    starting_equity: float,
    n_active_cycles: int,
    active_cycle_unrealised_pnl: float,
    proposed_order_size: float,
    proposed_order_type: str,
    thresholds: SafetyThresholds | None = None,
    stress_test_decline: float = 0.0,
    candle_index: int = 0,
) -> SafetyVerdict:
    """Convenience function — build a SafetyContext and evaluate."""
    th = thresholds or SafetyThresholds()
    ctx = SafetyContext(
        current_equity=current_equity,
        free_margin=free_margin,
        margin_in_use=margin_in_use,
        peak_equity=peak_equity,
        starting_equity=starting_equity,
        n_active_cycles=n_active_cycles,
        active_cycle_unrealised_pnl=active_cycle_unrealised_pnl,
        proposed_order_size=proposed_order_size,
        proposed_order_type=proposed_order_type,
        stress_test_decline=stress_test_decline or th.stress_test_decline,
        max_dd_pct=th.max_dd_pct,
        buffer_pct=th.buffer_pct,
        allow_overlap_when_break_even=th.allow_overlap_when_break_even,
        candle_index=candle_index,
    )
    return SafetyEngine(th).evaluate(ctx)
