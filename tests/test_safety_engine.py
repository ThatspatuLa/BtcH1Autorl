"""Tests for Stage 7 — Safety Engine."""
from __future__ import annotations

from safety.guards import (
    DCASafetyDecision,
    SafetyContext,
    check_dca_completion,
    check_drawdown,
    check_margin,
    check_overlap,
)
from safety.safety_engine import SafetyEngine, SafetyThresholds, evaluate_safety

# ============================================================
# Test: margin guard
# ============================================================

def test_margin_allows_when_within_bounds():
    ctx = SafetyContext(
        current_equity=10000, free_margin=5000, margin_in_use=1000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=500, proposed_order_type="OPEN_CYCLE",
    )
    v = check_margin(ctx)
    assert v.is_allowed
    assert v.guard_name == "margin"


def test_margin_blocks_when_projected_exceeds_free():
    ctx = SafetyContext(
        current_equity=10000, free_margin=5000, margin_in_use=4800,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=500, proposed_order_type="ADD_LAYER",
    )
    v = check_margin(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK
    assert "projected_margin" in v.reason


def test_margin_allows_close_cycle_always():
    ctx = SafetyContext(
        current_equity=5000, free_margin=0, margin_in_use=5000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=-2000,
        proposed_order_size=5000, proposed_order_type="CLOSE_CYCLE",
    )
    v = check_margin(ctx)
    assert v.is_allowed  # close never blocked by margin


def test_margin_allows_zero_size():
    ctx = SafetyContext(
        current_equity=10000, free_margin=0, margin_in_use=10000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=0, proposed_order_type="OPEN_CYCLE",
    )
    v = check_margin(ctx)
    assert v.is_allowed


# ============================================================
# Test: drawdown guard
# ============================================================

def test_drawdown_allows_within_bounds():
    ctx = SafetyContext(
        current_equity=9000, free_margin=9000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        max_dd_pct=0.35,
    )
    v = check_drawdown(ctx)
    assert v.is_allowed


def test_drawdown_hard_blocks_above_max():
    ctx = SafetyContext(
        current_equity=6000, free_margin=6000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="ADD_LAYER",
        max_dd_pct=0.35,
    )
    v = check_drawdown(ctx)
    assert v.decision == DCASafetyDecision.HARD_BLOCK
    assert "drawdown" in v.reason


def test_drawdown_hard_blocks_at_77_percent():
    """BTC 2022 stress event: 77% drawdown → hard block."""
    ctx = SafetyContext(
        current_equity=2300, free_margin=2300, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        max_dd_pct=0.35,
    )
    v = check_drawdown(ctx)
    assert v.is_hard_block


def test_drawdown_allows_close_always():
    """Even at extreme drawdown, we want to be able to close positions."""
    ctx = SafetyContext(
        current_equity=2000, free_margin=2000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=-8000,
        proposed_order_size=2000, proposed_order_type="CLOSE_CYCLE",
        max_dd_pct=0.35,
    )
    v = check_drawdown(ctx)
    assert v.is_allowed


def test_drawdown_allows_with_no_peak():
    """First candle: no peak established yet."""
    ctx = SafetyContext(
        current_equity=10000, free_margin=10000, margin_in_use=0,
        peak_equity=0, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    v = check_drawdown(ctx)
    assert v.is_allowed


# ============================================================
# Test: DCA completion guard
# ============================================================

def test_dca_completion_allows_open_when_margin_positive():
    ctx = SafetyContext(
        current_equity=10000, free_margin=5000, margin_in_use=5000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        buffer_pct=0.20,
    )
    v = check_dca_completion(ctx)
    assert v.is_allowed


def test_dca_completion_blocks_open_when_no_margin():
    ctx = SafetyContext(
        current_equity=5000, free_margin=0, margin_in_use=5000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    v = check_dca_completion(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


def test_dca_completion_blocks_add_layer_below_buffer():
    """20% buffer means: don't add layers if equity < 80% of starting."""
    ctx = SafetyContext(
        current_equity=7500, free_margin=7500, margin_in_use=0,
        peak_equity=8000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="ADD_LAYER",
        buffer_pct=0.20,  # buffer_floor = 8000
    )
    v = check_dca_completion(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


def test_dca_completion_allows_add_layer_above_buffer():
    ctx = SafetyContext(
        current_equity=8500, free_margin=8500, margin_in_use=0,
        peak_equity=9000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="ADD_LAYER",
        buffer_pct=0.20,  # buffer_floor = 8000
    )
    v = check_dca_completion(ctx)
    assert v.is_allowed


def test_dca_completion_buffer_pct_configurable():
    """buffer_pct=0.50 → only add layers if equity >= 50% of starting."""
    ctx = SafetyContext(
        current_equity=4500, free_margin=4500, margin_in_use=0,
        peak_equity=5000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="ADD_LAYER",
        buffer_pct=0.50,  # buffer_floor = 5000
    )
    v = check_dca_completion(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


# ============================================================
# Test: overlap guard
# ============================================================

def test_overlap_allows_no_active_cycles():
    ctx = SafetyContext(
        current_equity=10000, free_margin=10000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    v = check_overlap(ctx)
    assert v.is_allowed


def test_overlap_blocks_when_active_cycle_negative():
    """Existing cycle at a loss → no new cycle until it closes."""
    ctx = SafetyContext(
        current_equity=9500, free_margin=9500, margin_in_use=500,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=-200,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    v = check_overlap(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


def test_overlap_allows_when_active_cycle_break_even():
    ctx = SafetyContext(
        current_equity=10050, free_margin=10050, margin_in_use=500,
        peak_equity=10050, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=50,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    v = check_overlap(ctx)
    assert v.is_allowed


def test_overlap_blocks_when_cannot_fund_under_stress():
    """If margin_in_use × (1 + stress) > free_margin, no new cycle."""
    ctx = SafetyContext(
        current_equity=10000, free_margin=5000, margin_in_use=4000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        stress_test_decline=0.50,  # stressed_margin = 4000 × 1.5 = 6000 > 5000
    )
    v = check_overlap(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


def test_overlap_allows_when_fundable_under_stress():
    ctx = SafetyContext(
        current_equity=10000, free_margin=8000, margin_in_use=4000,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        stress_test_decline=0.50,  # stressed_margin = 6000 < 8000
    )
    v = check_overlap(ctx)
    assert v.is_allowed


def test_overlap_disabled_in_genome():
    """If genome disables overlap, block all multi-cycle scenarios."""
    ctx = SafetyContext(
        current_equity=10000, free_margin=10000, margin_in_use=100,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=50,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        allow_overlap_when_break_even=False,
    )
    v = check_overlap(ctx)
    assert v.decision == DCASafetyDecision.SOFT_BLOCK


def test_overlap_allows_non_open_orders():
    """ADD_LAYER and CLOSE_CYCLE are not subject to overlap rules."""
    ctx = SafetyContext(
        current_equity=9000, free_margin=9000, margin_in_use=500,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=-500,
        proposed_order_size=100, proposed_order_type="ADD_LAYER",
    )
    v = check_overlap(ctx)
    assert v.is_allowed


# ============================================================
# Test: SafetyEngine composition
# ============================================================

def test_engine_allows_when_all_guards_pass():
    engine = SafetyEngine()
    verdict = engine.evaluate(SafetyContext(
        current_equity=10000, free_margin=10000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    ))
    assert verdict.is_allowed
    assert verdict.guard_name == "all"
    assert verdict.reason == "all_guards_passed"


def test_engine_short_circuits_on_drawdown():
    """If drawdown is breached, the engine should return that first
    (HARD_BLOCK), even if margin would also be exceeded."""
    engine = SafetyEngine()
    verdict = engine.evaluate(SafetyContext(
        current_equity=2000, free_margin=0, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=1000, proposed_order_type="OPEN_CYCLE",
        max_dd_pct=0.35,
    ))
    assert verdict.decision == DCASafetyDecision.HARD_BLOCK
    assert verdict.guard_name == "drawdown"


def test_engine_drawdown_checked_before_margin():
    """If both drawdown and margin fail, drawdown (HARD_BLOCK) wins."""
    engine = SafetyEngine()
    verdict = engine.evaluate(SafetyContext(
        current_equity=2000, free_margin=0, margin_in_use=1500,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=1, active_cycle_unrealised_pnl=0,
        proposed_order_size=1000, proposed_order_type="OPEN_CYCLE",
    ))
    assert verdict.guard_name == "drawdown"


def test_engine_margin_short_circuits_on_soft_block():
    engine = SafetyEngine()
    verdict = engine.evaluate(SafetyContext(
        current_equity=10000, free_margin=100, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=500, proposed_order_type="OPEN_CYCLE",
    ))
    assert verdict.decision == DCASafetyDecision.SOFT_BLOCK
    assert verdict.guard_name == "margin"


# ============================================================
# Test: evaluate_safety convenience function
# ============================================================

def test_evaluate_safety_convenience():
    v = evaluate_safety(
        current_equity=10000, free_margin=10000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    assert v.is_allowed


def test_evaluate_safety_uses_default_thresholds():
    """Without explicit thresholds, defaults apply (max_dd=0.35, buffer=0.20)."""
    # Drawdown > 35% should block
    v = evaluate_safety(
        current_equity=5000, free_margin=5000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
    )
    assert v.is_hard_block


def test_evaluate_safety_with_custom_thresholds():
    """Custom thresholds: max_dd=0.60, buffer=0.10."""
    custom = SafetyThresholds(max_dd_pct=0.60, buffer_pct=0.10, stress_test_decline=0.5)
    # 40% DD — within 60% limit
    v = evaluate_safety(
        current_equity=6000, free_margin=6000, margin_in_use=0,
        peak_equity=10000, starting_equity=10000,
        n_active_cycles=0, active_cycle_unrealised_pnl=0,
        proposed_order_size=100, proposed_order_type="OPEN_CYCLE",
        thresholds=custom,
    )
    assert v.is_allowed


# ============================================================
# Test: defaults match locked Kanban
# ============================================================

def test_default_max_dd_is_35_percent():
    th = SafetyThresholds()
    assert th.max_dd_pct == 0.35


def test_default_buffer_pct_is_20_percent():
    """Per Kanban: buffer_pct default is 20%, NOT hardcoded."""
    th = SafetyThresholds()
    assert th.buffer_pct == 0.20


def test_default_stress_test_decline_is_btc_2022():
    """BTC 2021-11 to 2022-11 = -77.2% peak-to-trough."""
    th = SafetyThresholds()
    assert abs(th.stress_test_decline - 0.772) < 0.001


def test_default_overlap_allowed():
    th = SafetyThresholds()
    assert th.allow_overlap_when_break_even is True
