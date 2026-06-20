"""Tests for Stage 11 — TP/Exit Calculation Library."""
from __future__ import annotations

import pytest

from tp_calc.exits import (
    ExitContext,
    ExitDecision,
    _strip_method_prefix,
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

# ============================================================
# Test helpers
# ============================================================

def _ctx(price: float = 100.0, **overrides) -> ExitContext:
    """Build a default ExitContext with overrides."""
    defaults = {
        "current_price": 100.0,
        "avg_entry": 100.0,
        "cycle_high": 100.0,
        "cycle_low": 100.0,
        "n_layers_filled": 1,
        "candles_in_position": 10,
        "unrealised_pnl_pct": 0.0,
    }
    defaults.update(overrides)
    return ExitContext(**defaults)


# ============================================================
# Test: ExitDecision helpers
# ============================================================

def test_exit_decision_hold():
    d = ExitDecision.hold()
    assert d.should_close is False
    assert d.close_fraction == 0.0


def test_exit_decision_close_all():
    d = ExitDecision.close_all(105.0, "test")
    assert d.should_close is True
    assert d.target_price == 105.0
    assert d.close_fraction == 1.0
    assert d.reason == "test"


def test_exit_decision_close_partial():
    d = ExitDecision.close_partial(105.0, 0.5, "test")
    assert d.should_close is True
    assert d.close_fraction == 0.5


# ============================================================
# Test: fixed
# ============================================================

def test_exit_fixed_holds_below_target():
    d = exit_fixed({"tp_pct": 0.02}, _ctx(current_price=101.0, avg_entry=100.0))
    assert d.should_close is False


def test_exit_fixed_closes_at_target():
    d = exit_fixed({"tp_pct": 0.02}, _ctx(current_price=102.0, avg_entry=100.0))
    assert d.should_close is True
    assert d.target_price == pytest.approx(102.0)
    assert d.close_fraction == 1.0


def test_exit_fixed_closes_above_target():
    d = exit_fixed({"tp_pct": 0.02}, _ctx(current_price=110.0, avg_entry=100.0))
    assert d.should_close is True


# ============================================================
# Test: atr
# ============================================================

def test_exit_atr_holds_below_target():
    d = exit_atr({"atr_multiplier": 3.0}, _ctx(current_price=105.0, avg_entry=100.0, atr=2.0))
    assert d.should_close is False  # target = 100 + 6 = 106


def test_exit_atr_closes_at_target():
    d = exit_atr({"atr_multiplier": 3.0}, _ctx(current_price=107.0, avg_entry=100.0, atr=2.0))
    assert d.should_close is True
    assert d.target_price == pytest.approx(106.0)


def test_exit_atr_no_atr():
    d = exit_atr({"atr_multiplier": 3.0}, _ctx(current_price=200.0, atr=None))
    assert d.should_close is False


# ============================================================
# Test: vol_adjusted
# ============================================================

def test_exit_vol_adjusted_at_reference():
    d = exit_vol_adjusted(
        {"base_pct": 0.02, "vol_scale_factor": 0.5, "reference_vol": 0.02},
        _ctx(current_price=102.0, avg_entry=100.0, volatility=0.02, reference_vol=0.02),
    )
    assert d.should_close is True  # target = 100 * 1.02 = 102


def test_exit_vol_adjusted_wider_in_high_vol():
    """At 2x vol, target is 1.5x base_pct = 1.03."""
    d = exit_vol_adjusted(
        {"base_pct": 0.02, "vol_scale_factor": 0.5, "reference_vol": 0.02},
        _ctx(current_price=102.5, avg_entry=100.0, volatility=0.04, reference_vol=0.02),
    )
    assert d.should_close is False  # target = 100 * 1.03 = 103
    d = exit_vol_adjusted(
        {"base_pct": 0.02, "vol_scale_factor": 0.5, "reference_vol": 0.02},
        _ctx(current_price=103.5, avg_entry=100.0, volatility=0.04, reference_vol=0.02),
    )
    assert d.should_close is True


def test_exit_vol_adjusted_no_vol():
    d = exit_vol_adjusted({}, _ctx(current_price=200.0))
    assert d.should_close is False


# ============================================================
# Test: dca_depth_adjusted
# ============================================================

def test_exit_dca_depth_layer_1():
    d = exit_dca_depth_adjusted(
        {"base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10},
        _ctx(current_price=102.0, avg_entry=100.0, n_layers_filled=1),
    )
    assert d.should_close is True  # target = 100 * 1.02


def test_exit_dca_depth_layer_3():
    d = exit_dca_depth_adjusted(
        {"base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10},
        _ctx(current_price=102.5, avg_entry=100.0, n_layers_filled=3),
    )
    # effective = 0.02 + 2*0.005 = 0.03 → target = 103
    assert d.should_close is False
    d = exit_dca_depth_adjusted(
        {"base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10},
        _ctx(current_price=103.5, avg_entry=100.0, n_layers_filled=3),
    )
    assert d.should_close is True


def test_exit_dca_depth_capped():
    """At layer 20, target would be 0.02 + 19*0.005 = 0.115, capped at 0.10."""
    d = exit_dca_depth_adjusted(
        {"base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10},
        _ctx(current_price=109.5, avg_entry=100.0, n_layers_filled=20),
    )
    assert d.should_close is False  # target capped at 110
    d = exit_dca_depth_adjusted(
        {"base_pct": 0.02, "depth_multiplier": 0.005, "max_pct": 0.10},
        _ctx(current_price=110.5, avg_entry=100.0, n_layers_filled=20),
    )
    assert d.should_close is True


# ============================================================
# Test: partial_ladder
# ============================================================

def test_exit_partial_ladder_below_tp1():
    d = exit_partial_ladder(
        {"tp1_pct": 0.01, "tp1_fraction": 0.5, "tp2_pct": 0.03, "tp2_fraction": 1.0},
        _ctx(current_price=100.5, avg_entry=100.0),
    )
    assert d.should_close is False


def test_exit_partial_ladder_at_tp1():
    d = exit_partial_ladder(
        {"tp1_pct": 0.01, "tp1_fraction": 0.5, "tp2_pct": 0.03, "tp2_fraction": 1.0},
        _ctx(current_price=101.5, avg_entry=100.0),
    )
    assert d.should_close is True
    assert d.close_fraction == 0.5
    assert d.target_price == pytest.approx(101.0)


def test_exit_partial_ladder_at_tp2():
    d = exit_partial_ladder(
        {"tp1_pct": 0.01, "tp1_fraction": 0.5, "tp2_pct": 0.03, "tp2_fraction": 1.0},
        _ctx(current_price=103.5, avg_entry=100.0),
    )
    assert d.should_close is True
    assert d.close_fraction == 1.0
    assert d.target_price == pytest.approx(103.0)


# ============================================================
# Test: trailing
# ============================================================

def test_exit_trailing_below_activation():
    d = exit_trailing(
        {"trail_pct": 0.02, "activation_pct": 0.005},
        _ctx(current_price=100.3, avg_entry=100.0, cycle_high=100.5),
    )
    assert d.should_close is False  # below 100.5 activation


def test_exit_trailing_active_no_trigger():
    d = exit_trailing(
        {"trail_pct": 0.02, "activation_pct": 0.005},
        _ctx(current_price=101.0, avg_entry=100.0, cycle_high=102.0),
    )
    # 101 > 100.5 (activated), trail_stop = 102 * 0.98 = 99.96
    # 101 > 99.96 → hold
    assert d.should_close is False


def test_exit_trailing_triggered():
    d = exit_trailing(
        {"trail_pct": 0.02, "activation_pct": 0.005},
        _ctx(current_price=99.5, avg_entry=100.0, cycle_high=102.0),
    )
    # 99.5 < 99.96 → trigger
    assert d.should_close is True
    assert d.target_price == pytest.approx(99.96)


# ============================================================
# Test: break_even
# ============================================================

def test_exit_break_even_not_activated():
    """If cycle_high didn't reach min_profit threshold, don't activate."""
    d = exit_break_even(
        {"buffer_pct": 0.002, "min_profit_pct": 0.005},
        _ctx(current_price=100.0, avg_entry=100.0, cycle_high=100.3),
    )
    # cycle_high/avg = 1.003 < 1.005 → not activated
    assert d.should_close is False


def test_exit_break_even_active_holds():
    d = exit_break_even(
        {"buffer_pct": 0.002, "min_profit_pct": 0.005},
        _ctx(current_price=101.0, avg_entry=100.0, cycle_high=101.5),
    )
    # target = 100 * 1.002 = 100.2
    # 101 > 100.2 → hold
    assert d.should_close is False


def test_exit_break_even_triggers():
    d = exit_break_even(
        {"buffer_pct": 0.002, "min_profit_pct": 0.005},
        _ctx(current_price=100.1, avg_entry=100.0, cycle_high=101.5),
    )
    # 100.1 < 100.2 → trigger
    assert d.should_close is True
    assert d.target_price == pytest.approx(100.2)


# ============================================================
# Test: momentum_decay
# ============================================================

def test_exit_momentum_decay_no_rsi():
    d = exit_momentum_decay({"rsi_peak": 70.0, "rsi_exit": 50.0}, _ctx(rsi_value=None))
    assert d.should_close is False


def test_exit_momentum_decay_rsi_above_exit():
    d = exit_momentum_decay({"rsi_peak": 70.0, "rsi_exit": 50.0}, _ctx(rsi_value=60.0))
    assert d.should_close is False


def test_exit_momentum_decay_rsi_below_exit_no_profit():
    d = exit_momentum_decay({"rsi_peak": 70.0, "rsi_exit": 50.0}, _ctx(rsi_value=40.0, unrealised_pnl_pct=-0.05))
    assert d.should_close is False


def test_exit_momentum_decay_rsi_below_exit_with_profit():
    d = exit_momentum_decay({"rsi_peak": 70.0, "rsi_exit": 50.0}, _ctx(rsi_value=40.0, unrealised_pnl_pct=0.05))
    assert d.should_close is True


# ============================================================
# Test: exhaustion
# ============================================================

def test_exit_exhaustion_no_vol():
    d = exit_exhaustion({}, _ctx(volatility=None, reference_vol=0.02))
    assert d.should_close is False


def test_exit_exhaustion_low_vol_with_profit():
    d = exit_exhaustion({"vol_low_threshold": 0.5}, _ctx(volatility=0.005, reference_vol=0.02, unrealised_pnl_pct=0.05))
    # 0.005 < 0.5 * 0.02 = 0.01, and profit > 0
    assert d.should_close is True


def test_exit_exhaustion_no_profit():
    d = exit_exhaustion({"vol_low_threshold": 0.5}, _ctx(volatility=0.005, reference_vol=0.02, unrealised_pnl_pct=-0.05))
    assert d.should_close is False


def test_exit_exhaustion_climax():
    d = exit_exhaustion({"vol_spike_threshold": 2.0}, _ctx(volatility=0.05, reference_vol=0.02, unrealised_pnl_pct=0.05))
    # 0.05 > 2.0 * 0.02 = 0.04, and profit > 0
    assert d.should_close is True


# ============================================================
# Test: trend_hold
# ============================================================

def test_exit_trend_hold_no_ma():
    d = exit_trend_hold({"ma_distance_pct": 0.02}, _ctx(ma_value=None))
    assert d.should_close is False


def test_exit_trend_hold_below_min_profit():
    d = exit_trend_hold(
        {"ma_distance_pct": 0.02, "min_profit_pct": 0.005},
        _ctx(current_price=99.0, avg_entry=100.0, ma_value=100.0, unrealised_pnl_pct=0.0),
    )
    # pnl < 0.5%, current_price (99) < 100*0.98=98? No, 99 > 98
    # But min_profit filter kicks in
    assert d.should_close is False


def test_exit_trend_hold_below_ma():
    d = exit_trend_hold(
        {"ma_distance_pct": 0.02, "min_profit_pct": 0.005},
        _ctx(current_price=97.5, avg_entry=100.0, ma_value=100.0, cycle_high=101.5, unrealised_pnl_pct=0.05),
    )
    # pnl > 0.5%, current 97.5 < 98.0 → trigger
    assert d.should_close is True


# ============================================================
# Test: failed_continuation
# ============================================================

def test_exit_failed_continuation_peak_too_low():
    d = exit_failed_continuation(
        {"min_pnl_pct": 0.01, "reversal_pct": 0.005},
        _ctx(current_price=100.0, avg_entry=100.0, cycle_high=100.5),
    )
    # peak pnl = 0.5% < 1% → hold
    assert d.should_close is False


def test_exit_failed_continuation_holds_above_reversal():
    d = exit_failed_continuation(
        {"min_pnl_pct": 0.01, "reversal_pct": 0.005},
        _ctx(current_price=105.0, avg_entry=100.0, cycle_high=105.0),
    )
    # reversal_level = 105 * 0.995 = 104.475, 105 > 104.475 → hold
    assert d.should_close is False


def test_exit_failed_continuation_triggers():
    d = exit_failed_continuation(
        {"min_pnl_pct": 0.01, "reversal_pct": 0.005},
        _ctx(current_price=104.0, avg_entry=100.0, cycle_high=105.0),
    )
    # 104 < 104.475 → trigger
    assert d.should_close is True
    assert d.target_price == pytest.approx(104.475)


# ============================================================
# Test: time_in_position
# ============================================================

def test_exit_time_under_max():
    d = exit_time_in_position({"max_candles": 100}, _ctx(candles_in_position=50))
    assert d.should_close is False


def test_exit_time_at_max():
    d = exit_time_in_position({"max_candles": 100}, _ctx(candles_in_position=100))
    assert d.should_close is True


def test_exit_time_with_min_profit_filter():
    d = exit_time_in_position(
        {"max_candles": 100, "min_profit_pct": 0.01},
        _ctx(candles_in_position=100, unrealised_pnl_pct=0.005),
    )
    # pnl 0.5% < 1% → hold
    assert d.should_close is False


# ============================================================
# Test: hybrid
# ============================================================

def test_exit_hybrid_or_logic():
    """OR logic: any sub-method triggers close."""
    d = exit_hybrid(
        {
            "methods": "fixed,trailing",
            "fixed_tp_pct": 0.02,
            "trailing_trail_pct": 0.015,
            "trailing_activation_pct": 0.005,
        },
        _ctx(current_price=101.0, avg_entry=100.0, cycle_high=101.0),
    )
    # fixed target = 102, not hit
    # trailing: cycle_high 101 < activation 100.5? No, 101 > 100.5, activated
    # trail_stop = 101 * 0.985 = 99.485, 101 > 99.485 → hold
    assert d.should_close is False


def test_exit_hybrid_fixed_triggers_first():
    d = exit_hybrid(
        {
            "methods": "fixed,trailing",
            "fixed_tp_pct": 0.02,
            "trailing_trail_pct": 0.015,
            "trailing_activation_pct": 0.005,
        },
        _ctx(current_price=102.5, avg_entry=100.0, cycle_high=102.5),
    )
    # fixed at 102 → triggers
    assert d.should_close is True
    assert "fixed" in d.reason


def test_exit_hybrid_trailing_triggers_first():
    """If fixed doesn't trigger but trailing does, trailing wins."""
    d = exit_hybrid(
        {
            "methods": "trailing,fixed",
            "fixed_tp_pct": 0.05,  # not hit
            "trailing_trail_pct": 0.01,
            "trailing_activation_pct": 0.005,
        },
        _ctx(current_price=100.3, avg_entry=100.0, cycle_high=102.0),
    )
    # fixed target = 105, not hit
    # trailing: cycle_high 102 >= activation 100.5, activated
    # trail_stop = 102 * 0.99 = 100.98, 100.3 < 100.98 → trigger
    assert d.should_close is True
    assert "trailing" in d.reason


def test_exit_hybrid_unknown_method_skipped():
    d = exit_hybrid(
        {
            "methods": "unknown_method,fixed",
            "fixed_tp_pct": 0.02,
        },
        _ctx(current_price=102.5, avg_entry=100.0),
    )
    # unknown skipped, fixed triggers
    assert d.should_close is True


# ============================================================
# Test: _strip_method_prefix
# ============================================================

def test_strip_method_prefix_basic():
    full = {"fixed_tp_pct": 0.02, "trail_pct": 0.015, "methods": "fixed,trailing"}
    sub = _strip_method_prefix("fixed", full)
    assert sub == {"tp_pct": 0.02}


def test_strip_method_prefix_no_match():
    full = {"other_param": 1.0}
    sub = _strip_method_prefix("fixed", full)
    assert sub == {}


def test_strip_method_prefix_multiple():
    full = {"trailing_trail_pct": 0.015, "trailing_activation_pct": 0.005}
    sub = _strip_method_prefix("trailing", full)
    assert sub == {"trail_pct": 0.015, "activation_pct": 0.005}


# ============================================================
# Test: dispatcher
# ============================================================

def test_compute_exit_decision_dispatcher():
    d = compute_exit_decision("fixed", {"tp_pct": 0.02}, _ctx(current_price=102.0, avg_entry=100.0))
    assert d.should_close is True


def test_compute_exit_decision_unknown():
    with pytest.raises(ValueError, match="Unknown exit_method"):
        compute_exit_decision("nonexistent", {}, _ctx())
