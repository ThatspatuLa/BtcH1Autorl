"""Tests for Stage 8 — DCA Calculation Library."""
from __future__ import annotations

import pytest

from dca_calc.allocation import (
    AllocationContext,
    allocation_controlled_exp,
    allocation_drawdown_adjusted,
    allocation_equal,
    allocation_linear_increasing,
    allocation_volatility_adjusted,
    compute_layer_allocation,
    compute_total_position_size,
)
from dca_calc.confirmation import (
    ConfirmationContext,
    check_all_confirmations,
    check_confirmation,
    confirm_ma_above,
    confirm_ma_below,
    confirm_rsi_above,
    confirm_rsi_below,
    confirm_volatility_high,
    confirm_volatility_low,
)
from dca_calc.grid_spacing import (
    GridContext,
    compute_next_layer_price,
    grid_atr,
    grid_drawdown_from_high,
    grid_fixed_pct,
    grid_ma_distance,
    grid_rsi_oversold,
    grid_trend_adjusted,
    grid_volatility,
    grid_z_score,
)

# ============================================================
# Test: grid spacing - fixed_pct
# ============================================================

def test_grid_fixed_pct_basic():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    price = grid_fixed_pct({"pct": 0.015}, ctx)
    assert price == pytest.approx(98.5)


def test_grid_fixed_pct_rejects_zero_pct():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    with pytest.raises(ValueError):
        grid_fixed_pct({"pct": 0.0}, ctx)


def test_grid_fixed_pct_uses_avg_entry():
    """Trigger is relative to avg_entry, not current_price."""
    ctx = GridContext(current_price=80, avg_entry=100, cycle_high=110, layers_filled=1, n_layers_total=5)
    price = grid_fixed_pct({"pct": 0.02}, ctx)
    assert price == pytest.approx(98.0)  # 100 * 0.98


# ============================================================
# Test: grid spacing - atr
# ============================================================

def test_grid_atr_basic():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5, atr=2.0)
    price = grid_atr({"atr_multiplier": 2.0}, ctx)
    assert price == pytest.approx(96.0)  # 100 - 2*2


def test_grid_atr_returns_none_without_atr():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    assert grid_atr({"atr_multiplier": 2.0}, ctx) is None


def test_grid_atr_rejects_zero_multiplier():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5, atr=1.0)
    with pytest.raises(ValueError):
        grid_atr({"atr_multiplier": 0.0}, ctx)


# ============================================================
# Test: grid spacing - volatility
# ============================================================

def test_grid_volatility_basic():
    """At reference vol (2%), spacing = base_pct."""
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5, volatility=0.02)
    price = grid_volatility({"base_pct": 0.01, "vol_scale_factor": 0.5}, ctx)
    assert price == pytest.approx(99.0)


def test_grid_volatility_wider_in_high_vol():
    """2x reference vol → spacing scales up."""
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5, volatility=0.04)
    price = grid_volatility({"base_pct": 0.01, "vol_scale_factor": 0.5}, ctx)
    # effective_pct = 0.01 * (1 + 0.5 * (0.04/0.02 - 1)) = 0.01 * (1 + 0.5) = 0.015
    assert price == pytest.approx(98.5)


def test_grid_volatility_returns_none_without_vol():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    assert grid_volatility({"base_pct": 0.01}, ctx) is None


# ============================================================
# Test: grid spacing - drawdown_from_high
# ============================================================

def test_grid_drawdown_from_high_uses_high():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=120, layers_filled=0, n_layers_total=5)
    price = grid_drawdown_from_high({"drawdown_pct": 0.05}, ctx)
    assert price == pytest.approx(114.0)  # 120 * 0.95


def test_grid_drawdown_from_high_uses_reference_high():
    """reference_high takes priority over cycle_high."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=120,
        layers_filled=0, n_layers_total=5, reference_high=110,
    )
    price = grid_drawdown_from_high({"drawdown_pct": 0.05}, ctx)
    assert price == pytest.approx(104.5)  # 110 * 0.95


# ============================================================
# Test: grid spacing - ma_distance
# ============================================================

def test_grid_ma_distance_basic():
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, ma_value=200,
    )
    price = grid_ma_distance({"ma_distance_pct": 0.03}, ctx)
    assert price == pytest.approx(194.0)  # 200 * 0.97


def test_grid_ma_distance_returns_none_without_ma():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    assert grid_ma_distance({"ma_distance_pct": 0.03}, ctx) is None


# ============================================================
# Test: grid spacing - rsi_oversold
# ============================================================

def test_grid_rsi_oversold_when_below_threshold():
    """RSI=20 (oversold) → return avg_entry-based price."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, rsi_value=20.0,
    )
    price = grid_rsi_oversold({"rsi_threshold": 30.0, "oversold_depth_pct": 0.02}, ctx)
    assert price == pytest.approx(98.0)


def test_grid_rsi_oversold_returns_deeper_target_when_not_oversold():
    """RSI=60 (not oversold) → return a much deeper target."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, rsi_value=60.0,
    )
    price = grid_rsi_oversold({"rsi_threshold": 30.0, "oversold_depth_pct": 0.02}, ctx)
    # ctx.current_price * (1 - 0.02 * 5) = 100 * 0.9 = 90
    assert price == pytest.approx(90.0)
    # And it should be deeper than the oversold-target (avg_entry * 0.98 = 98)
    assert price < 98.0


# ============================================================
# Test: grid spacing - z_score
# ============================================================

def test_grid_z_score_when_oversold():
    """z_score < threshold → avg_entry-based price."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, z_score=2.0,
    )
    price = grid_z_score({"z_threshold": 1.5, "lookback_std": 0.02}, ctx)
    # 100 * (1 - 0.02 * 1.5) = 100 * 0.97 = 97.0
    assert price == pytest.approx(97.0)


def test_grid_z_score_when_not_oversold():
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, z_score=0.5,
    )
    price = grid_z_score({"z_threshold": 1.5, "lookback_std": 0.02}, ctx)
    # current_price * (1 - 0.02 * 1.5) = 100 * 0.97 = 97.0
    assert price == pytest.approx(97.0)


# ============================================================
# Test: grid spacing - trend_adjusted
# ============================================================

def test_grid_trend_adjusted_downtrend_widens():
    """In downtrend, grid widens (deeper trigger)."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, trend_strength=-0.5,
    )
    price = grid_trend_adjusted({"base_pct": 0.015, "trend_multiplier": 0.5}, ctx)
    # effective_pct = 0.015 * (1 + 0.5 * 0.5) = 0.015 * 1.25 = 0.01875
    # price = 100 * (1 - 0.01875) = 98.125
    assert price < 100 * 0.985  # wider than base_pct trigger (98.5)


def test_grid_trend_adjusted_uptrend_tightens():
    """In uptrend, grid tightens (shallower trigger)."""
    ctx = GridContext(
        current_price=100, avg_entry=100, cycle_high=100,
        layers_filled=0, n_layers_total=5, trend_strength=0.5,
    )
    price = grid_trend_adjusted({"base_pct": 0.015, "trend_multiplier": 0.5}, ctx)
    # effective_pct = 0.015 * max(0.1, 1 - 0.5*0.5) = 0.015 * 0.75 = 0.01125
    # price = 100 * 0.98875 = 98.875
    assert price > 100 * 0.985


# ============================================================
# Test: dispatcher
# ============================================================

def test_compute_next_layer_price_dispatcher():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    price = compute_next_layer_price("fixed_pct", {"pct": 0.02}, ctx)
    assert price == pytest.approx(98.0)


def test_compute_next_layer_price_unknown_method():
    ctx = GridContext(current_price=100, avg_entry=100, cycle_high=100, layers_filled=0, n_layers_total=5)
    with pytest.raises(ValueError, match="Unknown grid_method"):
        compute_next_layer_price("nonexistent", {}, ctx)


# ============================================================
# Test: allocation - equal
# ============================================================

def test_allocation_equal():
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100)
    assert allocation_equal({}, ctx) == 100.0
    ctx.layer_index = 3
    assert allocation_equal({}, ctx) == 100.0


# ============================================================
# Test: allocation - linear_increasing
# ============================================================

def test_allocation_linear_increasing_layer_0():
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100)
    size = allocation_linear_increasing({"increment_pct": 0.20}, ctx)
    assert size == pytest.approx(100.0)  # 1 + 0*0.20 = 1.0


def test_allocation_linear_increasing_layer_3():
    ctx = AllocationContext(base_stake=100, layer_index=3, layers_filled=3, n_layers_total=5, current_price=100, avg_entry=100)
    size = allocation_linear_increasing({"increment_pct": 0.20}, ctx)
    assert size == pytest.approx(160.0)  # 1 + 3*0.20 = 1.6


# ============================================================
# Test: allocation - controlled_exp
# ============================================================

def test_allocation_controlled_exp_grows():
    """Layer 0 = 100, layer 1 = 150, layer 2 = 225, etc."""
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100)
    assert allocation_controlled_exp({"multiplier": 1.5, "max_layer_size_pct": 5.0}, ctx) == 100.0

    ctx.layer_index = 1
    assert allocation_controlled_exp({"multiplier": 1.5, "max_layer_size_pct": 5.0}, ctx) == 150.0

    ctx.layer_index = 2
    assert allocation_controlled_exp({"multiplier": 1.5, "max_layer_size_pct": 5.0}, ctx) == 225.0


def test_allocation_controlled_exp_capped():
    """Layer 3 = 100 * 1.5^3 = 337.5, but cap is 5*100 = 500. Within cap."""
    ctx = AllocationContext(base_stake=100, layer_index=3, layers_filled=3, n_layers_total=5, current_price=100, avg_entry=100)
    assert allocation_controlled_exp({"multiplier": 1.5, "max_layer_size_pct": 5.0}, ctx) == pytest.approx(337.5)

    """Layer 5 = 100 * 1.5^5 = 759.4, cap at 500."""
    ctx.layer_index = 5
    assert allocation_controlled_exp({"multiplier": 1.5, "max_layer_size_pct": 5.0}, ctx) == 500.0


def test_allocation_controlled_exp_rejects_multiplier_le_1():
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100)
    with pytest.raises(ValueError):
        allocation_controlled_exp({"multiplier": 1.0, "max_layer_size_pct": 5.0}, ctx)


# ============================================================
# Test: allocation - drawdown_adjusted
# ============================================================

def test_allocation_drawdown_adjusted_zero_dd():
    """At zero drawdown, size = base_stake."""
    ctx = AllocationContext(base_stake=100, layer_index=1, layers_filled=1, n_layers_total=5, current_price=100, avg_entry=100, current_dd_pct=0.0)
    size = allocation_drawdown_adjusted({"sensitivity": 2.0, "min_size_pct": 0.5, "max_size_pct": 5.0}, ctx)
    assert size == pytest.approx(100.0)


def test_allocation_drawdown_adjusted_10pct_dd():
    """At 10% drawdown, sensitivity 2 → scale = 1 + 2*0.10 = 1.2 → 120."""
    ctx = AllocationContext(base_stake=100, layer_index=1, layers_filled=1, n_layers_total=5, current_price=100, avg_entry=100, current_dd_pct=0.10)
    size = allocation_drawdown_adjusted({"sensitivity": 2.0, "min_size_pct": 0.5, "max_size_pct": 5.0}, ctx)
    assert size == pytest.approx(120.0)


def test_allocation_drawdown_adjusted_capped():
    """At 50% drawdown, scale = 1 + 1.0 = 2.0 → 200. Within 5x cap."""
    ctx = AllocationContext(base_stake=100, layer_index=1, layers_filled=1, n_layers_total=5, current_price=100, avg_entry=100, current_dd_pct=0.50)
    size = allocation_drawdown_adjusted({"sensitivity": 2.0, "min_size_pct": 0.5, "max_size_pct": 5.0}, ctx)
    assert size == pytest.approx(200.0)


# ============================================================
# Test: allocation - volatility_adjusted
# ============================================================

def test_allocation_volatility_adjusted_at_reference():
    """At reference vol, size = base_stake."""
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100, volatility=0.02)
    size = allocation_volatility_adjusted({"reference_vol": 0.02, "min_size_pct": 0.5, "max_size_pct": 3.0}, ctx)
    assert size == pytest.approx(100.0)


def test_allocation_volatility_adjusted_high_vol_smaller():
    """High vol → smaller position."""
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100, volatility=0.04)
    size = allocation_volatility_adjusted({"reference_vol": 0.02, "min_size_pct": 0.5, "max_size_pct": 3.0}, ctx)
    # scale = 0.02 / 0.04 = 0.5 → size = 50
    assert size == pytest.approx(50.0)


def test_allocation_volatility_adjusted_no_vol():
    """No vol provided → fall back to base_stake."""
    ctx = AllocationContext(base_stake=100, layer_index=0, layers_filled=0, n_layers_total=5, current_price=100, avg_entry=100)
    size = allocation_volatility_adjusted({"reference_vol": 0.02}, ctx)
    assert size == pytest.approx(100.0)


# ============================================================
# Test: dispatcher + total size
# ============================================================

def test_compute_layer_allocation_dispatcher():
    ctx = AllocationContext(base_stake=100, layer_index=1, layers_filled=1, n_layers_total=5, current_price=100, avg_entry=100)
    assert compute_layer_allocation("equal", {}, ctx) == 100.0
    assert compute_layer_allocation("linear_increasing", {"increment_pct": 0.20}, ctx) == 120.0


def test_compute_total_position_size_equal():
    """5 equal layers of 100 = 500 total."""
    total = compute_total_position_size("equal", {}, base_stake=100, n_layers=5, current_price=100)
    assert total == pytest.approx(500.0)


def test_compute_total_position_size_controlled_exp():
    """5 layers with mult=1.5, cap=5: 100, 150, 225, 337.5, 500 (capped)."""
    total = compute_total_position_size(
        "controlled_exp", {"multiplier": 1.5, "max_layer_size_pct": 5.0},
        base_stake=100, n_layers=5, current_price=100,
    )
    assert total == pytest.approx(100 + 150 + 225 + 337.5 + 500)


# ============================================================
# Test: confirmation
# ============================================================

def test_confirm_rsi_below_passes():
    ctx = ConfirmationContext(rsi_value=25.0)
    assert confirm_rsi_below({"threshold": 30.0}, ctx) is True


def test_confirm_rsi_below_fails():
    ctx = ConfirmationContext(rsi_value=50.0)
    assert confirm_rsi_below({"threshold": 30.0}, ctx) is False


def test_confirm_rsi_below_no_value():
    ctx = ConfirmationContext()
    assert confirm_rsi_below({"threshold": 30.0}, ctx) is False


def test_confirm_rsi_above_passes():
    ctx = ConfirmationContext(rsi_value=80.0)
    assert confirm_rsi_above({"threshold": 70.0}, ctx) is True


def test_confirm_ma_above_passes():
    ctx = ConfirmationContext(ma_value=100.0, current_price=110.0)
    assert confirm_ma_above({}, ctx) is True


def test_confirm_ma_below_passes():
    ctx = ConfirmationContext(ma_value=100.0, current_price=90.0)
    assert confirm_ma_below({}, ctx) is True


def test_confirm_volatility_high():
    ctx = ConfirmationContext(volatility=0.04, reference_vol=0.02)
    assert confirm_volatility_high({"threshold": 1.5}, ctx) is True  # 0.04 > 0.03


def test_confirm_volatility_low():
    ctx = ConfirmationContext(volatility=0.01, reference_vol=0.02)
    assert confirm_volatility_low({"threshold": 0.5}, ctx) is False  # 0.01 > 0.01 (strict)
    ctx.volatility = 0.005
    assert confirm_volatility_low({"threshold": 0.5}, ctx) is True  # 0.005 < 0.01


def test_check_confirmation_dispatcher():
    ctx = ConfirmationContext(rsi_value=20.0)
    assert check_confirmation("rsi_below", {"threshold": 30.0}, ctx) is True
    assert check_confirmation("rsi_above", {"threshold": 70.0}, ctx) is False


def test_check_confirmation_unknown():
    ctx = ConfirmationContext()
    with pytest.raises(ValueError):
        check_confirmation("nonexistent", {}, ctx)


def test_check_all_confirmations_and_logic():
    """ALL must pass."""
    ctx = ConfirmationContext(rsi_value=20.0, ma_value=100.0, current_price=90.0)
    indicators = ["rsi_below", "ma_below"]
    params_map = {"rsi_below": {"threshold": 30.0}, "ma_below": {}}
    all_pass, failed = check_all_confirmations(indicators, params_map, ctx)
    assert all_pass is True
    assert failed == []


def test_check_all_confirmations_one_fails():
    ctx = ConfirmationContext(rsi_value=20.0, ma_value=100.0, current_price=110.0)  # price above MA
    indicators = ["rsi_below", "ma_below"]
    params_map = {"rsi_below": {"threshold": 30.0}, "ma_below": {}}
    all_pass, failed = check_all_confirmations(indicators, params_map, ctx)
    assert all_pass is False
    assert failed == ["ma_below"]
