"""Phase B — discovery_fitness.py tests (TDD for the new aggregator).

Discovery Fitness v2 formula:
    discovery_fitness = 0.60·full_period_base_score
                      + 0.20·recovery_score
                      + 0.10·consistency_score
                      + 0.05·stability_score
                      + 0.05·concentration_score
"""
from __future__ import annotations

import pytest

from fitness.discovery_fitness import (
    DISCOVERY_WEIGHTS,
    compute_concentration_score,
    compute_discovery_fitness,
    compute_stability_score,
)


# ============================================================
# Task B1: compute_discovery_fitness skeleton
# ============================================================

def test_discovery_weights_sum_to_one():
    """The 5 weights must sum to 1.0 (sanity)."""
    assert abs(sum(DISCOVERY_WEIGHTS.values()) - 1.0) < 1e-9


def test_discovery_weights_locked_values():
    """Weights are locked at 60/20/10/5/5 per Six's spec."""
    assert DISCOVERY_WEIGHTS["full_period_base_score"] == pytest.approx(0.60, abs=1e-9)
    assert DISCOVERY_WEIGHTS["recovery_score"] == pytest.approx(0.20, abs=1e-9)
    assert DISCOVERY_WEIGHTS["consistency_score"] == pytest.approx(0.10, abs=1e-9)
    assert DISCOVERY_WEIGHTS["stability_score"] == pytest.approx(0.05, abs=1e-9)
    assert DISCOVERY_WEIGHTS["concentration_score"] == pytest.approx(0.05, abs=1e-9)


def test_compute_discovery_fitness_weights_match_formula():
    """Verify the aggregator computes the weighted sum correctly."""
    # 0.60·0.85 + 0.20·0.50 + 0.10·0.55 + 0.05·0.33 + 0.05·0.875
    expected = (
        0.60 * 0.85
        + 0.20 * 0.50
        + 0.10 * 0.55
        + 0.05 * 0.33
        + 0.05 * 0.875
    )
    out = compute_discovery_fitness(
        full_period_base_score=0.85,
        recovery_score=0.50,
        consistency_score=0.55,
        stability_score=0.33,
        concentration_score=0.875,
    )
    assert out == pytest.approx(expected, abs=1e-9)
    # = 0.510 + 0.100 + 0.055 + 0.0165 + 0.04375 = 0.72525


def test_compute_discovery_fitness_all_ones_score_one():
    """All components = 1.0 → discovery_fitness = 1.0."""
    out = compute_discovery_fitness(
        full_period_base_score=1.0,
        recovery_score=1.0,
        consistency_score=1.0,
        stability_score=1.0,
        concentration_score=1.0,
    )
    assert out == pytest.approx(1.0, abs=1e-9)


def test_compute_discovery_fitness_all_zeros_score_zero():
    """All components = 0.0 → discovery_fitness = 0.0."""
    out = compute_discovery_fitness(
        full_period_base_score=0.0,
        recovery_score=0.0,
        consistency_score=0.0,
        stability_score=0.0,
        concentration_score=0.0,
    )
    assert out == pytest.approx(0.0, abs=1e-9)


def test_compute_discovery_fitness_output_in_unit_range():
    """Output must always be in [0, 1] even with weird inputs."""
    # Negative inputs shouldn't happen in practice but defensive
    out = compute_discovery_fitness(
        full_period_base_score=0.5,
        recovery_score=0.5,
        consistency_score=0.5,
        stability_score=0.5,
        concentration_score=0.5,
    )
    assert 0.0 <= out <= 1.0


# ============================================================
# Task B2: stability_score and concentration_score
# ============================================================

def test_compute_stability_score_zero_stddev_returns_one():
    """Constant monthly scores → stddev=0 → stability=1.0."""
    scores = [0.5] * 12
    s = compute_stability_score(scores)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_compute_stability_score_high_stddev_returns_zero():
    """Stddev ≥ 0.3 → stability=0.0."""
    scores = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # high variance
    s = compute_stability_score(scores)
    assert s == 0.0


def test_compute_stability_score_interpolation():
    """Stddev between 0 and 0.3 → linear interpolation."""
    # Stddev of [0.0, 0.2] = 0.1 → 1 - 0.1/0.3 = 0.667
    scores = [0.0, 0.2]
    s = compute_stability_score(scores)
    assert s == pytest.approx(1.0 - 0.1 / 0.3, abs=1e-3)


def test_compute_stability_score_empty_returns_neutral():
    """Empty list → neutral 1.0 (avoid bias)."""
    assert compute_stability_score([]) == 1.0


def test_compute_concentration_score_low_concentration_returns_one():
    """Top month ≤ 30% of profit → concentration_score = 1.0."""
    # top=30, total=100 → share=0.3 → 1.0
    monthly_profits = [30.0, 25.0, 20.0, 15.0, 10.0]
    s = compute_concentration_score(monthly_profits)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_compute_concentration_score_high_concentration_returns_zero():
    """Top month ≥ 70% of profit → concentration_score = 0.0."""
    # top=80, total=100 → share=0.8 → 0.0
    monthly_profits = [80.0, 10.0, 5.0, 3.0, 2.0]
    s = compute_concentration_score(monthly_profits)
    assert s == 0.0


def test_compute_concentration_score_interpolation():
    """Top share between 0.3 and 0.7 → linear penalty."""
    # top=50, total=100 → share=0.5 → 1 - (0.5-0.3)/0.4 = 0.5
    monthly_profits = [50.0, 25.0, 15.0, 5.0, 5.0]
    s = compute_concentration_score(monthly_profits)
    assert s == pytest.approx(0.5, abs=1e-3)


def test_compute_concentration_score_no_profit_returns_one():
    """All months lost or zero profit → no concentration penalty (1.0)."""
    monthly_profits = [-10.0, -5.0, 0.0]
    s = compute_concentration_score(monthly_profits)
    # No positive profit → no top month → no penalty
    assert s == 1.0


def test_compute_concentration_score_empty_returns_one():
    """Empty list → no penalty (1.0)."""
    assert compute_concentration_score([]) == 1.0


# ============================================================
# Task B3: Integration — realistic scenario
# ============================================================

def test_realistic_recovering_strategy_scores_higher_than_one_lucky_month():
    """A recovering strategy should score higher than one with a single lucky month."""
    # Strategy A: steady recovery, no concentration
    out_a = compute_discovery_fitness(
        full_period_base_score=0.80,
        recovery_score=0.80,      # fast recovery, high bounce
        consistency_score=0.55,   # consistent
        stability_score=0.80,     # low stddev
        concentration_score=1.00, # no concentration
    )
    # Strategy B: high base score but driven by one lucky month, no recovery
    out_b = compute_discovery_fitness(
        full_period_base_score=0.95,  # higher full-period score
        recovery_score=0.10,          # poor recovery
        consistency_score=0.30,       # inconsistent
        stability_score=0.20,         # high stddev
        concentration_score=0.10,     # 70% concentration
    )
    # Despite B having higher base score, A should win because recovery/concentration
    # are heavily penalised and consistency/stability reward A.
    assert out_a > out_b, (
        f"Strategy A (recovering) should beat Strategy B (lucky month): "
        f"A={out_a:.4f}, B={out_b:.4f}"
    )
