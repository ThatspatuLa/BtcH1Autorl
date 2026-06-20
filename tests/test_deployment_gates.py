"""Stage 6.5 — Discovery vs Deployment Gates.

Tests for the penalty curve, two-stage fitness, deployment gate logic,
and the closest_to_passing_score diagnostic.
"""
from __future__ import annotations

import pytest

from fitness.deployment_gates import (
    CONSISTENCY_PENALTY_TABLE,
    DEPLOYMENT_MAX_DD_PCT,
    DEPLOYMENT_MIN_CONSISTENCY,
    DEPLOYMENT_MIN_TOTAL_TRADES,
    DEPLOYMENT_MIN_TRADES_PER_MONTH,
    compute_deployment_gates,
    consistency_multiplier,
)

# ============================================================
# consistency_multiplier
# ============================================================

class TestConsistencyMultiplier:
    """The locked penalty curve."""

    def test_above_50_pct_full_credit(self):
        assert consistency_multiplier(1.0) == 1.00
        assert consistency_multiplier(0.75) == 1.00
        assert consistency_multiplier(0.50) == 1.00  # inclusive boundary

    def test_40_to_50_band(self):
        assert consistency_multiplier(0.49) == 0.85
        assert consistency_multiplier(0.45) == 0.85
        assert consistency_multiplier(0.40) == 0.85  # inclusive boundary

    def test_30_to_40_band(self):
        assert consistency_multiplier(0.39) == 0.65
        assert consistency_multiplier(0.35) == 0.65
        assert consistency_multiplier(0.30) == 0.65  # inclusive boundary

    def test_20_to_30_band(self):
        assert consistency_multiplier(0.29) == 0.40
        assert consistency_multiplier(0.25) == 0.40
        assert consistency_multiplier(0.20) == 0.40  # inclusive boundary

    def test_below_20(self):
        assert consistency_multiplier(0.19) == 0.15
        assert consistency_multiplier(0.10) == 0.15
        assert consistency_multiplier(0.0) == 0.15

    def test_curve_is_monotonically_nondecreasing(self):
        """The curve should only ever go up as consistency increases."""
        last = 0.0
        for c in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            m = consistency_multiplier(c)
            assert m >= last, f"non-monotonic at consistency={c}: {m} < {last}"
            last = m


# ============================================================
# compute_deployment_gates
# ============================================================

class TestComputeDeploymentGates:
    """The full two-stage fitness picture."""

    def test_perfect_candidate_passes_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is True
        assert g.failed_deployment_gates == []
        assert g.deployment_fitness == pytest.approx(0.75)  # = discovery (consistency ≥ 0.50 → mult 1.0)
        assert g.discovery_fitness == pytest.approx(0.75)

    def test_low_consistency_blocks_deployment_but_not_discovery(self):
        g = compute_deployment_gates(
            consistency_ratio=0.30,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        # Discovery fitness = base × mult (0.65 in 0.30-0.40 band)
        assert g.consistency_multiplier == pytest.approx(0.65)
        assert g.discovery_fitness == pytest.approx(0.75 * 0.65)
        # Deployment fails on consistency
        assert g.deployment_pass is False
        assert g.failed_deployment_gates[0].startswith("consistency<")
        assert g.deployment_fitness == 0.0

    def test_drawdown_above_35_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.40,  # > 35%
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert any("drawdown>" in g for g in g.failed_deployment_gates)

    def test_margin_failure_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=True,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert "margin_failure" in g.failed_deployment_gates

    def test_dca_completion_failure_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=True,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert "dca_completion_failure" in g.failed_deployment_gates

    def test_invalid_equity_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=True,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert "invalid_equity" in g.failed_deployment_gates

    def test_low_tpm_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=2.0,  # < 5
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert any("tpm<" in g for g in g.failed_deployment_gates)

    def test_low_total_trades_blocks_deployment(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=10,  # < 30
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert g.deployment_pass is False
        assert any("total_trades<" in g for g in g.failed_deployment_gates)

    def test_no_trades_skips_volume_gates(self):
        """A candidate with zero trades should not be double-penalised
        — volume gates are skipped when there's no volume to measure."""
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=0.0,
            total_trades=0,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        # Volume gates skipped (total_trades=0)
        assert not any("total_trades<" in x for x in g.failed_deployment_gates)
        assert not any("tpm<" in x for x in g.failed_deployment_gates)
        # No consistency fail, no DD fail, no safety fail → deployment passes
        assert g.deployment_pass is True

    def test_undetected_dd_does_not_penalise(self):
        """If DD is -1.0 (not measured), the DD gate should not fire."""
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=-1.0,  # not measured
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        assert not any("drawdown>" in g for g in g.failed_deployment_gates)


# ============================================================
# closest_to_passing_score
# ============================================================

class TestClosestToPassingScore:
    """The 0..1 diagnostic for near-miss candidates."""

    def test_passing_candidate_is_1(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        # All gates pass → should be at or near 1.0
        assert g.closest_to_passing_score == pytest.approx(1.0)

    def test_just_below_consistency_floor(self):
        g = compute_deployment_gates(
            consistency_ratio=0.49,  # just below 0.50
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        # Very close to passing — almost full credit on consistency
        assert g.closest_to_passing_score > 0.85

    def test_very_low_consistency_low_score(self):
        g = compute_deployment_gates(
            consistency_ratio=0.0,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=False,
            has_margin_failure=False,
            has_dca_completion_failure=False,
            base_aggregate_fitness=0.75,
        )
        # consistency=0 → 0.0 headroom on consistency (40% weight)
        # safety=3/3=1.0 (20% weight)
        # DD=0.20 < 0.35 → 1.0 (20% weight)
        # volume=1.0 (20% weight)
        # closest = 0.0*0.4 + 1.0*0.2 + 1.0*0.2 + 1.0*0.2 = 0.6
        assert g.closest_to_passing_score == pytest.approx(0.6)

    def test_safety_failures_zero_safety_headroom(self):
        g = compute_deployment_gates(
            consistency_ratio=0.80,
            max_drawdown_pct=0.20,
            trades_per_month=8.0,
            total_trades=200,
            has_invalid_equity=True,
            has_margin_failure=True,
            has_dca_completion_failure=True,
            base_aggregate_fitness=0.75,
        )
        # All safety gates fail → safety headroom = 0
        # Consistency passes → 1.0
        # DD passes → 1.0
        # Volume passes → 1.0
        # closest = 1.0*0.4 + 0.0*0.2 + 1.0*0.2 + 1.0*0.2 = 0.8
        assert g.closest_to_passing_score == pytest.approx(0.8)


# ============================================================
# to_dict / round-trip
# ============================================================

def test_deployment_gate_result_to_dict_has_all_fields():
    """All 13 reporting fields must be present in the dict output."""
    g = compute_deployment_gates(
        consistency_ratio=0.45,
        max_drawdown_pct=0.25,
        trades_per_month=6.0,
        total_trades=120,
        has_invalid_equity=False,
        has_margin_failure=False,
        has_dca_completion_failure=False,
        base_aggregate_fitness=0.65,
    )
    d = g.to_dict()
    required = {
        "consistency_ratio",
        "max_drawdown_pct",
        "trades_per_month",
        "total_trades",
        "has_invalid_equity",
        "has_margin_failure",
        "has_dca_completion_failure",
        "consistency_multiplier",
        "base_aggregate_fitness",
        "discovery_fitness",
        "deployment_fitness",
        "deployment_pass",
        "failed_deployment_gates",
        "closest_to_passing_score",
    }
    assert required.issubset(d.keys())
    assert len(required) == 14  # catch accidental additions


# ============================================================
# Constants
# ============================================================

def test_constants_locked():
    """The locked deployment floors must not drift."""
    assert DEPLOYMENT_MIN_CONSISTENCY == 0.50
    assert DEPLOYMENT_MAX_DD_PCT == 0.35
    assert DEPLOYMENT_MIN_TRADES_PER_MONTH == 5.0
    assert DEPLOYMENT_MIN_TOTAL_TRADES == 30
    # Penalty table is the locked 5-band curve
    assert len(CONSISTENCY_PENALTY_TABLE) == 5
    assert CONSISTENCY_PENALTY_TABLE[0] == (0.50, 1.00)
    assert CONSISTENCY_PENALTY_TABLE[-1] == (-1.00, 0.15)
