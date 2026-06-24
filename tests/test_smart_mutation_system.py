"""Tests for the smart mutation system (Pitfall #13, 2026-06-25).

Covers:
- IslandIntelligenceTracker: niche, correlations, backtest patterns
- FamilyReasoning: per-family hints + helpers
- BacktestReader: regime classification + analysis
- SmartMutator: strategy computation + family-aware mutation
- SmartMutator behavior per island (one test per family)
- Edge cases: cold start, stagnation, niche-leaving
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest


# ============================================================
# Test helpers — build minimal CandidateGenome / EvaluationResult
# ============================================================

def _make_genome(
    genome_id: str,
    grid_method: str = "fixed_pct",
    grid_params: dict | None = None,
    allocation_method: str = "equal",
    allocation_params: dict | None = None,
    indicators: list[str] | None = None,
    indicator_params: dict | None = None,
    island_id: int = 1,
    generation_index: int = 1,
):
    """Build a minimal CandidateGenome with lineage tags."""
    from genome.schema import (
        AllocationMethod,
        CandidateGenome,
        ConfirmationIndicator,
        DcaGenome,
        GridMethod,
        LineageMetadata,
        TpGenome,
    )
    if indicators is None:
        indicators = []
    if indicator_params is None:
        indicator_params = {}
    if allocation_params is None:
        allocation_params = {}
    if grid_params is None:
        grid_params = {"pct": 0.005, "max_layers": 5, "tp_pct": 0.005, "cooldown_candles": 4}

    # Convert string indicator names to enum
    indicator_enums = []
    for name in indicators:
        try:
            indicator_enums.append(ConfirmationIndicator(name))
        except ValueError:
            pass

    dca = DcaGenome(
        grid_method=GridMethod(grid_method),
        grid_params=dict(grid_params),
        allocation_method=AllocationMethod(allocation_method),
        allocation_params=dict(allocation_params),
        confirmation_indicators=indicator_enums,
        indicator_params=dict(indicator_params),
        max_dca_layers=grid_params.get("max_layers", 5),
    )
    tp = TpGenome(exit_method="fixed", exit_params={"tp_pct": grid_params.get("tp_pct", 0.005)})
    lineage = LineageMetadata(
        generation_index=generation_index,
        mutation_ops=[{"op": "island_assign", "island_id": island_id}],
    )
    return CandidateGenome(
        genome_id=genome_id,
        dca_genome=dca,
        tp_genome=tp,
        lineage=lineage,
    )


def _make_eval_result(
    candidate_id: str,
    genome_id: str,
    discovery_fitness: float = 0.7,
    monthly_scores: list | None = None,
    recovery_breakdown: dict | None = None,
):
    """Build a minimal EvaluationResult for testing."""
    from evolution.evaluator import EvaluationResult
    from fitness.monthly_fitness import MonthlyFitnessResult, MonthlyScore

    if monthly_scores is None:
        monthly_scores = []
    if recovery_breakdown is None:
        recovery_breakdown = {}

    mf = MonthlyFitnessResult(
        candidate_id=candidate_id,
        experiment_slug="test",
        monthly_scores=monthly_scores,
        n_months=len(monthly_scores),
        n_profitable_months=sum(1 for m in monthly_scores if not m.rejected and m.net_profit_pct > 0),
        n_rejected_months=sum(1 for m in monthly_scores if m.rejected),
        consistency_ratio=0.6,
        median_monthly_score=0.65,
        worst_month_score=0.4,
        stddev_monthly_score=0.1,
        variance_penalty=0.05,
        worst_floor_multiplier=1.0,
        base_aggregate_fitness=discovery_fitness,
        discovery_fitness=discovery_fitness,
        consistency_multiplier=1.0,
        full_period_base_score=discovery_fitness,
        recovery_score=0.7,
        stability_score=0.9,
        concentration_score=1.0,
        recovery_breakdown=recovery_breakdown,  # forward test kwarg
        per_month_recovery=[],
        deployment_fitness=discovery_fitness if discovery_fitness >= 0.65 else 0.0,
        deployment_pass=discovery_fitness >= 0.65,
        failed_deployment_gates=[],
        closest_to_passing_score=1.0,
        full_period_score=discovery_fitness,
        full_period_rejected=False,
        final_fitness=discovery_fitness,
        rejected=False,
        reject_reason=None,
    )
    return EvaluationResult(
        candidate_id=candidate_id,
        genome_id=genome_id,
        discovery_fitness=discovery_fitness,
        deployment_fitness=discovery_fitness,
        deployment_pass=discovery_fitness >= 0.65,
        failed_deployment_gates=[],
        closest_to_passing_score=1.0,
        consistency_ratio=0.6,
        consistency_multiplier=1.0,
        full_period_base_score=discovery_fitness,
        recovery_score=0.7,
        stability_score=0.9,
        concentration_score=1.0,
        recovery_breakdown=recovery_breakdown,
        rejected=False,
        reject_reason=None,
        elapsed_seconds=0.1,
        monthly_fitness=mf,
        score_breakdown=None,
        raw_metrics={},
        n_cycles_closed=100,
        final_equity=10500.0,
        max_dd_pct=0.05,
    )


def _make_monthly_score(
    month_label: str = "2024-01",
    net_profit_pct: float = 2.0,
    max_drawdown_pct: float = 3.0,
    monthly_score: float = 0.65,
    rejected: bool = False,
    trades_per_month: float = 20.0,
) -> "MonthlyScore":
    from fitness.monthly_fitness import MonthlyScore
    return MonthlyScore(
        month_index=1,
        month_label=month_label,
        start="2024-01-01T00:00:00",
        end="2024-01-31T23:59:59",
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        trades_per_month=trades_per_month,
        total_trades=int(trades_per_month),
        monthly_score=monthly_score,
        rejected=rejected,
        reject_reason=None,
        final_equity=10000.0 * (1 + net_profit_pct / 100),
        initial_equity=10000.0,
        recovery_subscores={},
    )


# ============================================================
# FamilyReasoning tests
# ============================================================

class TestFamilyReasoning:
    def test_all_8_families_have_hints(self):
        from evolution.family_reasoning import all_family_hints, FAMILY_HINTS
        assert len(FAMILY_HINTS) == 8
        expected = {"fixed_pct", "atr", "volatility_or_dd", "trend",
                    "oscillator", "vola_adj_alloc", "ctrl_exp_alloc", "tight_dca"}
        assert set(FAMILY_HINTS.keys()) == expected

    def test_get_hint_for_island(self):
        from evolution.family_reasoning import get_hint_for_island
        for iid in range(1, 9):
            h = get_hint_for_island(iid)
            assert h is not None
            assert h.bias_name  # non-empty

    def test_trend_hint_recommends_volatility_high(self):
        """I4 (trend) should suggest volatility_high for chop filtering."""
        from evolution.family_reasoning import get_hint_for_island
        h = get_hint_for_island(4)  # trend
        assert "volatility_high" in h.indicator_suggestions

    def test_volatility_or_dd_recommends_low_max_layers(self):
        """I3 (volatility_or_dd) should dampen max_layers."""
        from evolution.family_reasoning import get_hint_for_island
        h = get_hint_for_island(3)  # volatility_or_dd
        assert "max_layers" in h.params_to_dampen

    def test_fixed_pct_freezes_grid_method(self):
        """I1 (fixed_pct) should never swap grid_method."""
        from evolution.family_reasoning import get_hint_for_island
        h = get_hint_for_island(1)  # fixed_pct
        assert "grid_method" in h.params_to_freeze

    def test_vola_adj_alloc_freezes_allocation(self):
        """I6 (vola_adj_alloc) should never swap allocation_method."""
        from evolution.family_reasoning import get_hint_for_island
        h = get_hint_for_island(6)  # vola_adj_alloc
        assert "allocation_method" in h.params_to_freeze

    def test_regime_aware_flag(self):
        """I3, I4 (trend/volatility) are regime-aware; I1, I5, I8 are not."""
        from evolution.family_reasoning import get_hint_for_island
        assert get_hint_for_island(3).regime_aware is True   # volatility
        assert get_hint_for_island(4).regime_aware is True   # trend
        assert get_hint_for_island(1).regime_aware is False  # fixed_pct
        assert get_hint_for_island(5).regime_aware is False  # oscillator
        assert get_hint_for_island(8).regime_aware is False  # tight_dca

    def test_param_std_multipliers_defined(self):
        """Every family should have at least one std multiplier override."""
        from evolution.family_reasoning import all_family_hints
        for name, h in all_family_hints().items():
            assert len(h.param_std_multipliers) > 0, f"{name} missing multipliers"


# ============================================================
# BacktestReader tests
# ============================================================

class TestRegimeClassification:
    def test_bull_regime(self):
        from evolution.backtest_reader import classify_regime
        assert classify_regime(10.0) == "bull"
        assert classify_regime(5.5) == "bull"

    def test_bear_regime(self):
        from evolution.backtest_reader import classify_regime
        assert classify_regime(-5.0) == "bear"
        assert classify_regime(-2.5) == "bear"

    def test_chop_regime(self):
        from evolution.backtest_reader import classify_regime
        assert classify_regime(2.0) == "chop"
        assert classify_regime(-1.0) == "chop"
        assert classify_regime(0.0) == "chop"


class TestBacktestReader:
    def test_analyze_with_empty_results(self):
        from evolution.backtest_reader import BacktestReader
        reader = BacktestReader()
        summary = reader.analyze(island_id=1, gen_index=0, results=[])
        assert summary.sample_size == 0
        assert summary.monthly_consistency == 0.0

    def test_analyze_with_mixed_regimes(self):
        from evolution.backtest_reader import BacktestReader
        monthly = [
            _make_monthly_score("2024-01", net_profit_pct=10.0, monthly_score=0.8),  # bull
            _make_monthly_score("2024-02", net_profit_pct=-5.0, monthly_score=0.4),  # bear
            _make_monthly_score("2024-03", net_profit_pct=2.0, monthly_score=0.6),   # chop
            _make_monthly_score("2024-04", net_profit_pct=12.0, monthly_score=0.85), # bull
            _make_monthly_score("2024-05", net_profit_pct=-3.0, monthly_score=0.45), # bear
        ]
        results = [_make_eval_result("c1", "g1", monthly_scores=monthly)]
        summary = BacktestReader().analyze(1, 0, results)
        assert summary.regime.bull_n_months == 2
        assert summary.regime.bear_n_months == 2
        assert summary.regime.chop_n_months == 1
        assert summary.regime.best_regime == "bull"  # avg 0.825 > bear 0.425 > chop 0.6
        assert summary.dd.max_dd_pct > 0
        assert summary.frequency.total_trades_recent > 0

    def test_recovery_metrics(self):
        from evolution.backtest_reader import BacktestReader
        rb = {
            "drawdown_recovery_speed": 0.75,
            "equity_high_reclaim_rate": 0.65,
            "post_loss_month_bounce_rate": 0.85,
            "cycle_recovery_health": 0.95,
        }
        results = [_make_eval_result("c1", "g1", recovery_breakdown=rb)]
        summary = BacktestReader().analyze(1, 0, results)
        assert summary.recovery.avg_recovery_speed == 0.75
        assert summary.recovery.avg_equity_high_reclaim_rate == 0.65
        assert summary.recovery.avg_post_loss_bounce_rate == 0.85

    def test_dd_hotspots_sorted_descending(self):
        from evolution.backtest_reader import BacktestReader
        monthly = [
            _make_monthly_score("2024-01", max_drawdown_pct=2.0),
            _make_monthly_score("2024-02", max_drawdown_pct=8.0),
            _make_monthly_score("2024-03", max_drawdown_pct=5.0),
            _make_monthly_score("2024-04", max_drawdown_pct=12.0),
            _make_monthly_score("2024-05", max_drawdown_pct=3.0),
            _make_monthly_score("2024-06", max_drawdown_pct=10.0),
        ]
        results = [_make_eval_result("c1", "g1", monthly_scores=monthly)]
        summary = BacktestReader().analyze(1, 0, results)
        dd_months = summary.dd.worst_5_months
        assert dd_months[0]["dd_pct"] >= dd_months[-1]["dd_pct"]
        assert summary.dd.max_dd_pct == 12.0


# ============================================================
# IslandIntelligence tests
# ============================================================

class TestIslandIntelligence:
    def test_niche_fingerprint_from_elites(self):
        from evolution.island_intelligence import IslandIntelligenceTracker
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tracker = IslandIntelligenceTracker(output_dir=tmp)
            elites = [
                _make_genome(f"g{i}", grid_params={"pct": 0.005 + i * 0.0001, "max_layers": 5})
                for i in range(10)
            ]
            niche = tracker._build_niche(elites, gen_index=0)
            assert "pct" in niche.centroid
            assert niche.centroid["pct"] == pytest.approx(0.00545, abs=1e-4)
            assert niche.sample_size == 10

    def test_cold_start_returns_empty_intel(self):
        from evolution.island_intelligence import IslandIntelligence
        intel = IslandIntelligence(island_id=1, bias_name="fixed_pct")
        assert intel.niche.sample_size == 0
        assert len(intel.correlations.correlations) == 0

    def test_to_from_dict_roundtrip(self):
        from evolution.island_intelligence import IslandIntelligence, BacktestPatterns
        intel = IslandIntelligence(island_id=5, bias_name="trend")
        intel.backtest.recent_best_fitness.append(0.71)
        intel.backtest.bull_avg_score = 0.8
        intel.niche.centroid["pct"] = 0.006
        intel.correlations.saturated_params.add("cooldown_candles")
        d = intel.to_dict()
        restored = IslandIntelligence.from_dict(d)
        assert restored.island_id == 5
        assert restored.bias_name == "trend"
        assert restored.backtest.bull_avg_score == 0.8
        assert "pct" in restored.niche.centroid
        assert "cooldown_candles" in restored.correlations.saturated_params

    def test_save_and_load(self):
        from evolution.island_intelligence import IslandIntelligenceTracker, IslandIntelligence
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tracker = IslandIntelligenceTracker(output_dir=tmp)
            intel = IslandIntelligence(island_id=3, bias_name="volatility_or_dd")
            intel.niche.centroid["drawdown_pct"] = 0.05
            tracker.save(intel)
            loaded = tracker.load(3)
            assert loaded is not None
            assert loaded.bias_name == "volatility_or_dd"
            assert loaded.niche.centroid["drawdown_pct"] == 0.05


# ============================================================
# SmartMutator tests
# ============================================================

class TestSmartMutatorStrategy:
    def test_cold_start_uses_exploration_heavy_strategy(self):
        """When no intelligence, strategy should bias toward exploration."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=1, intelligence=None)
        strategy = sm.compute_strategy()
        assert strategy.in_niche_probability < 0.5  # cold start = more exploration
        assert strategy.niche_leaving_probability > 0.10

    def test_stagnation_boosts_global_std(self):
        """Stagnant fitness should boost std to escape plateau."""
        from evolution.island_intelligence import IslandIntelligence, BacktestPatterns
        from evolution.smart_mutator import SmartMutator
        intel = IslandIntelligence(island_id=4, bias_name="trend")
        # Simulate stagnant fitness
        for f in [0.70, 0.7001, 0.7000, 0.7002, 0.6999]:
            intel.backtest.recent_best_fitness.append(f)
        sm = SmartMutator(island_id=4, intelligence=intel)
        strategy = sm.compute_strategy()
        assert strategy.global_std_multiplier >= 1.5  # stagnation boost
        assert any("Stagnation" in n for n in strategy.reasoning_notes)

    def test_improving_dampens_global_std(self):
        """Improving fitness should dampen std to fine-tune."""
        from evolution.island_intelligence import IslandIntelligence, BacktestPatterns
        from evolution.smart_mutator import SmartMutator
        intel = IslandIntelligence(island_id=1, bias_name="fixed_pct")
        # Simulate improving fitness
        for f in [0.60, 0.65, 0.70, 0.75, 0.80]:
            intel.backtest.recent_best_fitness.append(f)
        sm = SmartMutator(island_id=1, intelligence=intel)
        strategy = sm.compute_strategy()
        assert strategy.global_std_multiplier <= 1.0  # dampened or unchanged

    def test_family_hint_boosts_explore_more_params(self):
        """Params in 'explore_more' should get >1.0 std multiplier."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=3, intelligence=None)  # volatility_or_dd
        strategy = sm.compute_strategy()
        # I3's hint has drawdown_pct in explore_more
        assert strategy.per_param_std_multiplier.get("drawdown_pct", 1.0) > 1.0

    def test_family_hint_dampens_dampen_params(self):
        """Params in 'dampen' should get <1.0 std multiplier."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=3, intelligence=None)
        strategy = sm.compute_strategy()
        # I3's hint has max_layers in dampen
        assert strategy.per_param_std_multiplier.get("max_layers", 1.0) < 1.0

    def test_std_multiplier_clamped_to_max(self):
        """Per-param std multiplier should never exceed MAX_STD_MULT."""
        from evolution.island_intelligence import IslandIntelligence, BacktestPatterns
        from evolution.smart_mutator import SmartMutator
        intel = IslandIntelligence(island_id=4, bias_name="trend")
        for f in [0.70, 0.70, 0.70, 0.70, 0.70]:
            intel.backtest.recent_best_fitness.append(f)
        sm = SmartMutator(island_id=4, intelligence=intel)
        strategy = sm.compute_strategy()
        for mult in strategy.per_param_std_multiplier.values():
            assert mult <= sm.MAX_STD_MULT


class TestSmartMutatorBehavior:
    """Per-family behavior tests."""

    def _make_simple_parent(self, island_id: int = 1):
        return _make_genome(
            "parent_1",
            grid_method="fixed_pct",
            grid_params={"pct": 0.005, "max_layers": 5, "tp_pct": 0.005, "cooldown_candles": 4},
            island_id=island_id,
        )

    def test_island_1_fixed_pct_never_swaps_grid_method(self):
        """Fixed_pct has frozen grid_method — 100 mutations, never swapped."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=1)
        rng = random.Random(42)
        parent = self._make_simple_parent(island_id=1)
        # 100 mutations, check grid_method
        for _ in range(100):
            child = sm.mutate(parent, rng=rng)
            assert child.dca_genome.grid_method.value == "fixed_pct"

    def test_island_6_vola_adj_alloc_never_swaps_allocation(self):
        """vola_adj_alloc has frozen allocation_method — 100 mutations, never swapped."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=6)
        rng = random.Random(42)
        parent = _make_genome(
            "p",
            grid_method="atr",
            allocation_method="volatility_adjusted",
            allocation_params={"reference_vol": 0.02, "min_size_pct": 0.5, "max_size_pct": 3.0},
            island_id=6,
        )
        for _ in range(100):
            child = sm.mutate(parent, rng=rng)
            assert child.dca_genome.allocation_method.value == "volatility_adjusted"

    def test_mutation_produces_valid_genome(self):
        """Mutation result should have valid params (within ranges)."""
        from evolution.smart_mutator import SmartMutator
        from evolution.operators import DCA_PARAM_RANGES, GLOBAL_MAX_DCA_LAYERS
        sm = SmartMutator(island_id=4)  # trend
        rng = random.Random(42)
        parent = self._make_simple_parent(island_id=4)
        for _ in range(50):
            child = sm.mutate(parent, rng=rng)
            # Check pct in range
            lo, hi = DCA_PARAM_RANGES["grid_pct"]
            assert lo <= child.dca_genome.grid_params["pct"] <= hi
            # Check max_layers in range and <= cap
            assert 2 <= child.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS
            # Check tp_pct in range
            lo, hi = DCA_PARAM_RANGES["tp_pct"]
            assert lo <= child.dca_genome.grid_params["tp_pct"] <= hi

    def test_lineage_records_smart_mutation(self):
        """Child lineage should record smart_mutate op with island_id and zone."""
        from evolution.smart_mutator import SmartMutator
        sm = SmartMutator(island_id=4)
        rng = random.Random(42)
        parent = self._make_simple_parent(island_id=4)
        child = sm.mutate(parent, rng=rng)
        ops = child.lineage.mutation_ops
        smart_op = next((op for op in ops if op.get("op") == "smart_mutate"), None)
        assert smart_op is not None
        assert smart_op["island_id"] == 4
        assert smart_op["zone"] in ("in_niche", "boundary", "leaving")

    def test_crossover_then_smart_mutate_works(self):
        """SmartMutator should handle children produced by crossover."""
        from evolution.smart_mutator import SmartMutator
        from evolution.operators import crossover
        sm = SmartMutator(island_id=2)
        rng = random.Random(42)
        a = self._make_simple_parent(island_id=2)
        b = _make_genome(
            "p2",
            grid_method="atr",
            grid_params={"pct": 0.008, "max_layers": 6, "tp_pct": 0.007, "cooldown_candles": 2,
                         "atr_multiplier": 2.5},
            island_id=2,
        )
        child = crossover(a, b, rng=rng)
        # SmartMutator should mutate it without crashing
        mutated = sm.mutate(child, rng=rng)
        assert mutated.dca_genome.grid_method.value in ("fixed_pct", "atr")

    def test_niche_pull_when_niche_defined(self):
        """When niche centroid is set, in-niche mutations should pull toward it."""
        from evolution.island_intelligence import IslandIntelligence, NicheFingerprint
        from evolution.smart_mutator import SmartMutator
        intel = IslandIntelligence(island_id=1, bias_name="fixed_pct")
        intel.niche = NicheFingerprint(
            centroid={"pct": 0.010},  # target higher than parent
            std={"pct": 0.001},
            sample_size=20,  # well-defined niche
        )
        intel.backtest.recent_best_fitness.extend([0.70, 0.71, 0.72])  # improving
        sm = SmartMutator(island_id=1, intelligence=intel)
        strategy = sm.compute_strategy()
        # In-niche prob should be high
        assert strategy.in_niche_probability >= 0.6
        rng = random.Random(42)
        parent = _make_genome(
            "p", grid_method="fixed_pct",
            grid_params={"pct": 0.005, "max_layers": 5, "tp_pct": 0.005, "cooldown_candles": 4},
            island_id=1,
        )
        # 100 mutations; check that pct drifts upward (toward 0.010)
        results = []
        for _ in range(100):
            child = sm.mutate(parent, rng=rng, strategy=strategy)
            results.append(child.dca_genome.grid_params["pct"])
        # Average pct should be > parent's 0.005 (pull toward 0.010)
        avg_pct = sum(results) / len(results)
        assert avg_pct > 0.005, f"avg_pct={avg_pct} (expected > 0.005)"


class TestSmartMutatorPerIsland:
    """One test per island — verifies family-specific behavior is wired correctly."""

    @pytest.mark.parametrize("island_id,bias_name,expected_method", [
        (1, "fixed_pct", "fixed_pct"),
        (2, "atr", "atr"),
        (3, "volatility_or_dd", None),  # either volatility or drawdown_from_high
        (4, "trend", None),              # either ma_distance or trend_adjusted
        (5, "oscillator", None),         # either rsi_oversold or z_score
        (6, "vola_adj_alloc", None),     # any grid, forced allocation
        (7, "ctrl_exp_alloc", None),     # any grid, forced allocation
        (8, "tight_dca", None),          # any grid, max_layers capped
    ])
    def test_island_family_hint_resolved(self, island_id, bias_name, expected_method):
        """Every island should have a non-None FamilyHint with correct bias."""
        from evolution.smart_mutator import SmartMutator
        from evolution.family_reasoning import get_hint_for_island
        h = get_hint_for_island(island_id)
        assert h is not None
        assert h.bias_name == bias_name
        sm = SmartMutator(island_id=island_id)
        assert sm.family_hint is not None
        assert sm.family_hint.bias_name == bias_name


# ============================================================
# Integration test — all 5 components working together
# ============================================================

class TestSmartMutationIntegration:
    def test_full_loop_one_gen(self):
        """Build a small genome population, run smart mutator on each, verify it works end-to-end."""
        from evolution.smart_mutator import SmartMutator
        from evolution.island_intelligence import (
            IslandIntelligence, IslandIntelligenceTracker, NicheFingerprint
        )
        from evolution.backtest_reader import BacktestReader
        from evolution.operators import GLOBAL_MAX_DCA_LAYERS
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # Set up intel tracker
            tracker = IslandIntelligenceTracker(output_dir=tmp)

            # Build 20 elites for I4 (trend) with varied params
            elites = []
            for i in range(20):
                g = _make_genome(
                    f"elite_{i}",
                    grid_method=["ma_distance", "trend_adjusted"][i % 2],
                    grid_params={
                        "pct": 0.005 + i * 0.0001,
                        "max_layers": 4 + (i % 4),
                        "tp_pct": 0.006,
                        "cooldown_candles": 4,
                        "ma_distance_pct": 0.025 + i * 0.001,
                    },
                    island_id=4,
                )
                elites.append(g)

            # Build some eval results with monthly scores
            monthly = [
                _make_monthly_score(f"2024-{m:02d}", net_profit_pct=m - 5, monthly_score=0.6 + m * 0.01)
                for m in range(1, 13)
            ]
            results = [
                _make_eval_result(f"c{i}", f"elite_{i}", monthly_scores=monthly)
                for i in range(20)
            ]

            # Update intel
            intel = tracker.update(island_id=4, gen_index=1, eval_results=results, elites=elites)
            assert intel.niche.sample_size == 20
            assert "ma_distance_pct" in intel.niche.centroid

            # Now run smart mutator
            sm = SmartMutator(island_id=4, intelligence=intel)
            strategy = sm.compute_strategy()
            assert strategy.global_std_multiplier > 0  # sanity

            # Mutate each elite
            rng = random.Random(42)
            children = []
            for e in elites:
                child = sm.mutate(e, rng=rng, strategy=strategy)
                children.append(child)
            assert len(children) == 20
            # Each child should have a valid grid_method (any — swap is allowed)
            for c in children:
                assert c.dca_genome.grid_method is not None
                assert c.dca_genome.max_dca_layers <= GLOBAL_MAX_DCA_LAYERS
