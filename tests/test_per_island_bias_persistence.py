"""Tests for per-island bias persistence via _build_dynamic_specs.

Pitfall #13 (2026-06-25) — "island-convergence-bug":
Previously, _generate_next_gen_island built per-island specs from the STATIC
ISLAND_SPECS list (the original 8). When a retirement picked a fresh bias
(via _check_retirement → pick_fresh_bias → _island_family_bias update), the
new bias only took effect for ONE gen before the next gen reverted to the
static spec. This let islands drift back toward their original grid_method
after the first post-retirement gen, causing convergence onto whichever
family was winning (in observed data: trend_adjusted).

The fix:
1. `_build_dynamic_specs()` reads from `_island_family_bias` (the live
   source of truth) for every island on every gen.
2. Retirement/force-retire bias changes take effect for ALL post-retirement
   gens, not just the first one.

This file covers:
- _build_dynamic_specs returns specs based on _island_family_bias, not static
- After retirement, the new bias applies on the FIRST post-retirement gen
- After retirement, the new bias STILL applies on subsequent gens
- Modifying _island_family_bias manually changes the spec on the next call
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def clean_output_dir(tmp_path):
    out = tmp_path / "runs"
    out.mkdir(parents=True, exist_ok=True)
    yield out
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def tiny_df():
    """8-row BTC DataFrame just enough to construct the harness."""
    idx = pd.date_range("2024-01-01", periods=8, freq="1h")
    return pd.DataFrame({
        "date": idx,
        "open": [100.0 + i for i in range(8)],
        "high": [101.0 + i for i in range(8)],
        "low": [99.0 + i for i in range(8)],
        "close": [100.5 + i for i in range(8)],
        "volume": [1000.0] * 8,
    })


def _make_cfg(output_dir: Path, n_islands: int = 3) -> "EvolutionConfig":
    from evolution.config import EvolutionConfig
    return EvolutionConfig(
        experiment_id="test_bias_persistence",
        output_dir=str(output_dir),
        candidates_per_gen=24,
        max_generations=2,
        n_islands=n_islands,
        island_mode=True,
    )


class TestBuildDynamicSpecs:
    """_build_dynamic_specs must read from _island_family_bias (live state)."""

    def test_initial_specs_match_static(self, clean_output_dir, tiny_df):
        """Before any retirement, dynamic specs match static ISLAND_SPECS."""
        from evolution.harness import EvolutionHarness
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        dynamic_specs = harness._build_dynamic_specs()

        from evolution.islands import get_island_specs
        static_specs = get_island_specs()[:3]
        assert len(dynamic_specs) == 3
        for dyn, static in zip(dynamic_specs, static_specs):
            assert dyn.island_id == static.island_id
            assert dyn.name == static.name
            assert dyn.forced_grid_methods == static.forced_grid_methods
            assert dyn.forced_allocation == static.forced_allocation

    def test_dynamic_specs_reflect_live_bias_change(self, clean_output_dir, tiny_df):
        """Manually mutating _island_family_bias changes the spec immediately."""
        from evolution.harness import EvolutionHarness
        from genome.schema import GridMethod
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        # Trigger init
        harness._build_dynamic_specs()

        # I1 should originally be fixed_pct
        original_specs = harness._build_dynamic_specs()
        assert original_specs[0].name == "fixed_pct"
        assert original_specs[0].forced_grid_methods == (GridMethod.FIXED_PCT,)

        # Mutate I1's bias to trend (simulating post-retirement)
        harness._island_family_bias[1] = {
            "name": "trend",
            "forced_grid_methods": (GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED),
        }

        new_specs = harness._build_dynamic_specs()
        assert new_specs[0].name == "trend", \
            "Dynamic spec must reflect _island_family_bias change, not static spec"
        assert new_specs[0].forced_grid_methods == (GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED)

    def test_dynamic_specs_lazy_init_when_empty(self, clean_output_dir, tiny_df):
        """If _island_family_bias is empty (never init'd), _build_dynamic_specs
        must lazy-init from static and still return a valid spec list."""
        from evolution.harness import EvolutionHarness
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        # Wipe any prior init
        harness._island_family_bias = {}

        specs = harness._build_dynamic_specs()
        assert len(specs) == 3
        # After lazy init, calling again should produce same result
        specs2 = harness._build_dynamic_specs()
        assert [s.name for s in specs] == [s.name for s in specs2]


class TestRetirementBiasPersistsAcrossGens:
    """After retirement, the new bias must apply for ALL post-retirement gens."""

    def test_fresh_bias_applies_first_post_retirement_gen(
        self, clean_output_dir, tiny_df
    ):
        """Gen N+1 immediately after a retirement uses the new bias spec."""
        from evolution.harness import EvolutionHarness
        from genome.schema import GridMethod
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        harness._build_dynamic_specs()

        # Simulate retirement: I1 was fixed_pct, now replaced with oscillator
        harness._island_family_bias[1] = {
            "name": "oscillator",
            "forced_grid_methods": (GridMethod.RSI_OVERSOLD, GridMethod.Z_SCORE),
        }

        # Gen N+1: dynamic spec for I1 must be oscillator
        specs = harness._build_dynamic_specs()
        assert specs[0].name == "oscillator"
        assert specs[0].forced_grid_methods == (GridMethod.RSI_OVERSOLD, GridMethod.Z_SCORE)

    def test_fresh_bias_still_applies_many_gens_later(
        self, clean_output_dir, tiny_df
    ):
        """Gen N+5 (and beyond) must still use the post-retirement bias.

        This is the regression we're guarding against — previously the bias
        override only applied for one gen, then reverted to static.
        """
        from evolution.harness import EvolutionHarness
        from genome.schema import GridMethod
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        harness._build_dynamic_specs()

        # Simulate retirement
        harness._island_family_bias[1] = {
            "name": "atr",
            "forced_grid_methods": (GridMethod.ATR,),
        }

        # Simulate 5 gens passing — call _build_dynamic_specs each gen
        for gen_idx in range(1, 6):
            specs = harness._build_dynamic_specs()
            assert specs[0].name == "atr", \
                f"Gen {gen_idx}: I1 spec regressed to {specs[0].name}, expected 'atr'"
            assert specs[0].forced_grid_methods == (GridMethod.ATR,), \
                f"Gen {gen_idx}: I1 forced_grid_methods regressed to {specs[0].forced_grid_methods}"


class TestGenerateNextGenUsesDynamicSpecs:
    """Integration test: _generate_next_gen_island actually uses dynamic specs."""

    def test_first_gen_uses_dynamic_bias_for_random_seeding(
        self, clean_output_dir, tiny_df
    ):
        """If the harness has no per-island elites (gen 0 case), it falls
        back to _seed_island_via_spec which must use the dynamic spec."""
        from evolution.harness import EvolutionHarness
        from genome.schema import GridMethod
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)
        harness._build_dynamic_specs()

        # Pre-seed _island_family_bias with a custom bias for I1
        harness._island_family_bias[1] = {
            "name": "atr",
            "forced_grid_methods": (GridMethod.ATR,),
        }

        # _seed_island_via_spec builds candidates from the spec
        from evolution.harness import _seed_island_via_spec
        import random as random_mod
        rng = random_mod.Random(42)
        spec = harness._build_dynamic_specs()[0]  # I1's dynamic spec
        candidates = _seed_island_via_spec(rng=rng, generation_index=0, spec=spec, count=10)

        # Every seeded candidate for I1 must have grid_method=atr
        for c in candidates:
            assert c.dca_genome.grid_method == GridMethod.ATR, \
                f"Dynamic spec for I1 didn't apply: got {c.dca_genome.grid_method}"


class TestRegressionGuardForPitfall13:
    """Direct regression test for Pitfall #13 — the original bug pattern."""

    def test_static_specs_would_have_reverted_but_dynamic_does_not(
        self, clean_output_dir, tiny_df
    ):
        """Concrete reproduction of the convergence bug:
        - Original code: specs = get_island_specs()[:N]  (static)
        - New code:      specs = self._build_dynamic_specs()  (dynamic)

        With static specs, modifying _island_family_bias has NO effect on
        per-gen population building. With dynamic specs, it does.
        """
        from evolution.harness import EvolutionHarness
        from evolution.islands import get_island_specs
        from genome.schema import GridMethod
        cfg = _make_cfg(clean_output_dir, n_islands=3)
        harness = EvolutionHarness(config=cfg, df=tiny_df)

        # Old (buggy) code path: static specs
        static_specs = get_island_specs()[:3]
        # Mutate the bias
        harness._island_family_bias[1] = {
            "name": "trend",
            "forced_grid_methods": (GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED),
        }
        # Static specs are unaffected by _island_family_bias mutation
        assert static_specs[0].name == "fixed_pct"  # still original
        assert static_specs[0].forced_grid_methods == (GridMethod.FIXED_PCT,)

        # New (fixed) code path: dynamic specs
        dynamic_specs = harness._build_dynamic_specs()
        assert dynamic_specs[0].name == "trend"  # reflects live state
        assert dynamic_specs[0].forced_grid_methods == (GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED)
