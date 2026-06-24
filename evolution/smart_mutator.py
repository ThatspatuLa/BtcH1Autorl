"""Smart mutator — family-aware, backtest-informed mutation.

Replaces blind Gaussian noise with intelligent per-island mutation that:
1. Reads IslandIntelligence (niche fingerprint, correlations, backtest patterns)
2. Reads FamilyHint (Six's domain knowledge per DCA family)
3. Adapts per-param mutation std based on:
   - Whether param is promising (high fitness correlation) → boost std
   - Whether param is saturated (low variance) → dampen std
   - Whether param is in family "explore_more" list → boost std
   - Whether param is in family "dampen" list → dampen std
4. Detects stagnation and boosts overall std to escape plateau
5. Biases mutations toward niche centroid (soft isolation)
6. Allows controlled niche-leaving (10% of mutations) for discovery

This is the "brain" of the smart mutation system. The user described wanting:
"for each family (DNA) of calculating the weight of the DCA it should be
different towards the next Island on how it mutates next"

Each family gets its own smart mutator instance with its own intelligence
state. Mutations are computed independently per island.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

from evolution.family_reasoning import FamilyHint, get_hint_for_island
from evolution.island_intelligence import (
    BacktestPatterns,
    IslandIntelligence,
    NicheFingerprint,
    ParamCorrelations,
)
from evolution.operators import (
    ALL_ALLOCATION_METHODS,
    ALL_CONFIRMATION_INDICATORS,
    ALL_GRID_METHODS,
    ALLOCATION_DEFAULT_PARAMS,
    ALLOCATION_PARAM_RANGES,
    DCA_PARAM_RANGES,
    GLOBAL_MAX_DCA_LAYERS,
    GRID_METHOD_DEFAULT_PARAMS,
    INDICATOR_DEFAULT_PARAMS,
    _random_allocation_params,
)
from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    DcaGenome,
    GridMethod,
    TpGenome,
)


# ============================================================
# Smart mutation strategy decision
# ============================================================

@dataclass
class MutationStrategy:
    """Per-gen mutation strategy for one island."""
    # Base std multipliers (applied on top of family hint multipliers)
    global_std_multiplier: float = 1.0
    # Per-param std multipliers (combined: family × correlation × strategy)
    per_param_std_multiplier: dict[str, float] = field(default_factory=dict)
    # In-niche probability (0-1)
    in_niche_probability: float = 0.70
    # Niche-leaving probability (forces exploration outside niche)
    niche_leaving_probability: float = 0.10
    # Boundary probability (push toward niche edge)
    boundary_probability: float = 0.20
    # Should we try adding/removing indicators this gen?
    indicator_change_probability: float = 0.225  # matches blind mutator default
    # Should we try grid_method swap this gen?
    grid_method_swap_probability: float = 0.0675  # 15% of mutation_rate (0.45)
    # Should we try allocation_method swap this gen?
    allocation_swap_probability: float = 0.0675
    # Reasoning notes (for debugging / Discord)
    reasoning_notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"std_mult={self.global_std_multiplier:.2f} | "
            f"in_niche={self.in_niche_probability:.0%} | "
            f"leaving={self.niche_leaving_probability:.0%} | "
            f"notes={len(self.reasoning_notes)}"
        )


# ============================================================
# Smart mutator
# ============================================================

class SmartMutator:
    """Family-aware mutator that uses island intelligence + family hints."""

    # Stagnation thresholds
    STAGNANT_FITNESS_DELTA = 0.005  # < this = stagnant
    IMPROVING_FITNESS_DELTA = 0.010  # > this = improving

    # Std multiplier bounds
    MIN_STD_MULT = 0.3
    MAX_STD_MULT = 3.0
    STAGNATION_BOOST = 2.0  # multiply std when stagnant
    IMPROVEMENT_DAMPEN = 0.7  # reduce std when improving

    # Mutation rate (passed from harness config)
    DEFAULT_MUTATION_RATE = 0.45

    def __init__(
        self,
        island_id: int,
        intelligence: IslandIntelligence | None = None,
        family_hint: FamilyHint | None = None,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
    ):
        self.island_id = island_id
        self.intelligence = intelligence
        self.family_hint = family_hint or get_hint_for_island(island_id)
        self.mutation_rate = mutation_rate

    def compute_strategy(self) -> MutationStrategy:
        """Decide this gen's mutation strategy based on intel + hint."""
        strategy = MutationStrategy()

        # 1) Adjust global std based on fitness trend
        trend = self.intelligence.backtest.trend() if self.intelligence else "unknown"
        if trend == "stagnant":
            strategy.global_std_multiplier = self.STAGNATION_BOOST
            strategy.reasoning_notes.append(
                f"Stagnation detected → boosted global std to {self.STAGNATION_BOOST}x"
            )
        elif trend == "improving":
            strategy.global_std_multiplier = self.IMPROVEMENT_DAMPEN
            strategy.reasoning_notes.append(
                f"Improving trend → dampened global std to {self.IMPROVEMENT_DAMPEN}x"
            )
        elif trend == "declining":
            strategy.global_std_multiplier = 1.5
            strategy.reasoning_notes.append("Declining trend → boosted std to 1.5x")

        # 2) Per-param std multipliers from family hint
        if self.family_hint:
            for param in self.family_hint.params_to_explore_more:
                # 1.5x boost for "explore more" params
                strategy.per_param_std_multiplier[param] = (
                    strategy.per_param_std_multiplier.get(param, 1.0) * 1.5
                )
            for param in self.family_hint.params_to_dampen:
                # 0.6x dampen for "dampen" params
                strategy.per_param_std_multiplier[param] = (
                    strategy.per_param_std_multiplier.get(param, 1.0) * 0.6
                )
            # Apply hint-specific multipliers
            for param, mult in self.family_hint.param_std_multipliers.items():
                strategy.per_param_std_multiplier[param] = (
                    strategy.per_param_std_multiplier.get(param, 1.0) * mult
                )

        # 3) Per-param std multipliers from correlations
        if self.intelligence:
            for param in self.intelligence.correlations.promising_params:
                # Promising → 1.3x boost
                strategy.per_param_std_multiplier[param] = (
                    strategy.per_param_std_multiplier.get(param, 1.0) * 1.3
                )
            for param in self.intelligence.correlations.saturated_params:
                # Saturated → 0.5x dampen
                strategy.per_param_std_multiplier[param] = (
                    strategy.per_param_std_multiplier.get(param, 1.0) * 0.5
                )

        # 4) Indicator change probability boost (if family suggests indicators)
        if self.family_hint and self.family_hint.indicator_suggestions:
            strategy.indicator_change_probability = 0.30
            strategy.reasoning_notes.append(
                f"Indicator suggestions: {self.family_hint.indicator_suggestions}"
            )

        # 5) In-niche probability — stronger when niche is well-defined
        if self.intelligence and self.intelligence.niche.sample_size >= 5:
            strategy.in_niche_probability = 0.70
            strategy.boundary_probability = 0.20
            strategy.niche_leaving_probability = 0.10
        else:
            # Cold start — more exploration
            strategy.in_niche_probability = 0.40
            strategy.boundary_probability = 0.30
            strategy.niche_leaving_probability = 0.30
            strategy.reasoning_notes.append(
                "Cold start (niche undefined) → higher exploration rate"
            )

        # Clamp std multipliers
        for param in strategy.per_param_std_multiplier:
            mult = strategy.per_param_std_multiplier[param]
            mult = max(self.MIN_STD_MULT, min(self.MAX_STD_MULT, mult))
            strategy.per_param_std_multiplier[param] = mult

        return strategy

    def mutate(
        self,
        parent: CandidateGenome,
        rng: random.Random,
        strategy: MutationStrategy | None = None,
        child_id: str | None = None,
    ) -> CandidateGenome:
        """Produce a smart mutation of parent.

        Uses the strategy (or computes one) to bias mutations intelligently.
        """
        if strategy is None:
            strategy = self.compute_strategy()

        parent_dca = parent.dca_genome

        # --- Normalize enum fields (in case parent loaded from JSON) ---
        current_grid_method = self._normalize_grid_method(parent_dca.grid_method)
        current_alloc_method = self._normalize_alloc_method(parent_dca.allocation_method)

        # --- Decide mutation zone (in-niche / boundary / leaving) ---
        zone = self._sample_zone(rng, strategy)

        # --- Build new grid_params ---
        new_grid_method = current_grid_method
        # Structural grid_method swap (rare, respects frozen list)
        if (
            self.family_hint
            and "grid_method" in self.family_hint.params_to_freeze
        ):
            pass  # never swap
        elif rng.random() < strategy.grid_method_swap_probability:
            new_grid_method = rng.choice(ALL_GRID_METHODS)

        new_grid_params = self._mutate_grid_params(
            rng=rng,
            parent_params=dict(parent_dca.grid_params),
            new_grid_method=new_grid_method,
            strategy=strategy,
            zone=zone,
        )

        # --- Build new allocation ---
        new_alloc_method = current_alloc_method
        new_alloc_params = dict(parent_dca.allocation_params)
        if (
            self.family_hint
            and "allocation_method" in self.family_hint.params_to_freeze
        ):
            pass  # never swap
        elif rng.random() < strategy.allocation_swap_probability:
            # Bias toward preferred methods if hint exists
            if self.family_hint and self.family_hint.preferred_allocation_methods:
                # 70% chance to pick preferred, 30% chance any
                if rng.random() < 0.7:
                    new_alloc_method = AllocationMethod(
                        rng.choice(self.family_hint.preferred_allocation_methods)
                    )
                else:
                    new_alloc_method = rng.choice(ALL_ALLOCATION_METHODS)
            else:
                new_alloc_method = rng.choice(ALL_ALLOCATION_METHODS)
            new_alloc_params = _random_allocation_params(rng, new_alloc_method)
        else:
            # Tweak existing allocation params
            for key in list(new_alloc_params.keys()):
                if rng.random() < self.mutation_rate:
                    lo_hi = ALLOCATION_PARAM_RANGES.get(new_alloc_method, {}).get(key)
                    if lo_hi:
                        lo, hi = lo_hi
                        std_mult = self._param_std_multiplier(key, strategy)
                        span = (hi - lo) * 0.15 * std_mult
                        new_alloc_params[key] = max(
                            lo, min(hi, new_alloc_params[key] + rng.gauss(0, span))
                        )

        # --- Confirmation indicators ---
        new_indicators = list(parent_dca.confirmation_indicators)
        new_ind_params = dict(parent_dca.indicator_params)
        if rng.random() < strategy.indicator_change_probability:
            if new_indicators and rng.random() < 0.5:
                # Remove one
                idx = rng.randint(0, len(new_indicators) - 1)
                removed = new_indicators.pop(idx)
                new_ind_params.pop(removed.value, None)
            else:
                # Add one — bias toward family suggestions
                candidates = [i for i in ALL_CONFIRMATION_INDICATORS if i not in new_indicators]
                if not candidates:
                    candidates = ALL_CONFIRMATION_INDICATORS
                if candidates and len(new_indicators) < 3:
                    if (
                        self.family_hint
                        and self.family_hint.indicator_suggestions
                        and rng.random() < 0.7
                    ):
                        # Pick from family suggestions
                        suggested_names = set(self.family_hint.indicator_suggestions)
                        suggested = [
                            i for i in candidates if i.value in suggested_names
                        ]
                        if suggested:
                            added = rng.choice(suggested)
                        else:
                            added = rng.choice(candidates)
                    else:
                        added = rng.choice(candidates)
                    new_indicators.append(added)
                    if added.value in INDICATOR_DEFAULT_PARAMS:
                        new_ind_params[added.value] = dict(
                            INDICATOR_DEFAULT_PARAMS[added.value]
                        )

        # Tweak indicator thresholds
        for ind_name in list(new_ind_params.keys()):
            if rng.random() < self.mutation_rate * 0.3:
                for param_key in new_ind_params[ind_name]:
                    current_val = new_ind_params[ind_name][param_key]
                    std_mult = self._param_std_multiplier(ind_name, strategy)
                    std = abs(current_val) * 0.10 * std_mult
                    new_ind_params[ind_name][param_key] = current_val + rng.gauss(0, std)

        # --- Build child ---
        max_layers = min(int(new_grid_params.get("max_layers", 5)), GLOBAL_MAX_DCA_LAYERS)
        new_grid_params["max_layers"] = max_layers

        new_dca = DcaGenome(
            grid_method=new_grid_method,
            grid_params=new_grid_params,
            allocation_method=new_alloc_method,
            allocation_params=new_alloc_params,
            combo_method=parent_dca.combo_method,
            combo_params=dict(parent_dca.combo_params),
            trigger_mode=parent_dca.trigger_mode,
            confirmation_indicators=new_indicators,
            indicator_params=new_ind_params,
            max_dca_layers=max_layers,
        )

        # TP genome — keep parent's exit_method, may tweak tp_pct via grid_params
        new_tp_genome = TpGenome(
            exit_method=parent.tp_genome.exit_method,
            exit_params=dict(parent.tp_genome.exit_params),
            sub_exits=list(parent.tp_genome.sub_exits),
        )

        cid = child_id or f"genome_G{parent.lineage.generation_index + 1}_{rng.randint(0, 1_000_000):06d}"

        from genome.schema import LineageMetadata
        new_lineage = LineageMetadata(
            parent_a_id=parent.genome_id,
            parent_b_id=None,
            generation_index=parent.lineage.generation_index + 1,
            mutation_seed=rng.randint(0, 1_000_000),
            mutation_ops=[
                {"op": "smart_mutate", "island_id": self.island_id, "zone": zone},
                *parent.lineage.mutation_ops,
            ],
            created_at=None,
        )

        return CandidateGenome(
            genome_id=cid,
            dca_genome=new_dca,
            tp_genome=new_tp_genome,
            safety_genome=parent.safety_genome,
            settings_overrides=parent.settings_overrides,
            lineage=new_lineage,
        )

    # ----- Internal helpers -----

    def _normalize_grid_method(self, raw: Any) -> GridMethod:
        if isinstance(raw, str):
            try:
                return GridMethod(raw)
            except ValueError:
                return GridMethod.FIXED_PCT
        return raw

    def _normalize_alloc_method(self, raw: Any) -> AllocationMethod:
        if isinstance(raw, str):
            try:
                return AllocationMethod(raw)
            except ValueError:
                return AllocationMethod.EQUAL
        return raw

    def _sample_zone(self, rng: random.Random, strategy: MutationStrategy) -> str:
        """Decide which mutation zone to use."""
        roll = rng.random()
        if roll < strategy.in_niche_probability:
            return "in_niche"
        elif roll < strategy.in_niche_probability + strategy.boundary_probability:
            return "boundary"
        return "leaving"

    def _param_std_multiplier(self, param: str, strategy: MutationStrategy) -> float:
        """Get the combined std multiplier for a param."""
        return strategy.per_param_std_multiplier.get(param, 1.0) * strategy.global_std_multiplier

    def _mutate_grid_params(
        self,
        rng: random.Random,
        parent_params: dict[str, float],
        new_grid_method: GridMethod,
        strategy: MutationStrategy,
        zone: str,
    ) -> dict[str, float]:
        """Mutate grid_params with intelligence + zone bias."""
        new_params = dict(parent_params)

        # --- Zone-based bias for main params ---
        # In-niche: small steps near current value
        # Boundary: larger steps toward niche edge
        # Leaving: bigger steps toward unexplored regions
        zone_std_mult = {
            "in_niche": 1.0,
            "boundary": 1.5,
            "leaving": 2.5,
        }.get(zone, 1.0)

        # --- pct / grid_pct ---
        current_pct = float(new_params.get("pct", 0.015))
        if rng.random() < self.mutation_rate:
            lo, hi = DCA_PARAM_RANGES["grid_pct"]
            span = hi - lo
            std_mult = self._param_std_multiplier("pct", strategy) * zone_std_mult
            # Bias toward niche centroid if available
            if (
                self.intelligence
                and self.intelligence.niche.sample_size >= 5
                and "pct" in self.intelligence.niche.centroid
            ):
                centroid = self.intelligence.niche.centroid["pct"]
                # Mix of pull toward centroid + random walk
                pull_strength = 0.3 if zone == "in_niche" else 0.0
                pull = (centroid - current_pct) * pull_strength
                current_pct = current_pct + pull + rng.gauss(0, span * 0.20 * std_mult)
            else:
                current_pct = current_pct + rng.gauss(0, span * 0.20 * std_mult)
            current_pct = max(lo, min(hi, current_pct))
        new_params["pct"] = current_pct

        # --- max_layers ---
        current_layers = int(new_params.get("max_layers", 5))
        current_layers = min(current_layers, GLOBAL_MAX_DCA_LAYERS)
        if rng.random() < self.mutation_rate:
            lo, hi = DCA_PARAM_RANGES["max_layers"]
            std_mult = self._param_std_multiplier("max_layers", strategy) * zone_std_mult
            # Bigger jumps when leaving
            delta_choices = [-2, -1, 1, 2] if zone != "leaving" else [-3, -2, 2, 3]
            delta = rng.choice(delta_choices)
            current_layers = max(lo, min(hi, current_layers + delta))
        current_layers = min(current_layers, GLOBAL_MAX_DCA_LAYERS)
        new_params["max_layers"] = current_layers

        # --- tp_pct ---
        current_tp = float(new_params.get("tp_pct", 0.02))
        if rng.random() < self.mutation_rate:
            lo, hi = DCA_PARAM_RANGES["tp_pct"]
            span = hi - lo
            std_mult = self._param_std_multiplier("tp_pct", strategy) * zone_std_mult
            current_tp = current_tp + rng.gauss(0, span * 0.20 * std_mult)
            current_tp = max(lo, min(hi, current_tp))
        new_params["tp_pct"] = current_tp

        # --- cooldown_candles ---
        current_cd = int(new_params.get("cooldown_candles", 0))
        if rng.random() < self.mutation_rate:
            lo, hi = DCA_PARAM_RANGES["cooldown_candles"]
            std_mult = self._param_std_multiplier("cooldown_candles", strategy) * zone_std_mult
            delta_choices = [-2, -1, 0, 1, 2] if zone != "leaving" else [-3, -2, 2, 3]
            delta = rng.choice(delta_choices)
            current_cd = max(lo, min(hi, current_cd + delta))
        new_params["cooldown_candles"] = current_cd

        # --- Grid-method-specific params ---
        method_defaults = GRID_METHOD_DEFAULT_PARAMS.get(new_grid_method.value, {})
        for param_key in method_defaults:
            if param_key in ("pct", "max_layers", "tp_pct", "cooldown_candles"):
                continue
            if rng.random() < self.mutation_rate:
                default_val = float(method_defaults[param_key])
                current_val = float(new_params.get(param_key, default_val))
                std_mult = self._param_std_multiplier(param_key, strategy) * zone_std_mult
                # Pull toward niche centroid if available
                if (
                    self.intelligence
                    and self.intelligence.niche.sample_size >= 5
                    and param_key in self.intelligence.niche.centroid
                ):
                    centroid = self.intelligence.niche.centroid[param_key]
                    pull_strength = 0.3 if zone == "in_niche" else 0.0
                    pull = (centroid - current_val) * pull_strength
                    new_val = current_val + pull + rng.gauss(0, abs(default_val) * 0.20 * std_mult)
                else:
                    new_val = current_val + rng.gauss(0, abs(default_val) * 0.20 * std_mult)
                new_params[param_key] = new_val

        return new_params
