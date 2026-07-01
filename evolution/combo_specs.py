"""Stage 2 combo specs — generate per-layer zone configs from top-N families.

This module is the entry point for Stage 2 of the family hyperopt pipeline:

    Stage 1 (single families)  →  top-5 ranking
                                 ↓
    Stage 2 (combos)           ←  build pairs (C(5,2)=10) + triples (C(5,3)=10)
                                 ↓
                                 for each base combo × 3 split strategies
                                   → ComboSpec (family list + split kind + zones)

A ComboSpec is the contract that the population builder + OrderManager consume:
the combo's zones list is rendered into a list[GridZoneSpec] on the candidate's
DcaGenome, and the OrderManager switches grid_method+grid_params at each layer
boundary according to that zone list.

Split strategies:
  - "3_split": contiguous zones. For N families in the combo, layers 1..max_layers
    are divided into N contiguous chunks. N=3 with 10 layers → 3-3-4 split.
                N=2 with 10 layers → 5-5 split.
  - "weighted_blend": same contiguous chunks, but layer assignments are based on
                exponential weighting — shallow gets small weight, deep gets large
                weight (so the "best for deep dip" family owns more layers).
                For N=3 with 10 layers → 2-3-5 split (weighted 0.2/0.3/0.5).
  - "alternating": round-robin assignment of families to layers (family index = layer
                index mod N). For N=3 with 10 layers → A-B-C-A-B-C-A-B-C-A.

All splits produce zones that satisfy the validator (contiguous, non-overlapping,
cover 1..max_dca_layers). For N=2 (pairs), the "3_split" degenerates to a 5-5
split, "alternating" to A-B-A-B-A-B-A-B-A-B, "weighted_blend" to 3-7 split.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from genome.schema import AllocationMethod, ConfirmationIndicator, GridMethod, GridZoneSpec
from evolution.hyperopt_config import FamilySpec, build_family_specs


# ============================================================
# Constants
# ============================================================

# Default max_dca_layers for Stage 2 combos — matches Stage 1 policy cap.
COMBO_DEFAULT_MAX_DCA_LAYERS: int = 10

# Available split strategies — all combos are evaluated under all 3.
COMBO_SPLIT_STRATEGIES: tuple[str, ...] = ("3_split", "weighted_blend", "alternating")

# Cached registry of all spacing families by name — avoids rebuilding FamilySpec list.
_FAMILY_CACHE: dict[str, FamilySpec] | None = None


def _get_family_by_name(name: str) -> FamilySpec:
    """Resolve a family name to its FamilySpec."""
    global _FAMILY_CACHE
    if _FAMILY_CACHE is None:
        _FAMILY_CACHE = {f.name: f for f in build_family_specs()}
    if name not in _FAMILY_CACHE:
        raise KeyError(f"Unknown family: {name!r}. Available: {list(_FAMILY_CACHE.keys())}")
    return _FAMILY_CACHE[name]


# ============================================================
# ComboSpec — contract for one combo run
# ============================================================


@dataclass
class ComboSpec:
    """A Stage 2 combo: list of family names + split strategy + zone spec.

    `zones` is the rendered list[GridZoneSpec] ready to attach to a DcaGenome.
    The zones are deterministic — same family list + same split always produces
    the same zones.
    """

    name: str
    families: list[str]  # ordered list of family names (zone order)
    split_strategy: str  # one of COMBO_SPLIT_STRATEGIES
    max_dca_layers: int
    zones: list[GridZoneSpec]
    family_grid_methods: list[list[str]] = field(default_factory=list)
    description: str = ""
    iteration: int = 0

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def is_pair(self) -> bool:
        return self.n_families == 2

    @property
    def is_triple(self) -> bool:
        return self.n_families == 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "families": list(self.families),
            "split_strategy": self.split_strategy,
            "max_dca_layers": self.max_dca_layers,
            "zones": [z.to_dict() for z in self.zones],
            "family_grid_methods": self.family_grid_methods,
            "n_families": self.n_families,
            "description": self.description,
            "iteration": self.iteration,
        }

    @property
    def deterministic_seed(self) -> int:
        """Reproducible seed per combo name (iteration included)."""
        h = hashlib.sha256(self.name.encode()).hexdigest()
        return int(h[:8], 16) + self.iteration

    def family_grid_methods_for(self, family_name: str) -> list[str]:
        """Return the allowed grid methods for one family (from its FamilySpec)."""
        spec = _get_family_by_name(family_name)
        return [m.value for m in spec.forced_grid_methods] if spec.forced_grid_methods else []


# ============================================================
# Layer-split helpers
# ============================================================


def _compute_3_split_layers(n_families: int, max_layers: int) -> list[int]:
    """Contiguous chunks for the 3-split strategy.

    For N=3 with max_layers=10: [3, 3, 4]  (sum=10)
    For N=2 with max_layers=10: [5, 5]
    For N=1 with max_layers=10: [10]
    """
    if n_families < 1:
        raise ValueError(f"n_families must be >= 1, got {n_families}")
    if max_layers < n_families:
        raise ValueError(
            f"max_layers={max_layers} < n_families={n_families}; "
            f"need at least 1 layer per family"
        )
    # Integer division distributes evenly; remainder goes to the LAST family
    # (so the deepest zone absorbs the extra layers — fits the "deep dip = more
    # layers" intuition).
    base, extra = divmod(max_layers, n_families)
    counts = [base] * n_families
    for i in range(extra):
        counts[n_families - 1 - i] += 1
    return counts


def _compute_weighted_blend_layers(n_families: int, max_layers: int) -> list[int]:
    """Weighted split: exponential weighting gives the deepest zone more layers.

    Weights per family at index i (i=0=shallow): w[i] = 2^i / sum(2^j)
    Then assign layers = round(w[i] * max_layers) and fix the rounding to sum
    exactly max_layers.

    For N=3 with max_layers=10: weights ≈ [0.14, 0.29, 0.57] → layers [1, 3, 6]
    (but rounds to [1, 3, 6] sum=10). Larger families get the deep zone.
    For N=2 with max_layers=10: weights [0.33, 0.67] → layers [3, 7].
    """
    if n_families < 1:
        raise ValueError(f"n_families must be >= 1, got {n_families}")
    if max_layers < n_families:
        raise ValueError(
            f"max_layers={max_layers} < n_families={n_families}; "
            f"need at least 1 layer per family"
        )
    weights = [2 ** i for i in range(n_families)]
    total_w = sum(weights)
    raw = [w / total_w * max_layers for w in weights]
    # Floor each, then distribute the remainder to the largest family(ies).
    counts = [int(r) for r in raw]
    leftover = max_layers - sum(counts)
    # Add leftover to the LAST family (deepest zone) — keeps "deep = more layers"
    if leftover > 0:
        counts[-1] += leftover
    elif leftover < 0:
        # Rounding overshoot — remove from last
        counts[-1] += leftover
    return counts


def _compute_alternating_layers(n_families: int, max_layers: int) -> list[int]:
    """Round-robin: family[i] owns layer index i+1, (i+n_families)+1, etc.

    For N=3 with max_layers=10: A-B-C-A-B-C-A-B-C-A → counts [4, 3, 3]
    For N=2 with max_layers=10: A-B-A-B-A-B-A-B-A-B → counts [5, 5]
    """
    if n_families < 1:
        raise ValueError(f"n_families must be >= 1, got {n_families}")
    counts = [0] * n_families
    for layer in range(1, max_layers + 1):
        idx = (layer - 1) % n_families
        counts[idx] += 1
    return counts


# ============================================================
# Zone builders
# ============================================================


def _family_default_grid_params(
    spec: FamilySpec,
    rng_seed: int,
) -> dict[str, float]:
    """Pick a sensible default grid_params dict for one of the family's methods.

    Used as the starting point when rendering zones for a family that has multiple
    allowed grid methods (hybrid families). We pick the first method and a simple
    default param set; mutate() in operators.py will perturb these within range.
    """
    if not spec.forced_grid_methods:
        return {"pct": 0.015}
    method = spec.forced_grid_methods[0]
    defaults: dict[str, float] = {"pct": 0.015}
    if method == GridMethod.ATR:
        defaults["atr_multiplier"] = 2.0
    elif method == GridMethod.VOLATILITY:
        defaults["base_pct"] = 0.01
        defaults["vol_scale_factor"] = 0.5
    elif method == GridMethod.DRAWDOWN_FROM_HIGH:
        defaults["drawdown_pct"] = 0.05
    elif method == GridMethod.MA_DISTANCE:
        defaults["ma_distance_pct"] = 0.03
    elif method == GridMethod.RSI_OVERSOLD:
        defaults["rsi_threshold"] = 30.0
        defaults["oversold_depth_pct"] = 0.02
    elif method == GridMethod.Z_SCORE:
        defaults["z_threshold"] = 1.5
        defaults["lookback_std"] = 0.02
    elif method == GridMethod.TREND_ADJUSTED:
        defaults["base_pct"] = 0.015
        defaults["trend_multiplier"] = 0.5
    return defaults


def build_zones_for_combo(
    families: list[str],
    split_strategy: str,
    max_dca_layers: int = COMBO_DEFAULT_MAX_DCA_LAYERS,
    family_grid_methods: list[list[str]] | None = None,
    rng_seed: int = 42,
) -> list[GridZoneSpec]:
    """Render a list of (family, split) into a list of GridZoneSpec ready to attach.

    Args:
        families: ordered list of family names (order = zone order; families[0] is the
                  shallowest zone, families[-1] is the deepest).
        split_strategy: one of "3_split", "weighted_blend", "alternating".
        max_dca_layers: total layers for this combo.
        family_grid_methods: optional override list parallel to families, each entry is
                  a list of allowed GridMethod values for that family. If None, the
                  family's forced_grid_methods are used.
        rng_seed: deterministic seed for picking grid methods when families have multiple.

    Returns:
        List of GridZoneSpec covering layers 1..max_dca_layers.
    """
    if split_strategy not in COMBO_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split_strategy {split_strategy!r}. "
            f"Choose from {COMBO_SPLIT_STRATEGIES}"
        )
    n = len(families)
    if split_strategy == "3_split":
        counts = _compute_3_split_layers(n, max_dca_layers)
    elif split_strategy == "weighted_blend":
        counts = _compute_weighted_blend_layers(n, max_dca_layers)
    else:  # alternating
        counts = _compute_alternating_layers(n, max_dca_layers)

    zones: list[GridZoneSpec] = []
    cursor = 1
    for i, fam_name in enumerate(families):
        spec = _get_family_by_name(fam_name)
        # Pick a grid method for this family
        if family_grid_methods is not None and family_grid_methods[i]:
            method_str = family_grid_methods[i][0]
        elif spec.forced_grid_methods:
            method_str = spec.forced_grid_methods[0].value
        else:
            method_str = GridMethod.FIXED_PCT.value
        method = GridMethod(method_str)
        grid_params = _family_default_grid_params(spec, rng_seed + i)
        zone = GridZoneSpec(
            layer_start=cursor,
            layer_count=counts[i],
            grid_method=method,
            grid_params=grid_params,
        )
        zones.append(zone)
        cursor += counts[i]
    return zones


# ============================================================
# Combo generation
# ============================================================


def build_pairs(family_names: list[str]) -> list[list[str]]:
    """All pairs from N families: C(N,2) = 10 for N=5."""
    if len(family_names) < 2:
        return []
    # Sort for determinism
    sorted_names = sorted(family_names)
    return [list(pair) for pair in combinations(sorted_names, 2)]


def build_triples(family_names: list[str]) -> list[list[str]]:
    """All triples from N families: C(N,3) = 10 for N=5."""
    if len(family_names) < 3:
        return []
    sorted_names = sorted(family_names)
    return [list(triple) for triple in combinations(sorted_names, 3)]


def build_stage2_combos(
    top_family_names: list[str],
    splits: tuple[str, ...] = COMBO_SPLIT_STRATEGIES,
    include_pairs: bool = True,
    include_triples: bool = True,
    max_dca_layers: int = COMBO_DEFAULT_MAX_DCA_LAYERS,
    iteration: int = 0,
) -> list[ComboSpec]:
    """Build the full Stage 2 combo set.

    For 5 families × {pairs (10) + triples (10)} × 3 splits = 60 ComboSpecs total.
    Iteration number is appended to the name when > 0 (for re-runs).

    Args:
        top_family_names: list of family names (typically the Stage 1 top-5).
        splits: which split strategies to include.
        include_pairs: include C(N,2) pair combos.
        include_triples: include C(N,3) triple combos.
        max_dca_layers: total layers per combo (policy: 10).
        iteration: 0 for first run, 1+ for re-runs.
    """
    if len(top_family_names) < 2:
        raise ValueError(
            f"Need at least 2 families for combos, got {len(top_family_names)}"
        )

    combos: list[ComboSpec] = []
    base_combos: list[list[str]] = []
    if include_pairs:
        base_combos.extend(build_pairs(top_family_names))
    if include_triples:
        base_combos.extend(build_triples(top_family_names))

    for base in base_combos:
        n = len(base)
        kind = "pair" if n == 2 else ("triple" if n == 3 else f"{n}tuple")
        for split in splits:
            zones = build_zones_for_combo(base, split, max_dca_layers=max_dca_layers)
            # Per-family allowed grid methods for downstream constraint checks
            family_grid_methods = []
            for fam in base:
                spec = _get_family_by_name(fam)
                if spec.forced_grid_methods:
                    family_grid_methods.append([m.value for m in spec.forced_grid_methods])
                else:
                    family_grid_methods.append([])
            name = f"combo_{kind}_{split}_{'_'.join(base)}"
            if iteration > 0:
                name = f"{name}_iter{iteration:02d}"
            combos.append(
                ComboSpec(
                    name=name,
                    families=list(base),
                    split_strategy=split,
                    max_dca_layers=max_dca_layers,
                    zones=zones,
                    family_grid_methods=family_grid_methods,
                    description=(
                        f"{kind} combo ({split}) over {max_dca_layers} layers — "
                        f"{' → '.join(base)}"
                    ),
                    iteration=iteration,
                )
            )
    return combos


def select_top_families_from_stage1(
    stage1_results: list[dict[str, Any]],
    top_n: int = 5,
) -> list[str]:
    """Pick the top-N family names from a Stage 1 results list.

    Each result dict must have 'family' and 'best_fitness' keys (as produced by
    scripts/run_family_hyperopt.py). Returns the family names sorted by fitness
    descending. Raises if fewer than top_n families are available.
    """
    sorted_results = sorted(stage1_results, key=lambda r: r.get("best_fitness", 0.0), reverse=True)
    if len(sorted_results) < top_n:
        raise ValueError(
            f"Need at least {top_n} Stage 1 results, got {len(sorted_results)}"
        )
    return [r["family"] for r in sorted_results[:top_n]]