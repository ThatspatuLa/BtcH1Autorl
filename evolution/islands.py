"""Island model — splits the population into N parallel sub-populations, each
with a forced specialization. Migration every M generations prevents inbreeding.

Effective 2026-06-22 per Six's plan: 8 islands × 62 candidates = 496, plus 4
random. Each island has enough mass (~3 elites + 49 children + 10 random) to
evolve within its niche, and migration lets successful DNA cross over.

Islands (each island gets 62 cands; forced bias during seeding):

  1. fixed_pct             — current ATB family (exploit deeper)
  2. atr                   — second-best dynamic grid
  3. volatility_or_dd      — volatility-relative spacing (volatility, drawdown_from_high)
  4. trend                 — trend-following DCA (ma_distance, trend_adjusted)
  5. oscillator            — mean-reversion DCA (rsi_oversold, z_score)
  6. vola_adj_alloc        — any grid, forced volatility_adjusted allocation
  7. ctrl_exp_alloc        — any grid, forced controlled_exp allocation
  8. tight_dca             — max_layers <= 8 (fast-exit / low-DCA)

Migration: every 5 generations, top 4 of each island → round-robin to
neighbors. Elites always stay in their home island.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    GridMethod,
)


# ============================================================
# Island specs
# ============================================================

@dataclass
class IslandSpec:
    """Defines the forced bias for one island."""
    island_id: int
    name: str
    n_candidates: int
    # Forced biases (None = no bias, draw uniformly from full enum)
    forced_grid_methods: tuple[GridMethod, ...] | None = None
    forced_allocation: AllocationMethod | None = None
    forced_confirmations: tuple[ConfirmationIndicator, ...] | None = None
    max_dca_layers_cap: int | None = None  # if set, cap max_layers at this value
    # Optional note for logging
    note: str = ""

    def describe(self) -> str:
        parts = [f"Island {self.island_id} ({self.name})"]
        if self.forced_grid_methods:
            gms = ", ".join(g.value for g in self.forced_grid_methods)
            parts.append(f"grid=[{gms}]")
        if self.forced_allocation:
            parts.append(f"alloc={self.forced_allocation.value}")
        if self.forced_confirmations:
            cis = ", ".join(c.value for c in self.forced_confirmations)
            parts.append(f"conf=[{cis}]")
        if self.max_dca_layers_cap is not None:
            parts.append(f"max_layers<=<={self.max_dca_layers_cap}")
        return " | ".join(parts)


# 8 islands, 62 candidates each = 496. Plus 4 random = 500.
ISLAND_SPECS: list[IslandSpec] = [
    IslandSpec(
        island_id=1, name="fixed_pct", n_candidates=62,
        forced_grid_methods=(GridMethod.FIXED_PCT,),
        note="current ATB family — exploit deeper",
    ),
    IslandSpec(
        island_id=2, name="atr", n_candidates=62,
        forced_grid_methods=(GridMethod.ATR,),
        note="second-best dynamic grid",
    ),
    IslandSpec(
        island_id=3, name="volatility_or_dd", n_candidates=62,
        forced_grid_methods=(GridMethod.VOLATILITY, GridMethod.DRAWDOWN_FROM_HIGH),
        note="volatility-relative spacing",
    ),
    IslandSpec(
        island_id=4, name="trend", n_candidates=62,
        forced_grid_methods=(GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED),
        note="trend-following DCA",
    ),
    IslandSpec(
        island_id=5, name="oscillator", n_candidates=62,
        forced_grid_methods=(GridMethod.RSI_OVERSOLD, GridMethod.Z_SCORE),
        note="mean-reversion DCA",
    ),
    IslandSpec(
        island_id=6, name="vola_adj_alloc", n_candidates=62,
        forced_allocation=AllocationMethod.VOLATILITY_ADJUSTED,
        note="cross-grid: any grid, forced volatility_adjusted allocation",
    ),
    IslandSpec(
        island_id=7, name="ctrl_exp_alloc", n_candidates=62,
        forced_allocation=AllocationMethod.CONTROLLED_EXP,
        note="cross-grid: any grid, forced controlled_exp allocation",
    ),
    IslandSpec(
        island_id=8, name="tight_dca", n_candidates=62,
        max_dca_layers_cap=8,
        note="fast-exit / low-DCA strategies",
    ),
]


def get_island_specs() -> list[IslandSpec]:
    """Return the canonical 8-island spec list."""
    return list(ISLAND_SPECS)


def get_island_spec(island_id: int) -> IslandSpec:
    for spec in ISLAND_SPECS:
        if spec.island_id == island_id:
            return spec
    raise ValueError(f"No island with id {island_id}")


# ============================================================
# Migration
# ============================================================

@dataclass
class MigrationResult:
    """Result of one migration step."""
    migrants_per_island: dict[int, int] = field(default_factory=dict)
    migrants_received_per_island: dict[int, int] = field(default_factory=dict)
    n_migrants_total: int = 0


def select_migrants(
    island_id: int,
    island_elites: list[CandidateGenome],
    n_migrants: int = 4,
    rng: random.Random | None = None,
) -> list[CandidateGenome]:
    """Pick the top-N migrants from one island. If fewer elites than n_migrants,
    return what we have. Mark them as migrants via lineage.metadata."""
    if not island_elites:
        return []
    rng = rng or random.Random()
    # Take the first n_migrants (caller is responsible for sorting elites by fitness desc)
    selected = island_elites[:n_migrants]
    return selected


def distribute_migrants(
    migrants_by_source: dict[int, list[CandidateGenome]],
    n_islands: int,
    rng: random.Random | None = None,
) -> dict[int, list[CandidateGenome]]:
    """Round-robin migrate the top-4 from each island to its neighbors.

    For island i, migrants come from islands (i-1) % N and (i+1) % N.
    Each neighbor contributes min(4, len(elites)) migrants.

    Returns: {island_id: list of incoming migrants}
    """
    rng = rng or random.Random()
    received: dict[int, list[CandidateGenome]] = {i: [] for i in range(1, n_islands + 1)}

    for source_id, migrants in migrants_by_source.items():
        # Each source sends its migrants to its two neighbors
        left = (source_id - 2) % n_islands + 1  # i-1 in 1..N
        right = (source_id % n_islands) + 1     # i+1 in 1..N
        # Split migrants: half to left, half to right (or alternating)
        for idx, m in enumerate(migrants):
            target = left if idx % 2 == 0 else right
            # Tag the genome as a migrant via lineage.mutation_ops
            m = _mark_as_migrant(m, source_id)
            received[target].append(m)

    return received


def _mark_as_migrant(genome: CandidateGenome, source_island: int) -> CandidateGenome:
    """Add a lineage tag so we can trace which island a migrant came from."""
    if genome.lineage.mutation_ops is None:
        genome.lineage.mutation_ops = []
    genome.lineage.mutation_ops.append({
        "op": "migrate",
        "from_island": source_island,
    })
    return genome


def count_migrants_received(received: dict[int, list[CandidateGenome]]) -> dict[int, int]:
    """Helper: how many migrants did each island receive?"""
    return {i: len(v) for i, v in received.items()}


# ============================================================
# Island tagging on CandidateGenome
# ============================================================
# We don't modify CandidateGenome schema (locked). Instead, islands are tracked
# via lineage.mutation_ops and via a parallel island_id map keyed by genome_id.

class IslandTracker:
    """Track which island each genome belongs to, across generations.

    Used by the harness to route candidates back to their island after eval,
    and to drive per-island leaderboards + migration.
    """
    def __init__(self) -> None:
        # genome_id -> island_id
        self._genome_to_island: dict[str, int] = {}
        # island_id -> list of evaluated CandidateGenome (this gen)
        self._island_pool: dict[int, list[CandidateGenome]] = {}

    def register(self, genome_id: str, island_id: int) -> None:
        self._genome_to_island[genome_id] = island_id

    def route(
        self,
        candidates: list[CandidateGenome],
        island_assignment: dict[str, int],
    ) -> dict[int, list[CandidateGenome]]:
        """Bucket candidates by island using the assignment map.

        island_assignment: maps genome_id -> island_id
        """
        buckets: dict[int, list[CandidateGenome]] = {}
        for c in candidates:
            iid = island_assignment.get(c.genome_id, 0)
            buckets.setdefault(iid, []).append(c)
            self._genome_to_island[c.genome_id] = iid
        return buckets

    def get_island_id(self, genome_id: str) -> int | None:
        return self._genome_to_island.get(genome_id)

    def get_all_for_island(self, island_id: int) -> list[CandidateGenome]:
        return list(self._island_pool.get(island_id, []))


def island_assignment_for_population(
    candidates: list[CandidateGenome],
    specs: list[IslandSpec],
    random_count: int = 4,
) -> dict[str, int]:
    """Compute which island each candidate belongs to.

    First (sum of island sizes) candidates are split across islands by spec.
    Final `random_count` candidates get island_id=0 (random bag, treated as
    a separate mini-island by the harness).

    Returns: {genome_id: island_id}
    """
    assignment: dict[str, int] = {}
    idx = 0
    for spec in specs:
        for _ in range(spec.n_candidates):
            if idx >= len(candidates):
                break
            assignment[candidates[idx].genome_id] = spec.island_id
            idx += 1
    # Remaining go to random island (0)
    for c in candidates[idx:]:
        assignment[c.genome_id] = 0
    return assignment


def per_island_elites(
    results_by_island: dict[int, list[tuple[CandidateGenome, float]]],
    elite_per_island: int = 3,
) -> dict[int, list[CandidateGenome]]:
    """Sort each island's results by fitness desc and take top-N.

    results_by_island: {island_id: [(genome, fitness), ...]}
    Returns: {island_id: [elite_genome, ...]}
    """
    elites: dict[int, list[CandidateGenome]] = {}
    for iid, results in results_by_island.items():
        sorted_results = sorted(results, key=lambda r: r[1], reverse=True)
        elites[iid] = [r[0] for r in sorted_results[:elite_per_island]]
    return elites
