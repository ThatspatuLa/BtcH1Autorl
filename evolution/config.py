"""EvolutionConfig — single source of truth for the GA loop.

All stop conditions and parameters are configurable here. v1 defaults
are locked per Kanban (8h wall-time, 20 generations, stagnation = 5 gens).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Locked v1 defaults — do not evolve
DEFAULT_WALL_TIME_SECONDS: int = 8 * 60 * 60        # 8 hours
DEFAULT_MAX_GENERATIONS: int = 20
DEFAULT_STAGNATION_GENERATIONS: int = 5             # stop if no improvement for 5 gens
DEFAULT_ALL_REJECTED_GENERATIONS: int = 3           # stop if all rejected for 3 gens
DEFAULT_CANDIDATES_PER_GEN: int = 500               # LOCKED
DEFAULT_ELITE_COUNT: int = 20                       # top 4% survive
DEFAULT_RANDOM_INJECTION: int = 120                 # fresh random per gen
DEFAULT_MUTATION_RATE: float = 0.30                # chance any single param mutates
DEFAULT_CROSSOVER_RATE: float = 0.50               # fraction of children from crossover
DEFAULT_PARALLEL_WORKERS: int = 8                  # process pool size


@dataclass
class EvolutionConfig:
    """All knobs for the evolution loop."""
    # Population
    candidates_per_gen: int = DEFAULT_CANDIDATES_PER_GEN
    elite_count: int = DEFAULT_ELITE_COUNT
    random_injection: int = DEFAULT_RANDOM_INJECTION
    # Genetic operators
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    # Stop conditions (any of these halts the run)
    wall_time_seconds: int = DEFAULT_WALL_TIME_SECONDS
    max_generations: int = DEFAULT_MAX_GENERATIONS
    stagnation_generations: int = DEFAULT_STAGNATION_GENERATIONS
    all_rejected_generations: int = DEFAULT_ALL_REJECTED_GENERATIONS
    # Parallelism
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    # Reproducibility
    base_seed: int = 42
    # Paths
    output_dir: str = "results/evolution"
    experiment_id: str = "exp_default"
    # Reporting
    leaderboard_top_n: int = 20
    # Stage 9 specific
    tp_pct: float = 0.02  # the locked fixed-TP baseline value
    # Island mode (Plan B, effective 2026-06-22)
    island_mode: bool = False  # if True, use 8-island sub-population model
    n_islands: int = 8
    migration_every_n_gens: int = 5
    migrants_per_island: int = 4
    # Stagnation: if True, stagnation guard fires per-island (not globally)
    per_island_stagnation: bool = True
    # Elite quality gate (Fix B, 2026-06-22): a candidate is "elite-eligible"
    # only if consistency_ratio >= this OR discovery_fitness >= min_discovery_for_elite.
    # Soft-passed candidates stay in the population for diversity but are not
    # used as breeding seed. Prevents a single mediocre 0.28 candidate from
    # becoming the seed of a 4-gen plateau.
    min_consistency_for_elite: float = 0.50
    min_discovery_for_elite: float = 0.70
    # Island retirement (effective 2026-06-22, Six's plan B extension).
    # When any island's per-island top fitness crosses `retirement_threshold`,
    # that island's full state is archived to `retirement_archive_dir` and the
    # slot is re-seeded with a fresh family bias from a 16-bias rotation pool.
    # This expands the retired-islands archive over time without ending the run.
    # 2026-06-25: lowered default 0.80 → 0.75 for cap-10 era (Pitfall #11 recommendation).
    # Cap-5 era peaked at 0.71; with cap=10 search space expanded, 0.75 is a
    # reachable threshold that lets the archive grow during early cap-10 cycles.
    # Bump back to 0.80 once cap-10 fitness crosses 0.75 regularly.
    retirement_enabled: bool = False
    retirement_threshold: float = 0.75
    retirement_archive_dir: str = "runs/retired_islands"
    max_retired_per_cycle: int = 999  # 999 = no cap (default; user can tighten)
    # Checkpoints (every-N-min snapshots for restart safety).
    # Written to <project_root>/checkpoints/. Default 20 min = ~6 saves per
    # 2h cycle. Set to 0 to disable (not recommended for long runs).
    checkpoint_interval_minutes: int = 20
    # Force-retire on per-island stagnation (Plan: 2026-06-24, Six).
    # If an island's per-island best fitness hasn't improved for this many
    # gens AND its current best is below force_retire_min_fitness, the slot
    # is re-seeded from the bias rotation pool (logged as "stagnation retire").
    force_retire_after_gens: int = 8
    force_retire_min_fitness: float = 0.70  # skip if already near the bar
    # Quick Win 3 (2026-06-25): Mid-stagnation soft intervention.
    # When a single island's per-island stagnation counter reaches this
    # value (but not yet force_retire_after_gens), boost that island's
    # random_injection for ONE gen — soft escape attempt before force-retire.
    # 0 = disabled. Default 8 (between stagnation_generations=5 and
    # force_retire_after_gens=15).
    mid_stagnation_threshold: int = 8
    # What fraction of the island's population to replace with random on
    # mid-stagnation. 0.50 = half. Set to 0 to disable the random boost
    # (the threshold check still runs but applies no boost).
    mid_stagnation_random_frac: float = 0.50

    def __post_init__(self) -> None:
        if self.candidates_per_gen < 1:
            raise ValueError("candidates_per_gen must be >= 1")
        if self.elite_count < 1:
            raise ValueError("elite_count must be >= 1")
        if self.elite_count > self.candidates_per_gen:
            raise ValueError("elite_count must be <= candidates_per_gen")
        if self.random_injection < 0:
            raise ValueError("random_injection must be >= 0")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if self.wall_time_seconds < 60:
            raise ValueError("wall_time_seconds must be >= 60")
        if self.max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation_generations must be >= 1")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")

    # Children per generation = elites (carried) + mutated/crossover + random
    @property
    def children_per_gen(self) -> int:
        return self.candidates_per_gen - self.elite_count - self.random_injection

    @property
    def crossover_children(self) -> int:
        return int(self.children_per_gen * self.crossover_rate)

    @property
    def mutation_children(self) -> int:
        return self.children_per_gen - self.crossover_children

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_per_gen": self.candidates_per_gen,
            "elite_count": self.elite_count,
            "random_injection": self.random_injection,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "wall_time_seconds": self.wall_time_seconds,
            "max_generations": self.max_generations,
            "stagnation_generations": self.stagnation_generations,
            "all_rejected_generations": self.all_rejected_generations,
            "parallel_workers": self.parallel_workers,
            "base_seed": self.base_seed,
            "tp_pct": self.tp_pct,
            "island_mode": self.island_mode,
            "n_islands": self.n_islands,
            "migration_every_n_gens": self.migration_every_n_gens,
            "migrants_per_island": self.migrants_per_island,
            "per_island_stagnation": self.per_island_stagnation,
            "min_consistency_for_elite": self.min_consistency_for_elite,
            "min_discovery_for_elite": self.min_discovery_for_elite,
            "retirement_enabled": self.retirement_enabled,
            "retirement_threshold": self.retirement_threshold,
            "retirement_archive_dir": self.retirement_archive_dir,
            "max_retired_per_cycle": self.max_retired_per_cycle,
            "checkpoint_interval_minutes": self.checkpoint_interval_minutes,
            "force_retire_after_gens": self.force_retire_after_gens,
            "force_retire_min_fitness": self.force_retire_min_fitness,
            "mid_stagnation_threshold": self.mid_stagnation_threshold,
            "mid_stagnation_random_frac": self.mid_stagnation_random_frac,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionConfig:
        return cls(
            candidates_per_gen=d.get("candidates_per_gen", DEFAULT_CANDIDATES_PER_GEN),
            elite_count=d.get("elite_count", DEFAULT_ELITE_COUNT),
            random_injection=d.get("random_injection", DEFAULT_RANDOM_INJECTION),
            mutation_rate=d.get("mutation_rate", DEFAULT_MUTATION_RATE),
            crossover_rate=d.get("crossover_rate", DEFAULT_CROSSOVER_RATE),
            wall_time_seconds=d.get("wall_time_seconds", DEFAULT_WALL_TIME_SECONDS),
            max_generations=d.get("max_generations", DEFAULT_MAX_GENERATIONS),
            stagnation_generations=d.get("stagnation_generations", DEFAULT_STAGNATION_GENERATIONS),
            all_rejected_generations=d.get("all_rejected_generations", DEFAULT_ALL_REJECTED_GENERATIONS),
            parallel_workers=d.get("parallel_workers", DEFAULT_PARALLEL_WORKERS),
            base_seed=d.get("base_seed", 42),
            tp_pct=d.get("tp_pct", 0.02),
            output_dir=d.get("output_dir", "results/evolution"),
            experiment_id=d.get("experiment_id", "exp_default"),
            leaderboard_top_n=d.get("leaderboard_top_n", 20),
            island_mode=d.get("island_mode", False),
            n_islands=d.get("n_islands", 8),
            migration_every_n_gens=d.get("migration_every_n_gens", 5),
            migrants_per_island=d.get("migrants_per_island", 4),
            per_island_stagnation=d.get("per_island_stagnation", True),
            min_consistency_for_elite=d.get("min_consistency_for_elite", 0.50),
            min_discovery_for_elite=d.get("min_discovery_for_elite", 0.70),
            retirement_enabled=d.get("retirement_enabled", False),
            retirement_threshold=d.get("retirement_threshold", 0.75),
            retirement_archive_dir=d.get("retirement_archive_dir", "runs/retired_islands"),
            max_retired_per_cycle=d.get("max_retired_per_cycle", 999),
            checkpoint_interval_minutes=d.get("checkpoint_interval_minutes", 20),
            force_retire_after_gens=d.get("force_retire_after_gens", 8),
            force_retire_min_fitness=d.get("force_retire_min_fitness", 0.70),
            mid_stagnation_threshold=d.get("mid_stagnation_threshold", 8),
            mid_stagnation_random_frac=d.get("mid_stagnation_random_frac", 0.50),
        )
