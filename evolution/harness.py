"""EvolutionHarness — the main GA loop with all stop conditions.

Stop conditions (any of these halts the run):
1. Wall-time cap (default 8h)
2. Max generations reached (default 20)
3. Stagnation: no fitness improvement for N generations (default 5)
4. All-rejected: every candidate rejected for N generations (default 3)
5. KeyboardInterrupt → save and exit cleanly

Each generation:
1. Generate N candidates (random for gen 0, mutate/crossover + random for gen 1+)
2. Evaluate each (backtest + fitness)
3. Sort by fitness, take top-K as elites
4. Persist gen record (atomic write)
5. Check stop conditions
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import random
import signal
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evolution.config import EvolutionConfig
from evolution.evaluator import CandidateEvaluator, EvaluationResult, _evaluate_one
from evolution.islands import (
    distribute_migrants,
    get_island_specs,
    select_migrants,
)
from evolution.operators import crossover, mutate, random_candidate_genome
from evolution.persistence import (
    GenerationHistory,
    GenerationRecord,
    RunSummary,
    UnfinishedStatus,
    save_best_genome,
    save_leaderboard,
    save_rejection_report,
    save_run_summary,
    save_state,
    save_unfinished_status,
)
from evolution.population_builder import (
    build_island_population,
    build_population,
    get_island_id_for_genome,
)
from genome.schema import CandidateGenome


def _seed_island_via_spec(
    rng: random.Random,
    generation_index: int,
    spec,
    count: int,
) -> list[CandidateGenome]:
    """Module-level helper: seed N candidates biased by island_spec.

    Used when an island has no elites (e.g. first gen or all-rejected prior gen).
    Delegates to population_builder.build_island_population for consistency.
    """
    return build_island_population(
        rng=rng,
        generation_index=generation_index,
        island_specs=[spec],
        gid_start=0,
        random_count=0,
    )[:count]


def _make_island_spec_from_bias(island_id: int, bias: dict, n_candidates: int):
    """Build a transient IslandSpec from a bias dict (used for retirement re-seeds).

    bias dict keys (any subset):
      - name: str (for logging)
      - forced_grid_methods: tuple[GridMethod, ...]
      - forced_allocation: AllocationMethod
      - forced_confirmations: tuple[ConfirmationIndicator, ...]
      - max_dca_layers_cap: int
    """
    from evolution.islands import IslandSpec
    return IslandSpec(
        island_id=island_id,
        name=bias.get("name", f"retired_replaced_{island_id}"),
        n_candidates=n_candidates,
        forced_grid_methods=bias.get("forced_grid_methods"),
        forced_allocation=bias.get("forced_allocation"),
        forced_confirmations=bias.get("forced_confirmations"),
        max_dca_layers_cap=bias.get("max_dca_layers_cap"),
        note=f"re-seeded after retirement (was: {bias.get('name', '?')})",
    )


@dataclass
class HarnessHooks:
    """Optional callbacks for observability/testing."""
    on_generation_start: Callable[[int], None] | None = None
    on_generation_end: Callable[[GenerationRecord], None] | None = None
    on_candidate_evaluated: Callable[[EvaluationResult], None] | None = None
    on_termination: Callable[[str, GenerationHistory], None] | None = None


class EvolutionHarness:
    """The bounded, resumable GA loop."""
    def __init__(
        self,
        config: EvolutionConfig,
        df: pd.DataFrame,
        hooks: HarnessHooks | None = None,
        seeded_population: list[CandidateGenome] | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.df = df
        self.hooks = hooks or HarnessHooks()
        self.evaluator = CandidateEvaluator(df, experiment_slug=config.experiment_id)
        self._interrupted = False
        self._seeded_population = seeded_population
        self._rng = rng or random.Random()
        self._setup_signal_handler()
        # Island-mode state (Plan B, effective 2026-06-22)
        # {island_id: best_fitness_so_far} for per-island stagnation tracking
        self._island_best_fitness: dict[int, float] = {}
        # {island_id: gens_since_improvement} for per-island stagnation guard
        self._island_stagnation_counter: dict[int, int] = {}
        # Last migration generation (for migration_every_n_gens)
        self._last_migration_gen: int = -1
        # Incoming migrants from last migration step
        self._incoming_migrants: dict[int, list[CandidateGenome]] = {}
        # Retirement state (effective 2026-06-22, Six's plan B extension).
        # {island_id: family_bias_dict}. Updated when an island is retired.
        self._island_family_bias: dict[int, dict[str, Any]] = {}
        # All retirement records produced in this run (one entry per retired slot).
        self._retired_records: list[Any] = []  # list[RetiredIslandRecord]
        # Cycle ID used as the prefix for retired island directories.
        self._cycle_id: str = time.strftime("%Y%m%d_%H%M%S")
        # Recent bias names (anti-clustering when picking fresh biases)
        self._recent_bias_names: list[str] = []
        # Wall-clock minutes since cycle start at the last checkpoint save.
        # Tracks elapsed-min, not unix time, so wall-clock skew doesn't matter.
        self._last_checkpoint_min: float = -1e9  # sentinel → first save at gen 0
        # Force-retire log: {island_id: gen_when_force_retired}, so we don't
        # immediately force-retire a freshly re-seeded island.
        self._force_retired_at_gen: dict[int, int] = {}
        # Reference to the GenerationHistory object during run(), used by
        # retirement to read per-island history slice.
        self._last_history: Any = None  # type: ignore[name-defined]  # noqa: F821

    def _setup_signal_handler(self) -> None:
        """SIGINT (Ctrl+C) saves state and exits cleanly."""
        def handler(signum: int, frame: Any) -> None:
            self._interrupted = True
        with contextlib.suppress(ValueError):
            # Not in main thread — skip
            signal.signal(signal.SIGINT, handler)

    # ------------------------------------------------------------------
    # Stagnation (Fix A, 2026-06-22)
    # ------------------------------------------------------------------

    def _check_stagnation(
        self,
        gen_record: GenerationRecord,
        gen_idx: int,
        last_improvement_gen: int,
    ) -> bool:
        """Check whether the run has stagnated. Returns True if so.

        Behavior depends on config.per_island_stagnation:
        - True (island mode): track per-island best fitness. Stagnation only
          fires when ALL islands with at least one elite-eligible candidate
          have failed to improve for stagnation_generations consecutive gens.
        - False (single-pop): track global best fitness. Stagnation fires
          after stagnation_generations consecutive gens with no global
          improvement. (Original behavior — preserved for non-island runs.)
        """
        if not self.config.per_island_stagnation:
            # Original global stagnation logic
            gens_since_improvement = gen_idx - last_improvement_gen
            return gens_since_improvement >= self.config.stagnation_generations

        # Per-island stagnation
        # 1. Read per-island best fitness from this gen's record (computed in
        #    _run_generation by iterating candidates and reading their
        #    lineage.island_assign tag).
        per_island_best_this_gen = dict(gen_record.per_island_best_fitness or {})

        # 2. Update per-island best-ever + per-island counter
        for iid, fit in per_island_best_this_gen.items():
            prev_best = self._island_best_fitness.get(iid, 0.0)
            if fit > prev_best:
                self._island_best_fitness[iid] = fit
                self._island_stagnation_counter[iid] = 0
            else:
                self._island_stagnation_counter[iid] = (
                    self._island_stagnation_counter.get(iid, 0) + 1
                )

        # 3. Only consider islands that had at least one elite-eligible
        #    candidate this gen. Islands that produced 0 elites are being
        #    re-seeded — they shouldn't count toward the stagnation trigger.
        active_islands = [
            iid for iid in per_island_best_this_gen.keys()
            if (gen_record.per_island_elite_count or {}).get(iid, 0) > 0
        ]
        if not active_islands:
            # Nobody produced an elite — that's a separate signal handled by
            # the all-rejected check. Don't double-fire stagnation here.
            return False

        # 4. Stagnation fires only when ALL active islands have stagnated
        #    for stagnation_generations consecutive gens. If even one island
        #    is still improving, we keep going.
        all_stagnant = all(
            self._island_stagnation_counter.get(iid, 0) >= self.config.stagnation_generations
            for iid in active_islands
        )
        return all_stagnant

    # ------------------------------------------------------------------
    # Force-retire on per-island stagnation (Plan: 2026-06-24, Six)
    # ------------------------------------------------------------------

    def _check_force_retire(
        self,
        gen_record: GenerationRecord,
        gen_idx: int,
        candidates: list,
        rng: random.Random,
    ) -> tuple[list[dict], dict[int, str]]:
        """Per-island force-retire: kill dead islands, re-seed from pool.

        Rule (per Six's spec, 2026-06-24):
            If an island's stagnation_counter >= force_retire_after_gens
            AND its top fitness is < force_retire_min_fitness
            AND it hasn't been re-seeded in the last 3 gens
            → archive it (reason="stagnation_force") and re-seed with
              a fresh bias from the 17-bias pool.

        This complements `_check_stagnation` (which terminates the WHOLE
        run when ALL islands stagnate) and `_check_retirement` (which
        archives on fitness ≥ threshold). Force-retire keeps individual
        dead islands from holding slots while the rest of the population
        is still improving.

        Returns:
            (force_retired_dicts, bias_overrides)
            Same shape as `_check_retirement`.
        """
        if not self.config.retirement_enabled:
            return [], {}
        if self.config.force_retire_after_gens < 1:
            return [], {}

        from evolution.retirement import (
            RetirementPolicy,
            archive_island,
            pick_fresh_bias,
        )

        # Lazy init (handles resume + first-call)
        if not self._island_family_bias:
            self._init_island_family_bias()

        # Find islands eligible for force-retire
        to_force_retire: list[int] = []
        for iid, counter in self._island_stagnation_counter.items():
            if counter < self.config.force_retire_after_gens:
                continue
            island_best = self._island_best_fitness.get(iid, 0.0)
            # Skip if fitness is already near the bar — let it keep trying
            if island_best >= self.config.force_retire_min_fitness:
                continue
            # Skip if recently re-seeded (grace period of 3 gens)
            last_reseed_gen = self._force_retired_at_gen.get(iid, -999)
            if gen_idx - last_reseed_gen < 3:
                continue
            to_force_retire.append(iid)

        if not to_force_retire:
            return [], {}

        policy = RetirementPolicy(
            enabled=True,
            threshold=self.config.retirement_threshold,
            archive_dir=self.config.retirement_archive_dir,
            max_retired_per_cycle=self.config.max_retired_per_cycle,
        )

        cand_by_id = {c.genome_id: c for c in candidates}

        retired_records: list[Any] = []
        bias_overrides: dict[int, str] = {}

        for iid in to_force_retire:
            # Get the elites for this island (same logic as _check_retirement)
            elites: list[tuple[Any, float]] = []
            for entry in gen_record.leaderboard:
                gid = entry["genome_id"]
                cand = cand_by_id.get(gid)
                if cand is None:
                    continue
                from evolution.population_builder import get_island_id_for_genome
                if get_island_id_for_genome(cand) == iid:
                    elites.append((cand, entry["discovery_fitness"]))

            if not elites:
                # Nothing to archive — just re-seed and reset counters
                fresh = pick_fresh_bias(
                    rng,
                    exclude_recent=self._recent_bias_names[-policy.recent_bias_window:],
                )
                old_name = self._island_family_bias.get(iid, {}).get("name", "?")
                self._island_family_bias[iid] = fresh
                self._island_best_fitness.pop(iid, None)
                self._island_stagnation_counter[iid] = 0
                self._force_retired_at_gen[iid] = gen_idx
                self._recent_bias_names.append(fresh.get("name", "?"))
                if len(self._recent_bias_names) > 8:
                    self._recent_bias_names = self._recent_bias_names[-8:]
                bias_overrides[iid] = fresh.get("name", "?")
                print(
                    f"[evo] 🔁 FORCE-RETIRED island {iid} ({old_name}) "
                    f"@ gen {gen_idx} (stagnation={self.config.force_retire_after_gens}, "
                    f"no elites to archive) → {fresh.get('name', '?')}",
                    flush=True,
                )
                continue

            bias = self._island_family_bias.get(iid, {"name": f"unknown_{iid}"})
            record = archive_island(
                policy=policy,
                cycle_id=self._cycle_id,
                cycle_output_dir=self.config.output_dir,
                island_id=iid,
                retired_at_gen=gen_idx,
                family_bias=bias,
                per_island_top_fitness=self._island_best_fitness.get(iid, 0.0),
                elites=elites,
                generations_evolved=gen_idx + 1,
            )
            retired_records.append(record)
            self._retired_records.append(record)

            old_name = bias.get("name", "?")
            fresh = pick_fresh_bias(
                rng,
                exclude_recent=self._recent_bias_names[-policy.recent_bias_window:],
            )
            self._island_family_bias[iid] = fresh
            self._island_best_fitness.pop(iid, None)
            self._island_stagnation_counter[iid] = 0
            self._force_retired_at_gen[iid] = gen_idx
            self._recent_bias_names.append(fresh.get("name", "?"))
            if len(self._recent_bias_names) > 8:
                self._recent_bias_names = self._recent_bias_names[-8:]
            bias_overrides[iid] = fresh.get("name", "?")

            print(
                f"[evo] 🔁 FORCE-RETIRED island {iid} ({old_name}) "
                f"@ gen {gen_idx} fitness={record.per_island_top_fitness:.4f} "
                f"(stagnation={self.config.force_retire_after_gens}) "
                f"→ {fresh.get('name', '?')}",
                flush=True,
            )

        # Return shape mirrors _check_retirement: dicts (not RetiredIslandRecord)
        return (
            [r.to_dict() for r in retired_records],
            bias_overrides,
        )

    # ------------------------------------------------------------------
    # Retirement (effective 2026-06-22, Six's plan B extension)
    # ------------------------------------------------------------------

    def _init_island_family_bias(self) -> None:
        """Seed _island_family_bias from static ISLAND_SPECS at startup."""
        if self._island_family_bias:
            return  # already initialized (e.g. on resume)
        from evolution.islands import get_island_specs
        for spec in get_island_specs()[:self.config.n_islands]:
            bias = {"name": spec.name}
            if spec.forced_grid_methods is not None:
                bias["forced_grid_methods"] = spec.forced_grid_methods
            if spec.forced_allocation is not None:
                bias["forced_allocation"] = spec.forced_allocation
            if spec.forced_confirmations is not None:
                bias["forced_confirmations"] = spec.forced_confirmations
            if spec.max_dca_layers_cap is not None:
                bias["max_dca_layers_cap"] = spec.max_dca_layers_cap
            self._island_family_bias[spec.island_id] = bias

    def _check_retirement(
        self,
        gen_record: GenerationRecord,
        candidates: list,
        rng: random.Random,
    ) -> tuple[list[dict], dict[int, str]]:
        """Check per-island top fitness against retirement threshold.

        For each island whose per_island_best_fitness crosses the threshold,
        archive the island's current top elites + per-island history, then
        mark the slot for re-seeding with a fresh bias.

        Returns:
            (retired_records, bias_overrides_for_next_gen)
        """
        if not self.config.retirement_enabled:
            return [], {}

        # Lazy init (handles resume + first-call)
        if not self._island_family_bias:
            self._init_island_family_bias()

        from evolution.retirement import (
            RetirementPolicy,
            check_for_retirements,
        )

        policy = RetirementPolicy(
            enabled=True,
            threshold=self.config.retirement_threshold,
            archive_dir=self.config.retirement_archive_dir,
            max_retired_per_cycle=self.config.max_retired_per_cycle,
        )

        # Build elites_by_island: {island_id: [(genome, fitness), ...]}
        # We need the full CandidateGenome objects for the archive, not just IDs.
        # Re-derive from gen_record.evaluated_candidate_ids + the original candidates.
        cand_by_id = {c.genome_id: c for c in candidates}
        elites_by_island: dict[int, list[tuple]] = {}

        # Iterate leaderboard (top by discovery_fitness) and bucket by island_id
        from evolution.population_builder import get_island_id_for_genome
        for entry in gen_record.leaderboard:
            gid = entry["genome_id"]
            cand = cand_by_id.get(gid)
            if cand is None:
                continue
            iid = get_island_id_for_genome(cand)
            if iid == 0:
                continue
            elites_by_island.setdefault(iid, []).append((cand, entry["discovery_fitness"]))

        # Per-island history slice — just the per-island_best_fitness over gens
        per_island_history: dict[int, list[dict]] = {}
        last_history = getattr(self, "_last_history", None)
        if last_history is not None:
            for gen in last_history.generations:
                for iid, fit in (gen.per_island_best_fitness or {}).items():
                    per_island_history.setdefault(iid, []).append({
                        "gen": gen.generation_index,
                        "best_fitness": fit,
                        "n_passed": gen.per_island_best_count.get(iid, 0),
                        "n_elite": gen.per_island_elite_count.get(iid, 0),
                    })

        retired_records, new_assignments = check_for_retirements(
            policy=policy,
            cycle_id=self._cycle_id,
            cycle_output_dir=self.config.output_dir,
            gen_record=gen_record,
            elites_by_island=elites_by_island,
            family_bias_by_island=self._island_family_bias,
            per_island_history_by_island=per_island_history,
            rng=rng,
        )

        # Update internal state
        for rec in retired_records:
            self._retired_records.append(rec)
            # Replace bias
            old_bias = self._island_family_bias.get(rec.island_id, {})
            old_name = old_bias.get("name", "?")
            fresh = new_assignments.get(rec.island_id, {})
            self._island_family_bias[rec.island_id] = fresh
            new_name = fresh.get("name", "?")
            self._recent_bias_names.append(new_name)
            # Trim recent list
            if len(self._recent_bias_names) > 8:
                self._recent_bias_names = self._recent_bias_names[-8:]
            print(
                f"[evo] 🏝️ RETIRED island {rec.island_id} ({old_name}) "
                f"@ gen {rec.retired_at_gen} fitness={rec.per_island_top_fitness:.4f} "
                f"→ replaced with {new_name}",
                flush=True,
            )

        bias_overrides = {iid: b.get("name", "?") for iid, b in new_assignments.items()}
        # Also reset per-island state for retired islands
        for rec in retired_records:
            self._island_best_fitness.pop(rec.island_id, None)
            self._island_stagnation_counter.pop(rec.island_id, None)

        return (
            [r.to_dict() for r in retired_records],
            bias_overrides,
        )

    # _per_island_best_from_record removed: per-island best fitness is now
    # computed inline in _run_generation and stored on gen_record directly.

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, resume: bool = True) -> RunSummary:
        """Run the evolution loop. Returns a RunSummary.

        resume: if True and generation_history.json exists, pick up where
        we left off. If False, start fresh (overwrites any existing state).
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load or init history
        existing = self._load_history() if resume else None
        if existing is not None:
            history = existing
        else:
            history = GenerationHistory(
                experiment_id=self.config.experiment_id,
                config=self.config.to_dict(),
                started_at=time.time(),
            )

        # Resume from the next generation
        start_gen = len(history.generations)
        rng = self._rng if self._rng is not None else random.Random(self.config.base_seed + start_gen)

        # Initialize island family-bias map (used by retirement)
        if self.config.island_mode:
            self._init_island_family_bias()
        # Keep last history accessible for retirement's per-island history slice
        self._last_history = history

        # Track stagnation state across the run
        last_improvement_gen = start_gen  # any improvement since start is "now"
        all_rejected_streak = 0

        t_start = time.time()
        termination_reason = "completed"

        try:
            for gen_idx in range(start_gen, self.config.max_generations):
                if self._interrupted:
                    termination_reason = "interrupted"
                    break

                # Wall-time check
                elapsed = time.time() - t_start
                if elapsed >= self.config.wall_time_seconds:
                    termination_reason = "wall_time"
                    break

                if self.hooks.on_generation_start:
                    self.hooks.on_generation_start(gen_idx)

                # === Run one generation ===
                gen_record = self._run_generation(gen_idx, history, rng, t_start)
                history.generations.append(gen_record)

                # Update best-ever (global)
                if gen_record.best_fitness > history.best_fitness_ever:
                    history.best_fitness_ever = gen_record.best_fitness
                    history.best_genome_id_ever = gen_record.best_genome_id
                    history.best_candidate_id_ever = gen_record.best_candidate_id
                    last_improvement_gen = gen_idx

                # Stagnation check (Fix A, 2026-06-22): per-island vs global
                stagnation_hit = self._check_stagnation(
                    gen_record, gen_idx, last_improvement_gen,
                )
                if stagnation_hit:
                    termination_reason = "stagnation"
                    if self.hooks.on_termination:
                        self.hooks.on_termination(termination_reason, history)
                    break

                # Check all-rejected
                if gen_record.n_passed == 0:
                    all_rejected_streak += 1
                    if all_rejected_streak >= self.config.all_rejected_generations:
                        termination_reason = "all_rejected"
                        if self.hooks.on_termination:
                            self.hooks.on_termination(termination_reason, history)
                        break
                else:
                    all_rejected_streak = 0

                # Persist state after every gen
                save_state(history, self.config.output_dir)

                # Periodic checkpoint (every N minutes, default 20).
                # Write to <project_root>/checkpoints/ so a computer restart
                # mid-cycle can resume from the latest snapshot.
                if self.config.checkpoint_interval_minutes > 0:
                    elapsed_min = (time.time() - t_start) / 60.0
                    if elapsed_min - self._last_checkpoint_min >= self.config.checkpoint_interval_minutes:
                        from evolution.persistence import save_checkpoint
                        retired_dicts = [r.to_dict() for r in self._retired_records]
                        rng_state = self._rng.getstate() if self._rng else None
                        save_checkpoint(
                            cycle_id=self._cycle_id,
                            gen_idx=gen_idx,
                            wall_time_used=time.time() - t_start,
                            per_island_best_fitness=self._island_best_fitness,
                            per_island_stagnation_counter=self._island_stagnation_counter,
                            retired_so_far=retired_dicts,
                            rng_state=rng_state,
                            extra={"termination_reason": termination_reason},
                        )
                        self._last_checkpoint_min = elapsed_min

                if self.hooks.on_generation_end:
                    self.hooks.on_generation_end(gen_record)

        except KeyboardInterrupt:
            termination_reason = "interrupted"
        except Exception:
            termination_reason = "error"
            traceback.print_exc()
            # Save state anyway
            save_state(history, self.config.output_dir)
            raise

        # === Wrap up ===
        t_end = time.time()
        total_runtime = t_end - t_start

        # Build summary
        total_evaluated = sum(g.n_candidates for g in history.generations)
        summary = RunSummary(
            experiment_id=self.config.experiment_id,
            started_at=history.started_at,
            finished_at=t_end,
            total_runtime_seconds=total_runtime,
            generations_completed=len(history.generations),
            generations_planned=self.config.max_generations,
            total_candidates_evaluated=total_evaluated,
            best_fitness_ever=history.best_fitness_ever,
            best_genome_id_ever=history.best_genome_id_ever,
            best_candidate_id_ever=history.best_candidate_id_ever,
            termination_reason=termination_reason,
            output_dir=self.config.output_dir,
        )
        save_run_summary(summary, self.config.output_dir)

        # If we didn't complete all generations, write unfinished_status.json
        if termination_reason != "completed":
            status = UnfinishedStatus(
                reason=termination_reason,
                generations_completed=len(history.generations),
                max_generations=self.config.max_generations,
                wall_time_seconds_used=total_runtime,
                wall_time_seconds_cap=self.config.wall_time_seconds,
                best_fitness_ever=history.best_fitness_ever,
                best_genome_id_ever=history.best_genome_id_ever,
                best_candidate_id_ever=history.best_candidate_id_ever,
                finished_at=t_end,
            )
            save_unfinished_status(status, self.config.output_dir)

        return summary

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        gen_idx: int,
        history: GenerationHistory,
        rng: random.Random,
        t_start: float,
    ) -> GenerationRecord:
        """Run one generation. Returns a GenerationRecord."""
        gen_started = time.time()

        # 1) Generate candidates
        if gen_idx == 0:
            candidates = self._generate_gen0(rng, gen_idx)
        else:
            candidates = self._generate_next_gen(history, rng, gen_idx)

        # 2) Evaluate each — parallel via ProcessPoolExecutor
        results: list[EvaluationResult] = []
        n_workers = max(1, self.config.parallel_workers)

        if n_workers == 1:
            # Sequential (debugging / single-core mode)
            for cand in candidates:
                history.candidate_counter += 1
                cand_id = f"cand_{history.candidate_counter:06d}"
                res = self.evaluator.evaluate(cand, cand_id)
                results.append(res)
                if self.hooks.on_candidate_evaluated:
                    self.hooks.on_candidate_evaluated(res)
        else:
            # Parallel: spawn workers, each gets a copy of the dataframe
            # Pickle df once and pass to each worker
            ctx = mp.get_context("spawn")  # safer than fork on Linux
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                # Submit all jobs
                futures = {}
                for cand in candidates:
                    history.candidate_counter += 1
                    cand_id = f"cand_{history.candidate_counter:06d}"
                    fut = pool.submit(
                        _evaluate_one,
                        self.df,
                        cand,
                        cand_id,
                        self.config.experiment_id,
                        time.time(),
                    )
                    futures[fut] = cand_id

                # Collect results as they complete
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as e:
                        # Worker crashed — log and continue
                        cand_id = futures[fut]
                        res = EvaluationResult(
                            candidate_id=cand_id,
                            genome_id=cand_id,
                            discovery_fitness=0.0,
                            deployment_fitness=0.0,
                            deployment_pass=False,
                            failed_deployment_gates=["worker_error"],
                            closest_to_passing_score=0.0,
                            consistency_ratio=0.0,
                            consistency_multiplier=0.0,
                            rejected=True,
                            reject_reason="worker_error",
                            rejection_source="worker",
                            elapsed_seconds=0.0,
                            monthly_fitness=self._empty_fitness(cand_id),
                            score_breakdown=None,
                            raw_metrics={},
                            n_cycles_closed=0,
                            final_equity=0.0,
                            max_dd_pct=1.0,
                            error=str(e),
                        )
                    results.append(res)
                    if self.hooks.on_candidate_evaluated:
                        self.hooks.on_candidate_evaluated(res)

        # 3) Sort + select. With the discovery/deployment split:
        #    - "passed" = not hard-rejected (eligible for diversity, may still
        #      be sub-deployment).
        #    - "elite_eligible" = subset of passed that also meets the
        #      elite quality gate (Fix B, 2026-06-22): consistency >= min OR
        #      discovery_fitness >= min. Prevents a 0.28-fitness candidate
        #      from becoming the breeding seed for the next 4 gens.
        #    - Breeding sort: by discovery_fitness descending (across elite_eligible).
        #    - Deployment sort: by deployment_fitness descending (only those
        #      with deployment_pass=True).
        passed = [r for r in results if not r.rejected]
        elite_eligible = [
            r for r in passed
            if r.consistency_ratio >= self.config.min_consistency_for_elite
            or r.discovery_fitness >= self.config.min_discovery_for_elite
        ]
        # If no one qualifies as elite (all soft-passed), fall back to all-passed
        # so we never produce an empty breeding pool.
        breeding_pool = elite_eligible if elite_eligible else passed
        breeding_pool.sort(key=lambda r: r.discovery_fitness, reverse=True)
        elites = breeding_pool[:self.config.elite_count]

        # Top-N by discovery (the "almost passing" diagnostic)
        top_discovery = passed[:self.config.leaderboard_top_n]

        # Top-N by deployment (only deployment-passing candidates)
        deployment_passing = [r for r in results if r.deployment_pass]
        deployment_passing.sort(key=lambda r: r.deployment_fitness, reverse=True)
        top_deployment = deployment_passing[:self.config.leaderboard_top_n]

        # 4) Build GenerationRecord
        reason_counts = Counter(r.reject_reason for r in results if r.rejected)
        all_reasons: dict[str, int] = {
            str(k): int(v) for k, v in reason_counts.items() if k is not None
        }
        best = elites[0] if elites else None
        discovery_fitnesses = [r.discovery_fitness for r in passed]
        median_fitness = (
            sorted(discovery_fitnesses)[len(discovery_fitnesses) // 2]
            if discovery_fitnesses else 0.0
        )
        leaderboard = [
            {
                "rank": i + 1,
                "candidate_id": r.candidate_id,
                "genome_id": r.genome_id,
                "discovery_fitness": r.discovery_fitness,
                "deployment_fitness": r.deployment_fitness,
                "deployment_pass": r.deployment_pass,
                "failed_deployment_gates": r.failed_deployment_gates,
                "closest_to_passing_score": r.closest_to_passing_score,
                "consistency_ratio": r.consistency_ratio,
                "consistency_multiplier": r.consistency_multiplier,
                "n_cycles_closed": r.n_cycles_closed,
                "final_equity": r.final_equity,
                "max_dd_pct": r.max_dd_pct,
                # Phase D — Discovery Fitness v2 component scores
                "full_period_base_score": r.full_period_base_score,
                "recovery_score": r.recovery_score,
                "stability_score": r.stability_score,
                "concentration_score": r.concentration_score,
                "recovery_breakdown": dict(r.recovery_breakdown or {}),
            }
            for i, r in enumerate(top_discovery)
        ]
        deployment_leaderboard = [
            {
                "rank": i + 1,
                "candidate_id": r.candidate_id,
                "genome_id": r.genome_id,
                "deployment_fitness": r.deployment_fitness,
                "discovery_fitness": r.discovery_fitness,
                "consistency_ratio": r.consistency_ratio,
            }
            for i, r in enumerate(top_deployment)
        ]
        # Per-island top-1 (Fix A, 2026-06-22): we know the island_id from the
        # genome (tagged in lineage.mutation_ops at population build time).
        # Build a {island_id: best_fitness} map by iterating passed candidates.
        per_island_best_fitness: dict[int, float] = {}
        per_island_best_count: dict[int, int] = {}
        per_island_elite_count: dict[int, int] = {}
        for r in passed:
            # Find the genome in the candidates list to read island_id
            cand = next((c for c in candidates if c.genome_id == r.genome_id), None)
            if cand is None:
                continue
            iid = get_island_id_for_genome(cand)
            # Skip island 0 (random bag) for stagnation tracking
            if iid == 0:
                continue
            fit = r.discovery_fitness
            if iid not in per_island_best_fitness or fit > per_island_best_fitness[iid]:
                per_island_best_fitness[iid] = fit
            per_island_best_count[iid] = per_island_best_count.get(iid, 0) + 1
            if (r in elite_eligible):
                per_island_elite_count[iid] = per_island_elite_count.get(iid, 0) + 1
        record = GenerationRecord(
            generation_index=gen_idx,
            started_at=gen_started,
            ended_at=time.time(),
            n_candidates=len(results),
            n_rejected=len(results) - len(passed),
            n_passed=len(passed),
            n_elite_eligible=len(elite_eligible),
            n_deployment_passing=len(deployment_passing),
            best_fitness=best.discovery_fitness if best else 0.0,
            median_fitness=median_fitness,
            best_candidate_id=best.candidate_id if best else "",
            best_genome_id=best.genome_id if best else "",
            wall_time_seconds_used=time.time() - t_start,
            rejection_reasons=all_reasons,
            evaluated_candidate_ids=[r.candidate_id for r in results],
            leaderboard=leaderboard,
            deployment_leaderboard=deployment_leaderboard,
            per_island_best_fitness=per_island_best_fitness,
            per_island_best_count=per_island_best_count,
            per_island_elite_count=per_island_elite_count,
            per_island_stagnation_counter=dict(self._island_stagnation_counter),
        )

        # 5) Persist per-generation artifacts
        save_leaderboard(gen_idx, leaderboard, self.config.output_dir)
        save_rejection_report(gen_idx, all_reasons, self.config.output_dir)
        if best:
            best_genome = next(
                (c for c in candidates if c.genome_id == best.genome_id), None
            )
            if best_genome is not None:
                save_best_genome(gen_idx, best_genome.to_dict(), self.config.output_dir)
        else:
            (Path(self.config.output_dir) / "best_genomes").mkdir(parents=True, exist_ok=True)

        # 6) Retirement check (effective 2026-06-22, Six's plan B extension).
        #    Only fires when retirement_enabled=True. Archives any island whose
        #    per_island_best_fitness crossed the threshold this gen, then marks
        #    the slot for re-seeding with a fresh bias.
        retired_dicts, bias_overrides = self._check_retirement(record, candidates, rng)
        record.retired_islands = retired_dicts
        record.island_bias_overrides = bias_overrides

        # 6b) Force-retire on per-island stagnation (Plan: 2026-06-24, Six).
        #     Fires AFTER fitness-retirement so the priority order is:
        #       fitness ≥ threshold → archive with reason "fitness"
        #       stagnation ≥ N gens + fitness < min → archive with reason "stagnation_force"
        #     A force-retire in the same gen as a fitness-retire is fine —
        #     different islands, different rules.
        force_dicts, force_bias_overrides = self._check_force_retire(
            record, gen_idx, candidates, rng,
        )
        if force_dicts:
            # Merge into the record's retirement lists so it persists + reports
            record.retired_islands.extend(force_dicts)
            record.island_bias_overrides.update(force_bias_overrides)

        return record

    # ------------------------------------------------------------------
    # Generation construction
    # ------------------------------------------------------------------

    def _generate_gen0(
        self,
        rng: random.Random,
        gen_idx: int,
    ) -> list[CandidateGenome]:
        """Gen 0: use seeded population if provided, else all-random.

        Island mode: if seeded_population is provided AND island_mode is on,
        partition the seeded pop across islands using build_island_population.
        Otherwise (island_mode with no seeded pop, or single-pop mode),
        fall through to the original path.
        """
        if self._seeded_population is not None:
            return self._seeded_population

        if self.config.island_mode:
            specs = get_island_specs()[:self.config.n_islands]
            return build_island_population(
                rng=rng,
                generation_index=gen_idx,
                island_specs=specs,
                gid_start=0,
                random_count=4,
            )

        # Single-population fallback (original behavior)
        return [
            random_candidate_genome(
                rng=rng,
                generation_index=gen_idx,
                tp_pct=self.config.tp_pct,
            )
            for _ in range(self.config.candidates_per_gen)
        ]

    def _generate_next_gen(
        self,
        history: GenerationHistory,
        rng: random.Random,
        gen_idx: int,
    ) -> list[CandidateGenome]:
        """Build next gen from elites + crossover + mutation + random injection.

        Island mode: each island's elites breed among themselves (with migrants
        from neighbors received in the last migration step). Random injection
        stays per-island for diversity.

        If there are no elites (all-rejected prior gen), fall back to fully
        random — we still produce candidates_per_gen of them.
        """
        prev_gen = history.generations[-1]
        target = self.config.candidates_per_gen

        # Island mode branch
        if self.config.island_mode:
            return self._generate_next_gen_island(
                prev_gen, gen_idx, rng, target,
            )

        elite_genomes = self._load_elite_genomes(prev_gen, gen_idx, rng)

        # No elites → all random
        if not elite_genomes:
            return [
                random_candidate_genome(
                    rng=rng, generation_index=gen_idx, tp_pct=self.config.tp_pct,
                )
                for _ in range(target)
            ]

        n_crossover = self.config.crossover_children
        n_mutation = self.config.mutation_children
        n_random = self.config.random_injection

        children: list[CandidateGenome] = []

        # Crossover children
        for _ in range(n_crossover):
            if len(elite_genomes) < 2:
                # Not enough elites — fall back to mutation
                children.append(mutate(elite_genomes[0], rng=rng, mutation_rate=self.config.mutation_rate))
                continue
            a = rng.choice(elite_genomes)
            b = rng.choice(elite_genomes)
            children.append(crossover(a, b, rng=rng))

        # Mutation children
        for _ in range(n_mutation):
            parent = rng.choice(elite_genomes)
            children.append(mutate(parent, rng=rng, mutation_rate=self.config.mutation_rate))

        # Random injection (fresh genomes for diversity)
        for _ in range(n_random):
            children.append(
                random_candidate_genome(rng=rng, generation_index=gen_idx, tp_pct=self.config.tp_pct)
            )

        # Pad to target (shouldn't be needed but defensive)
        while len(children) < target:
            children.append(
                random_candidate_genome(rng=rng, generation_index=gen_idx, tp_pct=self.config.tp_pct)
            )

        return children[:target]

    # ------------------------------------------------------------------
    # Island mode — per-island breeding + migration
    # ------------------------------------------------------------------

    def _generate_next_gen_island(
        self,
        prev_gen: GenerationRecord,
        gen_idx: int,
        rng: random.Random,
        target: int,
    ) -> list[CandidateGenome]:
        """Breed each island independently from its own elites + migrants.

        Island allocation per generation:
          - 3 elites (carried over, may include migrants from last migration)
          - crossover_children + mutation_children (per-island)
          - random_injection (per-island, fresh DNA)

        Total target = sum across all islands = candidates_per_gen.
        """
        specs = get_island_specs()[:self.config.n_islands]

        # 1. Migration step (every migration_every_n_gens generations)
        if (
            gen_idx > 0
            and gen_idx - self._last_migration_gen >= self.config.migration_every_n_gens
        ):
            self._do_migration(prev_gen, gen_idx, rng)

        # 1b. Apply retirement bias overrides from previous gen.
        #     If island X was retired last gen, use the fresh-bias spec instead
        #     of the static ISLAND_SPECS[X] spec.
        bias_overrides: dict[int, str] = dict(prev_gen.island_bias_overrides or {})  # type: ignore[arg-type]
        if bias_overrides:
            for iid, bias_name in bias_overrides.items():  # type: ignore[union-attr]
                if 1 <= iid <= len(specs):
                    bias = self._island_family_bias.get(iid, {})
                    specs[iid - 1] = _make_island_spec_from_bias(
                        iid, bias, specs[iid - 1].n_candidates,
                    )
                    print(
                        f"[evo] gen {gen_idx}: island {iid} using fresh bias "
                        f"'{bias_name}' (post-retirement re-seed)",
                        flush=True,
                    )

        # 2. Load per-island elites from previous generation's leaderboard
        per_island_elite_genomes = self._load_per_island_elites(prev_gen, gen_idx)

        # 3. Breed each island
        all_children: list[CandidateGenome] = []
        for spec in specs:
            elites = per_island_elite_genomes.get(spec.island_id, [])
            if not elites:
                # No elites for this island — fall back to seeding new random
                # biased by island spec via build_island_population's helper
                all_children.extend(_seed_island_via_spec(rng, gen_idx, spec, count=spec.n_candidates))
                continue

            n_iso_total = spec.n_candidates
            n_iso_random = max(2, int(n_iso_total * (self.config.random_injection / self.config.candidates_per_gen)))
            n_iso_crossover = int((n_iso_total - n_iso_random) * self.config.crossover_rate)
            n_iso_mutation = (n_iso_total - n_iso_random) - n_iso_crossover

            island_children: list[CandidateGenome] = []

            # Crossover
            for _ in range(n_iso_crossover):
                if len(elites) < 2:
                    island_children.append(mutate(elites[0], rng=rng, mutation_rate=self.config.mutation_rate))
                    continue
                a = rng.choice(elites)
                b = rng.choice(elites)
                island_children.append(crossover(a, b, rng=rng))

            # Mutation
            for _ in range(n_iso_mutation):
                parent = rng.choice(elites)
                island_children.append(mutate(parent, rng=rng, mutation_rate=self.config.mutation_rate))

            # Random injection (per-island fresh random)
            for _ in range(n_iso_random):
                island_children.append(
                    random_candidate_genome(rng=rng, generation_index=gen_idx, tp_pct=self.config.tp_pct)
                )

            # Track island assignment for these new candidates
            for c in island_children:
                c.lineage.mutation_ops = list(c.lineage.mutation_ops) + [{
                    "op": "island_assign", "island_id": spec.island_id,
                }]

            all_children.extend(island_children)

        # Pure-random bag (island 0)
        n_random_bag = max(0, target - len(all_children))
        for _ in range(n_random_bag):
            c = random_candidate_genome(rng=rng, generation_index=gen_idx, tp_pct=self.config.tp_pct)
            c.lineage.mutation_ops = list(c.lineage.mutation_ops) + [{
                "op": "island_assign", "island_id": 0,
            }]
            all_children.append(c)

        return all_children[:target]

    def _do_migration(
        self,
        prev_gen: GenerationRecord,
        gen_idx: int,
        rng: random.Random,
    ) -> None:
        """Migrate top-K elites from each island to its two neighbors.

        Stores migrants as `self._incoming_migrants: {island_id: [genome]}`
        so `_generate_next_gen_island` can fold them into the next gen's
        elites.
        """
        per_island_elite_genomes = self._load_per_island_elites(prev_gen, gen_idx)

        migrants_by_source: dict[int, list[CandidateGenome]] = {}
        for iid, elites in per_island_elite_genomes.items():
            if not elites:
                continue
            top_n = select_migrants(
                iid, elites, n_migrants=self.config.migrants_per_island, rng=rng,
            )
            if top_n:
                migrants_by_source[iid] = top_n

        received = distribute_migrants(
            migrants_by_source=migrants_by_source,
            n_islands=self.config.n_islands,
            rng=rng,
        )

        self._incoming_migrants = received
        self._last_migration_gen = gen_idx

    def _load_per_island_elites(
        self,
        prev_gen: GenerationRecord,
        gen_idx: int,
    ) -> dict[int, list[CandidateGenome]]:
        """Load per-island elites for breeding.

        Currently reads the previous gen's best_genome (single-pop legacy)
        and uses it as the seed for every island. Migrants from the last
        migration step are added on top. Future Stage 14 work: persist
        per-island best_genome files so each island can breed from its
        own actual #1.
        """
        out_dir = Path(self.config.output_dir)
        best_file = out_dir / "best_genomes" / f"gen_{prev_gen.generation_index:04d}.json"
        if not best_file.exists():
            return {}
        try:
            best_dict = json.loads(best_file.read_text())
        except Exception:
            return {}

        from genome.schema import CandidateGenome, DcaGenome, TpGenome, ConfirmationIndicator

        try:
            dca_d = best_dict["dca_genome"]
            tp_d = best_dict["tp_genome"]
            # Coerce confirmation_indicators back to enums (JSON round-trip strips them)
            raw_inds = dca_d.get("confirmation_indicators", [])
            coerced_inds = []
            for x in raw_inds:
                if isinstance(x, str):
                    try:
                        coerced_inds.append(ConfirmationIndicator(x))
                    except ValueError:
                        pass
                else:
                    coerced_inds.append(x)
            dca = DcaGenome(
                grid_method=dca_d.get("grid_method", "fixed_pct"),
                grid_params=dca_d.get("grid_params", {}),
                allocation_method=dca_d.get("allocation_method", "equal"),
                allocation_params=dca_d.get("allocation_params", {}),
                max_dca_layers=dca_d.get("max_dca_layers", 3),
                confirmation_indicators=coerced_inds,
                indicator_params=dca_d.get("indicator_params", {}),
            )
            tp = TpGenome(
                exit_method=tp_d.get("exit_method", "fixed"),
                exit_params=tp_d.get("exit_params", {}),
            )
            best = CandidateGenome(
                genome_id=best_dict.get("genome_id", f"genome_G{prev_gen.generation_index}_best"),
                dca_genome=dca,
                tp_genome=tp,
            )
        except Exception:
            return {}

        result: dict[int, list[CandidateGenome]] = {}
        for iid in range(1, self.config.n_islands + 1):
            elites = [best]
            if hasattr(self, "_incoming_migrants") and self._incoming_migrants:
                for m in self._incoming_migrants.get(iid, []):
                    elites.append(m)
            result[iid] = elites

        return result

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _empty_fitness(self, candidate_id: str):
        """Empty MonthlyFitnessResult for error cases."""
        from fitness.monthly_fitness import MonthlyFitnessResult
        return MonthlyFitnessResult(
            candidate_id=candidate_id,
            experiment_slug=self.config.experiment_id,
            monthly_scores=[],
            n_months=0,
            n_profitable_months=0,
            n_rejected_months=0,
            consistency_ratio=0.0,
            median_monthly_score=0.0,
            worst_month_score=0.0,
            stddev_monthly_score=0.0,
            variance_penalty=0.0,
            worst_floor_multiplier=0.0,
            base_aggregate_fitness=0.0,
            discovery_fitness=0.0,
            consistency_multiplier=0.0,
            deployment_fitness=0.0,
            deployment_pass=False,
            failed_deployment_gates=["evaluation_error"],
            closest_to_passing_score=0.0,
            final_fitness=0.0,
            rejected=True,
            reject_reason="no_data",
            full_period_score=None,
            full_period_rejected=True,
        )

    def _load_elite_genomes(
        self,
        prev_gen: GenerationRecord,
        gen_idx: int,
        rng: random.Random,
    ) -> list[CandidateGenome]:
        """Load the top-K genomes from the previous generation.

        We wrote best_genomes/gen_NNNN.json with the #1 genome. For others,
        we re-derive from the leaderboard entries. The leaderboard only
        stores IDs, not the full genome — but we have the candidate's
        params (grid_pct, max_layers) implicit in the evaluation pipeline.
        Simpler approach: store full genomes in the leaderboard too.
        For now, reconstruct top-K from the persisted best_genome file by
        re-using the best one's params and re-randomising the rest as a
        proxy. The full re-construction will land in Stage 14.
        """
        # Read best genome from disk
        out = Path(self.config.output_dir) / "best_genomes" / f"gen_{prev_gen.generation_index:04d}.json"
        if not out.exists():
            return []
        with open(out) as f:
            best_dict = json.load(f)
        try:
            from genome.schema import (
                CandidateGenome,
                DcaGenome,
                TpGenome,
                ConfirmationIndicator,
            )
            dca_d = best_dict["dca_genome"]
            tp_d = best_dict["tp_genome"]
            raw_inds = dca_d.get("confirmation_indicators", [])
            coerced_inds = []
            for x in raw_inds:
                if isinstance(x, str):
                    try:
                        coerced_inds.append(ConfirmationIndicator(x))
                    except ValueError:
                        pass
                else:
                    coerced_inds.append(x)
            dca = DcaGenome(
                grid_method=dca_d.get("grid_method", "fixed_pct"),
                grid_params=dca_d.get("grid_params", {}),
                allocation_method=dca_d.get("allocation_method", "equal"),
                allocation_params=dca_d.get("allocation_params", {}),
                max_dca_layers=dca_d.get("max_dca_layers", 3),
                confirmation_indicators=coerced_inds,
                indicator_params=dca_d.get("indicator_params", {}),
            )
            tp = TpGenome(
                exit_method=tp_d.get("exit_method", "fixed"),
                exit_params=tp_d.get("exit_params", {}),
            )
            best = CandidateGenome(
                genome_id=best_dict.get("genome_id", f"genome_G{prev_gen.generation_index}_best"),
                dca_genome=dca,
                tp_genome=tp,
            )
            return [best]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_history(self) -> GenerationHistory | None:
        from evolution.persistence import load_state
        return load_state(self.config.output_dir)
