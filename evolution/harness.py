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
import random
import signal
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evolution.config import EvolutionConfig
from evolution.evaluator import CandidateEvaluator, EvaluationResult
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
from genome.schema import CandidateGenome


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
    ):
        self.config = config
        self.df = df
        self.hooks = hooks or HarnessHooks()
        self.evaluator = CandidateEvaluator(df, experiment_slug=config.experiment_id)
        self._interrupted = False
        self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        """SIGINT (Ctrl+C) saves state and exits cleanly."""
        def handler(signum: int, frame: Any) -> None:
            self._interrupted = True
        with contextlib.suppress(ValueError):
            # Not in main thread — skip
            signal.signal(signal.SIGINT, handler)

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
        rng = random.Random(self.config.base_seed + start_gen)

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

                # Update best-ever
                if gen_record.best_fitness > history.best_fitness_ever:
                    history.best_fitness_ever = gen_record.best_fitness
                    history.best_genome_id_ever = gen_record.best_genome_id
                    history.best_candidate_id_ever = gen_record.best_candidate_id
                    last_improvement_gen = gen_idx
                else:
                    # Check stagnation
                    gens_since_improvement = gen_idx - last_improvement_gen
                    if gens_since_improvement >= self.config.stagnation_generations:
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

        # 2) Evaluate each
        results: list[EvaluationResult] = []
        for cand in candidates:
            history.candidate_counter += 1
            cand_id = f"cand_{history.candidate_counter:06d}"
            res = self.evaluator.evaluate(cand, cand_id)
            results.append(res)
            if self.hooks.on_candidate_evaluated:
                self.hooks.on_candidate_evaluated(res)

        # 3) Sort + select. With the discovery/deployment split:
        #    - "passed" = not hard-rejected (eligible for breeding, with non-zero
        #      discovery_fitness; may still be sub-deployment).
        #    - Breeding sort: by discovery_fitness descending.
        #    - Deployment sort: by deployment_fitness descending (only those
        #      with deployment_pass=True).
        passed = [r for r in results if not r.rejected]
        passed.sort(key=lambda r: r.discovery_fitness, reverse=True)
        elites = passed[:self.config.elite_count]

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
        record = GenerationRecord(
            generation_index=gen_idx,
            started_at=gen_started,
            ended_at=time.time(),
            n_candidates=len(results),
            n_rejected=len(results) - len(passed),
            n_passed=len(passed),
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

        return record

    # ------------------------------------------------------------------
    # Generation construction
    # ------------------------------------------------------------------

    def _generate_gen0(
        self,
        rng: random.Random,
        gen_idx: int,
    ) -> list[CandidateGenome]:
        """All-random gen 0."""
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

        If there are no elites (all-rejected prior gen), fall back to fully
        random — we still produce candidates_per_gen of them.
        """
        prev_gen = history.generations[-1]
        elite_genomes = self._load_elite_genomes(prev_gen, gen_idx, rng)
        target = self.config.candidates_per_gen

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
            )
            dca_d = best_dict["dca_genome"]
            tp_d = best_dict["tp_genome"]
            dca = DcaGenome(
                grid_method=dca_d.get("grid_method", "fixed_pct"),
                grid_params=dca_d.get("grid_params", {}),
                allocation_method=dca_d.get("allocation_method", "equal"),
                allocation_params=dca_d.get("allocation_params", {}),
                max_dca_layers=dca_d.get("max_dca_layers", 3),
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
