"""Evolution harness — bounded, resumable GA loop for Stage 10+.

This is the infrastructure, not the evolution itself. The actual DCA
evolution (Stage 10) uses these primitives; the TP evolution (Stage 12)
and joint evolution (Stage 14) will reuse the same harness with
different operator configurations.

Stop conditions (any of these halts the run):
- 8h wall-time cap (configurable; default 28800s)
- Generation cap reached (configurable; default 20)
- Stagnation: no fitness improvement for N generations (configurable; default 5)
- All candidates rejected for N generations (configurable; default 3)

Persistence (resumable):
- After every generation: write generation_history.json
- On wall-time cap / stagnation: write unfinished_status.json
- On clean completion: write final_status.json
- load_state() can resume from a previous run

Reporting (writes after each gen):
- leaderboard.json: top-20 candidates per generation
- best_genome.json: best of the generation
- rejection_report.json: count of each reject reason
"""
from __future__ import annotations

from .config import EvolutionConfig
from .evaluator import CandidateEvaluator, EvaluationResult
from .harness import EvolutionHarness, GenerationRecord, HarnessHooks, RunSummary
from .operators import crossover, mutate, random_candidate_genome, random_dca_genome
from .persistence import (
    GenerationHistory,
    UnfinishedStatus,
    checkpoint_age_seconds,
    list_checkpoints,
    load_latest_checkpoint,
    load_state,
    save_checkpoint,
    save_state,
)

__all__ = [
    "CandidateEvaluator",
    "EvaluationResult",
    "EvolutionConfig",
    "EvolutionHarness",
    "GenerationHistory",
    "GenerationRecord",
    "HarnessHooks",
    "RunSummary",
    "UnfinishedStatus",
    "checkpoint_age_seconds",
    "crossover",
    "list_checkpoints",
    "load_latest_checkpoint",
    "load_state",
    "mutate",
    "random_candidate_genome",
    "random_dca_genome",
    "save_checkpoint",
    "save_state",
]
