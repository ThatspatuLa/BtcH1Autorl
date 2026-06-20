"""Persistence — resumable generation history + status files.

Files written per evolution run (in output_dir):
- generation_history.json    — full state for resume
- leaderboard.json           — top-N per generation (for reporting)
- best_genome.json           — best of each generation
- rejection_report.json      — count of each reject reason
- unfinished_status.json     — written if run halts before completion
- final_status.json          — written on clean completion
- run_summary.json           — written on completion OR halt
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRecord:
    """Per-generation record — what gets persisted after each gen."""
    generation_index: int
    started_at: float                       # unix timestamp
    ended_at: float | None
    n_candidates: int
    n_rejected: int
    n_passed: int
    best_fitness: float
    median_fitness: float
    best_candidate_id: str
    best_genome_id: str
    wall_time_seconds_used: float
    rejection_reasons: dict[str, int]       # reason → count
    # IDs of all evaluated candidates (so resume can skip them)
    evaluated_candidate_ids: list[str] = field(default_factory=list)
    # Top-N leaderboard for this generation
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRecord:
        return cls(**d)


@dataclass
class UnfinishedStatus:
    """Written when the evolution halts before completing all generations."""
    reason: str                             # "wall_time", "stagnation", "all_rejected", "interrupted"
    generations_completed: int
    max_generations: int
    wall_time_seconds_used: float
    wall_time_seconds_cap: int
    best_fitness_ever: float
    best_genome_id_ever: str
    best_candidate_id_ever: str
    finished_at: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Final summary — written on completion OR halt."""
    experiment_id: str
    started_at: float
    finished_at: float
    total_runtime_seconds: float
    generations_completed: int
    generations_planned: int
    total_candidates_evaluated: int
    best_fitness_ever: float
    best_genome_id_ever: str
    best_candidate_id_ever: str
    termination_reason: str         # "completed", "wall_time", "stagnation", "all_rejected", "interrupted"
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationHistory:
    """The full resumable state of an evolution run."""
    experiment_id: str
    config: dict[str, Any]
    started_at: float
    generations: list[GenerationRecord] = field(default_factory=list)
    best_fitness_ever: float = 0.0
    best_genome_id_ever: str = ""
    best_candidate_id_ever: str = ""
    candidate_counter: int = 0
    genome_counter: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "started_at": self.started_at,
            "generations": [g.to_dict() for g in self.generations],
            "best_fitness_ever": self.best_fitness_ever,
            "best_genome_id_ever": self.best_genome_id_ever,
            "best_candidate_id_ever": self.best_candidate_id_ever,
            "candidate_counter": self.candidate_counter,
            "genome_counter": self.genome_counter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationHistory:
        return cls(
            experiment_id=d["experiment_id"],
            config=d["config"],
            started_at=d["started_at"],
            generations=[GenerationRecord.from_dict(g) for g in d.get("generations", [])],
            best_fitness_ever=d.get("best_fitness_ever", 0.0),
            best_genome_id_ever=d.get("best_genome_id_ever", ""),
            best_candidate_id_ever=d.get("best_candidate_id_ever", ""),
            candidate_counter=d.get("candidate_counter", 0),
            genome_counter=d.get("genome_counter", 0),
        )


# ============================================================
# File I/O
# ============================================================

def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically — write to .tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def save_state(history: GenerationHistory, output_dir: str | Path) -> None:
    """Write the generation history to disk (atomic)."""
    out = Path(output_dir)
    _atomic_write(out / "generation_history.json", history.to_dict())


def load_state(output_dir: str | Path) -> GenerationHistory | None:
    """Load generation history from disk, or None if not present."""
    path = Path(output_dir) / "generation_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return GenerationHistory.from_dict(json.load(f))


def save_unfinished_status(
    status: UnfinishedStatus,
    output_dir: str | Path,
) -> None:
    _atomic_write(Path(output_dir) / "unfinished_status.json", status.to_dict())


def save_run_summary(summary: RunSummary, output_dir: str | Path) -> None:
    _atomic_write(Path(output_dir) / "run_summary.json", summary.to_dict())


def save_leaderboard(
    generation_index: int,
    leaderboard: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Append a generation's leaderboard. Per-generation files for clarity."""
    out = Path(output_dir) / "leaderboards"
    _atomic_write(out / f"gen_{generation_index:04d}.json", {
        "generation_index": generation_index,
        "leaderboard": leaderboard,
    })


def save_best_genome(
    generation_index: int,
    best_genome_dict: dict[str, Any],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir) / "best_genomes"
    _atomic_write(out / f"gen_{generation_index:04d}.json", best_genome_dict)


def save_rejection_report(
    generation_index: int,
    rejection_reasons: dict[str, int],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir) / "rejection_reports"
    _atomic_write(out / f"gen_{generation_index:04d}.json", rejection_reasons)
