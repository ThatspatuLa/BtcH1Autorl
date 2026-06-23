"""Persistence — resumable generation history + status files.

Files written per evolution run (in output_dir):
- generation_history.json    — full state for resume
- leaderboard.json           — top-N per generation (for reporting)
- best_genome.json           — best of each generation
- rejection_report.json      — count of each reject reason
- unfinished_status.json     — written if run halts before completion
- final_status.json          — written on clean completion
- run_summary.json           — written on completion OR halt

Checkpoints (per-cycle, under checkpoints/ in project root):
- checkpoint_<cycle>_<HHMM>.json  — snapshot every N minutes (default 20)
- latest.json                     — symlink/copy of the most recent checkpoint
"""
from __future__ import annotations

import json
import time
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
    n_passed: int                          # not hard-rejected (eligible for breeding)
    n_deployment_passing: int              # subset of passed that also cleared deployment gates
    best_fitness: float                    # best discovery_fitness this gen
    median_fitness: float
    best_candidate_id: str
    best_genome_id: str
    wall_time_seconds_used: float
    rejection_reasons: dict[str, int]       # reason → count
    # Fix B (2026-06-22): elite-eligible count (passed + meets quality gate)
    n_elite_eligible: int = 0
    # IDs of all evaluated candidates (so resume can skip them)
    evaluated_candidate_ids: list[str] = field(default_factory=list)
    # Top-N leaderboard by discovery_fitness (the "almost passing" diagnostic)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    # Top-N by deployment_fitness (only deployment_pass=True candidates)
    deployment_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    # Per-island best fitness (Fix A, 2026-06-22): {island_id: best_fitness_this_gen}
    per_island_best_fitness: dict[int, float] = field(default_factory=dict)
    per_island_best_count: dict[int, int] = field(default_factory=dict)
    per_island_elite_count: dict[int, int] = field(default_factory=dict)
    # Retirement (2026-06-22): list of retired-island manifest summaries this gen
    # (filled in by harness when retirement fires). Stored as lightweight dicts
    # to avoid coupling persistence to the retirement module's classes.
    retired_islands: list[dict[str, Any]] = field(default_factory=list)
    # Map of re-seeded island slots this gen: {island_id: fresh_bias_name}
    # Used by the harness to override island biases when building the next gen.
    island_bias_overrides: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRecord:
        # Forward-compat: missing new fields default to zero / empty
        return cls(
            generation_index=d["generation_index"],
            started_at=d["started_at"],
            ended_at=d.get("ended_at"),
            n_candidates=d["n_candidates"],
            n_rejected=d["n_rejected"],
            n_passed=d["n_passed"],
            n_elite_eligible=d.get("n_elite_eligible", 0),
            n_deployment_passing=d.get("n_deployment_passing", 0),
            best_fitness=d["best_fitness"],
            median_fitness=d["median_fitness"],
            best_candidate_id=d["best_candidate_id"],
            best_genome_id=d["best_genome_id"],
            wall_time_seconds_used=d["wall_time_seconds_used"],
            rejection_reasons=d["rejection_reasons"],
            evaluated_candidate_ids=d.get("evaluated_candidate_ids", []),
            leaderboard=d.get("leaderboard", []),
            deployment_leaderboard=d.get("deployment_leaderboard", []),
            per_island_best_fitness={int(k): float(v) for k, v in d.get("per_island_best_fitness", {}).items()},
            per_island_best_count={int(k): int(v) for k, v in d.get("per_island_best_count", {}).items()},
            per_island_elite_count={int(k): int(v) for k, v in d.get("per_island_elite_count", {}).items()},
            retired_islands=list(d.get("retired_islands", [])),
            island_bias_overrides={int(k): str(v) for k, v in d.get("island_bias_overrides", {}).items()},
        )


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


# ============================================================
# Checkpoints — every-N-minute snapshots for restart safety
# ============================================================
# Designed so that if the user restarts the computer mid-cycle, the next
# `minato_fast_tick.sh` invocation can detect the latest checkpoint and
# resume from the last completed generation instead of starting over.
#
# Checkpoints live under <project_root>/checkpoints/ (NOT in output_dir) so
# they survive a fresh cycle's output directory. The cycle_id is part of the
# filename so multiple cycles don't collide.

CHECKPOINT_ROOT = Path("checkpoints")


def _checkpoint_dir() -> Path:
    """Return (and create) the project-root checkpoint directory."""
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_ROOT


def save_checkpoint(
    cycle_id: str,
    gen_idx: int,
    wall_time_used: float,
    per_island_best_fitness: dict[int, float],
    per_island_stagnation_counter: dict[int, int],
    retired_so_far: list[dict[str, Any]],
    rng_state: tuple[Any, ...] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a snapshot of the current evolution state.

    Files written:
        checkpoints/checkpoint_<cycle_id>_<HHMM>.json
        checkpoints/latest.json   (copy, points at the most recent)

    The HHMM comes from the current wall-clock minute so successive saves
    within the same minute overwrite each other (cheap dedup). Saves that
    fall on the 20-min boundary produce a new file.

    Returns the path of the per-minute checkpoint file.
    """
    ck_dir = _checkpoint_dir()
    ts_minute = time.strftime("%H%M", time.localtime())
    filename = f"checkpoint_{cycle_id}_{ts_minute}.json"
    out_path = ck_dir / filename

    payload = {
        "cycle_id": cycle_id,
        "saved_at": time.time(),
        "generation_index": gen_idx,
        "wall_time_used": wall_time_used,
        "per_island_best_fitness": dict(per_island_best_fitness),
        "per_island_stagnation_counter": dict(per_island_stagnation_counter),
        "retired_so_far_count": len(retired_so_far),
        "retired_so_far": retired_so_far,
        "rng_state": list(rng_state) if rng_state is not None else None,
        "extra": extra or {},
    }
    _atomic_write(out_path, payload)

    # Also write `latest.json` (always the most recent snapshot).
    latest_path = ck_dir / "latest.json"
    _atomic_write(latest_path, payload)

    return out_path


def load_latest_checkpoint(cycle_id: str | None = None) -> dict[str, Any] | None:
    """Load the most recent checkpoint.

    If cycle_id is given, return the most recent checkpoint matching that
    cycle. Otherwise return the global `latest.json` snapshot (regardless
    of cycle_id, so the fast-tick can decide whether to resume or start
    fresh based on age).

    Returns None if no checkpoint exists.
    """
    ck_dir = _checkpoint_dir()
    latest_path = ck_dir / "latest.json"
    if not latest_path.exists():
        return None
    try:
        with open(latest_path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if cycle_id is not None and payload.get("cycle_id") != cycle_id:
        # latest.json is from a different cycle — caller should treat as no
        # resume candidate (or scan by cycle_id).
        return None
    return payload


def list_checkpoints(cycle_id: str | None = None) -> list[dict[str, Any]]:
    """List all checkpoints (newest first), optionally filtered by cycle_id.

    Returns lightweight dicts (metadata only) — full payload is in `latest.json`.
    """
    ck_dir = _checkpoint_dir()
    out: list[dict[str, Any]] = []
    for path in sorted(ck_dir.glob("checkpoint_*.json"), reverse=True):
        try:
            with open(path) as f:
                d = json.load(f)
            if cycle_id is None or d.get("cycle_id") == cycle_id:
                out.append({
                    "path": str(path),
                    "cycle_id": d.get("cycle_id"),
                    "generation_index": d.get("generation_index"),
                    "saved_at": d.get("saved_at"),
                })
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return out


def checkpoint_age_seconds(payload: dict[str, Any]) -> float:
    """How old is this checkpoint? Used by fast-tick to decide resume vs fresh."""
    saved_at = payload.get("saved_at", 0.0)
    return max(0.0, time.time() - float(saved_at))
