"""ID generators for experiments, candidates, genomes, and run metadata.

Locked formats (per Kanban Stage 1):
- experiment_id: YYYYMMDD_HHMMSS_gen{N}_{slug}
- candidate_id: {experiment_id}_cand{NNNN}  (NNNN zero-padded to 4 digits)
- genome_id: uuid4
- run_metadata_id: {experiment_id}_run{NNNN}_{NN}
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_SLUG_MAX_LEN = 40

_EXPERIMENT_RE = re.compile(r"^\d{8}_\d{6}_gen\d+_[a-z0-9_]+$")
_CANDIDATE_RE = re.compile(r"^\d{8}_\d{6}_gen\d+_[a-z0-9_]+_cand\d{4}$")
_GENOME_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _slugify(s: str) -> str:
    s = s.lower().strip().replace("-", "_").replace(" ", "_")
    s = _SLUG_RE.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:_SLUG_MAX_LEN] or "unnamed"


def make_experiment_id(
    generation: int,
    slug: str,
    when: datetime | None = None,
) -> str:
    """Format: YYYYMMDD_HHMMSS_gen{N}_{slug}

    generation is 0 for gen-0 (initial random), 1+ for evolved.
    slug must be lowercase alphanumeric + underscore.
    """
    if generation < 0:
        raise ValueError(f"generation must be >= 0, got {generation}")
    slug = _slugify(slug)
    if not slug:
        raise ValueError("slug must contain at least one alphanumeric character")
    if when is None:
        when = datetime.now(timezone.utc)
    return f"{when.strftime('%Y%m%d_%H%M%S')}_gen{generation}_{slug}"


def make_candidate_id(experiment_id: str, candidate_index: int) -> str:
    """Format: {experiment_id}_cand{NNNN} where NNNN is 4-digit zero-padded."""
    if not _EXPERIMENT_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id does not match required pattern: {experiment_id!r}. "
            f"Use make_experiment_id() to generate it."
        )
    if candidate_index < 0:
        raise ValueError(f"candidate_index must be >= 0, got {candidate_index}")
    return f"{experiment_id}_cand{candidate_index:04d}"


def make_genome_id() -> str:
    """Format: uuid4 lowercase string (with dashes)."""
    return str(uuid.uuid4())


def make_run_metadata_id(experiment_id: str, run_index: int, attempt_index: int = 1) -> str:
    """Format: {experiment_id}_run{NNNN}_{NN} where NNNN and NN are zero-padded.

    run_index = which run of this experiment (1-based)
    attempt_index = retry attempt within the same run (default 1)
    """
    if not _EXPERIMENT_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id does not match required pattern: {experiment_id!r}"
        )
    if run_index < 1:
        raise ValueError(f"run_index must be >= 1, got {run_index}")
    if attempt_index < 1:
        raise ValueError(f"attempt_index must be >= 1, got {attempt_index}")
    return f"{experiment_id}_run{run_index:04d}_{attempt_index:02d}"


def is_valid_experiment_id(s: str) -> bool:
    return bool(_EXPERIMENT_RE.match(s))


def is_valid_candidate_id(s: str) -> bool:
    return bool(_CANDIDATE_RE.match(s))


def is_valid_genome_id(s: str) -> bool:
    return bool(_GENOME_RE.match(s))
