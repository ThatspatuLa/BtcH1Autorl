"""Run metadata — JSON schema for per-run provenance.

Every candidate backtest produces a run_metadata.json containing:
- experiment_id, candidate_id, genome_id (from configs.ids)
- settings snapshot (full Settings object serialised)
- git_commit, python_version, started_at, finished_at, exit_reason
- input config paths
- safety_pass (per-candidate safety result, set later in Stage 7)

This is the canonical provenance record for the system.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.ids import make_run_metadata_id


@dataclass
class RunMetadata:
    experiment_id: str
    run_id: str
    candidate_id: str | None
    genome_id: str | None
    started_at: str  # ISO 8601 UTC
    finished_at: str | None  # ISO 8601 UTC, None until complete
    exit_reason: str | None  # success, hard_reject, safety_fail, error, timeout
    git_commit: str
    git_dirty: bool
    python_version: str
    freqtrade_version: str
    settings_snapshot_path: str
    settings_snapshot_inline: dict[str, Any]
    input_files: dict[str, str]  # source paths
    safety_pass: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _git_commit(repo_root: Path) -> tuple[str, bool]:
    """Return (commit_sha, dirty_bool). Empty sha + False if not a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty_str = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return commit, bool(dirty_str)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", False


def _freqtrade_version() -> str:
    try:
        import freqtrade
        return getattr(freqtrade, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def make_run_metadata(
    experiment_id: str,
    run_index: int,
    attempt_index: int,
    settings,
    candidate_id: str | None = None,
    genome_id: str | None = None,
    repo_root: Path | None = None,
) -> RunMetadata:
    """Create a fresh RunMetadata. Call finished() to stamp finished_at + exit_reason."""
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    commit, dirty = _git_commit(repo_root)
    settings_dict = settings.data.to_dict() if hasattr(settings, "data") else dict(settings)
    return RunMetadata(
        experiment_id=experiment_id,
        run_id=make_run_metadata_id(experiment_id, run_index, attempt_index),
        candidate_id=candidate_id,
        genome_id=genome_id,
        started_at=_now_utc(),
        finished_at=None,
        exit_reason=None,
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        freqtrade_version=_freqtrade_version(),
        settings_snapshot_path=str((repo_root / "results" / experiment_id / "settings_snapshot.json").resolve()),
        settings_snapshot_inline=settings_dict,
        input_files={
            "freqtrade_config": str(settings.sources["freqtrade"]),
            "experiment_config": str(settings.sources["experiment"]),
            "research_config": str(settings.sources["research"]),
        },
    )


def write_run_metadata(meta: RunMetadata, path: Path | None = None) -> Path:
    """Write metadata JSON. Defaults to settings_snapshot_path + run_meta.json."""
    if path is None:
        path = Path(meta.settings_snapshot_path).parent / f"{meta.run_id}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(meta.to_json())
    return path


def load_run_metadata(path: Path | str) -> RunMetadata:
    with open(path) as f:
        data = json.load(f)
    return RunMetadata(**data)
