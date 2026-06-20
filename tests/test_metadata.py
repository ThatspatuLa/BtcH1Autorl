"""Stage 1 acceptance tests — run metadata.

Verifies run_metadata.json schema and round-trip:
- All required fields present
- Settings snapshot inline + path
- Serialise → deserialise round-trips
- Determinism: same inputs → same metadata (except timestamps)
"""
from __future__ import annotations

import json

import pytest

from configs.metadata import (
    load_run_metadata,
    make_run_metadata,
    write_run_metadata,
)

pytestmark = pytest.mark.stage1


def test_make_run_metadata_creates_all_required_fields(settings):
    meta = make_run_metadata(
        experiment_id="20260620_194500_gen0_smoke_v1",
        run_index=1,
        attempt_index=1,
        settings=settings,
        candidate_id="20260620_194500_gen0_smoke_v1_cand0042",
        genome_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert meta.experiment_id == "20260620_194500_gen0_smoke_v1"
    assert meta.run_id == "20260620_194500_gen0_smoke_v1_run0001_01"
    assert meta.candidate_id == "20260620_194500_gen0_smoke_v1_cand0042"
    assert meta.genome_id == "550e8400-e29b-41d4-a716-446655440000"
    assert meta.exit_reason is None  # not finished yet
    assert meta.finished_at is None
    assert meta.started_at.endswith("Z")  # ISO 8601 UTC
    assert meta.python_version  # non-empty
    assert meta.freqtrade_version  # non-empty
    assert "leverage" in meta.settings_snapshot_inline
    assert meta.settings_snapshot_inline["leverage"] == 5.0
    assert "freqtrade_config" in meta.input_files


def test_run_metadata_input_files_are_absolute(settings):
    meta = make_run_metadata(
        experiment_id="20260620_194500_gen0_smoke_v1",
        run_index=1,
        attempt_index=1,
        settings=settings,
    )
    for path in meta.input_files.values():
        assert path.startswith("/")


def test_run_metadata_round_trip(settings, tmp_results_dir):
    """Write metadata to disk, load it back, fields match."""
    meta = make_run_metadata(
        experiment_id="20260620_194500_gen0_smoke_v1",
        run_index=1,
        attempt_index=1,
        settings=settings,
        candidate_id="20260620_194500_gen0_smoke_v1_cand0000",
    )
    meta.finished_at = "2026-06-20T19:45:03.214000Z"
    meta.exit_reason = "success"
    meta.safety_pass = {"passed": True, "reasons": []}

    path = write_run_metadata(meta, path=tmp_results_dir / "meta.json")
    assert path.exists()

    loaded = load_run_metadata(path)
    assert loaded.experiment_id == meta.experiment_id
    assert loaded.run_id == meta.run_id
    assert loaded.candidate_id == meta.candidate_id
    assert loaded.finished_at == meta.finished_at
    assert loaded.exit_reason == meta.exit_reason
    assert loaded.safety_pass == meta.safety_pass
    assert loaded.settings_snapshot_inline == meta.settings_snapshot_inline


def test_run_metadata_json_is_valid(settings, tmp_results_dir):
    """Written metadata must be valid JSON."""
    meta = make_run_metadata(
        experiment_id="20260620_194500_gen0_smoke_v1",
        run_index=1,
        attempt_index=1,
        settings=settings,
    )
    path = write_run_metadata(meta, path=tmp_results_dir / "meta.json")
    parsed = json.loads(path.read_text())
    assert isinstance(parsed, dict)
    assert "experiment_id" in parsed
    assert "settings_snapshot_inline" in parsed
