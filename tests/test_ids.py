"""Stage 1 acceptance tests — ID generators.

Verifies locked ID formats:
- experiment_id: YYYYMMDD_HHMMSS_gen{N}_{slug}
- candidate_id: {experiment_id}_cand{NNNN}
- genome_id: uuid4
- run_metadata_id: {experiment_id}_run{NNNN}_{NN}
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import pytest

from configs.ids import (
    is_valid_candidate_id,
    is_valid_experiment_id,
    is_valid_genome_id,
    make_candidate_id,
    make_experiment_id,
    make_genome_id,
    make_run_metadata_id,
)

pytestmark = pytest.mark.stage1


EXPERIMENT_RE = re.compile(r"^\d{8}_\d{6}_gen\d+_[a-z0-9_]+$")
CANDIDATE_RE = re.compile(r"^\d{8}_\d{6}_gen\d+_[a-z0-9_]+_cand\d{4}$")


def test_experiment_id_format_with_explicit_when():
    """experiment_id must match YYYYMMDD_HHMMSS_gen{N}_{slug}."""
    when = datetime(2026, 6, 20, 19, 45, 0, tzinfo=UTC)
    eid = make_experiment_id(generation=0, slug="smoke_v1", when=when)
    assert eid == "20260620_194500_gen0_smoke_v1"
    assert EXPERIMENT_RE.match(eid)
    assert is_valid_experiment_id(eid)


def test_experiment_id_format_with_auto_when():
    """experiment_id with auto-generated timestamp must also match."""
    eid = make_experiment_id(generation=5, slug="dca_evolution")
    assert EXPERIMENT_RE.match(eid)
    assert "gen5" in eid
    assert "dca_evolution" in eid


def test_experiment_id_slug_normalised():
    """Slug must be lowercased and stripped of non-alphanumerics."""
    eid = make_experiment_id(generation=0, slug="My Test Run 1!", when=datetime(2026, 6, 20, tzinfo=UTC))
    assert "my_test_run_1" in eid


def test_experiment_id_rejects_negative_generation():
    with pytest.raises(ValueError, match="generation must be >= 0"):
        make_experiment_id(generation=-1, slug="test")


def test_experiment_id_rejects_empty_slug():
    """Empty slug must either raise OR fall back to 'unnamed' (documented behaviour)."""
    # Current behaviour: falls back to 'unnamed' so the ID is still valid
    eid = make_experiment_id(generation=0, slug="")
    assert EXPERIMENT_RE.match(eid)
    assert "unnamed" in eid


def test_candidate_id_format():
    eid = "20260620_194500_gen0_smoke_v1"
    cid = make_candidate_id(experiment_id=eid, candidate_index=42)
    assert cid == "20260620_194500_gen0_smoke_v1_cand0042"
    assert CANDIDATE_RE.match(cid)
    assert is_valid_candidate_id(cid)


def test_candidate_id_zero_pads_to_4_digits():
    eid = "20260620_194500_gen0_smoke_v1"
    assert make_candidate_id(eid, 0).endswith("_cand0000")
    assert make_candidate_id(eid, 1).endswith("_cand0001")
    assert make_candidate_id(eid, 499).endswith("_cand0499")


def test_candidate_id_rejects_bad_experiment_id():
    with pytest.raises(ValueError, match="experiment_id does not match"):
        make_candidate_id("not-a-valid-experiment-id", 0)


def test_candidate_id_rejects_negative_index():
    eid = "20260620_194500_gen0_smoke_v1"
    with pytest.raises(ValueError, match="candidate_index must be >= 0"):
        make_candidate_id(eid, -1)


def test_genome_id_is_uuid4():
    gid = make_genome_id()
    # Parse as UUID
    parsed = uuid.UUID(gid)
    assert parsed.version == 4
    assert is_valid_genome_id(gid)


def test_genome_id_is_unique():
    ids = {make_genome_id() for _ in range(1000)}
    assert len(ids) == 1000  # all unique


def test_run_metadata_id_format():
    eid = "20260620_194500_gen0_smoke_v1"
    rid = make_run_metadata_id(eid, run_index=1, attempt_index=1)
    assert rid == "20260620_194500_gen0_smoke_v1_run0001_01"


def test_run_metadata_id_zero_pads():
    eid = "20260620_194500_gen0_smoke_v1"
    assert make_run_metadata_id(eid, 42, 3) == f"{eid}_run0042_03"


def test_run_metadata_id_rejects_zero_or_negative():
    eid = "20260620_194500_gen0_smoke_v1"
    with pytest.raises(ValueError, match="run_index must be >= 1"):
        make_run_metadata_id(eid, 0, 1)
    with pytest.raises(ValueError, match="attempt_index must be >= 1"):
        make_run_metadata_id(eid, 1, 0)


def test_validators_reject_bad_inputs():
    assert not is_valid_experiment_id("not_valid")
    assert not is_valid_candidate_id("not_valid_cand")
    assert not is_valid_genome_id("not-a-uuid")
