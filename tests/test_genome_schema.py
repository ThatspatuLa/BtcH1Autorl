"""Stage 4 acceptance tests — genome schema, serialisation, validation, Freqtrade export.

Verifies:
- CandidateGenome dataclass + all enums work
- JSON + msgpack round-trip is deterministic
- genome_hash is stable across runs
- Validation catches invalid genomes
- All 5 example genomes load without errors
- Freqtrade export generates valid dict structure
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from genome import (
    DEFAULT_DCA_GENOME,
    DEFAULT_SAFETY_GENOME,
    DEFAULT_TP_GENOME,
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    DcaGenome,
    GenomeValidationError,
    GridMethod,
    LineageMetadata,
    MarginMode,
    SettingsOverrides,
    TpExitMethod,
    TpGenome,
    TriggerMode,
    genome_from_json,
    genome_from_msgpack,
    genome_hash,
    genome_to_json,
    genome_to_msgpack,
    to_freqtrade_strategy_params,
    validate_genome,
)

pytestmark = pytest.mark.stage4


def _make_simple_genome(genome_id: str = "test-genome-001") -> CandidateGenome:
    return CandidateGenome(
        genome_id=genome_id,
        dca_genome=DcaGenome(
            grid_method=GridMethod.FIXED_PCT,
            grid_params={"grid_pct": 1.5, "max_layers": 5},
            allocation_method=AllocationMethod.EQUAL,
            allocation_params={"base_notional": 100.0, "allocation_cap_pct": 0.10},
        ),
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"tp_pct": 2.0},
        ),
    )


def test_default_genomes_load():
    assert DEFAULT_DCA_GENOME.grid_method == GridMethod.FIXED_PCT
    assert DEFAULT_TP_GENOME.exit_method == TpExitMethod.FIXED
    assert DEFAULT_SAFETY_GENOME.require_buffer_pct == 0.20


def test_simple_genome_validates():
    g = _make_simple_genome()
    validate_genome(g)  # must not raise


def test_genome_to_json_roundtrip():
    g = _make_simple_genome()
    j = genome_to_json(g)
    parsed = json.loads(j)
    assert parsed["genome_id"] == g.genome_id
    assert parsed["dca_genome"]["grid_method"] == "fixed_pct"
    assert parsed["tp_genome"]["exit_method"] == "fixed"

    g2 = genome_from_json(j)
    validate_genome(g2)
    assert g2.genome_id == g.genome_id
    assert g2.dca_genome.grid_method == g.dca_genome.grid_method
    assert g2.tp_genome.exit_params == g.tp_genome.exit_params


def test_genome_to_msgpack_roundtrip():
    g = _make_simple_genome()
    blob = genome_to_msgpack(g)
    assert isinstance(blob, bytes)
    g2 = genome_from_msgpack(blob)
    assert g2.genome_id == g.genome_id
    assert g2.dca_genome.grid_params == g.dca_genome.grid_params


def test_genome_hash_is_deterministic():
    g1 = _make_simple_genome("hash-test")
    g2 = _make_simple_genome("hash-test")  # same params, different lineage irrelevant
    h1 = genome_hash(g1)
    h2 = genome_hash(g2)
    # Same params must produce same hash (hash is over params, not genome_id)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
    assert re.match(r"^[0-9a-f]{64}$", h1)


def test_genome_hash_changes_with_params():
    g1 = _make_simple_genome("a")
    g2 = _make_simple_genome("b")
    # Different genome_id (which is excluded from hash) — should still hash same
    g3 = _make_simple_genome("c")
    g3.dca_genome.grid_params["grid_pct"] = 2.5  # different param
    assert genome_hash(g1) == genome_hash(g2)
    assert genome_hash(g1) != genome_hash(g3)


def test_validation_rejects_invalid_max_layers():
    g = _make_simple_genome()
    g.dca_genome.max_dca_layers = 0
    with pytest.raises(GenomeValidationError, match="max_dca_layers must be >= 1"):
        validate_genome(g)


def test_validation_rejects_excessive_max_layers():
    g = _make_simple_genome()
    g.dca_genome.max_dca_layers = 100
    with pytest.raises(GenomeValidationError, match="max_dca_layers must be <= 50"):
        validate_genome(g)


def test_validation_rejects_confirmation_without_indicators():
    g = _make_simple_genome()
    g.dca_genome.trigger_mode = TriggerMode.PRICE_WITH_CONFIRMATION
    g.dca_genome.confirmation_indicators = []
    with pytest.raises(GenomeValidationError, match="at least one confirmation_indicator"):
        validate_genome(g)


def test_validation_accepts_confirmation_with_indicators():
    g = _make_simple_genome()
    g.dca_genome.trigger_mode = TriggerMode.PRICE_WITH_CONFIRMATION
    g.dca_genome.confirmation_indicators = [ConfirmationIndicator.RSI_BELOW]
    validate_genome(g)  # must not raise


def test_validation_rejects_bad_buffer_pct():
    """AMENDMENT 2: buffer_pct must be in [0, 1), NOT hardcoded."""
    g = _make_simple_genome()
    g.safety_genome.require_buffer_pct = 1.5
    with pytest.raises(GenomeValidationError, match="require_buffer_pct must be in"):
        validate_genome(g)

    # Edge cases
    g.safety_genome.require_buffer_pct = 0.0
    validate_genome(g)  # 0 is valid (no buffer)
    g.safety_genome.require_buffer_pct = 0.99
    validate_genome(g)  # 0.99 is valid
    g.safety_genome.require_buffer_pct = 0.05
    validate_genome(g)  # any value in [0, 1) is valid


def test_validation_rejects_nan_params():
    import math
    g = _make_simple_genome()
    g.dca_genome.grid_params["grid_pct"] = math.nan
    with pytest.raises(GenomeValidationError, match="finite number"):
        validate_genome(g)

    g = _make_simple_genome()
    g.dca_genome.grid_params["grid_pct"] = math.inf
    with pytest.raises(GenomeValidationError, match="finite number"):
        validate_genome(g)


def test_settings_overrides_round_trip():
    g = _make_simple_genome()
    g.settings_overrides = SettingsOverrides(
        leverage=3.0,
        margin_mode=MarginMode.CROSS,
        fee_pct=0.001,
        buffer_pct=0.15,
    )
    j = genome_to_json(g)
    g2 = genome_from_json(j)
    assert g2.settings_overrides.leverage == 3.0
    assert g2.settings_overrides.margin_mode == MarginMode.CROSS
    assert g2.settings_overrides.fee_pct == 0.001
    assert g2.settings_overrides.buffer_pct == 0.15


def test_settings_overrides_strips_nones():
    g = _make_simple_genome()
    g.settings_overrides = SettingsOverrides(leverage=3.0)
    d = g.settings_overrides.to_dict()
    assert d == {"leverage": 3.0}  # no None keys


def test_freqtrade_export_structure():
    g = _make_simple_genome("freqtrade-export-test")
    ft = to_freqtrade_strategy_params(g)
    assert "strategy_name" in ft
    assert "dca" in ft
    assert "tp" in ft
    assert "safety" in ft
    assert "settings_overrides" in ft
    assert ft["dca"]["grid_method"] == "fixed_pct"
    assert ft["tp"]["exit_method"] == "fixed"
    assert ft["dca"]["max_dca_layers"] == 5


def test_lineage_round_trip():
    g = _make_simple_genome()
    g.lineage = LineageMetadata(
        parent_a_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        parent_b_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        generation_index=5,
        mutation_seed=42,
        mutation_ops=[{"op": "param_perturb", "gene": "grid_pct", "from": 1.5, "to": 1.7}],
    )
    j = genome_to_json(g)
    g2 = genome_from_json(j)
    assert g2.lineage.parent_a_id == g.lineage.parent_a_id
    assert g2.lineage.generation_index == 5
    assert g2.lineage.mutation_ops[0]["from"] == 1.5


# ============================================================
# Example genomes — must all load without errors
# ============================================================

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "genomes" / "examples"


@pytest.mark.parametrize("filename", [
    "simple_fixed.json",
    "atr_grid.json",
    "hybrid_trigger.json",
    "partial_ladder_tp.json",
    "trailing_tp.json",
])
def test_example_genome_loads(filename):
    """All 5 example genomes must validate and round-trip cleanly."""
    path = EXAMPLES_DIR / filename
    assert path.exists(), f"Example genome missing: {path}"
    g = genome_from_json(path.read_text())
    validate_genome(g)
    # Re-serialise and re-load — must be lossless
    j = genome_to_json(g)
    g2 = genome_from_json(j)
    validate_genome(g2)
    assert g2.dca_genome.grid_method == g.dca_genome.grid_method
    assert g2.tp_genome.exit_method == g.tp_genome.exit_method


def test_all_examples_are_distinct():
    """All 5 examples must have different genome_ids (so they hash differently)."""
    ids = set()
    for path in EXAMPLES_DIR.glob("*.json"):
        g = genome_from_json(path.read_text())
        assert g.genome_id not in ids, f"Duplicate genome_id: {g.genome_id}"
        ids.add(g.genome_id)
    assert len(ids) == 5
