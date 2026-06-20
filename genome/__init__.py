"""BTC H1 AutoRL — genome subpackage.

Stage 4 deliverable: CandidateGenome dataclass + serialisation + validation + Freqtrade export.

Genome sections:
- dca_genome: grid_method, allocation_method, combo_method, trigger_mode, params
- tp_genome: exit_method + params
- safety_genome: max_dca_layers, overlap_allowed, buffer_pct
- settings_overrides: optional per-candidate broker/cost overrides
- mutation_ops: lineage metadata
"""
from genome.schema import (
    # Helpers
    DEFAULT_DCA_GENOME,
    DEFAULT_SAFETY_GENOME,
    DEFAULT_TP_GENOME,
    AllocationMethod,
    # Top-level
    CandidateGenome,
    ComboMethod,
    ConfirmationIndicator,
    # Sections
    DcaGenome,
    GenomeValidationError,
    # Enums
    GridMethod,
    LineageMetadata,
    MarginMode,
    SafetyGenome,
    SettingsOverrides,
    TpExitMethod,
    TpGenome,
    TriggerMode,
    genome_from_json,
    genome_from_msgpack,
    # Hash + Freqtrade export
    genome_hash,
    # Serialisation
    genome_to_json,
    genome_to_msgpack,
    to_freqtrade_strategy_params,
    # Validation
    validate_genome,
)

__all__ = [
    "DEFAULT_DCA_GENOME",
    "DEFAULT_SAFETY_GENOME",
    "DEFAULT_TP_GENOME",
    "AllocationMethod",
    "CandidateGenome",
    "ComboMethod",
    "ConfirmationIndicator",
    "DcaGenome",
    "GenomeValidationError",
    "GridMethod",
    "LineageMetadata",
    "MarginMode",
    "SafetyGenome",
    "SettingsOverrides",
    "TpExitMethod",
    "TpGenome",
    "TriggerMode",
    "genome_from_json",
    "genome_from_msgpack",
    "genome_hash",
    "genome_to_json",
    "genome_to_msgpack",
    "to_freqtrade_strategy_params",
    "validate_genome",
]
