"""BTC H1 AutoRL — configs subpackage.

Stage 1 deliverable: immutable Settings loader that merges
Freqtrade-compatible config + experiment overrides + research-engine config.
"""
from configs.loader import Settings, FrozenSettingsDict
from configs.ids import (
    make_experiment_id,
    make_candidate_id,
    make_genome_id,
    make_run_metadata_id,
)
from configs.metadata import (
    RunMetadata,
    make_run_metadata,
    write_run_metadata,
    load_run_metadata,
)

__all__ = [
    "Settings",
    "FrozenSettingsDict",
    "make_experiment_id",
    "make_candidate_id",
    "make_genome_id",
    "make_run_metadata_id",
    "RunMetadata",
    "make_run_metadata",
    "write_run_metadata",
    "load_run_metadata",
]
