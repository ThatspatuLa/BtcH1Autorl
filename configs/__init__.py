"""BTC H1 AutoRL — configs subpackage.

Stage 1 deliverable: immutable Settings loader that merges
Freqtrade-compatible config + experiment overrides + research-engine config.
"""
from configs.ids import (
    make_candidate_id,
    make_experiment_id,
    make_genome_id,
    make_run_metadata_id,
)
from configs.loader import FrozenSettingsDict, Settings
from configs.metadata import (
    RunMetadata,
    load_run_metadata,
    make_run_metadata,
    write_run_metadata,
)

__all__ = [
    "FrozenSettingsDict",
    "RunMetadata",
    "Settings",
    "load_run_metadata",
    "make_candidate_id",
    "make_experiment_id",
    "make_genome_id",
    "make_run_metadata",
    "make_run_metadata_id",
    "write_run_metadata",
]
