"""Island retirement system.

Effective 2026-06-22 per Six's plan B extension: when any island's per-island
top fitness crosses `retirement_threshold` (default 0.80), the island's full
current state (top elites, lineage, history) is **archived** to the
retirement pool and the slot is **re-seeded** with a fresh random bias.

This expands the retired-islands archive over time, giving later stages
(stage 12 TP, stage 14 joint evolution) many known-good genomes to work with.

Each retired island is preserved in full as a JSON manifest at:
    {archive_dir}/retired_{cycle}_{island_id}_{gen_idx}/manifest.json

with sidecar files:
    - top_3_elites.json
    - generation_history.json
    - per_island_history.json
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    ConfirmationIndicator,
    GridMethod,
)


# ============================================================
# Bias pool — 16 distinct family biases the rotation can draw from
# ============================================================
# Same 8 from the static ISLAND_SPECS + 8 more for rotation coverage.
# When an island is retired, the new island picks a bias it didn't have
# in the previous 4 picks (anti-clustering).

BIAS_POOL: list[dict[str, Any]] = [
    # Original 8
    {"name": "fixed_pct", "forced_grid_methods": (GridMethod.FIXED_PCT,)},
    {"name": "atr", "forced_grid_methods": (GridMethod.ATR,)},
    {"name": "volatility_or_dd", "forced_grid_methods": (GridMethod.VOLATILITY, GridMethod.DRAWDOWN_FROM_HIGH)},
    {"name": "trend", "forced_grid_methods": (GridMethod.MA_DISTANCE, GridMethod.TREND_ADJUSTED)},
    {"name": "oscillator", "forced_grid_methods": (GridMethod.RSI_OVERSOLD, GridMethod.Z_SCORE)},
    {"name": "vola_adj_alloc", "forced_allocation": AllocationMethod.VOLATILITY_ADJUSTED},
    {"name": "ctrl_exp_alloc", "forced_allocation": AllocationMethod.CONTROLLED_EXP},
    {"name": "tight_dca", "max_dca_layers_cap": 8},
    # New 8 for rotation
    {"name": "equal_alloc", "forced_allocation": AllocationMethod.EQUAL},
    {"name": "linear_inc_alloc", "forced_allocation": AllocationMethod.LINEAR_INCREASING},
    {"name": "dd_adj_alloc", "forced_allocation": AllocationMethod.DRAWDOWN_ADJUSTED},
    {"name": "rsi_confirm", "forced_confirmations": (ConfirmationIndicator.RSI_BELOW, ConfirmationIndicator.RSI_ABOVE)},
    {"name": "ma_confirm", "forced_confirmations": (ConfirmationIndicator.MA_BELOW, ConfirmationIndicator.MA_ABOVE)},
    {"name": "vol_confirm", "forced_confirmations": (ConfirmationIndicator.VOLATILITY_HIGH, ConfirmationIndicator.VOLATILITY_LOW)},
    {"name": "no_confirm", "forced_confirmations": ()},
    {"name": "trend_only_tight", "forced_grid_methods": (GridMethod.TREND_ADJUSTED,), "max_dca_layers_cap": 6},
    {"name": "atr_low_tp", "forced_grid_methods": (GridMethod.ATR,), "max_dca_layers_cap": 10},
]


def pick_fresh_bias(
    rng: random.Random,
    exclude_recent: list[str] | None = None,
) -> dict[str, Any]:
    """Pick a fresh bias from BIAS_POOL, excluding recently-used names.

    exclude_recent: list of bias names to avoid (default: avoid last 4 picks).
    Returns a copy of the bias dict (with tuples intact).
    """
    exclude = set(exclude_recent or [])
    choices = [b for b in BIAS_POOL if b["name"] not in exclude]
    if not choices:
        # Fallback if everything is excluded
        choices = list(BIAS_POOL)
    picked = rng.choice(choices)
    # Return a shallow copy so caller can mutate without affecting the pool
    return {k: v for k, v in picked.items()}


# ============================================================
# Retirement policy
# ============================================================

@dataclass
class RetirementPolicy:
    """How and when to retire islands."""
    enabled: bool = True
    threshold: float = 0.80                  # per-island top fitness to trigger
    archive_dir: str = "runs/retired_islands"  # root for archived islands
    max_retired_per_cycle: int = 999         # 999 = no cap
    recent_bias_window: int = 4              # avoid re-using last N biases when picking fresh

    def should_retire(self, per_island_top_fitness: float) -> bool:
        if not self.enabled:
            return False
        return per_island_top_fitness >= self.threshold


# ============================================================
# Retired island record (the persisted manifest)
# ============================================================

@dataclass
class RetiredIslandRecord:
    """One archived island — frozen state of a retired slot."""
    island_id: int
    retired_at_cycle: str                       # e.g. "20260622_152341"
    retired_at_gen: int                         # gen_idx when retirement fired
    retired_at_timestamp: float                 # unix timestamp
    family_bias: dict[str, Any]                 # the bias dict it had
    per_island_top_fitness: float
    n_elites_archived: int
    n_generations_evolved: int
    top_3_elite_ids: list[str]
    top_3_elite_fitness: list[float]
    cycle_id: str                               # run-cycle id this came from
    cycle_output_dir: str                       # path to the original output dir

    def to_dict(self) -> dict[str, Any]:
        # Convert tuples to lists for JSON serialization
        d = asdict(self)
        if "family_bias" in d and isinstance(d["family_bias"], dict):
            for k, v in list(d["family_bias"].items()):
                if isinstance(v, tuple):
                    d["family_bias"][k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetiredIslandRecord":
        return cls(**d)


# ============================================================
# Archive + restore
# ============================================================

def archive_island(
    policy: RetirementPolicy,
    cycle_id: str,
    cycle_output_dir: str,
    island_id: int,
    retired_at_gen: int,
    family_bias: dict[str, Any],
    per_island_top_fitness: float,
    elites: list[tuple[CandidateGenome, float]],   # [(genome, fitness), ...]
    generations_evolved: int,
    per_island_history: list[dict[str, Any]] | None = None,
) -> RetiredIslandRecord:
    """Archive one island. Writes manifest + sidecar files. Returns the record.

    Directory structure created:
        {archive_dir}/retired_{cycle_id}_{island_id}_{gen_idx}/
            manifest.json
            top_3_elites.json
            generation_history.json  (per-island slice)
    """
    archive_root = Path(policy.archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)

    # Build dir name (cycle_id like "20260622_152341", island_id=3, gen_idx=12)
    dir_name = f"retired_{cycle_id}_{island_id}_{retired_at_gen}"
    island_dir = archive_root / dir_name
    island_dir.mkdir(parents=True, exist_ok=True)

    # Sort elites by fitness desc
    elites_sorted = sorted(elites, key=lambda x: x[1], reverse=True)
    top_3 = elites_sorted[:3]

    record = RetiredIslandRecord(
        island_id=island_id,
        retired_at_cycle=cycle_id,
        retired_at_gen=retired_at_gen,
        retired_at_timestamp=datetime.now().timestamp(),
        family_bias=family_bias,
        per_island_top_fitness=per_island_top_fitness,
        n_elites_archived=len(elites_sorted),
        n_generations_evolved=generations_evolved,
        top_3_elite_ids=[g.genome_id for g, _ in top_3],
        top_3_elite_fitness=[round(f, 6) for _, f in top_3],
        cycle_id=cycle_id,
        cycle_output_dir=cycle_output_dir,
    )

    # Write manifest
    with open(island_dir / "manifest.json", "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    # Write top-3 elites (full CandidateGenome dump)
    elite_dump = []
    for g, fit in top_3:
        elite_dump.append({
            "genome_id": g.genome_id,
            "fitness": fit,
            "genome": _genome_to_dict(g),
        })
    with open(island_dir / "top_3_elites.json", "w") as f:
        json.dump(elite_dump, f, indent=2)

    # Write per-island history slice (for forensic / future re-evolution)
    if per_island_history:
        with open(island_dir / "generation_history.json", "w") as f:
            json.dump(per_island_history, f, indent=2)

    return record


def _genome_to_dict(g: CandidateGenome) -> dict[str, Any]:
    """Convert a CandidateGenome to a JSON-serializable dict using its own to_dict()."""
    return g.to_dict()


def _safe_dict(obj: Any) -> dict[str, Any]:
    """Convert an attrs/dataclass object to a JSON-safe dict."""
    if obj is None:
        return {}
    if hasattr(obj, "__dict__"):
        out = {}
        for k, v in obj.__dict__.items():
            if hasattr(v, "value"):
                out[k] = v.value
            elif isinstance(v, tuple):
                out[k] = [getattr(x, "value", x) for x in v]
            elif isinstance(v, list):
                out[k] = [_safe_dict(x) if hasattr(x, "__dict__") else getattr(x, "value", x) for x in v]
            else:
                out[k] = v
        return out
    return {}


# ============================================================
# Retirement check — called after each generation
# ============================================================

def check_for_retirements(
    policy: RetirementPolicy,
    cycle_id: str,
    cycle_output_dir: str,
    gen_record: Any,                       # GenerationRecord (avoid hard import cycle)
    elites_by_island: dict[int, list[tuple[CandidateGenome, float]]],
    family_bias_by_island: dict[int, dict[str, Any]],
    per_island_history_by_island: dict[int, list[dict[str, Any]]] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[RetiredIslandRecord], dict[int, dict[str, Any]]]:
    """Check each island's per-island top fitness against the policy threshold.

    Returns:
        (retired_records, new_family_bias_assignments)
        - retired_records: list of newly-retired island archives
        - new_family_bias_assignments: {island_id: fresh_bias} for slots that were retired
                                       (only present for retired islands)
    """
    if not policy.enabled:
        return [], {}

    rng = rng or random.Random()
    retired: list[RetiredIslandRecord] = []
    new_assignments: dict[int, dict[str, Any]] = {}
    recent_bias_names: list[str] = []

    for island_id, top_fit in (gen_record.per_island_best_fitness or {}).items():
        if not policy.should_retire(top_fit):
            continue

        # Find the elites for this island
        elites = elites_by_island.get(island_id, [])
        if not elites:
            continue

        bias = family_bias_by_island.get(island_id, {"name": f"unknown_{island_id}"})
        history = (per_island_history_by_island or {}).get(island_id, [])

        record = archive_island(
            policy=policy,
            cycle_id=cycle_id,
            cycle_output_dir=cycle_output_dir,
            island_id=island_id,
            retired_at_gen=gen_record.generation_index,
            family_bias=bias,
            per_island_top_fitness=top_fit,
            elites=elites,
            generations_evolved=gen_record.generation_index + 1,
            per_island_history=history,
        )
        retired.append(record)
        recent_bias_names.append(bias.get("name", ""))

        # Pick a fresh bias for the slot
        fresh = pick_fresh_bias(rng, exclude_recent=recent_bias_names[-policy.recent_bias_window:])
        new_assignments[island_id] = fresh

    return retired, new_assignments


# ============================================================
# Listing + loading archived islands
# ============================================================

def list_retired_islands(archive_dir: str = "runs/retired_islands") -> list[RetiredIslandRecord]:
    """Load all archived island records from disk."""
    root = Path(archive_dir)
    if not root.exists():
        return []
    records = []
    for manifest_path in sorted(root.glob("retired_*/manifest.json")):
        try:
            with open(manifest_path) as f:
                d = json.load(f)
            records.append(RetiredIslandRecord.from_dict(d))
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return records


def retire_count(archive_dir: str = "runs/retired_islands") -> int:
    """Quick count of retired islands on disk."""
    root = Path(archive_dir)
    if not root.exists():
        return 0
    return len(list(root.glob("retired_*/manifest.json")))


def retire_summary(archive_dir: str = "runs/retired_islands") -> dict[str, Any]:
    """Summary stats for the archive."""
    records = list_retired_islands(archive_dir)
    if not records:
        return {
            "n_retired": 0,
            "avg_top_fitness": 0.0,
            "max_top_fitness": 0.0,
            "by_family": {},
            "by_cycle": {},
        }
    fits = [r.per_island_top_fitness for r in records]
    by_family: dict[str, int] = {}
    by_cycle: dict[str, int] = {}
    for r in records:
        by_family[r.family_bias.get("name", "?")] = by_family.get(r.family_bias.get("name", "?"), 0) + 1
        by_cycle[r.cycle_id] = by_cycle.get(r.cycle_id, 0) + 1
    return {
        "n_retired": len(records),
        "avg_top_fitness": sum(fits) / len(fits),
        "max_top_fitness": max(fits),
        "by_family": by_family,
        "by_cycle": by_cycle,
    }
