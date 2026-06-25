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
    # Rotation 9 (original 9 — for diversity)
    {"name": "equal_alloc", "forced_allocation": AllocationMethod.EQUAL},
    {"name": "linear_inc_alloc", "forced_allocation": AllocationMethod.LINEAR_INCREASING},
    {"name": "dd_adj_alloc", "forced_allocation": AllocationMethod.DRAWDOWN_ADJUSTED},
    {"name": "rsi_confirm", "forced_confirmations": (ConfirmationIndicator.RSI_BELOW, ConfirmationIndicator.RSI_ABOVE)},
    {"name": "ma_confirm", "forced_confirmations": (ConfirmationIndicator.MA_BELOW, ConfirmationIndicator.MA_ABOVE)},
    {"name": "vol_confirm", "forced_confirmations": (ConfirmationIndicator.VOLATILITY_HIGH, ConfirmationIndicator.VOLATILITY_LOW)},
    {"name": "no_confirm", "forced_confirmations": ()},
    {"name": "trend_only_tight", "forced_grid_methods": (GridMethod.TREND_ADJUSTED,), "max_dca_layers_cap": 6},
    {"name": "atr_low_tp", "forced_grid_methods": (GridMethod.ATR,), "max_dca_layers_cap": 10},
    # EXPANDED 2026-06-25 (Six's request: "enough islands in queue").
    # Added 7 more niche biases for force-retire rotation. With 24 total
    # biases and 8 active islands, that's 16 in rotation — plenty of fresh
    # material so the same bias doesn't get re-used too quickly.
    {"name": "fixed_pct_tight", "forced_grid_methods": (GridMethod.FIXED_PCT,), "max_dca_layers_cap": 6},
    {"name": "vol_only", "forced_grid_methods": (GridMethod.VOLATILITY,), "max_dca_layers_cap": 10},
    {"name": "dd_only", "forced_grid_methods": (GridMethod.DRAWDOWN_FROM_HIGH,), "max_dca_layers_cap": 10},
    {"name": "ma_only", "forced_grid_methods": (GridMethod.MA_DISTANCE,), "max_dca_layers_cap": 10},
    {"name": "rsi_only", "forced_grid_methods": (GridMethod.RSI_OVERSOLD,), "max_dca_layers_cap": 10},
    {"name": "zscore_only", "forced_grid_methods": (GridMethod.Z_SCORE,), "max_dca_layers_cap": 10},
    {"name": "loose_dca", "max_dca_layers_cap": 10},  # no forced grid — full freedom
    # EXPANDED 2026-06-25 2nd pass (Six: "make sure we have enough islands in queue").
    # Adding 8 more exotic combinations for the 500-gen run. These combine
    # grid + allocation + confirmation patterns that haven't been explored
    # in the original 24-bias pool. With 32 biases and 8 active islands,
    # that's 24 in rotation — enough for ~3 full retirement cycles before
    # bias reuse becomes a concern.
    {"name": "vol_confirm_tight", "forced_grid_methods": (GridMethod.VOLATILITY,),
     "forced_confirmations": (ConfirmationIndicator.VOLATILITY_LOW,),
     "max_dca_layers_cap": 5},
    {"name": "dd_confirm_exp", "forced_grid_methods": (GridMethod.DRAWDOWN_FROM_HIGH,),
     "forced_allocation": AllocationMethod.CONTROLLED_EXP,
     "max_dca_layers_cap": 10},
    {"name": "trend_rsi_hybrid", "forced_grid_methods": (GridMethod.MA_DISTANCE,),
     "forced_confirmations": (ConfirmationIndicator.RSI_BELOW, ConfirmationIndicator.MA_ABOVE),
     "max_dca_layers_cap": 10},
    {"name": "atr_linear_tight", "forced_grid_methods": (GridMethod.ATR,),
     "forced_allocation": AllocationMethod.LINEAR_INCREASING,
     "max_dca_layers_cap": 5},
    {"name": "vol_drawdown_combo", "forced_grid_methods": (GridMethod.VOLATILITY, GridMethod.DRAWDOWN_FROM_HIGH),
     "forced_allocation": AllocationMethod.DRAWDOWN_ADJUSTED,
     "max_dca_layers_cap": 10},
    {"name": "zscore_no_confirm", "forced_grid_methods": (GridMethod.Z_SCORE,),
     "forced_confirmations": (),
     "max_dca_layers_cap": 10},
    {"name": "trend_dd_combo", "forced_grid_methods": (GridMethod.TREND_ADJUSTED, GridMethod.DRAWDOWN_FROM_HIGH),
     "forced_allocation": AllocationMethod.VOLATILITY_ADJUSTED,
     "max_dca_layers_cap": 10},
    {"name": "deep_dca_exp", "max_dca_layers_cap": 10,
     "forced_allocation": AllocationMethod.CONTROLLED_EXP,
     "forced_grid_methods": (GridMethod.MA_DISTANCE, GridMethod.ATR)},
]  # TOTAL: 32 biases (8 active + 24 rotation)


def pick_fresh_bias(
    rng: random.Random,
    exclude_recent: list[str] | None = None,
    exclude_families: list[str] | None = None,
) -> dict[str, Any]:
    """Pick a fresh bias from BIAS_POOL, excluding recently-used names AND
    families currently active on other islands.

    exclude_recent: list of bias names to avoid (anti-clustering on recent picks).
    exclude_families: list of bias names currently active on OTHER islands.
        The retiring island's CURRENT family is auto-added to this list,
        so a retirement ALWAYS picks a different family.
        This prevents the failure mode where I1 (fixed_pct) retires and gets
        reseeded with a fresh fixed_pct seed — that's not a new family.

    Returns a copy of the bias dict (with tuples intact).
    """
    exclude = set(exclude_recent or []) | set(exclude_families or [])
    choices = [b for b in BIAS_POOL if b["name"] not in exclude]
    if not choices:
        # Fallback if everything is excluded (shouldn't happen with 17 families + 8 islands)
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
    # Phase F7: minimum deployment-passing candidates required to retire.
    # A high-fitness island with ZERO deployment-passing candidates is suspicious
    # (could be curve-fit, over-fit, or not robust enough to clear all gates).
    # Default 1 = at least one must have passed.
    min_deployment_passing: int = 1

    def should_retire(self, per_island_top_fitness: float) -> bool:
        if not self.enabled:
            return False
        return per_island_top_fitness >= self.threshold

    def check_eligibility(
        self,
        island_id: int,
        per_island_top_fitness: float,
        deployment_passing_count: int,
    ) -> bool:
        """Phase F7 — return True iff the island is eligible to be retired.

        Two conditions, both must be true:
        1. per_island_top_fitness >= threshold
        2. deployment_passing_count >= min_deployment_passing

        Returns False if the policy is disabled, fitness is too low, or
        the island hasn't produced any deployment-passing candidates.
        """
        if not self.enabled:
            return False
        if per_island_top_fitness < self.threshold:
            return False
        if deployment_passing_count < self.min_deployment_passing:
            return False
        return True


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
    # Pitfall #11 (2026-06-25): why this island was retired.
    # "fitness_threshold" = crossed retirement_threshold
    # "stagnation_force"   = stagnation_counter >= force_retire_after_gens
    reason: str = "fitness_threshold"

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

def elite_signature(g: CandidateGenome) -> tuple:
    """Build a dedup signature for a CandidateGenome based on its actual params.

    Used to collapse clones that share fitness + params but differ only by
    genome_id (e.g. migration copies, mutation re-seeds of the same lineage).
    """
    dca = g.dca_genome
    grid = dca.grid_params or {}
    alloc = dca.allocation_params or {}
    tp = g.tp_genome.exit_params or {} if g.tp_genome else {}
    return (
        dca.grid_method.value if hasattr(dca.grid_method, "value") else str(dca.grid_method),
        round(float(grid.get("pct", 0.0)), 6),
        round(float(grid.get("drawdown_pct", 0.0)), 6),
        round(float(grid.get("tp_pct", 0.0)), 6),
        int(round(float(grid.get("max_layers", 0)))),
        int(round(float(grid.get("cooldown_candles", 0)))),
        dca.allocation_method.value if hasattr(dca.allocation_method, "value") else str(dca.allocation_method),
        round(float(alloc.get("multiplier", 0.0)), 4),
        round(float(alloc.get("max_layer_size_pct", 0.0)), 4),
        tuple(sorted(c.value if hasattr(c, "value") else str(c)
                     for c in (dca.confirmation_indicators or []))),
        round(float(tp.get("tp_pct", 0.0)), 6),
        dca.combo_method.value if hasattr(dca.combo_method, "value") else str(dca.combo_method),
        dca.trigger_mode.value if hasattr(dca.trigger_mode, "value") else str(dca.trigger_mode),
    )


def dedup_elites_by_signature(
    elites: list[tuple[CandidateGenome, float]],
) -> list[tuple[CandidateGenome, float]]:
    """Deduplicate elites by params signature, keeping the highest-fitness copy.

    elites: [(genome, fitness), ...] — may contain clones that share params
            but differ in genome_id (e.g. migration copies).
    Returns: list sorted by fitness desc with one entry per unique signature.
    """
    seen: dict[tuple, tuple[CandidateGenome, float]] = {}
    for g, f in elites:
        sig = elite_signature(g)
        prev = seen.get(sig)
        if prev is None or f > prev[1]:
            seen[sig] = (g, f)
    return sorted(seen.values(), key=lambda x: x[1], reverse=True)


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
    top_n: int = 3,
    reason: str = "fitness_threshold",  # Pitfall #11: "fitness_threshold" or "stagnation_force"
) -> RetiredIslandRecord:
    """Archive one island. Writes manifest + sidecar files. Returns the record.

    Directory structure created:
        {archive_dir}/retired_{cycle_id}_{island_id}_{gen_idx}/
            manifest.json
            top_3_elites.json
            generation_history.json  (per-island slice)

    Elites are deduplicated by params signature before taking the top-N, so
    migration copies / re-seeded clones don't collapse the archive.
    """
    archive_root = Path(policy.archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)

    # Build dir name (cycle_id like "20260622_152341", island_id=3, gen_idx=12)
    dir_name = f"retired_{cycle_id}_{island_id}_{retired_at_gen}"
    island_dir = archive_root / dir_name
    island_dir.mkdir(parents=True, exist_ok=True)

    # DEDUP by params signature (not by genome_id) so migration copies of the
    # same fit lineage don't collapse to N copies of one genome. This was the
    # Bug #1 root cause — archive top-3 were 3x the same elite.
    elites_dedup = dedup_elites_by_signature(elites)
    elites_sorted = elites_dedup  # already sorted desc by fitness
    top_3 = elites_sorted[:top_n]

    record = RetiredIslandRecord(
        island_id=island_id,
        retired_at_cycle=cycle_id,
        retired_at_gen=retired_at_gen,
        retired_at_timestamp=datetime.now().timestamp(),
        family_bias=family_bias,
        per_island_top_fitness=per_island_top_fitness,
        n_elites_archived=len(elites_dedup),
        n_generations_evolved=generations_evolved,
        top_3_elite_ids=[g.genome_id for g, _ in top_3],
        top_3_elite_fitness=[round(f, 6) for _, f in top_3],
        cycle_id=cycle_id,
        cycle_output_dir=cycle_output_dir,
        reason=reason,
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
        # Phase F7: also require deployment-passing count >= min threshold.
        # Use the check_eligibility method on policy to combine both gates.
        # The deployment_passing count comes from per_island_best_count on the
        # gen_record (populated by harness when iterating passed candidates).
        deploy_passing = 0
        if gen_record is not None and hasattr(gen_record, "per_island_best_count"):
            deploy_passing = (gen_record.per_island_best_count or {}).get(island_id, 0)

        if not policy.check_eligibility(
            island_id=island_id,
            per_island_top_fitness=top_fit,
            deployment_passing_count=deploy_passing,
        ):
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
            reason="fitness_threshold",
        )
        retired.append(record)
        recent_bias_names.append(bias.get("name", ""))

        # Pick a fresh bias for the slot.
        # The replacement MUST be a different family from:
        #   1. The retiring island's current bias (so we don't reseed with same family)
        #   2. All other currently-active islands' biases (so we don't duplicate
        #      an active family across multiple islands)
        # This implements Six's directive: "the replacement island should be a
        # NEW family, not the same grid method."
        current_bias_name = bias.get("name", "")
        active_families = [
            b.get("name", "")
            for iid, b in family_bias_by_island.items()
            if iid != island_id and b.get("name")
        ]
        fresh = pick_fresh_bias(
            rng,
            exclude_recent=recent_bias_names[-policy.recent_bias_window:],
            exclude_families=[current_bias_name] + active_families,
        )
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
