"""Genetic operators — random genome generation, mutation, crossover.

Stage 10 only varies the DCA genome. The TP genome stays fixed at the
Stage 9 baseline. Stage 12 will add TP operators; Stage 14 will combine.

Mutation strategy: per-parameter Gaussian noise with probability = mutation_rate.
For enum fields, mutation means re-rolling from the enum.

Crossover strategy: per-parameter uniform crossover. Each param has a 50%
chance of coming from parent A vs parent B.
"""
from __future__ import annotations

import random

from genome.schema import (
    AllocationMethod,
    CandidateGenome,
    DcaGenome,
    GridMethod,
    LineageMetadata,
    TpExitMethod,
    TpGenome,
)

# ============================================================
# Parameter ranges (the search space)
# ============================================================

# Stage 10 (expanded) search space.
#
# Why wider than v1? The first 3 runs all failed with rejection=tpm<5 —
# meaning candidates weren't generating enough closed trades in 5 years of
# H1 BTC data. The placeholder OrderManager only closes cycles when the
# price moves enough to hit either grid_pct (down for layers) or tp_pct
# (up for close). Tight ranges miss most market action.
#
# New ranges:
# - grid_pct: 0.003..0.08  (was 0.005..0.05)  — wider range, captures both
#   "buy the dip" and "deep dip" strategies
# - max_layers: 2..12      (was 2..8)        — more depth for martingale
# - tp_pct: 0.005..0.05    (was locked 0.02) — GA can pair TP with grid
DCA_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "grid_pct": (0.003, 0.08),
    "max_layers": (2, 12),
    "tp_pct": (0.005, 0.05),
}

# Allocation params (not used in Stage 9 baseline but kept for future Stage 10+)
ALLOCATION_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    AllocationMethod.EQUAL: {},
    AllocationMethod.LINEAR_INCREASING: {"increment_pct": (0.05, 0.5)},
    AllocationMethod.CONTROLLED_EXP: {
        "multiplier": (1.2, 3.0),
        "max_layer_size_pct": (2.0, 10.0),
    },
    AllocationMethod.DRAWDOWN_ADJUSTED: {
        "sensitivity": (1.0, 5.0),
        "min_size_pct": (0.3, 1.0),
        "max_size_pct": (2.0, 10.0),
    },
    AllocationMethod.VOLATILITY_ADJUSTED: {
        "reference_vol": (0.01, 0.05),
        "min_size_pct": (0.3, 1.0),
        "max_size_pct": (2.0, 5.0),
    },
}


# ============================================================
# Random genome
# ============================================================

def random_dca_genome(
    rng: random.Random | None = None,
    genome_id: str | None = None,
    generation_index: int = 0,
    tp_pct: float | None = None,
) -> DcaGenome:
    """Generate a random DcaGenome within the v1 search space.

    The genome carries (grid_pct, max_layers, tp_pct) in grid_params under
    the keys "pct", "max_layers", "tp_pct". When Stage 10 wires the full
    Stage 8 grid spacing, this is replaced with proper grid_method params.

    tp_pct default: random within DCA_PARAM_RANGES["tp_pct"] if not given.
    """
    rng = rng or random.Random()
    grid_pct = rng.uniform(*DCA_PARAM_RANGES["grid_pct"])
    max_layers_lo, max_layers_hi = DCA_PARAM_RANGES["max_layers"]
    max_layers = rng.randint(int(max_layers_lo), int(max_layers_hi))
    if tp_pct is None:
        tp_pct = rng.uniform(*DCA_PARAM_RANGES["tp_pct"])
    return DcaGenome(
        grid_method=GridMethod.FIXED_PCT,
        grid_params={"pct": grid_pct, "max_layers": max_layers, "tp_pct": tp_pct},
        allocation_method=AllocationMethod.EQUAL,
        allocation_params={},
        max_dca_layers=max_layers,
    )


def random_candidate_genome(
    rng: random.Random | None = None,
    genome_id: str | None = None,
    generation_index: int = 0,
    tp_pct: float = 0.02,
) -> CandidateGenome:
    """Generate a full random CandidateGenome.

    The genome carries a random tp_pct in dca_genome.grid_params["tp_pct"].
    The tp_genome uses the same tp_pct (so the Stage 9 baseline reads it
    correctly). When tp_pct is passed as an argument, it overrides the
    random generation (used by resume / deterministic seeding).
    """
    rng = rng or random.Random()
    dca = random_dca_genome(
        rng=rng, generation_index=generation_index, tp_pct=tp_pct,
    )
    gid = genome_id or f"genome_G{generation_index}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=gid,
        dca_genome=dca,
        tp_genome=TpGenome(
            exit_method=TpExitMethod.FIXED,
            exit_params={"tp_pct": dca.grid_params["tp_pct"]},
        ),
        lineage=LineageMetadata(
            parent_a_id=None,
            parent_b_id=None,
            generation_index=generation_index,
            mutation_seed=rng.randint(0, 2**31 - 1),
        ),
    )


# ============================================================
# Mutation
# ============================================================

def mutate(
    parent: CandidateGenome,
    rng: random.Random | None = None,
    mutation_rate: float = 0.30,
    child_id: str | None = None,
) -> CandidateGenome:
    """Mutate a parent genome to produce a child.

    Per-parameter: with probability `mutation_rate`, perturb the value.
    For floats: Gaussian noise scaled to the param range.
    For ints: gaussian-ish, clamped to range.
    """
    rng = rng or random.Random()
    parent_dca = parent.dca_genome

    # Mutate grid_pct
    new_grid_pct = parent_dca.grid_params.get("pct", 0.015)
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["grid_pct"]
        span = hi - lo
        new_grid_pct = new_grid_pct + rng.gauss(0, span * 0.20)
        new_grid_pct = max(lo, min(hi, new_grid_pct))

    # Mutate max_layers
    new_max_layers = parent_dca.max_dca_layers
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["max_layers"]
        delta = rng.choice([-2, -1, 1, 2]) if rng.random() < 0.20 else rng.choice([-1, 1])
        new_max_layers = max(lo, min(hi, new_max_layers + delta))

    # Mutate tp_pct (newly enabled in the expanded search space)
    new_tp_pct = float(parent_dca.grid_params.get("tp_pct", 0.02))
    if rng.random() < mutation_rate:
        lo, hi = DCA_PARAM_RANGES["tp_pct"]
        span = hi - lo
        new_tp_pct = new_tp_pct + rng.gauss(0, span * 0.20)
        new_tp_pct = max(lo, min(hi, new_tp_pct))

    # Build child genome
    new_dca = DcaGenome(
        grid_method=parent_dca.grid_method,
        grid_params={
            "pct": new_grid_pct,
            "max_layers": new_max_layers,
            "tp_pct": new_tp_pct,
        },
        allocation_method=parent_dca.allocation_method,
        allocation_params=dict(parent_dca.allocation_params),
        max_dca_layers=int(new_max_layers),
    )
    new_tp = TpGenome(
        exit_method=parent.tp_genome.exit_method,
        exit_params={"tp_pct": new_tp_pct},
    )
    cid = child_id or f"genome_G{parent.lineage.generation_index + 1}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=cid,
        dca_genome=new_dca,
        tp_genome=new_tp,
        lineage=LineageMetadata(
            parent_a_id=parent.genome_id,
            parent_b_id=None,
            generation_index=parent.lineage.generation_index + 1,
            mutation_seed=rng.randint(0, 2**31 - 1),
            mutation_ops=[{
                "op": "mutate",
                "parent_id": parent.genome_id,
            }],
        ),
    )


# ============================================================
# Crossover
# ============================================================

def crossover(
    parent_a: CandidateGenome,
    parent_b: CandidateGenome,
    rng: random.Random | None = None,
    child_id: str | None = None,
) -> CandidateGenome:
    """Uniform crossover: each param has 50% chance of coming from A or B.

    TP genome (fixed) is inherited from parent A so the Stage 9 baseline
    reads it correctly.
    """
    rng = rng or random.Random()

    # grid_pct: pick from one parent
    a_pct = parent_a.dca_genome.grid_params.get("pct", 0.015)
    b_pct = parent_b.dca_genome.grid_params.get("pct", 0.015)
    new_pct = a_pct if rng.random() < 0.5 else b_pct

    # max_layers: pick from one parent
    new_layers = parent_a.dca_genome.max_dca_layers if rng.random() < 0.5 else parent_b.dca_genome.max_dca_layers

    # tp_pct: pick from one parent
    a_tp = float(parent_a.dca_genome.grid_params.get("tp_pct", 0.02))
    b_tp = float(parent_b.dca_genome.grid_params.get("tp_pct", 0.02))
    new_tp = a_tp if rng.random() < 0.5 else b_tp

    new_dca = DcaGenome(
        grid_method=parent_a.dca_genome.grid_method,
        grid_params={"pct": new_pct, "max_layers": new_layers, "tp_pct": new_tp},
        allocation_method=parent_a.dca_genome.allocation_method,
        allocation_params={},
        max_dca_layers=new_layers,
    )
    new_tp_genome = TpGenome(
        exit_method=parent_a.tp_genome.exit_method,
        exit_params={"tp_pct": new_tp},
    )
    cid = child_id or f"genome_G{max(parent_a.lineage.generation_index, parent_b.lineage.generation_index) + 1}_{rng.randint(0, 1_000_000):06d}"
    return CandidateGenome(
        genome_id=cid,
        dca_genome=new_dca,
        tp_genome=new_tp_genome,
        lineage=LineageMetadata(
            parent_a_id=parent_a.genome_id,
            parent_b_id=parent_b.genome_id,
            generation_index=max(parent_a.lineage.generation_index, parent_b.lineage.generation_index) + 1,
            mutation_seed=rng.randint(0, 2**31 - 1),
            mutation_ops=[{
                "op": "crossover",
                "parent_a_id": parent_a.genome_id,
                "parent_b_id": parent_b.genome_id,
            }],
        ),
    )
