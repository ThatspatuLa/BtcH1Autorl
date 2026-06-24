"""Tests for the per-island Discord gen-end notification format.

User directive 2026-06-25: "I want to know what island has the top candidate
and what fitness level, also all the other islands and their top performing
candidate". Added per-island top fitness table + stagnation warnings to the
Discord on_gen_end hook in scripts/run_continuous_evolution.py.

This file locks in:
1. GenerationRecord now carries per_island_stagnation_counter (was only on
   harness internal state — hook couldn't see it).
2. The Discord message builder includes:
   - All 8 islands' top fitness sorted desc with medals (no genome IDs)
   - Stagnation warnings (⚡ at ≥5 gens, ⚠️ at ≥8 gens if fit < 0.70)
   - Retirement events from record.retired_islands
"""
from __future__ import annotations

import time

from evolution.persistence import GenerationRecord


def _make_record(
    gen_idx=1,
    n_candidates=500,
    n_passed=400,
    n_deployment_passing=10,
    best_fitness=0.68,
    median_fitness=0.62,
    per_island_best_fitness=None,
    per_island_stagnation_counter=None,
    retired_islands=None,
):
    """Helper to build a GenerationRecord with sensible defaults."""
    return GenerationRecord(
        generation_index=gen_idx,
        started_at=time.time() - 60,
        ended_at=time.time(),
        n_candidates=n_candidates,
        n_rejected=n_candidates - n_passed,
        n_passed=n_passed,
        n_deployment_passing=n_deployment_passing,
        best_fitness=best_fitness,
        median_fitness=median_fitness,
        best_candidate_id="",
        best_genome_id="",
        wall_time_seconds_used=60.0,
        rejection_reasons={},
        per_island_best_fitness=per_island_best_fitness or {},
        per_island_stagnation_counter=per_island_stagnation_counter or {},
        retired_islands=retired_islands or [],
    )


# ============================================================
# Source-of-truth: GenerationRecord carries the new field
# ============================================================

def test_generation_record_has_per_island_stagnation_counter():
    """Field added 2026-06-25 so the on_gen_end hook can warn before force-retire."""
    r = _make_record()
    assert hasattr(r, "per_island_stagnation_counter")
    assert r.per_island_stagnation_counter == {}


def test_generation_record_per_island_stagnation_counter_round_trips():
    """The dict must serialize + deserialize via from_dict."""
    r = _make_record(
        per_island_stagnation_counter={1: 8, 2: 5, 3: 0},
    )
    d = r.to_dict()
    assert d["per_island_stagnation_counter"] == {1: 8, 2: 5, 3: 0}
    r2 = GenerationRecord.from_dict(d)
    assert r2.per_island_stagnation_counter == {1: 8, 2: 5, 3: 0}


def test_generation_record_backward_compat_no_stagnation_key():
    """Old JSON without per_island_stagnation_counter must still load."""
    d = {
        "generation_index": 1,
        "started_at": 0.0,
        "ended_at": 1.0,
        "n_candidates": 100,
        "n_rejected": 50,
        "n_passed": 50,
        "n_elite_eligible": 10,
        "n_deployment_passing": 1,
        "best_fitness": 0.6,
        "median_fitness": 0.5,
        "best_candidate_id": "",
        "best_genome_id": "",
        "wall_time_seconds_used": 1.0,
        "rejection_reasons": {},
        # NB: no per_island_stagnation_counter key
    }
    r = GenerationRecord.from_dict(d)
    assert r.per_island_stagnation_counter == {}


# ============================================================
# Hook format tests (mirror production logic in on_gen_end)
# ============================================================

def _build_hook_message(record, n_deploy_total=0):
    """Replicates the body of on_gen_end in run_continuous_evolution.py."""
    from evolution.islands import get_island_specs

    island_bias_lookup = {}
    for spec in get_island_specs()[:8]:
        island_bias_lookup[spec.island_id] = spec.name

    per_island = record.per_island_best_fitness or {}
    if per_island:
        ranked = sorted(per_island.items(), key=lambda kv: kv[1], reverse=True)
        medals = ["🥇", "🥈", "🥉"]
        island_lines = []
        for rank, (iid, fit) in enumerate(ranked):
            medal = medals[rank] if rank < 3 else "  "
            bias_name = island_bias_lookup.get(iid, f"island_{iid}")
            island_lines.append(f"{medal} I{iid} ({bias_name}): {fit:.4f}")
        island_block = "\n".join(island_lines)
    else:
        island_block = "_no per-island data this gen_"

    stagnation = record.per_island_stagnation_counter or {}
    stagnation_warnings = []
    for iid, counter in sorted(stagnation.items()):
        island_fit = per_island.get(iid, 0.0)
        if counter >= 8 and island_fit < 0.70:
            bias_name = island_bias_lookup.get(iid, f"island_{iid}")
            stagnation_warnings.append(
                f"⚠️ I{iid} ({bias_name}): {counter} gens stagnant @ {island_fit:.4f} → force-retire imminent"
            )
        elif counter >= 5:
            bias_name = island_bias_lookup.get(iid, f"island_{iid}")
            stagnation_warnings.append(
                f"⚡ I{iid} ({bias_name}): {counter} gens stagnant"
            )

    stagnation_block = ""
    if stagnation_warnings:
        stagnation_block = "\n" + "\n".join(stagnation_warnings)

    retirement_block = ""
    if record.retired_islands:
        ret_lines = []
        for rec_dict in record.retired_islands:
            ret_lines.append(
                f"🏝️ Archived I{rec_dict.get('island_id', '?')} "
                f"({rec_dict.get('bias_name', '?')}): {rec_dict.get('reason', '?')}"
            )
        retirement_block = "\n" + "\n".join(ret_lines)

    return (
        f"📊 **Gen {record.generation_index} Summary** — Cap 10\n"
        f"Passed: {record.n_passed}/{record.n_candidates} | "
        f"Deploy-passing: {record.n_deployment_passing} | "
        f"Best: {record.best_fitness:.6f} | Median: {record.median_fitness:.6f}\n"
        f"Total deploy-passing so far: {n_deploy_total}\n\n"
        f"🏝️ **Per-Island Top Fitness:**\n"
        f"{island_block}"
        f"{stagnation_block}"
        f"{retirement_block}"
    )


def test_hook_message_contains_all_8_islands():
    """Six directive: 'all the other islands and their top performing candidate'."""
    record = _make_record(
        per_island_best_fitness={
            1: 0.68, 2: 0.65, 3: 0.63, 4: 0.61,
            5: 0.59, 6: 0.57, 7: 0.55, 8: 0.53,
        },
    )
    msg = _build_hook_message(record)
    for iid in range(1, 9):
        assert f"I{iid}" in msg, f"Island {iid} missing from message"
    for bias in ("fixed_pct", "atr", "volatility_or_dd", "trend",
                 "oscillator", "vola_adj_alloc", "ctrl_exp_alloc", "tight_dca"):
        assert bias in msg, f"Bias {bias} missing from message"


def test_hook_message_no_genome_ids():
    """Six directive 2026-06-25: 'get rid of the genome ID'."""
    record = _make_record(per_island_best_fitness={1: 0.68, 2: 0.65})
    msg = _build_hook_message(record)
    assert "genome_G" not in msg
    assert "genome_id" not in msg


def test_hook_message_sorted_by_fitness_desc():
    """🥇🥈🥉 medals go to top 3 fitness islands, in order."""
    record = _make_record(
        per_island_best_fitness={
            1: 0.55,  # 5th
            2: 0.68,  # 1st
            3: 0.60,  # 4th
            4: 0.66,  # 2nd
            5: 0.65,  # 3rd
        },
    )
    msg = _build_hook_message(record)
    pos = {iid: msg.find(f"I{iid} (") for iid in [2, 4, 5, 3, 1]}
    assert pos[2] < pos[4] < pos[5] < pos[3] < pos[1], (
        f"Islands not sorted by fitness desc, positions: {pos}"
    )
    assert "🥇 I2" in msg
    assert "🥈 I4" in msg
    assert "🥉 I5" in msg


def test_hook_message_stagnation_warning_at_5_gens():
    """⚡ symbol at 5-7 stagnant gens (advance warning)."""
    record = _make_record(
        per_island_best_fitness={3: 0.65},
        per_island_stagnation_counter={3: 5},
    )
    msg = _build_hook_message(record)
    assert "⚡ I3" in msg
    assert "5 gens stagnant" in msg
    assert "force-retire imminent" not in msg


def test_hook_message_force_retire_imminent_at_8_gens_below_threshold():
    """⚠️ symbol at ≥8 stagnant gens AND fitness < 0.70."""
    record = _make_record(
        per_island_best_fitness={5: 0.66},
        per_island_stagnation_counter={5: 8},
    )
    msg = _build_hook_message(record)
    assert "⚠️ I5" in msg
    assert "force-retire imminent" in msg


def test_hook_message_force_retire_skipped_if_fit_above_070():
    """At 8 gens stale, if fitness ≥ 0.70, do NOT warn (policy: skip if near bar)."""
    record = _make_record(
        best_fitness=0.75,
        per_island_best_fitness={5: 0.72},
        per_island_stagnation_counter={5: 8},
    )
    msg = _build_hook_message(record)
    assert "force-retire imminent" not in msg
    assert "⚠️" not in msg


def test_hook_message_empty_per_island_handled_gracefully():
    """Gen 0 / pre-evaluation shouldn't crash the hook."""
    record = _make_record(gen_idx=0, n_passed=0, best_fitness=0.0, median_fitness=0.0)
    msg = _build_hook_message(record)
    assert "no per-island data this gen" in msg


def test_hook_message_retirement_event_appears():
    """When an island retires this gen, show 🏝️ archive line."""
    record = _make_record(
        retired_islands=[{"island_id": 3, "bias_name": "volatility_or_dd", "reason": "stagnation_force"}],
    )
    msg = _build_hook_message(record)
    assert "🏝️ Archived I3" in msg
    assert "volatility_or_dd" in msg
    assert "stagnation_force" in msg
