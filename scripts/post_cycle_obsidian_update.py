#!/usr/bin/env python3
"""post_cycle_obsidian_update.py — deterministic Obsidian updater for Stage 10 cron.

Reads final_status.json from the just-completed evolution cycle and appends a
new entry to the Obsidian vault at:
    /home/spatula/Obsidian/ZenVault/01_Projects/Minato/02_Latest_Run_Results.md

Also reads retirement archive index from:
    /home/spatula/Projects/BtcH1Autorl/runs/retired_islands/

Designed to be called by the cron agent at cycle end, OR by run_continuous_evolution.py
directly via --post-cycle-obsidian-update. Idempotent: skips if today's entry
already exists for this cycle_id (so repeated runs don't duplicate).

Usage:
    python3 scripts/post_cycle_obsidian_update.py --cycle-dir runs/evo_continuous_20260622_170000

If --cycle-dir is omitted, finds the most recent evo_continuous_* dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Project paths
PROJECT_ROOT = Path("/home/spatula/Projects/BtcH1Autorl")
RETIRED_DIR = PROJECT_ROOT / "runs" / "retired_islands"
RUNS_DIR = PROJECT_ROOT / "runs"

# Obsidian paths
OBSIDIAN_VAULT = Path("/home/spatula/Obsidian/ZenVault")
MINATO_RUN_NOTE = OBSIDIAN_VAULT / "01_Projects" / "Minato" / "02_Latest_Run_Results.md"


def find_latest_cycle_dir() -> Path | None:
    """Find the most recent evo_continuous_* dir that has final_status.json."""
    candidates = sorted(RUNS_DIR.glob("evo_continuous_*"), reverse=True)
    for c in candidates:
        if (c / "final_status.json").exists():
            return c
    return None


def load_final_status(cycle_dir: Path) -> dict[str, Any]:
    """Read final_status.json from a cycle dir."""
    fs_path = cycle_dir / "final_status.json"
    if not fs_path.exists():
        raise FileNotFoundError(f"final_status.json missing in {cycle_dir}")
    return json.loads(fs_path.read_text())


def count_retired_islands_total() -> int:
    """Count retired island manifests on disk (across all cycles)."""
    if not RETIRED_DIR.exists():
        return 0
    return len(list(RETIRED_DIR.glob("retired_*/manifest.json")))


def count_retired_this_cycle(cycle_id: str) -> int:
    """Count retired islands with cycle_id prefix matching this cycle."""
    if not RETIRED_DIR.exists():
        return 0
    pattern = f"retired_{cycle_id}_*"
    return len(list(RETIRED_DIR.glob(f"{pattern}/manifest.json")))


def detect_all_time_best() -> float:
    """Scan all final_status.json files for the highest best_fitness_ever."""
    best = 0.0
    for fs in RUNS_DIR.glob("evo_continuous_*/final_status.json"):
        try:
            d = json.loads(fs.read_text())
            bf = float(d.get("best_fitness_ever", 0))
            if bf > best:
                best = bf
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    return best


def load_best_genome_v2_breakdown(cycle_dir: Path) -> dict[str, Any] | None:
    """Load v2 component breakdown for the best candidate from the latest leaderboard.

    Returns None if the cycle predates Phase D (pre-v2 — leaderboards lack the fields).
    """
    lb_dir = cycle_dir / "leaderboards"
    if not lb_dir.exists():
        return None
    # Find the latest generation's leaderboard
    candidates = sorted(lb_dir.glob("gen_*.json"), reverse=True)
    for lb_path in candidates:
        try:
            lb = json.loads(lb_path.read_text())
            entries = lb.get("leaderboard", [])
            if not entries:
                continue
            top = entries[0]
            # Check for v2 fields (Phase D onwards)
            if "full_period_base_score" in top:
                return {
                    "generation": lb.get("generation_index", "?"),
                    "candidate_id": top.get("candidate_id", "?"),
                    "genome_id": top.get("genome_id", "?"),
                    "discovery_fitness": top.get("discovery_fitness", 0),
                    "full_period_base_score": top.get("full_period_base_score", 0),
                    "recovery_score": top.get("recovery_score", 0),
                    "stability_score": top.get("stability_score", 0),
                    "concentration_score": top.get("concentration_score", 0),
                    "recovery_breakdown": top.get("recovery_breakdown", {}),
                }
            # Pre-v2 cycle — no breakdown available
            return None
        except (json.JSONDecodeError, OSError):
            continue
    return None


def render_cycle_section(
    cycle_id: str,
    fs: dict[str, Any],
    retired_this_cycle: int,
    retired_total: int,
    all_time_best: float,
    v2_breakdown: dict[str, Any] | None = None,
) -> str:
    """Render the markdown section for one cycle (deterministic, minimal)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = fs.get("termination_reason", "unknown")
    gens_done = fs.get("generations_completed", 0)
    gens_planned = fs.get("generations_planned", 0)
    n_cands = fs.get("total_candidates_evaluated", 0)
    best_fit = fs.get("best_fitness_ever", 0.0)
    best_genome = fs.get("best_genome_id_ever", "")
    best_cand = fs.get("best_candidate_id_ever", "")
    deploy_total = fs.get("n_deployment_passing_total", 0)
    runtime = fs.get("total_runtime_seconds", 0.0)
    output_dir = Path(fs.get("output_dir", "")).name or "?"

    # v2 breakdown block (Phase D) — only if available
    v2_block = ""
    if v2_breakdown is not None:
        rb = v2_breakdown.get("recovery_breakdown", {})
        rb_lines = []
        for k, v in rb.items():
            rb_lines.append(f"| {k} | {float(v):.4f} |")
        rb_table = "\n".join(rb_lines) if rb_lines else "_(no sub-metrics)_"
        v2_block = f"""
### Fitness v2 breakdown (best candidate, gen {v2_breakdown.get("generation", "?")})

| Component | Value |
|---|---|
| discovery_fitness (final) | **{float(v2_breakdown.get("discovery_fitness", 0)):.4f}** |
| full_period_base_score (60%) | {float(v2_breakdown.get("full_period_base_score", 0)):.4f} |
| recovery_score (20%) | {float(v2_breakdown.get("recovery_score", 0)):.4f} |
| stability_score (5%) | {float(v2_breakdown.get("stability_score", 0)):.4f} |
| concentration_score (5%) | {float(v2_breakdown.get("concentration_score", 0)):.4f} |

#### Recovery sub-metrics

| Sub-metric | Value |
|---|---|
{rb_table}

Candidate: `{v2_breakdown.get("candidate_id", "?")}` · Genome: `{v2_breakdown.get("genome_id", "?")}`
"""
    else:
        v2_block = "\n### Fitness v2 breakdown\n\n_(pre-Phase D cycle — no component breakdown available)_\n"

    return f"""## Cycle {cycle_id} — {ts}

| Metric | Value |
|---|---|
| Status | `{status}` |
| Generations | {gens_done} / {gens_planned} |
| Candidates evaluated | {n_cands:,} |
| Best fitness | **{best_fit:.6f}** |
| Best genome | `{best_genome}` |
| Best candidate | `{best_cand}` |
| Deploy-passing | {deploy_total} |
| Runtime | {runtime:.0f}s ({runtime / 60:.1f} min) |
| Output | `{output_dir}` |
| Retired this cycle | {retired_this_cycle} |
| **All-time best fitness** | **{all_time_best:.6f}** |
| Total retired islands (archive) | {retired_total} |
{v2_block}
---

"""


def is_already_posted(cycle_id: str, note_content: str) -> bool:
    """Idempotency check — skip if this cycle_id already has a section."""
    header_pattern = f"## Cycle {cycle_id} —"
    return header_pattern in note_content


def insert_section(note_content: str, new_section: str) -> str:
    """Insert a new cycle section after the frontmatter block, newest first."""
    # The note format is:
    #   # Latest Run Results
    #   > Auto-updated...
    #   ---
    #
    #   ## <existing sections>
    #
    # We insert the new section after the FIRST `---` line (the separator
    # following the frontmatter), before any existing cycle sections.
    lines = note_content.splitlines(keepends=True)

    # Find the first `---` separator line (frontmatter end)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            insert_idx = i + 1
            break

    # Skip blank lines after the separator so the new section sits cleanly
    while insert_idx < len(lines) and lines[insert_idx].strip() == "":
        insert_idx += 1

    # Insert new section followed by a blank line, then the existing content
    new_lines = lines[:insert_idx] + ["\n"] + [new_section] + lines[insert_idx:]
    return "".join(new_lines)


def post_cycle_to_obsidian(cycle_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Main entry — read cycle, render section, patch Obsidian note.

    Returns a status dict suitable for logging / Discord reporting.
    """
    fs = load_final_status(cycle_dir)
    cycle_id = cycle_dir.name.replace("evo_continuous_", "")
    retired_total = count_retired_islands_total()
    retired_this_cycle = count_retired_this_cycle(cycle_id)
    all_time_best = detect_all_time_best()
    v2_breakdown = load_best_genome_v2_breakdown(cycle_dir)

    new_section = render_cycle_section(
        cycle_id=cycle_id,
        fs=fs,
        retired_this_cycle=retired_this_cycle,
        retired_total=retired_total,
        all_time_best=all_time_best,
        v2_breakdown=v2_breakdown,
    )

    if not MINATO_RUN_NOTE.exists():
        return {
            "ok": False,
            "skipped": True,
            "reason": f"Obsidian note missing: {MINATO_RUN_NOTE}",
        }

    note_content = MINATO_RUN_NOTE.read_text()

    if is_already_posted(cycle_id, note_content):
        return {
            "ok": True,
            "skipped": True,
            "reason": f"Cycle {cycle_id} already posted (idempotent skip)",
        }

    patched = insert_section(note_content, new_section)

    if dry_run:
        return {
            "ok": True,
            "skipped": False,
            "dry_run": True,
            "preview_chars": len(new_section),
            "would_patch": str(MINATO_RUN_NOTE),
        }

    MINATO_RUN_NOTE.write_text(patched)
    return {
        "ok": True,
        "skipped": False,
        "patched_bytes": len(patched) - len(note_content),
        "obsidian_note": str(MINATO_RUN_NOTE),
        "cycle_id": cycle_id,
        "all_time_best": all_time_best,
        "retired_total": retired_total,
        "retired_this_cycle": retired_this_cycle,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycle-dir",
        type=Path,
        default=None,
        help="Path to evo_continuous_<id>/ dir (default: latest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying Obsidian",
    )
    args = parser.parse_args()

    cycle_dir = args.cycle_dir
    if cycle_dir is None:
        cycle_dir = find_latest_cycle_dir()
        if cycle_dir is None:
            print("[obsidian-post] ERROR: no evo_continuous_*/final_status.json found", file=sys.stderr)
            return 1

    if not cycle_dir.exists():
        print(f"[obsidian-post] ERROR: cycle dir does not exist: {cycle_dir}", file=sys.stderr)
        return 1

    result = post_cycle_to_obsidian(cycle_dir, dry_run=args.dry_run)
    print(f"[obsidian-post] {json.dumps(result, indent=2)}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
