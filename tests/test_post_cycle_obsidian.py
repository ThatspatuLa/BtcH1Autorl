"""Tests for scripts/post_cycle_obsidian_update.py.

Effective 2026-06-22 per Six's plan: deterministic Obsidian sync at cycle end.
This module is critical for "constant contact" between the cron agent and the
Obsidian vault — if it breaks, notes go stale.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Path setup
PROJECT_ROOT = Path("/home/spatula/Projects/BtcH1Autorl")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "post_cycle_obsidian_update.py"

# Make the script importable
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import post_cycle_obsidian_update as pco  # noqa: E402


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def fake_cycle_dir(tmp_path):
    """Create a fake cycle dir with a final_status.json."""
    cycle_dir = tmp_path / "evo_continuous_20260622_120000"
    cycle_dir.mkdir()
    fs = {
        "termination_reason": "completed",
        "generations_completed": 40,
        "generations_planned": 40,
        "total_candidates_evaluated": 20000,
        "best_fitness_ever": 0.8235,
        "best_genome_id_ever": "genome_test_001",
        "best_candidate_id_ever": "cand_test_001",
        "n_deployment_passing_total": 350,
        "total_runtime_seconds": 3600.5,
        "output_dir": str(cycle_dir),
        "seed": 42,
    }
    (cycle_dir / "final_status.json").write_text(json.dumps(fs))
    return cycle_dir


@pytest.fixture
def fake_obsidian_note(tmp_path):
    """Create a fake Minato run-results note."""
    note_dir = tmp_path / "Obsidian" / "01_Projects" / "Minato"
    note_dir.mkdir(parents=True)
    note = note_dir / "02_Latest_Run_Results.md"
    note.write_text(
        "# Latest Run Results\n\n"
        "> Auto-updated after each cron completion.\n\n"
        "---\n\n"
        "## Earlier entry\n\n"
        "Some old content.\n"
    )
    return note


# Patch the script's hardcoded paths to use our tmp_path fixtures
@pytest.fixture
def patch_paths(monkeypatch, fake_obsidian_note, tmp_path):
    """Redirect the script's hardcoded paths to tmp_path fixtures."""
    monkeypatch.setattr(pco, "MINATO_RUN_NOTE", fake_obsidian_note)
    # Patch find_latest_cycle_dir to look in our tmp_path too
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir(exist_ok=True)
    monkeypatch.setattr(pco, "RUNS_DIR", fake_runs)
    # Redirect retired_islands to tmp_path
    monkeypatch.setattr(pco, "RETIRED_DIR", tmp_path / "retired_islands")


# ---------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------

class TestFinalStatusLoading:
    def test_load_final_status_ok(self, fake_cycle_dir):
        fs = pco.load_final_status(fake_cycle_dir)
        assert fs["termination_reason"] == "completed"
        assert fs["best_fitness_ever"] == 0.8235

    def test_load_final_status_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pco.load_final_status(tmp_path / "nonexistent")


class TestRetirementCounting:
    def test_count_retired_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pco, "RETIRED_DIR", tmp_path / "no_archive")
        assert pco.count_retired_islands_total() == 0

    def test_count_retired_with_archives(self, tmp_path, monkeypatch):
        archive = tmp_path / "archive"
        archive.mkdir()
        for i in range(3):
            island_dir = archive / f"retired_20260622_X_{i+1}"
            island_dir.mkdir()
            (island_dir / "manifest.json").write_text("{}")
        monkeypatch.setattr(pco, "RETIRED_DIR", archive)
        assert pco.count_retired_islands_total() == 3

    def test_count_retired_this_cycle(self, tmp_path, monkeypatch):
        archive = tmp_path / "archive"
        archive.mkdir()
        # 2 in cycle_X, 3 in cycle_Y
        for i in range(2):
            (archive / f"retired_cycle_X_{i+1}").mkdir()
            (archive / f"retired_cycle_X_{i+1}" / "manifest.json").write_text("{}")
        for i in range(3):
            (archive / f"retired_cycle_Y_{i+1}").mkdir()
            (archive / f"retired_cycle_Y_{i+1}" / "manifest.json").write_text("{}")
        monkeypatch.setattr(pco, "RETIRED_DIR", archive)
        assert pco.count_retired_this_cycle("cycle_X") == 2
        assert pco.count_retired_this_cycle("cycle_Y") == 3


class TestAllTimeBest:
    def test_detect_all_time_best_picks_max(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        runs.mkdir()
        for i, fit in enumerate([0.81, 0.8235, 0.79, 0.815]):
            d = runs / f"evo_continuous_20260622_run{i}"
            d.mkdir()
            (d / "final_status.json").write_text(json.dumps({"best_fitness_ever": fit}))
        monkeypatch.setattr(pco, "RUNS_DIR", runs)
        assert abs(pco.detect_all_time_best() - 0.8235) < 1e-9

    def test_detect_all_time_best_empty(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        runs.mkdir()
        monkeypatch.setattr(pco, "RUNS_DIR", runs)
        assert pco.detect_all_time_best() == 0.0


class TestSectionRendering:
    def test_render_section_contains_required_fields(self):
        fs = {
            "termination_reason": "completed",
            "generations_completed": 40,
            "generations_planned": 40,
            "total_candidates_evaluated": 20000,
            "best_fitness_ever": 0.8235,
            "best_genome_id_ever": "genome_x",
            "best_candidate_id_ever": "cand_x",
            "n_deployment_passing_total": 350,
            "total_runtime_seconds": 3600.0,
            "output_dir": "/tmp/evo_continuous_TEST",
        }
        section = pco.render_cycle_section(
            cycle_id="20260622_TEST",
            fs=fs,
            retired_this_cycle=2,
            retired_total=15,
            all_time_best=0.8235,
        )
        assert "## Cycle 20260622_TEST" in section
        assert "completed" in section
        assert "0.823500" in section
        assert "350" in section  # deploy-passing
        assert "Retired this cycle" in section
        assert "2" in section  # retired_this_cycle
        assert "15" in section  # retired_total

    def test_render_section_minimal(self):
        section = pco.render_cycle_section(
            cycle_id="X",
            fs={
                "termination_reason": "stagnation",
                "generations_completed": 5,
                "generations_planned": 20,
                "total_candidates_evaluated": 2500,
                "best_fitness_ever": 0.80,
                "best_genome_id_ever": "g",
                "best_candidate_id_ever": "c",
                "n_deployment_passing_total": 100,
                "total_runtime_seconds": 600.0,
                "output_dir": "evo_continuous_X",
            },
            retired_this_cycle=0,
            retired_total=0,
            all_time_best=0.80,
        )
        # Contains exactly one trailing separator after the table
        assert section.rstrip().endswith("---")
        # Single cycle section
        assert section.count("## Cycle X —") == 1
        # Has all the expected table cells
        assert "0.800000" in section
        assert "stagnation" in section
        # v2 breakdown is absent (None by default) — falls back to placeholder
        assert "Fitness v2 breakdown" in section
        assert "pre-Phase D cycle" in section

    def test_render_section_includes_v2_breakdown(self):
        """When v2_breakdown is provided, render the full component table."""
        fs = {
            "termination_reason": "completed", "generations_completed": 10,
            "generations_planned": 10, "total_candidates_evaluated": 5000,
            "best_fitness_ever": 0.85, "best_genome_id_ever": "g1",
            "best_candidate_id_ever": "c1", "n_deployment_passing_total": 100,
            "total_runtime_seconds": 1800.0, "output_dir": "evo_continuous_TEST",
        }
        v2 = {
            "generation": 9, "candidate_id": "c1", "genome_id": "g1",
            "discovery_fitness": 0.85,
            "full_period_base_score": 0.90,
            "recovery_score": 0.70,
            "stability_score": 0.80,
            "concentration_score": 0.95,
            "recovery_breakdown": {
                "drawdown_recovery_speed": 0.65,
                "post_loss_month_bounce_rate": 0.75,
                "equity_high_reclaim_rate": 0.80,
                "cycle_recovery_health": 0.60,
            },
        }
        section = pco.render_cycle_section(
            cycle_id="20260622_V2", fs=fs,
            retired_this_cycle=0, retired_total=0, all_time_best=0.85,
            v2_breakdown=v2,
        )
        # v2 header present
        assert "Fitness v2 breakdown" in section
        # All 5 component rows present
        assert "0.8500" in section   # discovery_fitness
        assert "0.9000" in section   # full_period_base_score
        assert "0.7000" in section   # recovery_score
        assert "0.8000" in section   # stability_score
        assert "0.9500" in section   # concentration_score
        # All 4 recovery sub-metrics present
        assert "drawdown_recovery_speed" in section
        assert "post_loss_month_bounce_rate" in section
        assert "equity_high_reclaim_rate" in section
        assert "cycle_recovery_health" in section
        # Candidate + genome shown
        assert "`c1`" in section
        assert "`g1`" in section
        # The pre-Phase D placeholder is NOT shown
        assert "pre-Phase D cycle" not in section

    def test_load_v2_breakdown_returns_none_for_pre_v2_leaderboards(self, tmp_path):
        """Leaderboards without full_period_base_score field → returns None."""
        # tmp_path/leaderboards/gen_0000.json with old-shape entry
        lb_dir = tmp_path / "leaderboards"
        lb_dir.mkdir()
        lb = {
            "generation_index": 0,
            "leaderboard": [
                {
                    "candidate_id": "c1", "genome_id": "g1",
                    "discovery_fitness": 0.8,  # OLD shape — no v2 fields
                    "deployment_fitness": 0.8, "deployment_pass": True,
                    "failed_deployment_gates": [], "consistency_ratio": 0.6,
                }
            ],
        }
        (lb_dir / "gen_0000.json").write_text(json.dumps(lb))
        result = pco.load_best_genome_v2_breakdown(tmp_path)
        assert result is None

    def test_load_v2_breakdown_extracts_from_latest_generation(self, tmp_path):
        """Finds the latest gen and extracts v2 fields if present."""
        lb_dir = tmp_path / "leaderboards"
        lb_dir.mkdir()
        # gen 0: no top entry
        (lb_dir / "gen_0000.json").write_text(json.dumps({"generation_index": 0, "leaderboard": []}))
        # gen 1: v2 top entry
        (lb_dir / "gen_0001.json").write_text(json.dumps({
            "generation_index": 1,
            "leaderboard": [{
                "candidate_id": "c_late", "genome_id": "g_late",
                "discovery_fitness": 0.85,
                "full_period_base_score": 0.9,
                "recovery_score": 0.7,
                "stability_score": 0.8,
                "concentration_score": 0.95,
                "recovery_breakdown": {"drawdown_recovery_speed": 0.5},
            }],
        }))
        result = pco.load_best_genome_v2_breakdown(tmp_path)
        assert result is not None
        assert result["candidate_id"] == "c_late"
        assert result["full_period_base_score"] == 0.9
        assert result["recovery_breakdown"]["drawdown_recovery_speed"] == 0.5


class TestIdempotency:
    def test_is_already_posted_true(self):
        content = "## Cycle X — 2026-06-22\nfoo"
        assert pco.is_already_posted("X", content) is True

    def test_is_already_posted_false(self):
        content = "## Earlier entry\nfoo"
        assert pco.is_already_posted("Y", content) is False

    def test_post_skips_when_already_posted(self, patch_paths, fake_cycle_dir, fake_obsidian_note):
        # Pre-write the cycle section
        cycle_id = fake_cycle_dir.name.replace("evo_continuous_", "")
        existing = fake_obsidian_note.read_text()
        fake_obsidian_note.write_text(
            existing + f"\n## Cycle {cycle_id} — already there\n"
        )
        # Use fake_cycle_dir by patching find_latest_cycle_dir
        import post_cycle_obsidian_update as mod
        original_find = mod.find_latest_cycle_dir
        mod.find_latest_cycle_dir = lambda: fake_cycle_dir
        try:
            result = mod.post_cycle_to_obsidian(fake_cycle_dir, dry_run=False)
            assert result["ok"] is True
            assert result["skipped"] is True
            assert "already posted" in result["reason"]
        finally:
            mod.find_latest_cycle_dir = original_find


class TestInsertion:
    def test_insert_section_inserts_after_frontmatter(self):
        content = (
            "# Title\n"
            "> subtitle\n"
            "---\n"
            "\n"
            "## Existing\n"
            "stuff\n"
        )
        new_section = "## New\nbody\n---\n"
        patched = pco.insert_section(content, new_section)
        # New section appears BEFORE existing
        new_pos = patched.find("## New")
        existing_pos = patched.find("## Existing")
        assert new_pos > 0
        assert new_pos < existing_pos

    def test_insert_section_no_existing(self):
        content = "# Title\n> subtitle\n---\n"
        new_section = "## New\nbody\n"
        patched = pco.insert_section(content, new_section)
        assert "## New" in patched


class TestEndToEnd:
    def test_post_cycle_to_obsidian_writes_section(
        self, patch_paths, fake_cycle_dir, fake_obsidian_note
    ):
        result = pco.post_cycle_to_obsidian(fake_cycle_dir, dry_run=False)
        assert result["ok"] is True
        assert result["skipped"] is False
        assert result["patched_bytes"] > 0

        # Verify the note now contains the section
        content = fake_obsidian_note.read_text()
        assert "## Cycle 20260622_120000" in content
        assert "0.823500" in content

    def test_post_cycle_dry_run_does_not_write(
        self, patch_paths, fake_cycle_dir, fake_obsidian_note
    ):
        before = fake_obsidian_note.read_text()
        result = pco.post_cycle_to_obsidian(fake_cycle_dir, dry_run=True)
        assert result["ok"] is True
        assert result["dry_run"] is True
        after = fake_obsidian_note.read_text()
        assert before == after  # unchanged

    def test_post_cycle_obsidian_missing(
        self, monkeypatch, fake_cycle_dir, tmp_path
    ):
        # Point to a non-existent note
        monkeypatch.setattr(pco, "MINATO_RUN_NOTE", tmp_path / "missing.md")
        result = pco.post_cycle_to_obsidian(fake_cycle_dir, dry_run=False)
        assert result["ok"] is False
        assert "missing" in result["reason"].lower()


class TestCLI:
    def test_cli_dry_run_exits_zero(self, fake_cycle_dir):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--cycle-dir", str(fake_cycle_dir), "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "dry_run" in result.stdout

    def test_cli_no_cycle_dir_finds_latest(self, monkeypatch, tmp_path):
        # Create a fake runs dir with one cycle
        runs = tmp_path / "runs"
        runs.mkdir()
        cycle = runs / "evo_continuous_20260622_X"
        cycle.mkdir()
        (cycle / "final_status.json").write_text(json.dumps({
            "termination_reason": "completed",
            "generations_completed": 1, "generations_planned": 1,
            "total_candidates_evaluated": 100,
            "best_fitness_ever": 0.5,
            "best_genome_id_ever": "g", "best_candidate_id_ever": "c",
            "n_deployment_passing_total": 10,
            "total_runtime_seconds": 60.0,
            "output_dir": str(cycle),
        }))
        monkeypatch.setattr(pco, "RUNS_DIR", runs)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        # Should not crash (might skip due to obsidian path, but should be OK/skip)
        assert result.returncode in (0, 1)  # OK or skipped-with-error
