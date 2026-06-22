"""Tests for resolve_output_dir() in scripts/run_continuous_evolution.py (Bug #2 fix).

Bug: cron passed --output-dir runs (literal). The script used that string as
the actual output dir, so every cycle wrote into the SAME runs/ directory and
overwrote each other's generation_history.json / leaderboards / best_genomes.
You couldn't audit what cycle A did — only what cycle B is currently doing.

Fix: resolve_output_dir() detects "runs" (or "runs/") as a *container* path,
creates a fresh runs/evo_continuous_<ts>/ subdir per cycle, and updates a
runs/latest symlink to point at the most recent cycle. Each cycle now has its
own immutable output snapshot.

This file: targeted tests proving (a) "runs" gets a timestamped subdir,
(b) repeated calls produce distinct dirs, (c) "runs/latest" symlink tracks
the most recent, (d) custom paths pass through unchanged, (e) None falls
back to runs/evo_continuous_<ts>/ at cwd root.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.run_continuous_evolution import resolve_output_dir


@pytest.fixture
def fresh_cwd(tmp_path, monkeypatch):
    """Run each test in an isolated cwd with a clean runs/ dir."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    yield tmp_path


# ---------------------------------------------------------------
# Bug #2 root cause: --output-dir runs should NOT be used literally
# ---------------------------------------------------------------

class TestResolveOutputDir:
    def test_cron_runs_gets_timestamped_subdir(self, fresh_cwd):
        """The exact bug scenario: cron passes 'runs', we should create a subdir."""
        out, created = resolve_output_dir("runs", "20260622_180000")
        assert created is True
        assert out.name == "evo_continuous_20260622_180000"
        assert out.parent.name == "runs"
        assert out.exists()
        # Manifest of fix: NOT literally "runs"
        assert out.name != "runs"

    def test_runs_with_trailing_slash_gets_timestamped_subdir(self, fresh_cwd):
        out, created = resolve_output_dir("runs/", "20260622_180000")
        assert created is True
        assert out.name == "evo_continuous_20260622_180000"

    def test_two_calls_produce_distinct_dirs(self, fresh_cwd):
        """Two cron ticks 1 second apart must get different dirs (no overwrite)."""
        out1, _ = resolve_output_dir("runs", "20260622_180000")
        out2, _ = resolve_output_dir("runs", "20260622_180001")
        assert out1 != out2
        assert out1.exists()
        assert out2.exists()

    def test_latest_symlink_tracks_most_recent(self, fresh_cwd):
        """After two cycles, runs/latest points at the latest one."""
        out1, _ = resolve_output_dir("runs", "20260622_180000")
        out2, _ = resolve_output_dir("runs", "20260622_180001")
        latest = Path("runs") / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == out2.resolve()

    def test_three_cycles_latest_updates_each_time(self, fresh_cwd):
        resolve_output_dir("runs", "20260622_180000")
        resolve_output_dir("runs", "20260622_180001")
        out3, _ = resolve_output_dir("runs", "20260622_180002")
        latest = Path("runs") / "latest"
        assert latest.resolve() == out3.resolve()

    def test_custom_path_used_as_is(self, fresh_cwd):
        """If user passes a non-runs path, use it directly (no extra subdir)."""
        # Build a custom path with timestamp in name already
        custom = "runs/my_experiment_cycle_A"
        out, created = resolve_output_dir(custom, "20260622_180000")
        assert created is False
        assert str(out) == custom
        assert out.exists()

    def test_none_falls_back_to_timestamped_root_subdir(self, fresh_cwd):
        """No --output-dir: should create runs/evo_continuous_<ts>/ at cwd."""
        out, created = resolve_output_dir(None, "20260622_180000")
        assert created is True
        assert out.name == "evo_continuous_20260622_180000"
        assert out.parent.name == "runs"
        assert out.exists()

    def test_repeated_resolution_into_same_subdir_is_idempotent(self, fresh_cwd):
        """Calling resolve_output_dir twice with same args returns same dir,
        doesn't fail or overwrite."""
        out1, _ = resolve_output_dir("runs", "20260622_180000")
        # Add a sentinel file
        (out1 / "marker.txt").write_text("hello")
        out2, _ = resolve_output_dir("runs", "20260622_180000")
        assert out1 == out2
        # Marker preserved (not blown away)
        assert (out2 / "marker.txt").read_text() == "hello"

    def test_cron_runs_subdir_does_not_clobber_existing_retired_islands(self, fresh_cwd):
        """The fix must not interfere with runs/retired_islands/ (used by Bug #1 fix)."""
        # Simulate retirement archive already there
        retired = Path("runs/retired_islands")
        retired.mkdir(parents=True)
        (retired / "retired_X_1_5").mkdir()
        (retired / "retired_X_1_5" / "manifest.json").write_text("{}")

        out, _ = resolve_output_dir("runs", "20260622_180000")
        assert out.exists()
        # Retirement archive untouched
        assert (retired / "retired_X_1_5" / "manifest.json").exists()


class TestResolveOutputDirEdgeCases:
    def test_runs_subdir_path_not_treated_as_container(self, fresh_cwd):
        """If someone passes 'runs/evo_continuous_X', that's a custom path —
        not the container 'runs'. We should use it as-is."""
        out, created = resolve_output_dir("runs/evo_continuous_X", "20260622_180000")
        assert created is False
        assert out.name == "evo_continuous_X"

    def test_runs_symlink_failure_is_nonfatal(self, fresh_cwd, monkeypatch):
        """If symlink creation fails (e.g. permission denied on CIFS), still
        return a usable timestamped subdir — don't crash the run."""
        from scripts import run_continuous_evolution

        def _fail_unlink(*a, **kw):
            raise OSError("symlink not permitted")

        # Patch the symlink path; we expect the function to swallow OSError
        monkeypatch.setattr(
            "pathlib.Path.symlink_to",
            lambda self, *a, **kw: (_ for _ in ()).throw(OSError("nope")),
        )
        out, created = resolve_output_dir("runs", "20260622_180000")
        assert out.exists()
        assert out.name == "evo_continuous_20260622_180000"
