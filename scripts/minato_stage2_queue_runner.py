#!/usr/bin/env python3
"""Run the next unfinished Stage 2 Minato combo job.

Stage 2 = 60 combo runs (20 base × 3 splits), each = 5,000 epochs with smart-adjust.

Cron-neutral: starts at most one combo process per invocation. State is persisted to
runs/hyperopt/stage2_queue_state.json for crash detection by the cron daemon.

CRITICAL: Stage 2 takes hours per combo (60-240 hours total sequential). The queue
processes one combo at a time, but the cron should run this every ~10-15 min so we
catch crashes promptly without restarting too aggressively.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolution.combo_specs import build_stage2_combos, select_top_families_from_stage1
from scripts.run_family_hyperopt import (
    collect_phase1_results,
    phase2_combo_output_dir,
)

STATE_PATH = ROOT / "runs" / "hyperopt" / "stage2_queue_state.json"
LOG_DIR = ROOT / "runs" / "hyperopt" / "logs"


def _now() -> float:
    return time.time()


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True  # No exception → process is alive


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"state_error": f"Invalid JSON in {path}"}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _combo_complete(combo_name: str) -> bool:
    summary_path = phase2_combo_output_dir(combo_name) / "run_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    return summary.get("status") == "complete"


def _build_stage2_combos() -> list:
    """Reconstruct the canonical 60-combo list from Stage 1 top-5."""
    phase1_results = collect_phase1_results()
    top5 = select_top_families_from_stage1(phase1_results, top_n=5)
    return build_stage2_combos(top5)


def _build_status(state: dict[str, Any]) -> dict[str, Any]:
    # Defensive: if state was corrupted (json error), normalise to a dict so .get works.
    if not isinstance(state, dict):
        state = {}
    combos = _build_stage2_combos()
    combo_names = [c.name for c in combos]
    completed = [name for name in combo_names if _combo_complete(name)]
    current = state.get("current") if isinstance(state.get("current"), dict) else {}
    current_pid = current.get("pid")
    current_combo = current.get("combo")
    running = bool(current_combo and _pid_alive(current_pid))

    if current_combo and not running and current_combo not in completed:
        failures = list(state.get("failures", []))
        if not failures or failures[-1].get("combo") != current_combo:
            failures.append({
                "combo": current_combo,
                "pid": current_pid,
                "reason": "process_not_alive_without_complete_summary",
                "recorded_at": _now(),
            })
        state["failures"] = failures
        current = {}
        current_combo = None

    pending = [name for name in combo_names if name not in completed and name != current_combo]
    overall = "running" if running else ("complete" if not pending else "idle")

    return {
        "stage": 2,
        "status": overall,
        "total_combos": len(combo_names),
        "completed_count": len(completed),
        "completed": completed,
        "pending_count": len(pending),
        "pending": pending,
        "current": current if running else {},
        "failures": state.get("failures", []),
        "updated_at": _now(),
    }


def _start_combo(combo_name: str, dry_run: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_family_hyperopt.py"),
        "--phase", "stage2_combo",
        "--combo", combo_name,
    ]
    log_path = LOG_DIR / f"stage2_{combo_name}.log"
    if dry_run:
        return {
            "combo": combo_name,
            "command": command,
            "log_path": str(log_path),
            "dry_run": True,
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "combo": combo_name,
        "pid": proc.pid,
        "command": command,
        "log_path": str(log_path),
        "started_at": _now(),
    }


def run_queue_once(state_path: Path, dry_run: bool = False) -> dict[str, Any]:
    state = _load_state(state_path)
    status = _build_status(state)

    if status["status"] == "complete":
        status["next_action"] = "review_stage2_ranking"
        _write_state(state_path, status)
        return status

    if status["status"] == "running":
        status["next_action"] = "wait_for_current_combo"
        _write_state(state_path, status)
        return status

    next_combo = status["pending"][0] if status["pending"] else None
    if not next_combo:
        status["status"] = "complete"
        status["next_action"] = "review_stage2_ranking"
        _write_state(state_path, status)
        return status

    started = _start_combo(next_combo, dry_run=dry_run)
    status["status"] = "dry_run" if dry_run else "started"
    status["current"] = started
    status["next_action"] = "would_start_combo" if dry_run else "started_combo"
    if not dry_run:
        _write_state(state_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Minato Stage 2 combo queue runner")
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--status", action="store_true", help="Show queue status without starting work")
    parser.add_argument("--dry-run", action="store_true", help="Show the next start action without starting work")
    args = parser.parse_args()

    state = _load_state(args.state_path)
    if args.status:
        result = _build_status(state)
        _write_state(args.state_path, result)
    else:
        result = run_queue_once(args.state_path, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()