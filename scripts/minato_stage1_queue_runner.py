#!/usr/bin/env python3
"""Run the next unfinished Stage 1 Minato spacing-family job.

This is intentionally cron-neutral: it can be called manually, by cron, or by a
future supervisor. It starts at most one family process per invocation.
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

from evolution.hyperopt_config import build_family_specs
from scripts.run_family_hyperopt import phase1_output_dir

STATE_PATH = ROOT / "runs" / "hyperopt" / "stage1_queue_state.json"
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
    return True


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


def _family_complete(family_name: str) -> bool:
    summary_path = phase1_output_dir(family_name) / "run_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    return summary.get("status") == "complete"


def _build_status(state: dict[str, Any]) -> dict[str, Any]:
    families = [family.name for family in build_family_specs()]
    completed = [name for name in families if _family_complete(name)]
    current = state.get("current") if isinstance(state.get("current"), dict) else {}
    current_pid = current.get("pid")
    current_family = current.get("family")
    running = bool(current_family and _pid_alive(current_pid))

    if current_family and not running and current_family not in completed:
        failures = list(state.get("failures", []))
        if not failures or failures[-1].get("family") != current_family:
            failures.append({
                "family": current_family,
                "pid": current_pid,
                "reason": "process_not_alive_without_complete_summary",
                "recorded_at": _now(),
            })
        state["failures"] = failures
        current = {}
        current_family = None

    pending = [name for name in families if name not in completed and name != current_family]
    overall = "running" if running else ("complete" if not pending else "idle")

    return {
        "stage": 1,
        "status": overall,
        "total_families": len(families),
        "completed_count": len(completed),
        "completed": completed,
        "pending_count": len(pending),
        "pending": pending,
        "current": current if running else {},
        "failures": state.get("failures", []),
        "updated_at": _now(),
    }


def _start_family(family_name: str, dry_run: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_family_hyperopt.py"),
        "--phase",
        "1",
        "--family",
        family_name,
    ]
    log_path = LOG_DIR / f"stage1_{family_name}.log"
    if dry_run:
        return {
            "family": family_name,
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
        "family": family_name,
        "pid": proc.pid,
        "command": command,
        "log_path": str(log_path),
        "started_at": _now(),
    }


def run_queue_once(state_path: Path, dry_run: bool = False) -> dict[str, Any]:
    state = _load_state(state_path)
    status = _build_status(state)

    if status["status"] == "complete":
        status["next_action"] = "review_stage1_ranking"
        _write_state(state_path, status)
        return status

    if status["status"] == "running":
        status["next_action"] = "wait_for_current_family"
        _write_state(state_path, status)
        return status

    next_family = status["pending"][0] if status["pending"] else None
    if not next_family:
        status["status"] = "complete"
        status["next_action"] = "review_stage1_ranking"
        _write_state(state_path, status)
        return status

    started = _start_family(next_family, dry_run=dry_run)
    status["status"] = "dry_run" if dry_run else "started"
    status["current"] = started
    status["next_action"] = "would_start_family" if dry_run else "started_family"
    if not dry_run:
        _write_state(state_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Minato Stage 1 spacing-family queue runner")
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
