#!/usr/bin/env bash
# minato_stage2_queue_tick.sh — Cron wrapper for Stage 2 combo queue.
#
# Cron schedule: every 10 minutes, no_agent=true.
# Calls minato_stage2_queue_runner.py once per tick. The runner:
#   - If a combo is currently running: do nothing (wait)
#   - If all 60 combos are done: write next_action=review_stage2_ranking to state
#   - Otherwise: launch the next pending combo in the background
#
# Stage 2 combos run 5,000 epochs each — takes 1-4 hours per combo. Total: 60-240h
# sequential. The 10-minute tick is just a watchdog (detects crashes by PID liveness
# + run_summary.json presence); it does NOT throttle the running combo.
#
# Usage:
#   bash scripts/minato_stage2_queue_tick.sh              # normal tick
#   bash scripts/minato_stage2_queue_tick.sh --status     # show queue status, no spawn
#   bash scripts/minato_stage2_queue_tick.sh --dry-run    # show what would be started
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/spatula/freqtrade/.venv/bin/python3"
RUNNER="$PROJECT_ROOT/scripts/minato_stage2_queue_runner.py"

# Change to project root so relative imports work
cd "$PROJECT_ROOT"

# Forward all args to the runner. The runner handles --status, --dry-run, and
# the normal tick path itself.
exec "$PYTHON" -u "$RUNNER" "$@"