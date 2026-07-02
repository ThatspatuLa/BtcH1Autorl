#!/usr/bin/env bash
# minato_stage2_queue_tick.sh — Cron wrapper for Stage 2 combo queue.
#
# Cron schedule: every 10 minutes, no_agent=true.
# Calls minato_stage2_queue_runner.py once per tick. The runner:
#   - If a combo is currently running: do nothing (wait)
#   - If all 60 combos are done: write next_action=review_stage2_ranking to state
#   - Otherwise: launch the next pending combo in the background
#
# Stage 2 combos run 1,000 epochs each — takes 1-3 hours per combo.
# The 10-minute tick is just a watchdog (detects crashes by PID liveness
# + run_summary.json presence); it does NOT throttle the running combo.
#
# Usage:
#   bash scripts/minato_stage2_queue_tick.sh              # normal tick
#   bash scripts/minato_stage2_queue_tick.sh --status     # show queue status, no spawn
#   bash scripts/minato_stage2_queue_tick.sh --dry-run    # show what would be started
set -uo pipefail

# Resolve project root. This script can live at either:
#   ~/.hermes/scripts/minato_stage2_queue_tick.sh (cron-required location)
#   /home/spatula/Projects/BtcH1Autorl/scripts/minato_stage2_queue_tick.sh (project location)
# We hardcode the known project root since both are fixed paths on this system.
PROJECT_ROOT="/home/spatula/Projects/BtcH1Autorl"

if [ ! -f "$PROJECT_ROOT/scripts/run_family_hyperopt.py" ]; then
    echo "ERROR: project root not found at $PROJECT_ROOT" >&2
    exit 1
fi

PYTHON="/home/spatula/freqtrade/.venv/bin/python3"
RUNNER="$PROJECT_ROOT/scripts/minato_stage2_queue_runner.py"

# Change to project root so relative imports work
cd "$PROJECT_ROOT"

# Forward all args to the runner. The runner handles --status, --dry-run, and
# the normal tick path itself.
exec "$PYTHON" -u "$RUNNER" "$@"