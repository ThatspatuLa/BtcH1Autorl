#!/usr/bin/env bash
# minato_fast_tick.sh — Minato Stage 10 evolution tick.
#
# Cron job: ae4894508637 (every 1 minute, no_agent=true).
# This script is the parent-process supervisor for the evolution cycle.
# It:
#   1. Checks if a cycle is already running (lock file + alive PID). If yes, exits.
#   2. Checks for a recent checkpoint (< 30 min old) → resume in same dir.
#   3. Otherwise starts a fresh cycle with full 8-island + retirement flags.
#   4. Disowns so the parent shell exit doesn't kill the cycle.
#
# All flags match Six's policy 2026-06-25 (cap-10 era + per-island independence):
#   - 8-island model (--island-mode --n-islands 8) — each evolves its own niche
#   - Per-island top persistence (Fix 2026-06-25: islands-converged-bug)
#     Each island loads ITS OWN previous-gen #1 from
#     best_genomes/per_island_gen_<N>_island_<I>.json, not the global #1.
#     Prevents premature convergence across the 8 islands.
#   - Retirement on fitness >= 0.75 (lowered from 0.80 — cap-10 needs more
#     retirement signal to keep islands evolving independently)
#   - Push pressure: random_injection 220
#   - Checkpoints every 20 min
#   - Force-retire on per-island stagnation after 8 gens (skip if fitness >= 0.70)
#   - 80 generations + 4h wall time (cap-10 era — proportional to cap bump)
#
# Outputs:
#   /tmp/minato_fast_tick.log          — this script's log
#   /tmp/stage10_cycle_latest.log      — the actual cycle's log
#   /home/spatula/Projects/BtcH1Autorl/checkpoints/ — resume snapshots
#
# Constants (tweak here, not in cron):
set -euo pipefail

PROJECT_DIR="/home/spatula/Projects/BtcH1Autorl"
LOCK_FILE="${PROJECT_DIR}/runs/evolution.lock"
LOG_FILE="/tmp/minato_fast_tick.log"
CYCLE_LOG="/tmp/stage10_cycle_latest.log"
PYTHON="/home/spatula/freqtrade/.venv/bin/python3"
CHECKPOINT_RESUME_MAX_AGE_SEC=1800  # 30 min — fresh enough to resume

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

is_pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Step 1 — is a cycle already running?
# ---------------------------------------------------------------------------
if [ -f "$LOCK_FILE" ]; then
    existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if is_pid_alive "$existing_pid"; then
        # Healthy — cycle is running, nothing to do
        log "tick: cycle already running (pid=$existing_pid), exit"
        exit 0
    else
        # Stale lock — previous cycle crashed
        log "tick: stale lock detected (pid=$existing_pid dead), removing"
        rm -f "$LOCK_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2 — resume from latest checkpoint if recent enough
# ---------------------------------------------------------------------------
RESUME_FLAG=""
LATEST_CK="${PROJECT_DIR}/checkpoints/latest.json"
if [ -f "$LATEST_CK" ]; then
    ck_age=$( (cd "$PROJECT_DIR" && "$PYTHON" -c "
import json, time
with open('$LATEST_CK') as f:
    d = json.load(f)
print(int(time.time() - float(d.get('saved_at', 0.0))))
" 2>/dev/null) || echo "999999")
    if [ "$ck_age" -lt "$CHECKPOINT_RESUME_MAX_AGE_SEC" ]; then
        # Recent — resume in same cycle dir
        ck_cycle=$( (cd "$PROJECT_DIR" && "$PYTHON" -c "
import json
with open('$LATEST_CK') as f:
    d = json.load(f)
print(d.get('cycle_id', ''))
" 2>/dev/null) || echo "")
        if [ -n "$ck_cycle" ]; then
            resume_dir="${PROJECT_DIR}/runs/evo_continuous_${ck_cycle}"
            if [ -d "$resume_dir" ]; then
                log "tick: resuming checkpointed cycle $ck_cycle (age=${ck_age}s, dir=$resume_dir)"
                RESUME_FLAG="--output-dir ${resume_dir}"
            fi
        fi
    else
        log "tick: checkpoint too old (age=${ck_age}s), starting fresh"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3 — spawn the cycle
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

if [ -z "$RESUME_FLAG" ]; then
    # Fresh cycle — timestamped subdir + latest symlink
    log "tick: starting fresh 8-island cycle"
    "$PYTHON" -u scripts/run_continuous_evolution.py \
        --experiment-id stage10_continuous \
        --output-dir runs \
        --max-generations 80 \
        --wall-time 14400 \
        --candidates 500 \
        --island-mode \
        --n-islands 8 \
        --migration-every 5 \
        --elite-count 20 \
        --random-injection 220 \
        --mutation-rate 0.45 \
        --crossover-rate 0.40 \
        --workers 8 \
        --stagnation-generations 5 \
        --all-rejected-generations 3 \
        --retirement-enabled \
        --retirement-threshold 0.75 \
        --checkpoint-interval-min 20 \
        --force-retire-after-gens 8 \
        --force-retire-min-fitness 0.70 \
        >> "$CYCLE_LOG" 2>&1 &
else
    log "tick: resuming cycle in existing dir"
    "$PYTHON" -u scripts/run_continuous_evolution.py \
        --experiment-id stage10_continuous \
        $RESUME_FLAG \
        --max-generations 80 \
        --wall-time 14400 \
        --candidates 500 \
        --island-mode \
        --n-islands 8 \
        --migration-every 5 \
        --elite-count 20 \
        --random-injection 220 \
        --mutation-rate 0.45 \
        --crossover-rate 0.40 \
        --workers 8 \
        --stagnation-generations 5 \
        --all-rejected-generations 3 \
        --retirement-enabled \
        --retirement-threshold 0.75 \
        --checkpoint-interval-min 20 \
        --force-retire-after-gens 8 \
        --force-retire-min-fitness 0.70 \
        >> "$CYCLE_LOG" 2>&1 &
fi

cycle_pid=$!
disown $cycle_pid 2>/dev/null || true
log "tick: spawned cycle pid=$cycle_pid"
exit 0
