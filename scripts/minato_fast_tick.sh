#!/usr/bin/env bash
# minato_fast_tick.sh — Minato Stage 10 evolution tick (cron supervisor).
#
# Cron job: ae4894508637 (every 1 minute, no_agent=true).
# This script is the parent-process supervisor for the evolution cycle.
# It runs every 1 minute via cron and:
#   1. Detects if a healthy cycle is already running → exit (do nothing).
#   2. Detects a stalled cycle (0% CPU both samples, 60s apart) → kill + cooldown.
#   3. Detects multiple cycle dirs in runs/ → keep newest, kill older.
#   4. Detects a stale lock (>30 min, no process alive) → remove lock.
#   5. Respects a respawn cooldown to prevent spawn loops.
#   6. Resumes from a recent checkpoint (< 30 min) if available.
#   7. Otherwise starts a fresh cycle with full 8-island + retirement flags.
#
# All flags match Six's policy 2026-06-25 (cap-10 era + per-island independence):
#   - 8-island model (--island-mode --n-islands 8) — each evolves its own niche
#   - Per-island top persistence — each island loads ITS OWN previous-gen #1
#   - Retirement on fitness >= 0.75 (cap-10 era threshold)
#   - Push pressure: random_injection 220
#   - Checkpoints every 20 min
#   - Force-retire on per-island stagnation after 8 gens (skip if fitness >= 0.70)
#   - 80 generations + 4h wall time
#
# Outputs:
#   /tmp/minato_fast_tick.log  — this script's log
#   /tmp/stage10_cycle_latest.log — the actual cycle's log
#   /home/spatula/Projects/BtcH1Autorl/checkpoints/ — resume snapshots
set -uo pipefail

PROJECT_DIR="/home/spatula/Projects/BtcH1Autorl"
LOCK_FILE="${PROJECT_DIR}/runs/evolution.lock"
LOG_FILE="/tmp/minato_fast_tick.log"
CYCLE_LOG="/tmp/stage10_cycle_latest.log"
PYTHON="/home/spatula/freqtrade/.venv/bin/python3"
CHECKPOINT_RESUME_MAX_AGE_SEC=1800  # 30 min
RESPAWN_COOLDOWN_FILE="${PROJECT_DIR}/runs/.respawn_cooldown"
COOLDOWN_SEC=1800  # 30 min after a kill before respawn

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

# Get total CPU time (user+sys, in seconds) accumulated by a PID AND all its descendants.
# This catches multiprocessing Pool workers — the parent may show low CPU% but the
# workers are doing the real evaluation work.
get_cpu_time_tree() {
    local pid="$1"
    local total=0
    # Collect all descendants (recursive)
    local pids="$pid"
    local new_pids
    while :; do
        new_pids=$(pgrep -P $(echo $pids | tr ' ' ',') 2>/dev/null | tr '\n' ' ') || break
        [ -z "$new_pids" ] && break
        pids="$pids $new_pids"
        # Avoid infinite loop if pids don't grow
        local count=$(echo $pids | wc -w)
        [ "$count" -gt 200 ] && break
    done
    for p in $pids; do
        local t=$(ps -p "$p" -o cputime= 2>/dev/null | head -1)
        [ -z "$t" ] && continue
        local s=$(cputime_to_secs "$t")
        total=$(( total + s ))
    done
    echo "$total"
}

# Convert cputime format [DD-]HH:MM:SS to total seconds
cputime_to_secs() {
    local t="$1"
    awk -F'[:-]' '
    NF==4 { d=$1; h=$2; m=$3; s=$4 }
    NF==3 { d=0; h=$1; m=$2; s=$3 }
    { print d*86400 + h*3600 + m*60 + s }
    ' <<< "$t" 2>/dev/null
}

# Get the timestamp of the last log line written by the cycle process.
# Returns seconds-since-epoch, or 0 if unknown.
last_cycle_log_age() {
    local cycle_dir="$1"
    local log="${cycle_dir}/cycle.log"
    [ -f "$log" ] || return 0
    local last_line
    last_line=$(tail -n 100 "$log" | grep -E '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}' | tail -1)
    [ -z "$last_line" ] && return 0
    local ts
    ts=$(echo "$last_line" | head -1 | sed -E 's/^\[([^]]+)\].*/\1/')
    local epoch
    epoch=$(date -d "$ts" +%s 2>/dev/null) || return 0
    echo $(( $(date +%s) - epoch ))
}

# ---------------------------------------------------------------------------
# SAFEGUARD 1 — Respawn cooldown (after a recent kill)
# ---------------------------------------------------------------------------
if [ -f "$RESPAWN_COOLDOWN_FILE" ]; then
    cooldown_age=$( (cd "$PROJECT_DIR" && "$PYTHON" -c "
import time
print(int(time.time() - float(open('$RESPAWN_COOLDOWN_FILE').read().strip() or 0)))
" 2>/dev/null) || echo "999999")
    if [ "$cooldown_age" -lt "$COOLDOWN_SEC" ]; then
        log "tick: in respawn cooldown (${cooldown_age}s/${COOLDOWN_SEC}s), skip"
        exit 0
    else
        log "tick: respawn cooldown expired (${cooldown_age}s), removing"
        rm -f "$RESPAWN_COOLDOWN_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# SAFEGUARD 2 — Multiple cycles in runs/ → kill duplicate PIDs
# ---------------------------------------------------------------------------
# This catches two cases:
#   A. Multiple cycle dirs in runs/ (kill old dirs + their PIDs)
#   B. Multiple evolution processes running at the same time (regardless of cwd)
mapfile -t CYCLE_DIRS < <(find "${PROJECT_DIR}/runs" -maxdepth 1 -type d -name "evo_continuous_*" 2>/dev/null | sort)
if [ "${#CYCLE_DIRS[@]}" -gt 1 ]; then
    newest="${CYCLE_DIRS[-1]}"
    log "tick: ${#CYCLE_DIRS[@]} cycle dirs found, killing all but newest: $(basename "$newest")"
    for dir in "${CYCLE_DIRS[@]}"; do
        if [ "$dir" != "$newest" ]; then
            log "tick: killing old cycle dir: $(basename "$dir")"
            for pid in $(pgrep -f "run_continuous_evolution" 2>/dev/null || true); do
                pid_cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
                if [[ "$pid_cwd" == "$dir" ]] || [[ "$pid_cwd" == "${dir}/" ]]; then
                    log "tick: kill -9 $pid (cwd=$pid_cwd)"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        fi
    done
    sleep 2
fi

# Case B: count currently-running evolution processes. If >1, keep the NEWEST
# (highest start time / highest PID) and kill the rest. This handles the case
# where two cycles were spawned with the same `--output-dir runs` (same cwd)
# so safeguard 2A above couldn't distinguish them.
mapfile -t ALL_PIDS < <(pgrep -f "run_continuous_evolution" 2>/dev/null || true)
if [ "${#ALL_PIDS[@]}" -gt 1 ]; then
    log "tick: ${#ALL_PIDS[@]} evolution processes running concurrently — killing older ones"
    # Find the NEWEST PID (highest start time)
    newest_pid=""
    newest_start=0
    for pid in "${ALL_PIDS[@]}"; do
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "$PPID" ] && continue
        ps_start=$(stat -c %Y "/proc/$pid" 2>/dev/null || echo 0)
        if [ "$ps_start" -gt "$newest_start" ]; then
            newest_start="$ps_start"
            newest_pid="$pid"
        fi
    done
    log "tick: keeping newest pid=$newest_pid (started ${newest_start})"
    for pid in "${ALL_PIDS[@]}"; do
        [ "$pid" = "$newest_pid" ] && continue
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "$PPID" ] && continue
        log "tick: kill -9 $pid (older duplicate)"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
fi

# ---------------------------------------------------------------------------
# SAFEGUARD 3 — Single-cycle health check
# ---------------------------------------------------------------------------
# Find any running evolution process and check health.
RUNNING_PIDS=()
while IFS= read -r pid; do
    RUNNING_PIDS+=("$pid")
done < <(pgrep -f "run_continuous_evolution" 2>/dev/null || true)

# Also find the bash wrapper (e.g. /usr/bin/bash -lic ... run_continuous_evolution)
# because the python process is a child of bash, and pgrep -f only matches
# the python cmdline, not the bash wrapper. But Safeguard 3 operates on
# python PIDs, so this is fine.

if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    for pid in "${RUNNING_PIDS[@]}"; do
        pid_cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        # Skip if this is an ancestor shell of our own script
        [ "$pid" = "$$" ] && continue
        [ "$pid" = "$PPID" ] && continue

        # Check uptime of the process — if < 10 min, give it time
        # (was 5 min, increased to 10 min because gen 0 can take 8-10 min
        # to bootstrap multiprocessing workers on first launch)
        proc_start=$(stat -c %Y "/proc/$pid" 2>/dev/null || echo 0)
        if [ "$proc_start" -gt 0 ]; then
            proc_age=$(( $(date +%s) - proc_start ))
            if [ "$proc_age" -lt 600 ]; then
                log "tick: cycle pid=$pid is young (${proc_age}s), give it time"
                exit 0
            fi
        fi

        # Check wall-time progress via generation_history.json modtime.
        # If the file was modified within the last 5 min, the cycle is making
        # progress (gen 0 can take 8-10 min; subsequent gens ~3-5 min).
        # This is more reliable than CPU sampling because multiprocessing workers
        # aren't always descendants of the bash wrapper we can see in pgrep.
        cycle_dir="$pid_cwd"
        if [ -d "$cycle_dir" ] && [ -f "$cycle_dir/generation_history.json" ]; then
            history_mtime=$(stat -c %Y "$cycle_dir/generation_history.json" 2>/dev/null || echo 0)
            if [ "$history_mtime" -gt 0 ]; then
                history_age=$(( $(date +%s) - history_mtime ))
                if [ "$history_age" -lt 300 ]; then
                    log "tick: cycle pid=$pid healthy (history updated ${history_age}s ago)"
                    exit 0
                fi
            fi
        fi

        # Check CPU time accumulation over 60s — INCLUDES child workers
        cpu1=$(get_cpu_time_tree "$pid")
        sleep 120
        cpu2=$(get_cpu_time_tree "$pid")

        if [ -n "$cpu1" ] && [ -n "$cpu2" ]; then
            cpu_delta=$(( cpu2 - cpu1 ))
            log "tick: pid=$pid cpu_tree_delta over 120s = ${cpu_delta}s"

            if [ "$cpu_delta" -lt 30 ]; then
                log "tick: STALL DETECTED pid=$pid (cpu_delta=${cpu_delta}s in 120s) — killing"
                kill -9 "$pid" 2>/dev/null || true
                sleep 2
                rm -f "$LOCK_FILE"
                date +%s > "$RESPAWN_COOLDOWN_FILE"
                log "tick: respawn cooldown started (${COOLDOWN_SEC}s)"
                exit 0
            fi
        fi

        # Process is healthy — nothing to do
        log "tick: cycle pid=$pid healthy (cpu_delta=${cpu_delta:-?}s), exit"
        exit 0
    done
fi

# ---------------------------------------------------------------------------
# SAFEGUARD 4 — Stale lock cleanup
# ---------------------------------------------------------------------------
if [ -f "$LOCK_FILE" ]; then
    lock_mtime=$(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)
    if [ "$lock_mtime" -gt 0 ]; then
        lock_age=$(( $(date +%s) - lock_mtime ))
        if [ "$lock_age" -gt 1800 ]; then
            log "tick: stale lock detected (age=${lock_age}s), removing"
            rm -f "$LOCK_FILE"
        else
            existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
            if ! is_pid_alive "$existing_pid"; then
                log "tick: lock with dead pid ($existing_pid), removing"
                rm -f "$LOCK_FILE"
            else
                log "tick: lock held by live pid=$existing_pid (age=${lock_age}s)"
                exit 0
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Step — resume from latest checkpoint if recent enough
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
        ck_cycle=$( (cd "$PROJECT_DIR" && "$PYTHON" -c "
import json
with open('$LATEST_CK') as f:
    d = json.load(f)
print(d.get('cycle_id', ''))
" 2>/dev/null) || echo "")
        if [ -n "$ck_cycle" ]; then
            resume_dir="${PROJECT_DIR}/runs/evo_continuous_${ck_cycle}"
            if [ -d "$resume_dir" ]; then
                log "tick: resuming checkpointed cycle $ck_cycle (age=${ck_age}s)"
                RESUME_FLAG="--output-dir ${resume_dir}"
            fi
        fi
    else
        log "tick: checkpoint too old (age=${ck_age}s), starting fresh"
    fi
fi

# ---------------------------------------------------------------------------
# Spawn the cycle
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

if [ -z "$RESUME_FLAG" ]; then
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
        --force-retire-after-gens 15 \
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
        --force-retire-after-gens 15 \
        --force-retire-min-fitness 0.70 \
        >> "$CYCLE_LOG" 2>&1 &
fi

cycle_pid=$!
disown $cycle_pid 2>/dev/null || true
log "tick: spawned cycle pid=$cycle_pid"
exit 0
