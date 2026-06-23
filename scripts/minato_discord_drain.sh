#!/usr/bin/env bash
# minato_discord_drain.sh — Minato Discord message drainer.
#
# Cron job: ae4894508636 (every 1 minute, no_agent=true).
#
# Reads JSON message files written by run_continuous_evolution.py's
# send_discord() helper into runs/discord_queue/, and posts each one
# to Discord via `hermes send`. The "channel" field in the JSON overrides
# the default Discord target if present (forward-compat for multi-channel
# staging).
#
# Throttle: 1.5s between messages to avoid Discord rate limits.
# Skips non-JSON files (just logs the skip).
# Deletes the file on successful send.
#
# Outputs:
#   /tmp/minato_discord_drain.log  — this script's log
#
set -euo pipefail

PROJECT_DIR="/home/spatula/Projects/BtcH1Autorl"
QUEUE_DIR="${PROJECT_DIR}/runs/discord_queue"
LOG_FILE="/tmp/minato_discord_drain.log"
DEFAULT_DISCORD_TARGET="discord:1500437358934233219"
THROTTLE_SEC=1.5
PYTHON="/home/spatula/freqtrade/.venv/bin/python3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# Queue dir may not exist yet on first run
if [ ! -d "$QUEUE_DIR" ]; then
    mkdir -p "$QUEUE_DIR"
    log "drain: created queue dir $QUEUE_DIR"
    exit 0
fi

# Drain each JSON file in the queue
n_drained=0
n_failed=0
for msg_file in "$QUEUE_DIR"/msg_*.json; do
    [ -e "$msg_file" ] || continue  # no files

    # Read the message body out of the JSON to a temp file (preserves newlines).
    body_file="$(mktemp)"
    trap "rm -f '$body_file'" RETURN

    if ! "$PYTHON" -c "
import json
with open('$msg_file') as f:
    d = json.load(f)
with open('$body_file', 'w') as out:
    out.write(d.get('message', ''))
" 2>/dev/null; then
        log "drain: failed to parse $(basename "$msg_file")"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Get channel (just a short string, no newline issues).
    # run_continuous_evolution.py writes the bare channel ID (no platform
    # prefix) so we detect + prepend 'discord:' if missing.
    channel="$("$PYTHON" -c "
import json
with open('$msg_file') as f:
    d = json.load(f)
print(d.get('channel', '$DEFAULT_DISCORD_TARGET'))
" 2>/dev/null)"
    channel="${channel:-$DEFAULT_DISCORD_TARGET}"
    # If it doesn't have a 'platform:' prefix, assume Discord
    if [[ "$channel" != *:* ]]; then
        channel="discord:${channel}"
    fi

    body_size=$(wc -c < "$body_file")
    if [ "$body_size" -eq 0 ]; then
        log "drain: empty message in $(basename "$msg_file"), skipping"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Post via hermes send using --file for body (preserves newlines)
    if hermes send -t "$channel" -f "$body_file" >> "$LOG_FILE" 2>&1; then
        rm -f "$msg_file"
        n_drained=$((n_drained + 1))
        sleep "$THROTTLE_SEC"
    else
        log "drain: send failed for $(basename "$msg_file")"
        n_failed=$((n_failed + 1))
    fi
done

if [ "$n_drained" -gt 0 ] || [ "$n_failed" -gt 0 ]; then
    log "drain: drained=$n_drained failed=$n_failed"
fi

exit 0
