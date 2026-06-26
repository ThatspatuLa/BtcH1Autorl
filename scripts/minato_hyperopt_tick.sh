#!/usr/bin/env bash
# minato_hyperopt_tick.sh — Cron wrapper for Stage 10 family-budgeted hyperopt.
#
# Usage:
#   bash scripts/minato_hyperopt_tick.sh <phase> <family_name>
#   bash scripts/minato_hyperopt_tick.sh 1 pure_atr
#   bash scripts/minato_hyperopt_tick.sh 1 --all-families
#   bash scripts/minato_hyperopt_tick.sh 2 --all-families
#   bash scripts/minato_hyperopt_tick.sh 3 combo_pure_atr_alloc_equal_confirm_rsi --iteration 1
#   bash scripts/minato_hyperopt_tick.sh 3 --all-families
#   bash scripts/minato_hyperopt_tick.sh --status 1
#   bash scripts/minato_hyperopt_tick.sh --rank 1
#
# Manual phase transitions:
#   Phase 1 → Phase 2: cron runs all 22 families, then you manually start Phase 2
#   Phase 2 → Phase 3: cron runs all 5 deep families, then you manually start Phase 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/../freqtrade/.venv/bin/python3"
RUNNER="$PROJECT_ROOT/scripts/run_family_hyperopt.py"

# Change to project root
cd "$PROJECT_ROOT"

# Handle status/rank queries
if [[ "${1:-}" == "--status" ]]; then
    PHASE="${2:-1}"
    exec "$PYTHON" -u "$RUNNER" --phase "$PHASE" --status
fi

if [[ "${1:-}" == "--rank" ]]; then
    PHASE="${2:-1}"
    exec "$PYTHON" -u "$RUNNER" --phase "$PHASE" --rank
fi

# Normal run
PHASE="${1:-}"
FAMILY="${2:-}"
ITERATION="${3:-1}"

if [[ -z "$PHASE" ]]; then
    echo "Usage: $0 <phase> <family_name|--all-families> [iteration]"
    echo "       $0 --status <phase>"
    echo "       $0 --rank <phase>"
    exit 1
fi

if [[ "$FAMILY" == "--all-families" ]]; then
    exec "$PYTHON" -u "$RUNNER" --phase "$PHASE" --all-families
elif [[ -n "$FAMILY" ]]; then
    ARGS=(--phase "$PHASE" --family "$FAMILY")
    if [[ "$PHASE" == "3" ]]; then
        ARGS+=(--iteration "$ITERATION")
    fi
    exec "$PYTHON" -u "$RUNNER" "${ARGS[@]}"
else
    echo "Specify family name or --all-families"
    exit 1
fi
