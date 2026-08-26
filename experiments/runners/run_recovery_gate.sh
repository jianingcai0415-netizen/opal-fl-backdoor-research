#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-experiments/configs/stage5r_subspace_writein/stage5r_layer4_linear_recovery40.yaml}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/runs}"
RUN_ID="${RUN_ID:-opal_recovery_gate_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

if [ "${1:-}" = "--dry-run" ]; then
    echo "$PYTHON_BIN main.py --params $CONFIG"
    echo "run_id=$RUN_ID"
    echo "run_dir=$LOG_ROOT/$RUN_ID"
    exit 0
fi

if [ ! -f "main.py" ]; then
    echo "This showcase repo contains protocol code and configs, not the full private training tree." >&2
    echo "Set PROJECT_DIR to a full FCBA-compatible training checkout before launching a GPU run." >&2
    exit 2
fi

if [ ! -f "$CONFIG" ]; then
    echo "Missing config: $CONFIG" >&2
    exit 3
fi

mkdir -p "$LOG_ROOT/$RUN_ID"
"$PYTHON_BIN" main.py --params "$CONFIG" 2>&1 | tee "$LOG_ROOT/$RUN_ID/train.log"

