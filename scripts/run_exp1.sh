#!/bin/bash
# Crucible Experiment 1: Differential Fuzzing
# Tests HypeReca's index kernels on sm86 vs sm90
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp1_differential_fuzzing/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Crucible Exp1: Differential Fuzzing              ║"
echo "╚═══════════════════════════════════════════════════╝"

python3 "$ROOT_DIR/experiments/exp1_differential_fuzzing/run_fuzzing.py" \
    --num-configs 500 \
    --output "$RESULTS_DIR/fuzzing_report.json" \
    2>&1 | tee "$RESULTS_DIR/fuzzing.log"

echo "[DONE] Results saved to $RESULTS_DIR/"
