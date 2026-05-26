#!/bin/bash
# Crucible Experiment 2: Workload Partition
# Load-aware scheduling across A6000×2 + H100×1 vs static uniform
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp2_workload_partition/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Crucible Exp2: Workload Partition                ║"
echo "╚═══════════════════════════════════════════════════╝"

python3 "$ROOT_DIR/experiments/exp2_workload_partition/run_partition.py" \
    --num-iters 500 \
    --output "$RESULTS_DIR/partition_report.json" \
    2>&1 | tee "$RESULTS_DIR/partition.log"

echo "[DONE] Results saved to $RESULTS_DIR/"
