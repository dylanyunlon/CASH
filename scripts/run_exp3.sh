#!/bin/bash
# Crucible Experiment 3: Correctness × Performance Pareto Frontier
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp3_pareto_frontier/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Crucible Exp3: Pareto Frontier                   ║"
echo "╚═══════════════════════════════════════════════════╝"

for TOLERANCE in 0 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2; do
    echo "━━━ Tolerance: $TOLERANCE ━━━"
    python3 "$ROOT_DIR/experiments/exp3_pareto_frontier/run_pareto.py" \
        --tolerance $TOLERANCE \
        --num-iters 200 \
        --output "$RESULTS_DIR/pareto_tol_${TOLERANCE}.json" \
        2>&1 | tee "$RESULTS_DIR/pareto_tol_${TOLERANCE}.log"
done

python3 "$ROOT_DIR/experiments/exp3_pareto_frontier/plot_pareto.py" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR"

echo "[DONE] Results saved to $RESULTS_DIR/"
