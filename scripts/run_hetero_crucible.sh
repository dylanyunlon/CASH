#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Crucible Heterogeneous Cluster Test — Launch Script
#  Target: ags1 (A6000×2 sm86 + H100 NVL sm90 + EPYC 9354×2)
#
#  Usage:
#    bash scripts/run_hetero_crucible.sh              # Full run
#    bash scripts/run_hetero_crucible.sh --quick       # Smoke test
#    bash scripts/run_hetero_crucible.sh --exp exp1    # Fuzzing only
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${REPO_ROOT}/results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "═══════════════════════════════════════════════════"
echo "  Crucible Cross-Architecture Test"
echo "  Repo:    ${REPO_ROOT}"
echo "  Results: ${RESULTS_DIR}"
echo "  Time:    $(date)"
echo "═══════════════════════════════════════════════════"

# ── 1. Hardware probe ──
echo ""
echo ">>> GPU Architecture Detection..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,compute_cap \
        --format=csv,noheader 2>/dev/null || true

    echo ""
    echo ">>> Cross-GPU Topology:"
    nvidia-smi topo -m 2>/dev/null | head -12 || true
fi

# ── 2. Validate sm86 + sm90 ──
echo ""
SM86_COUNT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -c "8.6" || echo 0)
SM90_COUNT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -c "9.0" || echo 0)
echo "  sm86 (A6000) count: ${SM86_COUNT}"
echo "  sm90 (H100)  count: ${SM90_COUNT}"

if [ "$SM86_COUNT" -eq 0 ] || [ "$SM90_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠  Cross-architecture fuzzing requires at least 1 sm86 + 1 sm90."
    echo "   Running in degraded mode (same-arch comparison)."
fi

# ── 3. NUMA pinning — all GPUs on NUMA node 1 ──
NUMA_NODE=1
if command -v numactl &>/dev/null; then
    NUMA_PREFIX="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    echo ""
    echo ">>> NUMA pinning: node ${NUMA_NODE}"
else
    NUMA_PREFIX=""
fi

# ── 4. Environment ──
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo ""
echo ">>> Python environment:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || {
    echo "  ERROR: PyTorch not found"; exit 1
}

# ── 5. Run ──
echo ""
echo ">>> Launching Crucible experiments..."
EXTRA_ARGS="$@"

${NUMA_PREFIX} python3 "${REPO_ROOT}/experiments/run_hetero_crucible.py" \
    --output "${RESULTS_DIR}/results.json" \
    --seed 42 \
    ${EXTRA_ARGS}

EXIT_CODE=$?

# ── 6. Summary ──
echo ""
echo "═══════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ Experiment completed"
else
    echo "  ✗ Experiment failed (exit code ${EXIT_CODE})"
fi
echo "  Results: ${RESULTS_DIR}/results.json"
echo "═══════════════════════════════════════════════════"

python3 -c "
import json
try:
    with open('${RESULTS_DIR}/results.json') as f:
        r = json.load(f)
    exps = r.get('experiments', {})
    if 'regression' in exps:
        reg = exps['regression']
        p = sum(1 for t in reg['tests'].values() if t.get('passed'))
        print(f'  Regression: {p}/{len(reg[\"tests\"])} passed')
    if 'exp1_fuzzing' in exps:
        e1 = exps['exp1_fuzzing']
        print(f'  Exp1 Fuzzing: {e1.get(\"divergences_found\",\"?\")}/{e1.get(\"total_tests\",\"?\")} divergences')
    if 'exp2_partition' in exps:
        e2 = exps['exp2_partition']
        for w in e2.get('workloads', [])[:1]:
            print(f'  Exp2 Partition speedup: {w.get(\"speedup_static_over_uniform\",\"?\")}x')
    if 'exp3_pareto' in exps:
        pts = exps['exp3_pareto'].get('points', [])
        if pts:
            print(f'  Exp3 Pareto: {len(pts)} tolerance levels swept')
except Exception as e:
    print(f'  (parse error: {e})')
" 2>/dev/null || true

exit $EXIT_CODE
