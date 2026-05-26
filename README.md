# Crucible: Cross-Architecture Kernel Correctness Testing and Workload Scheduling for Heterogeneous GPU Clusters

> **Crucible** (坩锅): A vessel for testing materials under extreme conditions. Same kernel code may behave differently on sm86 (A6000) vs sm90 (H100) — Crucible puts them all in the furnace and finds what breaks.

## Overview

Crucible is a cross-architecture kernel validation and workload scheduling framework for heterogeneous GPU clusters. It automatically discovers silent correctness issues when the same CUDA kernel runs on different GPU architectures, and optimally partitions workloads across asymmetric hardware.

### Key Contributions

1. **Cross-Architecture Kernel Fuzzing**: Adapts SyzMini's influence-guided call removal from Linux syscall fuzzing to CUDA kernel launch parameter fuzzing. Automatically discovers kernel configurations (grid/block sizes, shared memory, data alignment) that are correct on sm86 but trigger silent corruption on sm90.

2. **Heterogeneous Workload Partitioning**: Extracts PIM-ANNS's per-core fine-grained scheduling philosophy to build a GPU workload partitioner that minimizes total completion time across H100 (high bandwidth, few) and A6000 (lower bandwidth, many).

3. **Correctness × Performance Pareto Analysis**: Quantifies the throughput cost of enforcing bitwise cross-architecture consistency vs relaxed consistency.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Crucible Runtime                    │
├────────────────────────┬─────────────────────────────┤
│  SyzMini Fuzzer Engine │  PIM-ANNS Scheduler Engine  │
│  (Correctness Testing) │  (Workload Partitioning)    │
├────────────────────────┴─────────────────────────────┤
│    sm86 (A6000×2)      │      sm90 (H100×1)          │
└────────────────────────┴─────────────────────────────┘
```

## Repository Structure

```
CASH/
├── core/
│   ├── syzmini/           # Cross-architecture kernel fuzzer (SyzMini)
│   └── pim_anns/          # Heterogeneous workload scheduler (PIM-ANNS)
├── crucible/
│   ├── fuzzer/            # CUDA kernel launch parameter fuzzer
│   ├── scheduler/         # Asymmetric GPU workload partitioner
│   ├── validator/         # Cross-architecture bitwise consistency checker
│   └── benchmarks/        # Micro-benchmarks
├── experiments/
│   ├── exp1_differential_fuzzing/   # sm86 vs sm90 kernel behavior diff
│   ├── exp2_workload_partition/     # Dynamic load-aware scheduling
│   └── exp3_pareto_frontier/        # Correctness × performance tradeoff
├── scripts/               # Launch & utility scripts
└── docs/                  # Documentation
```

## Hardware Requirements

- **Minimum**: 1× NVIDIA H100 (sm90) + 2× NVIDIA A6000 (sm86)
- **CUDA Toolkit**: ≥ 12.0 (multi-arch compilation support)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Experiment 1: Differential Fuzzing
bash scripts/run_exp1.sh

# Run Experiment 2: Workload Partition
bash scripts/run_exp2.sh

# Run Experiment 3: Pareto Frontier
bash scripts/run_exp3.sh
```

## Experiments

### Exp1: Differential Fuzzing
Compiles HypeReca's indexPut/indexGet/indexCopy kernels for sm86 and sm90. Uses SyzMini-style heuristics to generate boundary tensor shapes and detects architecture-dependent behavioral differences.

### Exp2: Workload Partition
Given embedding lookup + gradient update mixed workloads, applies PIM-ANNS-style load-aware scheduling across A6000×2 + H100×1. Compares against static uniform partitioning.

### Exp3: Correctness × Performance Pareto Frontier
Maps the Pareto frontier between enforcing bitwise cross-architecture consistency (maximum correctness) and relaxing it (maximum throughput).

## Related Work

- **Alloy** ([MPAlloy](https://github.com/dylanyunlon/MPAlloy)): Mixed-precision elastic embedding training system — Crucible validates Alloy's kernels across architectures and provides optimized workload partitions.

## Target Venue

ASPLOS / ISCA / ATC

## License

See individual component licenses in `core/*/LICENSE`.
