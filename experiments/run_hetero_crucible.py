#!/usr/bin/env python3
"""
Crucible Heterogeneous Cluster Integration Test
================================================
Target: ags1 server (A6000×2 sm86 + H100 NVL sm90)

Runs all three CASH experiments + regression tests for C001-C003 fixes.

Usage:
  numactl --cpunodebind=1 --membind=1 python3 experiments/run_hetero_crucible.py --output results/full.json
  numactl --cpunodebind=1 --membind=1 python3 experiments/run_hetero_crucible.py --quick --output results/smoke.json
"""

import argparse
import json
import os
import sys
import time
import gc
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from crucible.fuzzer.kernel_fuzzer import (
    CrucibleFuzzer, BoundaryTensorGenerator, KernelConfig, InfluenceGuide
)
from crucible.scheduler.workload_partitioner import (
    LoadAwarePartitioner, GPUProfile, WorkloadSpec, DeviceCalibrator,
    BitwiseConsistencyChecker
)


# ═══════════════════════════════════════════════════════════════════
#  Hardware Discovery
# ═══════════════════════════════════════════════════════════════════

def discover_hardware() -> Dict:
    hw = {
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': 0,
        'gpus': [],
        'warnings': [],
    }
    if not torch.cuda.is_available():
        hw['warnings'].append("CUDA not available")
        return hw

    hw['num_gpus'] = torch.cuda.device_count()
    for i in range(hw['num_gpus']):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        hw['gpus'].append({
            'index': i, 'name': name,
            'mem_gb': round(mem, 1),
            'sm': f"sm{cap[0]}{cap[1]}",
            'tier': 'sm90' if cap[0] >= 9 else 'sm86',
        })
    return hw


# ═══════════════════════════════════════════════════════════════════
#  Exp1: Differential Fuzzing (sm86 vs sm90)
# ═══════════════════════════════════════════════════════════════════

def simple_embedding_kernel(input_data: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """Simulated embedding kernel for fuzzing: indexing + scatter."""
    n = input_data.shape[0]
    dim = config.tensor_shape[1] if len(config.tensor_shape) > 1 else 128
    table = torch.randn(max(n, 1000), dim, device=input_data.device)
    indices = (input_data[:, 0].abs() * 999).long().clamp(0, 999)
    return table[indices]


def run_exp1_fuzzing(hw: Dict, quick: bool = False) -> Dict:
    print("\n" + "="*60)
    print("  CASH EXP1: Differential Fuzzing (sm86 vs sm90)")
    print("="*60)

    num_configs = 30 if quick else 200
    gpus = hw.get('gpus', [])
    sm86_devs = [g['index'] for g in gpus if g['tier'] == 'sm86']
    sm90_devs = [g['index'] for g in gpus if g['tier'] == 'sm90']

    if not sm86_devs or not sm90_devs:
        print("  Need at least 1 sm86 + 1 sm90 GPU. Running CPU mock.")
        dev_a = torch.device('cpu')
        dev_b = torch.device('cpu')
    else:
        dev_a = torch.device(f'cuda:{sm86_devs[0]}')
        dev_b = torch.device(f'cuda:{sm90_devs[0]}')

    fuzzer = CrucibleFuzzer(dev_a, dev_b)
    report = fuzzer.run_campaign(simple_embedding_kernel, num_configs=num_configs)

    print(f"\n  Divergences: {report['divergences_found']}/{report['total_tests']}")
    print(f"  High-influence params: {report['high_influence_params']}")

    return report


# ═══════════════════════════════════════════════════════════════════
#  Exp2: Workload Partition (static vs dynamic vs uniform)
# ═══════════════════════════════════════════════════════════════════

def run_exp2_partition(hw: Dict, quick: bool = False) -> Dict:
    print("\n" + "="*60)
    print("  CASH EXP2: Workload Partition")
    print("="*60)

    gpus = hw.get('gpus', [])
    profiles = []

    for g in gpus:
        idx = g['index']
        if torch.cuda.is_available():
            try:
                p = DeviceCalibrator.calibrate(idx, num_trials=10 if quick else 50)
                profiles.append(p)
                print(f"  GPU {idx} ({g['name']}): lookup={p.embedding_lookup_throughput:.0f}/s, "
                      f"update={p.gradient_update_throughput:.0f}/s")
            except Exception as e:
                print(f"  GPU {idx}: calibration failed ({e}), using defaults")
                profiles.append(GPUProfile(
                    device_id=idx, device_name=g['name'],
                    arch=g['sm'], memory_bandwidth_gbps=768 if g['tier'] == 'sm86' else 3350,
                    compute_tflops=38.7 if g['tier'] == 'sm86' else 67.0,
                    memory_total_gb=g['mem_gb'], pcie_bandwidth_gbps=32.0 if g['tier'] == 'sm86' else 64.0,
                    embedding_lookup_throughput=1e6, gradient_update_throughput=5e5,
                ))

    if not profiles:
        profiles = [
            GPUProfile(0, "mock_A6000", "sm86", 768, 38.7, 48, 32, 1e6, 5e5),
            GPUProfile(1, "mock_A6000", "sm86", 768, 38.7, 48, 32, 1e6, 5e5),
            GPUProfile(2, "mock_H100", "sm90", 3350, 67.0, 96, 64, 3e6, 1.5e6),
        ]

    partitioner = LoadAwarePartitioner(profiles)

    workloads = [
        WorkloadSpec(100_000, 50_000, 128, 4096, 26),
        WorkloadSpec(1_000_000, 500_000, 128, 65536, 26),
        WorkloadSpec(10_000_000, 5_000_000, 128, 65536, 26),
    ]

    results = {'workloads': []}
    for wl in workloads:
        static = partitioner.partition_static(wl)
        dynamic = partitioner.partition_dynamic(wl)
        uniform = partitioner.partition_uniform(wl)

        entry = {
            'num_lookups': wl.num_lookups,
            'static': {
                'assignments': {str(k): round(v, 4) for k, v in static.device_assignments.items()},
                'estimated_ms': round(static.estimated_time_ms, 3),
                'balance': round(static.load_balance_ratio, 4),
            },
            'dynamic': {
                'assignments': {str(k): round(v, 4) for k, v in dynamic.device_assignments.items()},
                'estimated_ms': round(dynamic.estimated_time_ms, 3),
                'balance': round(dynamic.load_balance_ratio, 4),
            },
            'uniform': {
                'assignments': {str(k): round(v, 4) for k, v in uniform.device_assignments.items()},
                'estimated_ms': round(uniform.estimated_time_ms, 3),
                'balance': round(uniform.load_balance_ratio, 4),
            },
            'speedup_static_over_uniform': round(
                uniform.estimated_time_ms / max(static.estimated_time_ms, 1e-9), 3),
        }
        results['workloads'].append(entry)
        print(f"  Lookups={wl.num_lookups}: static={static.estimated_time_ms:.1f}ms "
              f"uniform={uniform.estimated_time_ms:.1f}ms "
              f"speedup={entry['speedup_static_over_uniform']:.2f}x")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Exp3: Pareto Frontier (Correctness × Performance)
# ═══════════════════════════════════════════════════════════════════

def run_exp3_pareto(hw: Dict, quick: bool = False) -> Dict:
    print("\n" + "="*60)
    print("  CASH EXP3: Correctness × Performance Pareto Frontier")
    print("="*60)

    num_iters = 50 if quick else 200
    gpus = hw.get('gpus', [])
    num_gpus = len(gpus)

    dev_a = torch.device('cuda:0') if num_gpus >= 1 else torch.device('cpu')
    dev_b = torch.device(f'cuda:{num_gpus-1}') if num_gpus >= 2 else torch.device('cpu')

    tolerances = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    checker = BitwiseConsistencyChecker()

    embedding_dim = 128
    num_embeddings = 100_000
    batch_size = 4096

    results = {'points': []}

    for tol in tolerances:
        table_a = torch.randn(num_embeddings, embedding_dim, device=dev_a)
        table_b = table_a.to(dev_b)

        throughputs = []
        consistency_ok_count = 0

        for i in range(num_iters):
            indices = torch.randint(0, num_embeddings, (batch_size,))
            idx_a = indices.to(dev_a)
            idx_b = indices.to(dev_b)

            if dev_a.type == 'cuda':
                torch.cuda.synchronize(dev_a)
            if dev_b.type == 'cuda':
                torch.cuda.synchronize(dev_b)

            t0 = time.perf_counter()
            out_a = table_a[idx_a]
            out_b = table_b[idx_b]
            if dev_a.type == 'cuda':
                torch.cuda.synchronize(dev_a)
            if dev_b.type == 'cuda':
                torch.cuda.synchronize(dev_b)
            elapsed = time.perf_counter() - t0

            if tol == 0:
                check = checker.check_bitwise(out_a, out_b)
                ok = check['bitwise_identical']
            else:
                check = checker.check_with_tolerance(out_a, out_b, atol=tol)
                ok = check.get('within_tolerance', True)

            if ok:
                consistency_ok_count += 1

            sync_overhead = 0
            if not ok:
                t1 = time.perf_counter()
                table_b = table_a.to(dev_b)
                if dev_b.type == 'cuda':
                    torch.cuda.synchronize(dev_b)
                sync_overhead = time.perf_counter() - t1

            throughputs.append(batch_size / (elapsed + sync_overhead))

        point = {
            'tolerance': tol,
            'mean_throughput': round(float(np.mean(throughputs)), 1),
            'p50_throughput': round(float(np.median(throughputs)), 1),
            'consistency_rate': round(consistency_ok_count / num_iters, 4),
        }
        results['points'].append(point)
        print(f"  tol={tol:.0e}: throughput={point['mean_throughput']:.0f}/s "
              f"consistency={point['consistency_rate']:.1%}")

        del table_a, table_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════
#  C001-C003 Regression Tests
# ═══════════════════════════════════════════════════════════════════

def run_regression(hw: Dict) -> Dict:
    print("\n" + "="*60)
    print("  C001-C003 Regression Tests")
    print("="*60)

    results = {'tests': {}}

    # C001: LoadAwarePartitioner with zero-throughput profiles
    print("\n  C001: Zero-throughput guard...")
    try:
        zero_profiles = [
            GPUProfile(0, "uncalibrated", "sm86", 768, 38.7, 48, 32, 0.0, 0.0),
            GPUProfile(1, "uncalibrated", "sm90", 3350, 67.0, 96, 64, 0.0, 0.0),
        ]
        partitioner = LoadAwarePartitioner(zero_profiles)
        wl = WorkloadSpec(100_000, 50_000, 128, 4096, 26)
        plan = partitioner.partition_static(wl)
        ok = plan.estimated_time_ms > 0 and not any(
            v != v for v in plan.device_assignments.values())  # no NaN
        results['tests']['c001_zero_throughput'] = {'passed': ok}
        print(f"    Zero-throughput partitioning: {'✓' if ok else '✗'}")
    except Exception as e:
        results['tests']['c001_zero_throughput'] = {'passed': False, 'error': str(e)}
        print(f"    ✗ Exception: {e}")

    # C002: partition_dynamic with extreme utilization
    print("  C002: Extreme utilization guard...")
    try:
        profiles = [
            GPUProfile(0, "A6000", "sm86", 768, 38.7, 48, 32, 1e6, 5e5),
            GPUProfile(1, "H100", "sm90", 3350, 67.0, 96, 64, 3e6, 1.5e6),
        ]
        partitioner = LoadAwarePartitioner(profiles)
        wl = WorkloadSpec(100_000, 50_000, 128, 4096, 26)
        # Extreme: all devices at 200% utilization → adjustment goes negative
        extreme_util = {0: 2.0, 1: 2.0}
        plan = partitioner.partition_dynamic(wl, current_utilization=extreme_util)
        total_frac = sum(plan.device_assignments.values())
        ok = abs(total_frac - 1.0) < 1e-6 and plan.estimated_time_ms > 0
        results['tests']['c002_extreme_util'] = {'passed': ok, 'total_frac': total_frac}
        print(f"    Extreme utilization: total_frac={total_frac:.6f} {'✓' if ok else '✗'}")
    except Exception as e:
        results['tests']['c002_extreme_util'] = {'passed': False, 'error': str(e)}
        print(f"    ✗ Exception: {e}")

    # C003: fuzz_kernel shape mismatch handling
    print("  C003: Shape mismatch in fuzz_kernel...")
    try:
        def bad_kernel(input_data, config):
            # Returns different shapes depending on device
            if input_data.device.type == 'cpu':
                return torch.randn(10, 128)
            else:
                return torch.randn(10, 128, device=input_data.device)

        # Both on CPU for testing (same shape → no divergence, but proves no crash)
        fuzzer = CrucibleFuzzer(torch.device('cpu'), torch.device('cpu'))

        def shape_mismatch_kernel(input_data, config):
            # Deliberately return different shapes
            n = input_data.shape[0]
            if hasattr(shape_mismatch_kernel, '_call_count'):
                shape_mismatch_kernel._call_count += 1
            else:
                shape_mismatch_kernel._call_count = 1
            # Alternate: first call returns (n, 64), second returns (n, 128)
            dim = 64 if shape_mismatch_kernel._call_count % 2 == 1 else 128
            return torch.randn(n, dim)

        # Monkey-patch to force shape difference
        cfg = KernelConfig(
            grid_dim=(1, 1, 1), block_dim=(32, 1, 1),
            shared_mem_bytes=0, data_alignment=16,
            tensor_shape=(16, 128), dtype=torch.float32
        )
        input_data = torch.randn(16, 128)
        result = fuzzer.fuzz_kernel(shape_mismatch_kernel, cfg, input_data)
        ok = result.is_divergent and "Shape mismatch" in (result.error or "")
        results['tests']['c003_shape_mismatch'] = {'passed': ok}
        print(f"    Shape mismatch detection: {'✓' if ok else '✗'}")
    except Exception as e:
        results['tests']['c003_shape_mismatch'] = {'passed': False, 'error': str(e)}
        print(f"    ✗ Exception: {e}")

    all_passed = all(t.get('passed', False) for t in results['tests'].values())
    results['all_passed'] = all_passed
    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Crucible Hetero Integration Test')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'exp1', 'exp2', 'exp3', 'regression'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "="*60)
    print("  HARDWARE DISCOVERY")
    print("="*60)
    hw = discover_hardware()
    for g in hw['gpus']:
        print(f"  GPU {g['index']}: {g['name']} ({g['mem_gb']}GB, {g['sm']})")

    report = {'hardware': hw, 'args': vars(args), 'experiments': {}}

    try:
        if args.exp in ('all', 'regression'):
            report['experiments']['regression'] = run_regression(hw)
        if args.exp in ('all', 'exp1'):
            report['experiments']['exp1_fuzzing'] = run_exp1_fuzzing(hw, args.quick)
        if args.exp in ('all', 'exp2'):
            report['experiments']['exp2_partition'] = run_exp2_partition(hw, args.quick)
        if args.exp in ('all', 'exp3'):
            report['experiments']['exp3_pareto'] = run_exp3_pareto(hw, args.quick)
    except Exception as e:
        report['error'] = {'message': str(e), 'traceback': traceback.format_exc()}
        print(f"\n  ERROR: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
