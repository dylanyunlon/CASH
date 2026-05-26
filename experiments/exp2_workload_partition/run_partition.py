"""
Crucible Exp2: Workload Partition
Load-aware scheduling vs static uniform across A6000×2 + H100×1.
"""

import argparse
import json
import torch
import numpy as np
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from crucible.scheduler.workload_partitioner import (
    DeviceCalibrator, LoadAwarePartitioner, WorkloadSpec, GPUProfile
)


def create_mock_profiles():
    """Create mock profiles when real GPUs aren't available."""
    return [
        GPUProfile(0, "H100-SIM", "sm90", 3350, 67, 80, 64,
                   embedding_lookup_throughput=5e6, gradient_update_throughput=2e6),
        GPUProfile(1, "A6000-SIM-0", "sm86", 768, 38.7, 48, 32,
                   embedding_lookup_throughput=2e6, gradient_update_throughput=1e6),
        GPUProfile(2, "A6000-SIM-1", "sm86", 768, 38.7, 48, 32,
                   embedding_lookup_throughput=2e6, gradient_update_throughput=1e6),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', type=int, default=500)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Calibrate devices
    if num_gpus >= 2:
        print("  Calibrating GPUs...")
        profiles = [DeviceCalibrator.calibrate(i) for i in range(num_gpus)]
    else:
        print("  Using simulated GPU profiles...")
        profiles = create_mock_profiles()
    
    for p in profiles:
        print(f"  GPU {p.device_id}: {p.device_name} ({p.arch}) | "
              f"Lookup: {p.embedding_lookup_throughput:.0f}/s | "
              f"Update: {p.gradient_update_throughput:.0f}/s")
    
    partitioner = LoadAwarePartitioner(profiles)
    
    # Sweep workload sizes
    results = {'workloads': []}
    
    workload_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    
    for num_ops in workload_sizes:
        workload = WorkloadSpec(
            num_lookups=num_ops,
            num_gradient_updates=num_ops // 2,
            embedding_dim=128,
            batch_size=4096,
            num_tables=26,
        )
        
        # Compare strategies
        uniform = partitioner.partition_uniform(workload)
        static = partitioner.partition_static(workload)
        dynamic = partitioner.partition_dynamic(workload)
        
        entry = {
            'num_ops': num_ops,
            'uniform': {
                'assignments': uniform.device_assignments,
                'estimated_time_ms': uniform.estimated_time_ms,
                'bottleneck': uniform.bottleneck_device,
                'balance': uniform.load_balance_ratio,
            },
            'load_aware_static': {
                'assignments': static.device_assignments,
                'estimated_time_ms': static.estimated_time_ms,
                'bottleneck': static.bottleneck_device,
                'balance': static.load_balance_ratio,
            },
            'load_aware_dynamic': {
                'assignments': dynamic.device_assignments,
                'estimated_time_ms': dynamic.estimated_time_ms,
                'bottleneck': dynamic.bottleneck_device,
                'balance': dynamic.load_balance_ratio,
            },
            'speedup_static_vs_uniform': uniform.estimated_time_ms / max(static.estimated_time_ms, 1e-6),
            'speedup_dynamic_vs_uniform': uniform.estimated_time_ms / max(dynamic.estimated_time_ms, 1e-6),
        }
        
        results['workloads'].append(entry)
        print(f"  Ops={num_ops:>7d}: Uniform={uniform.estimated_time_ms:.2f}ms | "
              f"Static={static.estimated_time_ms:.2f}ms | "
              f"Dynamic={dynamic.estimated_time_ms:.2f}ms | "
              f"Speedup={entry['speedup_static_vs_uniform']:.2f}×")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()
