"""
Crucible Exp3: Correctness × Performance Pareto Frontier
Measures throughput under varying cross-architecture consistency constraints.
"""

import argparse
import json
import torch
import numpy as np
import time

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from crucible.scheduler.workload_partitioner import BitwiseConsistencyChecker


def run_with_constraint(tolerance: float, num_iters: int, device_a, device_b):
    """Run embedding workload with a given consistency tolerance."""
    embedding_dim = 128
    num_embeddings = 100000
    batch_size = 4096
    
    table_a = torch.randn(num_embeddings, embedding_dim, device=device_a)
    table_b = table_a.to(device_b)
    
    checker = BitwiseConsistencyChecker()
    
    throughputs = []
    consistency_checks = []
    
    for i in range(num_iters):
        indices = torch.randint(0, num_embeddings, (batch_size,))
        
        start = time.perf_counter()
        
        # Lookup on both devices
        idx_a = indices.to(device_a)
        idx_b = indices.to(device_b)
        out_a = table_a[idx_a]
        out_b = table_b[idx_b]
        
        if device_a.type == 'cuda':
            torch.cuda.synchronize(device_a)
        if device_b.type == 'cuda':
            torch.cuda.synchronize(device_b)
        
        elapsed = time.perf_counter() - start
        
        # Check consistency if enforcing strict constraints
        if tolerance == 0:
            result = checker.check_bitwise(out_a, out_b)
        else:
            result = checker.check_with_tolerance(out_a, out_b, atol=tolerance)
        
        # If not within tolerance, "re-sync" (simulated overhead)
        sync_overhead = 0
        if not result.get('within_tolerance', result.get('bitwise_identical', True)):
            sync_start = time.perf_counter()
            table_b = table_a.to(device_b)  # Re-sync
            if device_b.type == 'cuda':
                torch.cuda.synchronize(device_b)
            sync_overhead = time.perf_counter() - sync_start
        
        total_time = elapsed + sync_overhead
        throughputs.append(batch_size / total_time)
        
        if i % 50 == 0:
            consistency_checks.append({
                'step': i,
                'consistent': result.get('within_tolerance', result.get('bitwise_identical')),
                'max_abs_diff': result.get('max_abs_diff', 0),
            })
    
    del table_a, table_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'tolerance': tolerance,
        'mean_throughput': float(np.mean(throughputs)),
        'p50_throughput': float(np.median(throughputs)),
        'p99_throughput': float(np.percentile(throughputs, 1)),  # Low tail
        'consistency_checks': consistency_checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', type=float, required=True)
    parser.add_argument('--num-iters', type=int, default=200)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_a = torch.device('cuda:0') if num_gpus >= 1 else torch.device('cpu')
    device_b = torch.device('cuda:1') if num_gpus >= 2 else torch.device('cpu')
    
    print(f"  Tolerance: {args.tolerance} | Iters: {args.num_iters}")
    print(f"  Device A: {device_a} | Device B: {device_b}")
    
    result = run_with_constraint(args.tolerance, args.num_iters, device_a, device_b)
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Mean throughput: {result['mean_throughput']:.0f} samples/s")


if __name__ == '__main__':
    main()
