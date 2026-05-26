"""
Crucible Exp1: Differential Fuzzing
Tests HypeReca's indexPut/indexGet/indexCopy kernels on sm86 vs sm90.
"""

import argparse
import json
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from crucible.fuzzer.kernel_fuzzer import (
    CrucibleFuzzer, BoundaryTensorGenerator, KernelConfig
)


# ── Simulated HypeReca index kernels ──

def kernel_index_get(input_tensor: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """Simulates HypeReca's indexGet kernel: sparse embedding lookup."""
    indices = torch.randint(0, input_tensor.shape[0],
                           (min(config.tensor_shape[0], input_tensor.shape[0]),),
                           device=input_tensor.device)
    return input_tensor[indices]


def kernel_index_put(input_tensor: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """Simulates HypeReca's indexPut kernel: scatter update."""
    output = input_tensor.clone()
    indices = torch.randint(0, output.shape[0],
                           (min(config.tensor_shape[0], output.shape[0]),),
                           device=output.device)
    values = torch.randn(indices.shape[0], output.shape[1], device=output.device)
    output[indices] = values
    return output


def kernel_index_copy(input_tensor: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """Simulates HypeReca's indexCopy kernel: gather-scatter."""
    src_idx = torch.randint(0, input_tensor.shape[0],
                           (min(config.tensor_shape[0], input_tensor.shape[0]),),
                           device=input_tensor.device)
    output = input_tensor.clone()
    dst_idx = torch.randint(0, output.shape[0], (src_idx.shape[0],),
                           device=output.device)
    output[dst_idx] = input_tensor[src_idx]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-configs', type=int, default=500)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        print("[ERROR] Need at least 2 GPUs (sm86 + sm90)")
        print(f"  Found {num_gpus} GPUs")
        # Run in simulation mode
        print("  Running in simulation mode...")
    
    device_a = torch.device('cuda:0') if num_gpus >= 1 else torch.device('cpu')
    device_b = torch.device('cuda:1') if num_gpus >= 2 else torch.device('cpu')
    
    if num_gpus >= 1:
        for i in range(num_gpus):
            cap = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {name} (sm{cap[0]}{cap[1]})")
    
    kernels = {
        'indexGet': kernel_index_get,
        'indexPut': kernel_index_put,
        'indexCopy': kernel_index_copy,
    }
    
    all_reports = {}
    
    for kernel_name, kernel_fn in kernels.items():
        print(f"\n  ═══ Fuzzing kernel: {kernel_name} ═══")
        fuzzer = CrucibleFuzzer(device_a, device_b)
        report = fuzzer.run_campaign(kernel_fn, num_configs=args.num_configs // 3)
        all_reports[kernel_name] = report
        
        print(f"  Results: {report['divergences_found']}/{report['total_tests']} divergences "
              f"({report['divergence_rate']*100:.1f}%)")
        print(f"  High-influence params: {report['high_influence_params']}")
    
    # Aggregate
    total_divergences = sum(r['divergences_found'] for r in all_reports.values())
    total_tests = sum(r['total_tests'] for r in all_reports.values())
    
    output = {
        'summary': {
            'total_tests': total_tests,
            'total_divergences': total_divergences,
            'overall_divergence_rate': total_divergences / max(total_tests, 1),
        },
        'per_kernel': all_reports,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  ═══ Overall Summary ═══")
    print(f"  Total tests: {total_tests}")
    print(f"  Total divergences: {total_divergences}")
    print(f"  Divergence rate: {total_divergences/max(total_tests,1)*100:.1f}%")


if __name__ == '__main__':
    main()
