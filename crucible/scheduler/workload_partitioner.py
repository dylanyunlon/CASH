"""
Crucible Heterogeneous GPU Workload Partitioner
Adapts PIM-ANNS's per-core fine-grained scheduling for asymmetric GPU clusters.
Decides optimal workload split between H100 (high bandwidth, few) and A6000 (lower bandwidth, many).
"""

import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUProfile:
    """Performance profile of a GPU device."""
    device_id: int
    device_name: str
    arch: str  # "sm86" or "sm90"
    memory_bandwidth_gbps: float
    compute_tflops: float  # FP32 TFLOPS
    memory_total_gb: float
    pcie_bandwidth_gbps: float
    
    # Measured (populated via calibration)
    embedding_lookup_throughput: float = 0.0   # lookups/sec
    gradient_update_throughput: float = 0.0     # updates/sec
    allreduce_latency_us: float = 0.0           # microseconds


@dataclass
class WorkloadSpec:
    """Specification of a mixed embedding workload."""
    num_lookups: int
    num_gradient_updates: int
    embedding_dim: int
    batch_size: int
    num_tables: int
    

@dataclass
class PartitionPlan:
    """How to split workload across devices."""
    device_assignments: Dict[int, float]  # device_id → fraction of workload
    estimated_time_ms: float
    bottleneck_device: int
    load_balance_ratio: float  # 1.0 = perfectly balanced


class DeviceCalibrator:
    """
    Calibrates GPU performance by running micro-benchmarks.
    Produces GPUProfile with measured throughput numbers.
    """
    
    @staticmethod
    def calibrate(device_id: int, embedding_dim: int = 128,
                  num_embeddings: int = 100000, num_trials: int = 50) -> GPUProfile:
        """Run calibration benchmarks on a GPU."""
        device = torch.device(f'cuda:{device_id}')
        name = torch.cuda.get_device_name(device_id)
        props = torch.cuda.get_device_properties(device_id)
        cap = torch.cuda.get_device_capability(device_id)
        arch = f"sm{cap[0]}{cap[1]}"
        
        # Approximate specs (can be refined with actual measurements)
        memory_bw = {
            'sm86': 768,    # A6000: 768 GB/s GDDR6X
            'sm90': 3350,   # H100: 3.35 TB/s HBM3
        }.get(f"sm{cap[0]}0", 500)
        
        compute = {
            'sm86': 38.7,   # A6000: 38.7 TFLOPS FP32
            'sm90': 67.0,   # H100: 67 TFLOPS FP32
        }.get(f"sm{cap[0]}0", 20)
        
        pcie_bw = 32.0 if cap[0] < 9 else 64.0  # Gen4 vs Gen5
        
        # ── Measure embedding lookup throughput ──
        table = torch.randn(num_embeddings, embedding_dim, device=device)
        batch_size = 4096
        
        lookup_times = []
        for _ in range(num_trials):
            indices = torch.randint(0, num_embeddings, (batch_size,), device=device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = table[indices]
            torch.cuda.synchronize(device)
            lookup_times.append(time.perf_counter() - start)
        
        lookup_throughput = batch_size / np.mean(lookup_times)
        
        # ── Measure gradient update throughput ──
        optimizer = torch.optim.SGD([table.requires_grad_(True)], lr=0.01)
        
        update_times = []
        for _ in range(num_trials):
            indices = torch.randint(0, num_embeddings, (batch_size,), device=device)
            output = table[indices].sum()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize(device)
            update_times.append(time.perf_counter() - start)
        
        update_throughput = batch_size / np.mean(update_times)
        
        del table
        torch.cuda.empty_cache()
        
        profile = GPUProfile(
            device_id=device_id,
            device_name=name,
            arch=arch,
            memory_bandwidth_gbps=memory_bw,
            compute_tflops=compute,
            memory_total_gb=props.total_mem / (1024**3),
            pcie_bandwidth_gbps=pcie_bw,
            embedding_lookup_throughput=lookup_throughput,
            gradient_update_throughput=update_throughput,
        )
        
        logger.info(f"Calibrated GPU:{device_id} ({name}): "
                    f"lookup={lookup_throughput:.0f}/s, update={update_throughput:.0f}/s")
        
        return profile


class LoadAwarePartitioner:
    """
    Partitions workloads across heterogeneous GPUs to minimize total completion time.
    Uses a cost model based on calibrated device profiles.
    
    Core idea from PIM-ANNS: assign work proportional to each device's throughput,
    not equally. For asymmetric hardware, this is critical.
    """
    
    def __init__(self, profiles: List[GPUProfile]):
        self.profiles = {p.device_id: p for p in profiles}
        self.total_lookup_throughput = sum(p.embedding_lookup_throughput for p in profiles)
        self.total_update_throughput = sum(p.gradient_update_throughput for p in profiles)
    
    def partition_static(self, workload: WorkloadSpec) -> PartitionPlan:
        """
        Static partitioning: assign work proportional to measured throughput.
        This is the simplest strategy — ignores cross-device communication.
        """
        assignments = {}
        estimated_times = {}
        
        for dev_id, profile in self.profiles.items():
            # Fraction based on lookup throughput (dominant operation)
            frac = profile.embedding_lookup_throughput / self.total_lookup_throughput
            assignments[dev_id] = frac
            
            # Estimated time for this device's share
            lookups = workload.num_lookups * frac
            updates = workload.num_gradient_updates * frac
            t_lookup = lookups / profile.embedding_lookup_throughput
            t_update = updates / profile.gradient_update_throughput
            estimated_times[dev_id] = (t_lookup + t_update) * 1000  # ms
        
        # Total time is bottleneck (max across devices)
        bottleneck = max(estimated_times, key=estimated_times.get)
        total_time = estimated_times[bottleneck]
        
        # Load balance ratio: min_time / max_time (1.0 = perfect)
        times = list(estimated_times.values())
        balance = min(times) / max(times) if max(times) > 0 else 1.0
        
        return PartitionPlan(
            device_assignments=assignments,
            estimated_time_ms=total_time,
            bottleneck_device=bottleneck,
            load_balance_ratio=balance
        )
    
    def partition_dynamic(self, workload: WorkloadSpec,
                          current_utilization: Optional[Dict[int, float]] = None
                          ) -> PartitionPlan:
        """
        Dynamic partitioning: adjusts based on current device utilization.
        Re-balances when one device is more loaded than others.
        """
        assignments = {}
        
        for dev_id, profile in self.profiles.items():
            base_frac = profile.embedding_lookup_throughput / self.total_lookup_throughput
            
            if current_utilization and dev_id in current_utilization:
                # Reduce assignment for heavily utilized devices
                util = current_utilization[dev_id]
                adjustment = 1.0 - (util - 0.5) * 0.5  # Scale down if util > 50%
                adjusted_frac = base_frac * max(0.1, adjustment)
            else:
                adjusted_frac = base_frac
            
            assignments[dev_id] = adjusted_frac
        
        # Normalize
        total = sum(assignments.values())
        assignments = {k: v / total for k, v in assignments.items()}
        
        # Estimate time
        estimated_times = {}
        for dev_id, frac in assignments.items():
            profile = self.profiles[dev_id]
            t = (workload.num_lookups * frac / profile.embedding_lookup_throughput +
                 workload.num_gradient_updates * frac / profile.gradient_update_throughput) * 1000
            estimated_times[dev_id] = t
        
        bottleneck = max(estimated_times, key=estimated_times.get)
        times = list(estimated_times.values())
        
        return PartitionPlan(
            device_assignments=assignments,
            estimated_time_ms=estimated_times[bottleneck],
            bottleneck_device=bottleneck,
            load_balance_ratio=min(times) / max(times) if max(times) > 0 else 1.0
        )
    
    def partition_uniform(self, workload: WorkloadSpec) -> PartitionPlan:
        """Uniform partitioning baseline (equal split regardless of capability)."""
        n = len(self.profiles)
        assignments = {dev_id: 1.0 / n for dev_id in self.profiles}
        
        estimated_times = {}
        for dev_id, frac in assignments.items():
            profile = self.profiles[dev_id]
            t = (workload.num_lookups * frac / profile.embedding_lookup_throughput +
                 workload.num_gradient_updates * frac / profile.gradient_update_throughput) * 1000
            estimated_times[dev_id] = t
        
        bottleneck = max(estimated_times, key=estimated_times.get)
        times = list(estimated_times.values())
        
        return PartitionPlan(
            device_assignments=assignments,
            estimated_time_ms=estimated_times[bottleneck],
            bottleneck_device=bottleneck,
            load_balance_ratio=min(times) / max(times) if max(times) > 0 else 1.0
        )


class BitwiseConsistencyChecker:
    """
    Checks bitwise consistency of kernel outputs across architectures.
    Used in Exp3 to define the correctness × performance Pareto frontier.
    """
    
    @staticmethod
    def check_bitwise(output_a: torch.Tensor, output_b: torch.Tensor) -> Dict:
        """Check if two outputs are bitwise identical."""
        a_cpu = output_a.cpu()
        b_cpu = output_b.cpu()
        
        bitwise_match = torch.equal(a_cpu, b_cpu)
        
        if not bitwise_match:
            diff = (a_cpu.float() - b_cpu.float()).abs()
            return {
                'bitwise_identical': False,
                'max_abs_diff': diff.max().item(),
                'mean_abs_diff': diff.mean().item(),
                'num_different_elements': (diff > 0).sum().item(),
                'total_elements': diff.numel(),
                'mismatch_rate': (diff > 0).sum().item() / diff.numel(),
            }
        
        return {
            'bitwise_identical': True,
            'max_abs_diff': 0.0,
            'mean_abs_diff': 0.0,
            'num_different_elements': 0,
            'total_elements': a_cpu.numel(),
            'mismatch_rate': 0.0,
        }
    
    @staticmethod
    def check_with_tolerance(output_a: torch.Tensor, output_b: torch.Tensor,
                             atol: float = 1e-6, rtol: float = 1e-4) -> Dict:
        """Check consistency with tolerance (relaxed constraint)."""
        result = BitwiseConsistencyChecker.check_bitwise(output_a, output_b)
        
        a_cpu = output_a.float().cpu()
        b_cpu = output_b.float().cpu()
        within_tolerance = torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)
        
        result['within_tolerance'] = within_tolerance
        result['atol'] = atol
        result['rtol'] = rtol
        
        return result
