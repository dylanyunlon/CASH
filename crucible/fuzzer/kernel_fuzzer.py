"""
Crucible Cross-Architecture CUDA Kernel Fuzzer
Adapts SyzMini's influence-guided call removal from Linux syscall fuzzing
to CUDA kernel launch parameter fuzzing on sm86 (A6000) vs sm90 (H100).
"""

import torch
import numpy as np
import itertools
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ArchTarget(Enum):
    SM86 = "sm_86"  # A6000
    SM90 = "sm_90"  # H100


@dataclass
class KernelConfig:
    """CUDA kernel launch configuration."""
    grid_dim: Tuple[int, int, int]
    block_dim: Tuple[int, int, int]
    shared_mem_bytes: int
    data_alignment: int  # bytes
    tensor_shape: Tuple[int, ...]
    dtype: torch.dtype
    
    @property
    def total_threads(self) -> int:
        return (self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2] *
                self.block_dim[0] * self.block_dim[1] * self.block_dim[2])
    
    def to_dict(self) -> dict:
        return {
            'grid_dim': list(self.grid_dim),
            'block_dim': list(self.block_dim),
            'shared_mem_bytes': self.shared_mem_bytes,
            'data_alignment': self.data_alignment,
            'tensor_shape': list(self.tensor_shape),
            'dtype': str(self.dtype),
        }


@dataclass
class FuzzResult:
    """Result of a single fuzz test."""
    config: KernelConfig
    arch_a_output: Optional[torch.Tensor]
    arch_b_output: Optional[torch.Tensor]
    arch_a_time_ms: float
    arch_b_time_ms: float
    is_divergent: bool
    max_abs_diff: float
    max_rel_diff: float
    divergence_indices: List[int] = field(default_factory=list)
    error: Optional[str] = None


class InfluenceGuide:
    """
    Influence-guided parameter generation.
    Adapts SyzMini's influence scoring to CUDA kernel parameters:
    - Identifies which parameters most influence cross-arch divergence
    - Focuses fuzzing on high-influence parameter regions
    
    Uses bounded ring buffer for history to prevent memory leaks
    during long fuzzing campaigns (C004 fix).
    """
    
    MAX_HISTORY = 10000  # Cap history to prevent unbounded growth
    
    def __init__(self):
        self.param_scores: Dict[str, float] = {
            'grid_x': 0.5,
            'grid_y': 0.5,
            'block_x': 0.5,
            'block_y': 0.5,
            'block_z': 0.5,
            'shared_mem': 0.5,
            'alignment': 0.5,
            'shape_0': 0.5,
            'shape_1': 0.5,
        }
        self._history: List[Tuple[dict, bool]] = []
        self._total_updates: int = 0
        self._total_divergences: int = 0
    
    def update(self, config: KernelConfig, caused_divergence: bool):
        """Update influence scores based on observed divergence."""
        params = {
            'grid_x': config.grid_dim[0],
            'grid_y': config.grid_dim[1],
            'block_x': config.block_dim[0],
            'block_y': config.block_dim[1],
            'block_z': config.block_dim[2],
            'shared_mem': config.shared_mem_bytes,
            'alignment': config.data_alignment,
            'shape_0': config.tensor_shape[0] if len(config.tensor_shape) > 0 else 0,
            'shape_1': config.tensor_shape[1] if len(config.tensor_shape) > 1 else 0,
        }
        # Ring buffer: drop oldest when full
        if len(self._history) >= self.MAX_HISTORY:
            self._history.pop(0)
        self._history.append((params, caused_divergence))
        self._total_updates += 1
        
        if caused_divergence:
            self._total_divergences += 1
            # Increase influence score for parameters at boundary values
            for key, val in params.items():
                if self._is_boundary_value(key, val):
                    self.param_scores[key] = min(1.0, self.param_scores[key] + 0.1)
        else:
            # Slight decay for non-divergent runs
            for key in self.param_scores:
                self.param_scores[key] = max(0.1, self.param_scores[key] * 0.98)
    
    def _is_boundary_value(self, param_name: str, value: int) -> bool:
        """Check if a parameter value is at an architecture-relevant boundary."""
        boundaries = {
            'block_x': [32, 64, 128, 256, 512, 1024],
            'block_y': [1, 2, 4, 8, 16, 32],
            'shared_mem': [0, 1024, 4096, 16384, 49152, 65536, 100352, 163840, 227328],
            'alignment': [1, 2, 4, 8, 16, 32, 64, 128, 256],
            'grid_x': [1, 65535, 2147483647],
        }
        if param_name in boundaries:
            return value in boundaries[param_name] or any(
                abs(value - b) <= 1 for b in boundaries[param_name]
            )
        return False
    
    def get_high_influence_params(self, top_k: int = 3) -> List[str]:
        """Get parameters with highest influence scores."""
        sorted_params = sorted(self.param_scores.items(), key=lambda x: -x[1])
        return [p[0] for p in sorted_params[:top_k]]

    def get_summary(self) -> Dict:
        """Get influence analysis summary for reporting."""
        return {
            'total_updates': self._total_updates,
            'total_divergences': self._total_divergences,
            'divergence_rate': (self._total_divergences / max(self._total_updates, 1)),
            'scores': dict(self.param_scores),
            'top_3': self.get_high_influence_params(3),
            'history_size': len(self._history),
        }


class BoundaryTensorGenerator:
    """
    Generates boundary tensor shapes that are likely to trigger
    architecture-dependent behavior differences.
    """
    
    # SM86 (A6000) vs SM90 (H100) key differences
    ARCH_BOUNDARIES = {
        'sm86': {
            'max_threads_per_block': 1024,
            'max_shared_mem_per_block': 49152,  # 48 KB
            'warp_size': 32,
            'sm_count': 84,
            'max_registers_per_block': 65536,
        },
        'sm90': {
            'max_threads_per_block': 1024,
            'max_shared_mem_per_block': 228 * 1024,  # 228 KB with opt-in
            'warp_size': 32,
            'sm_count': 132,
            'max_registers_per_block': 65536,
        }
    }
    
    @classmethod
    def generate_boundary_configs(cls, num_configs: int = 100,
                                   influence_guide: Optional[InfluenceGuide] = None
                                   ) -> List[KernelConfig]:
        """Generate kernel configs at architecture boundary values."""
        configs = []
        
        # ── Shared memory boundaries (biggest divergence source) ──
        # sm86 max = 48KB, sm90 can go up to 228KB
        shared_mem_values = [0, 1024, 48 * 1024, 49152, 49153,  # At sm86 boundary
                            64 * 1024, 100 * 1024, 164 * 1024, 228 * 1024]
        
        # ── Block size boundaries ──
        block_sizes = [(32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1),
                       (512, 1, 1), (1024, 1, 1),
                       (32, 32, 1), (16, 16, 4), (8, 8, 16)]
        
        # ── Tensor shapes at alignment boundaries ──
        shape_dims = [1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
                      127, 128, 129, 255, 256, 257, 511, 512, 1024, 1025]
        
        for smem in shared_mem_values:
            for block in block_sizes[:3]:  # Limit combinations
                for dim0 in shape_dims[:5]:
                    for alignment in [1, 4, 16, 64, 128]:
                        grid = (max(1, 1024 // block[0]), 1, 1)
                        configs.append(KernelConfig(
                            grid_dim=grid,
                            block_dim=block,
                            shared_mem_bytes=smem,
                            data_alignment=alignment,
                            tensor_shape=(dim0, 128),
                            dtype=torch.float32
                        ))
                        if len(configs) >= num_configs:
                            return configs
        
        return configs[:num_configs]


class CrucibleFuzzer:
    """
    Main fuzzer engine that runs CUDA kernels on both architectures
    and compares outputs for divergence.
    """
    
    def __init__(self, device_sm86: torch.device, device_sm90: torch.device):
        self.device_a = device_sm86  # A6000
        self.device_b = device_sm90  # H100
        self.influence = InfluenceGuide()
        self.results: List[FuzzResult] = []
        self.divergences: List[FuzzResult] = []
    
    def fuzz_kernel(self, kernel_fn: Callable, config: KernelConfig,
                    input_data: torch.Tensor) -> FuzzResult:
        """
        Run a kernel on both architectures and compare outputs.
        Handles: shape mismatches (C003), CPU-only testing (C005).
        """
        try:
            # Run on sm86 (A6000)
            input_a = input_data.to(self.device_a)
            if self.device_a.type == 'cuda':
                torch.cuda.synchronize(self.device_a)
            start_a = time.perf_counter()
            output_a = kernel_fn(input_a, config)
            if self.device_a.type == 'cuda':
                torch.cuda.synchronize(self.device_a)
            time_a = (time.perf_counter() - start_a) * 1000
            
            # Run on sm90 (H100)
            input_b = input_data.to(self.device_b)
            if self.device_b.type == 'cuda':
                torch.cuda.synchronize(self.device_b)
            start_b = time.perf_counter()
            output_b = kernel_fn(input_b, config)
            if self.device_b.type == 'cuda':
                torch.cuda.synchronize(self.device_b)
            time_b = (time.perf_counter() - start_b) * 1000
            
            # Compare outputs
            out_a_cpu = output_a.float().cpu()
            out_b_cpu = output_b.float().cpu()
            
            # C003: Shape mismatch is itself a divergence
            if out_a_cpu.shape != out_b_cpu.shape:
                result = FuzzResult(
                    config=config,
                    arch_a_output=out_a_cpu, arch_b_output=out_b_cpu,
                    arch_a_time_ms=time_a, arch_b_time_ms=time_b,
                    is_divergent=True,
                    max_abs_diff=float('inf'),
                    max_rel_diff=float('inf'),
                    error=f"Shape mismatch: {out_a_cpu.shape} vs {out_b_cpu.shape}"
                )
                self.influence.update(config, True)
                self.results.append(result)
                self.divergences.append(result)
                return result
            
            abs_diff = (out_a_cpu - out_b_cpu).abs()
            max_abs = abs_diff.max().item()
            max_rel = (abs_diff / (out_a_cpu.abs() + 1e-10)).max().item()
            
            is_divergent = max_abs > 1e-6 or max_rel > 1e-4
            
            divergence_idx = []
            if is_divergent:
                divergence_idx = torch.where(abs_diff > 1e-6)[0][:10].tolist()
            
            result = FuzzResult(
                config=config,
                arch_a_output=out_a_cpu,
                arch_b_output=out_b_cpu,
                arch_a_time_ms=time_a,
                arch_b_time_ms=time_b,
                is_divergent=is_divergent,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                divergence_indices=divergence_idx
            )
            
        except Exception as e:
            result = FuzzResult(
                config=config,
                arch_a_output=None, arch_b_output=None,
                arch_a_time_ms=0, arch_b_time_ms=0,
                is_divergent=True,
                max_abs_diff=float('inf'),
                max_rel_diff=float('inf'),
                error=str(e)
            )
        
        # Update influence scores
        self.influence.update(config, result.is_divergent)
        self.results.append(result)
        if result.is_divergent:
            self.divergences.append(result)
        
        return result
    
    def run_campaign(self, kernel_fn: Callable, num_configs: int = 100) -> Dict:
        """Run a full fuzzing campaign."""
        configs = BoundaryTensorGenerator.generate_boundary_configs(
            num_configs, self.influence
        )
        
        print(f"  Running {len(configs)} fuzz configurations...")
        
        for i, config in enumerate(configs):
            input_data = torch.randn(*config.tensor_shape, dtype=config.dtype)
            result = self.fuzz_kernel(kernel_fn, config, input_data)
            
            if (i + 1) % 20 == 0:
                div_rate = len(self.divergences) / (i + 1) * 100
                print(f"  Progress: {i+1}/{len(configs)} | "
                      f"Divergences: {len(self.divergences)} ({div_rate:.1f}%)")
        
        report = {
            'total_tests': len(self.results),
            'divergences_found': len(self.divergences),
            'divergence_rate': len(self.divergences) / max(len(self.results), 1),
            'high_influence_params': self.influence.get_high_influence_params(),
            'influence_scores': self.influence.param_scores,
            'divergence_details': [
                {
                    'config': d.config.to_dict(),
                    'max_abs_diff': d.max_abs_diff,
                    'max_rel_diff': d.max_rel_diff,
                    'error': d.error,
                }
                for d in self.divergences[:50]
            ],
        }
        
        return report
