"""
Crucible Cross-Architecture Consistency Validator
=================================================
Python-side orchestration for the CUDA DifferentialCompareKernel and
ULP analysis pipeline.  This module bridges crucible/include/ CUDA
kernels with the experiment scripts.

Provides:
  - ULPAnalyzer: compute ULP distance distributions between two tensors
  - ConsistencyValidator: run multi-level consistency checks (bitwise / ULP / relative)
  - ParetoTracer: sweep tolerance levels and record throughput × correctness
"""

import torch
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

logger = logging.getLogger(__name__)


class CompareMode(IntEnum):
    """Mirrors crucible/include/crucible/differential_validator.cuh::CompareMode"""
    BITWISE = 0
    ULP_BOUNDED = 1
    RELATIVE = 2


@dataclass
class DivergenceReport:
    """Python mirror of crucible/include/crucible/differential_validator.cuh::DivergenceReport"""
    total_elements: int
    mismatched_elements: int
    max_abs_diff: float
    max_rel_diff: float
    first_mismatch_idx: int
    mode: CompareMode
    tolerance: float = 0.0


class ULPAnalyzer:
    """
    Compute ULP (Units in Last Place) distance distributions.
    
    Python implementation of crucible/include/crucible/ulp_analyzer.cuh.
    For production, the CUDA kernels should be used; this Python version
    enables testing and prototyping without compilation.
    
    ULP distance = |reinterpret_as_int(a) - reinterpret_as_int(b)|
    """
    
    ULP_HIST_BINS = 64  # Matches CUDA ULP_HIST_BINS
    
    @staticmethod
    def float_to_biased_int(x: torch.Tensor) -> torch.Tensor:
        """Convert float to biased integer representation for monotonic ULP distance."""
        # IEEE 754: reinterpret float bits as int32
        int_repr = x.view(torch.int32)
        # Negative floats: flip all bits except sign → monotonic
        mask = int_repr < 0
        int_repr = torch.where(mask, 0x7FFFFFFF - int_repr, int_repr)
        return int_repr.long()
    
    @classmethod
    def ulp_distance(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise ULP distance between two float32 tensors."""
        a_flat = a.float().contiguous().view(-1)
        b_flat = b.float().contiguous().view(-1)
        
        # Handle NaN: assign max distance
        nan_mask = torch.isnan(a_flat) | torch.isnan(b_flat)
        
        ia = cls.float_to_biased_int(a_flat)
        ib = cls.float_to_biased_int(b_flat)
        
        ulp = (ia - ib).abs()
        ulp[nan_mask] = 2**31 - 1  # Max representable
        
        return ulp
    
    @classmethod
    def ulp_histogram(cls, a: torch.Tensor, b: torch.Tensor,
                      num_bins: int = 64) -> Dict:
        """
        Compute ULP distance histogram — directly usable as a paper figure.
        
        Returns bin edges (log2 scale) and counts, matching the CUDA
        ULPHistogramKernel output format.
        """
        ulp = cls.ulp_distance(a, b)
        
        # Log2 binning: bin k contains ULP distances in [2^k, 2^(k+1))
        # Bin 0: exact match (ULP=0)
        ulp_np = ulp.cpu().numpy().astype(np.float64)
        
        exact_match = int((ulp_np == 0).sum())
        nonzero = ulp_np[ulp_np > 0]
        
        if len(nonzero) > 0:
            log2_ulp = np.log2(nonzero.clip(min=1))
            max_log = int(np.ceil(log2_ulp.max())) + 1
            bins = min(num_bins - 1, max_log)
            hist, edges = np.histogram(log2_ulp, bins=bins, range=(0, bins))
        else:
            hist = np.array([], dtype=np.int64)
            edges = np.array([0], dtype=np.float64)
        
        return {
            'exact_match_count': exact_match,
            'total_elements': len(ulp_np),
            'nonzero_count': len(nonzero),
            'max_ulp': int(ulp_np.max()) if len(ulp_np) > 0 else 0,
            'mean_ulp': float(ulp_np.mean()) if len(ulp_np) > 0 else 0,
            'histogram_counts': hist.tolist(),
            'histogram_edges_log2': edges.tolist(),
        }
    
    @classmethod
    def summary(cls, a: torch.Tensor, b: torch.Tensor) -> Dict:
        """Compute comprehensive ULP statistics."""
        ulp = cls.ulp_distance(a, b)
        ulp_np = ulp.cpu().numpy().astype(np.float64)
        
        return {
            'total_elements': len(ulp_np),
            'exact_matches': int((ulp_np == 0).sum()),
            'max_ulp': int(ulp_np.max()) if len(ulp_np) > 0 else 0,
            'mean_ulp': float(ulp_np.mean()),
            'median_ulp': float(np.median(ulp_np)),
            'p99_ulp': float(np.percentile(ulp_np, 99)) if len(ulp_np) > 0 else 0,
            'p999_ulp': float(np.percentile(ulp_np, 99.9)) if len(ulp_np) > 0 else 0,
        }


class ConsistencyValidator:
    """
    Multi-level cross-architecture consistency checker.
    
    Orchestrates the three comparison strategies from
    crucible/include/crucible/differential_validator.cuh:
      - BITWISE: exact bit equality
      - ULP_BOUNDED: within N ULP tolerance  
      - RELATIVE: within relative error bound
    """
    
    def __init__(self):
        self.ulp_analyzer = ULPAnalyzer()
        self._check_history: List[DivergenceReport] = []
    
    def check(self, output_a: torch.Tensor, output_b: torch.Tensor,
              mode: CompareMode = CompareMode.BITWISE,
              tolerance: float = 0.0) -> DivergenceReport:
        """
        Run a consistency check with the specified mode.
        
        Mirrors DifferentialCompareKernel dispatch:
          BITWISE     → bit-exact comparison
          ULP_BOUNDED → ULP distance ≤ tolerance
          RELATIVE    → |a-b|/max(|a|,ε) ≤ tolerance
        """
        a = output_a.float().cpu().contiguous().view(-1)
        b = output_b.float().cpu().contiguous().view(-1)
        
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        n = a.numel()
        
        if mode == CompareMode.BITWISE:
            # Reinterpret as int32, compare bits
            ia = a.view(torch.int32)
            ib = b.view(torch.int32)
            mismatches = (ia != ib)
        elif mode == CompareMode.ULP_BOUNDED:
            ulp = self.ulp_analyzer.ulp_distance(a, b)
            mismatches = ulp > tolerance
        elif mode == CompareMode.RELATIVE:
            rel = (a - b).abs() / (a.abs().clamp(min=1e-10))
            mismatches = rel > tolerance
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        mismatch_count = int(mismatches.sum().item())
        
        if mismatch_count > 0:
            diff = (a - b).abs()
            max_abs = float(diff.max().item())
            max_rel = float((diff / a.abs().clamp(min=1e-10)).max().item())
            first_idx = int(torch.where(mismatches)[0][0].item())
        else:
            max_abs = 0.0
            max_rel = 0.0
            first_idx = -1
        
        report = DivergenceReport(
            total_elements=n,
            mismatched_elements=mismatch_count,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            first_mismatch_idx=first_idx,
            mode=mode,
            tolerance=tolerance,
        )
        self._check_history.append(report)
        return report
    
    def check_all_modes(self, output_a: torch.Tensor, output_b: torch.Tensor,
                        ulp_tolerance: float = 4.0,
                        rel_tolerance: float = 1e-5) -> Dict[str, DivergenceReport]:
        """Run all three comparison modes and return results."""
        return {
            'bitwise': self.check(output_a, output_b, CompareMode.BITWISE),
            'ulp_bounded': self.check(output_a, output_b, CompareMode.ULP_BOUNDED, ulp_tolerance),
            'relative': self.check(output_a, output_b, CompareMode.RELATIVE, rel_tolerance),
        }


@dataclass
class ParetoPoint:
    """One point on the correctness × performance Pareto frontier."""
    tolerance_level: int
    tolerance_ulp: float
    require_bitwise: bool
    throughput_ops_per_sec: float
    achieved_max_ulp: float
    achieved_max_rel: float
    consistency_ok: bool
    latency_ms: float


class ParetoTracer:
    """
    Sweep the correctness × performance Pareto frontier.
    
    Mirrors crucible/include/crucible/pareto_solver.cuh::ToleranceLevel::presets
    but runs on Python for easy integration with experiment scripts.
    """
    
    # Matches ToleranceLevel::presets in pareto_solver.cuh
    TOLERANCE_PRESETS = [
        {'name': 'L0_bitwise',   'max_ulp': 0,     'max_rel': 0,    'require_bitwise': True},
        {'name': 'L1_1ulp',      'max_ulp': 1,     'max_rel': 1e-7, 'require_bitwise': False},
        {'name': 'L2_4ulp',      'max_ulp': 4,     'max_rel': 1e-6, 'require_bitwise': False},
        {'name': 'L3_16ulp',     'max_ulp': 16,    'max_rel': 1e-5, 'require_bitwise': False},
        {'name': 'L4_256ulp',    'max_ulp': 256,   'max_rel': 1e-4, 'require_bitwise': False},
        {'name': 'L5_64Kulp',    'max_ulp': 65536, 'max_rel': 1e-2, 'require_bitwise': False},
        {'name': 'L6_unconstrained', 'max_ulp': 1e12, 'max_rel': 1.0, 'require_bitwise': False},
    ]
    
    def __init__(self, device_a: torch.device, device_b: torch.device):
        self.device_a = device_a
        self.device_b = device_b
        self.validator = ConsistencyValidator()
        self.points: List[ParetoPoint] = []
    
    def sweep(self, num_embeddings: int = 100_000, embedding_dim: int = 128,
              batch_size: int = 4096, num_iters: int = 200) -> List[ParetoPoint]:
        """
        Sweep all tolerance presets and measure throughput at each level.
        Returns list of ParetoPoints for plotting.
        """
        self.points = []
        
        for lvl, preset in enumerate(self.TOLERANCE_PRESETS):
            table_a = torch.randn(num_embeddings, embedding_dim, device=self.device_a)
            table_b = table_a.to(self.device_b)
            
            throughputs = []
            max_ulp_observed = 0.0
            max_rel_observed = 0.0
            
            for i in range(num_iters):
                indices = torch.randint(0, num_embeddings, (batch_size,))
                idx_a = indices.to(self.device_a)
                idx_b = indices.to(self.device_b)
                
                if self.device_a.type == 'cuda':
                    torch.cuda.synchronize(self.device_a)
                if self.device_b.type == 'cuda':
                    torch.cuda.synchronize(self.device_b)
                
                t0 = time.perf_counter()
                out_a = table_a[idx_a]
                out_b = table_b[idx_b]
                if self.device_a.type == 'cuda':
                    torch.cuda.synchronize(self.device_a)
                if self.device_b.type == 'cuda':
                    torch.cuda.synchronize(self.device_b)
                elapsed = time.perf_counter() - t0
                
                # Verify at this tolerance level
                if preset['require_bitwise']:
                    report = self.validator.check(out_a, out_b, CompareMode.BITWISE)
                    ok = report.mismatched_elements == 0
                else:
                    report = self.validator.check(
                        out_a, out_b, CompareMode.ULP_BOUNDED, preset['max_ulp'])
                    ok = report.mismatched_elements == 0
                
                max_ulp_observed = max(max_ulp_observed, report.max_abs_diff)
                max_rel_observed = max(max_rel_observed, report.max_rel_diff)
                
                # Sync overhead if consistency violated
                if not ok:
                    t1 = time.perf_counter()
                    table_b = table_a.to(self.device_b)
                    if self.device_b.type == 'cuda':
                        torch.cuda.synchronize(self.device_b)
                    elapsed += time.perf_counter() - t1
                
                throughputs.append(batch_size * embedding_dim / elapsed)
            
            point = ParetoPoint(
                tolerance_level=lvl,
                tolerance_ulp=preset['max_ulp'],
                require_bitwise=preset['require_bitwise'],
                throughput_ops_per_sec=float(np.mean(throughputs)),
                achieved_max_ulp=max_ulp_observed,
                achieved_max_rel=max_rel_observed,
                consistency_ok=max_ulp_observed <= preset['max_ulp'] or preset['max_ulp'] >= 1e10,
                latency_ms=float(1000.0 * batch_size * embedding_dim / np.mean(throughputs)),
            )
            self.points.append(point)
            
            del table_a, table_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(
                f"Pareto {preset['name']}: {point.throughput_ops_per_sec:.0f} ops/s, "
                f"max_ulp={max_ulp_observed:.0f}, ok={point.consistency_ok}")
        
        return self.points
    
    def to_dict(self) -> Dict:
        """Export Pareto frontier for JSON serialization."""
        return {
            'num_levels': len(self.points),
            'points': [
                {
                    'level': p.tolerance_level,
                    'tolerance_ulp': p.tolerance_ulp,
                    'require_bitwise': p.require_bitwise,
                    'throughput_ops_per_sec': round(p.throughput_ops_per_sec, 1),
                    'achieved_max_ulp': round(p.achieved_max_ulp, 1),
                    'consistency_ok': p.consistency_ok,
                    'latency_ms': round(p.latency_ms, 3),
                }
                for p in self.points
            ],
        }
