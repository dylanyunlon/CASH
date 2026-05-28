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
    
    # Mirror of ulp_analyzer.cuh: ULP_HIST_BINS = 34 (bin 0 = exact match,
    # bin k in 1..32 = ULP in [2^(k-1), 2^k), bin 33 = NaN / catastrophic).
    # The CUDA path is fp32-only (__float_as_int / const float*), so this
    # mirror reinterprets in fp32 to stay bit-for-bit comparable with the
    # device kernel's output rather than inventing a separate scheme.
    ULP_HIST_BINS = 34

    # C020: ULP is an integer quantity. The maximum finite fp32 ULP distance is
    # |key(+inf) - key(-inf)| = 4278190081 < 2^32, so int64 holds every finite
    # value exactly with room to spare. The device kernel uses UINT64_MAX as its
    # NaN sentinel; in Python we use an explicit, exactly-representable in-range
    # sentinel (2^62) and ALSO return a NaN mask, so callers never rely on a
    # magic value surviving a float cast (the old 2^64-1 sentinel rounded to
    # 2^64 in float64 and did not round-trip). 2^62 is far above any finite ULP,
    # fits in int64, and is exact in float64 if a caller does cast.
    NAN_ULP = 1 << 62
    _MAX_FINITE_ULP = (1 << 32)  # strict upper bound on any finite fp32 ULP

    # Precomputed scalar constants (avoid reallocating tensors in hot paths).
    _MASK_NEG = 0xFFFFFFFF
    _MASK_POS = 0x80000000

    @classmethod
    def _order_key(cls, x_flat_i32: torch.Tensor) -> torch.Tensor:
        """Monotonic total-order key, mirroring float_to_biased_int in the .cuh.

        key = bits ^ (0xFFFFFFFF if sign else 0x80000000), computed in uint32
        space then widened to int64. Positives flip only the sign bit (ranking
        above all negatives); negatives flip every bit (reversing their
        descending IEEE order). |key(a) - key(b)| is then the exact
        representable-float count between a and b, monotonic across +/-0.

        Expects an already-flattened int32 bit view to avoid repeated reshape.
        """
        u = x_flat_i32.to(torch.int64) & 0xFFFFFFFF        # uint32 bit pattern
        mask = torch.where(u >= 0x80000000,                # sign bit set
                           u.new_tensor(cls._MASK_NEG),
                           u.new_tensor(cls._MASK_POS))
        return u ^ mask

    @classmethod
    def float_to_biased_int(cls, x: torch.Tensor) -> torch.Tensor:
        """Public alias kept for the cross-reference table in the review docs."""
        return cls._order_key(x.float().contiguous().view(-1).view(torch.int32))

    @classmethod
    def ulp_distance_with_mask(cls, a: torch.Tensor, b: torch.Tensor):
        """Return (ulp_int64, nan_mask_bool).

        C020: ULP is int64 and exact. C021: empty input returns empty tensors,
        never raising. NaN positions are flagged in nan_mask and set to the
        in-range NAN_ULP sentinel in the value tensor.
        """
        a_flat = a.float().contiguous().view(-1)
        b_flat = b.float().contiguous().view(-1)

        if a_flat.numel() != b_flat.numel():
            raise ValueError(
                f"ulp_distance shape mismatch: {a_flat.numel()} vs {b_flat.numel()}")

        nan_mask = torch.isnan(a_flat) | torch.isnan(b_flat)

        ia = cls._order_key(a_flat.view(torch.int32))
        ib = cls._order_key(b_flat.view(torch.int32))
        ulp = (ia - ib).abs()                 # int64, exact; <= 2^32-1 for finite

        if nan_mask.any():
            ulp = ulp.clone()
            ulp[nan_mask] = cls.NAN_ULP
        return ulp, nan_mask

    @classmethod
    def ulp_distance(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise ULP distance as int64, matching ULPDistanceKernel.

        NaN on either side yields the in-range NAN_ULP sentinel. Callers that
        must distinguish NaN from a (impossible-for-fp32) huge finite ULP should
        use ulp_distance_with_mask. Returns an empty int64 tensor for empty
        input rather than raising (C021).
        """
        ulp, _ = cls.ulp_distance_with_mask(a, b)
        return ulp

    @classmethod
    def _floor_log2_int(cls, v: int) -> int:
        """Exact floor(log2(v)) for v >= 1 via bit length (no float log2)."""
        return v.bit_length() - 1

    @classmethod
    def ulp_to_bin(cls, ulp: int) -> int:
        """Mirror of the device ulp_to_bin: fixed log2 bucketing into 34 bins.

        C023: uses exact integer log2 (bit_length), identical to the kernel's
        `while (t >>= 1)` loop, with no dependence on float log2 rounding.
        """
        if ulp == 0:
            return 0
        if ulp >= cls.NAN_ULP:
            return cls.ULP_HIST_BINS - 1
        bin_idx = cls._floor_log2_int(int(ulp))   # floor(log2(ulp))
        return min(bin_idx + 1, cls.ULP_HIST_BINS - 1)

    @classmethod
    def ulp_histogram(cls, a: torch.Tensor, b: torch.Tensor) -> Dict:
        """ULP distance histogram using the kernel's fixed 34-bin log2 schema.

        Returns counts of length ULP_HIST_BINS so the Python and CUDA
        histograms are directly comparable as the same paper figure. Every
        element lands in exactly one bin, so sum(counts) == total_elements.
        Binning is exact-integer (C023); NaN handled via mask (C020); empty
        input yields all-zero counts (C021).
        """
        ulp, nan_mask = cls.ulp_distance_with_mask(a, b)
        n = int(ulp.numel())

        counts = np.zeros(cls.ULP_HIST_BINS, dtype=np.int64)
        if n == 0:
            return {
                'exact_match_count': 0, 'nan_count': 0, 'total_elements': 0,
                'nonzero_count': 0, 'max_ulp': 0, 'mean_ulp': 0.0,
                'histogram_counts': counts.tolist(), 'histogram_bins': cls.ULP_HIST_BINS,
            }

        nan_np = nan_mask.cpu().numpy()
        ulp_np = ulp.cpu().numpy()                      # int64
        finite_mask = ~nan_np
        finite = ulp_np[finite_mask & (ulp_np > 0)]     # finite, nonzero

        nan_count = int(nan_np.sum())
        exact_count = int((ulp_np[finite_mask] == 0).sum())

        counts[0] = exact_count
        counts[cls.ULP_HIST_BINS - 1] += nan_count
        if finite.size:
            # C023: exact-integer log2 binning, no float log2. The bins are
            # [2^(k-1), 2^k) -> bin k. Equivalently, bin = (#power-of-two edges
            # <= v) + 1, which np.searchsorted computes exactly on integers.
            # edges = [2^1, 2^2, ..., 2^32]; searchsorted(side='right') counts
            # edges <= v. This matches the kernel's `while (t >>= 1)` loop on
            # every value (verified against the host emulation in C024).
            edges = (1 << np.arange(1, cls.ULP_HIST_BINS - 1, dtype=np.int64))  # 2^1..2^32
            b_idx = np.searchsorted(edges, finite, side='right') + 1
            b_idx = np.clip(b_idx, 1, cls.ULP_HIST_BINS - 2)
            np.add.at(counts, b_idx, 1)

        finite_all = ulp_np[finite_mask]
        return {
            'exact_match_count': exact_count,
            'nan_count': nan_count,
            'total_elements': n,
            'nonzero_count': int((ulp_np[finite_mask] > 0).sum()) + nan_count,
            'max_ulp': int(finite_all.max()) if finite_all.size else 0,
            'mean_ulp': float(finite_all.mean()) if finite_all.size else 0.0,
            'histogram_counts': counts.tolist(),
            'histogram_bins': cls.ULP_HIST_BINS,
        }
    
    @classmethod
    def summary(cls, a: torch.Tensor, b: torch.Tensor) -> Dict:
        """Comprehensive ULP statistics. NaN excluded from finite stats (C020);
        empty input returns zeros (C021)."""
        ulp, nan_mask = cls.ulp_distance_with_mask(a, b)
        n = int(ulp.numel())
        nan_count = int(nan_mask.sum().item())
        if n == 0:
            return {
                'total_elements': 0, 'exact_matches': 0, 'nan_count': 0,
                'max_ulp': 0, 'mean_ulp': 0.0, 'median_ulp': 0.0,
                'p99_ulp': 0.0, 'p999_ulp': 0.0,
            }
        finite = ulp[~nan_mask].cpu().numpy()      # int64, sentinel excluded
        if finite.size == 0:                        # every element was NaN
            return {
                'total_elements': n, 'exact_matches': 0, 'nan_count': nan_count,
                'max_ulp': 0, 'mean_ulp': 0.0, 'median_ulp': 0.0,
                'p99_ulp': 0.0, 'p999_ulp': 0.0,
            }
        return {
            'total_elements': n,
            'exact_matches': int((finite == 0).sum()),
            'nan_count': nan_count,
            'max_ulp': int(finite.max()),
            'mean_ulp': float(finite.mean()),
            'median_ulp': float(np.median(finite)),
            'p99_ulp': float(np.percentile(finite, 99)),
            'p999_ulp': float(np.percentile(finite, 99.9)),
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
            max_ulp_observed = 0
            max_rel_observed = 0.0
            nan_observed = 0
            
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
                
                # ── Validation (outside the timed region) ──
                # C022: compute the ULP distance ONCE and derive every decision
                # from it, instead of calling check() (which itself computes ULP
                # for ULP_BOUNDED) and then recomputing ULP a second time as the
                # earlier C012 code did. C020: NaN is tracked via its mask and
                # excluded from the finite max, so a single NaN can no longer
                # poison achieved_max_ulp with the sentinel value.
                ulp_t, nan_t = self.validator.ulp_analyzer.ulp_distance_with_mask(
                    out_a, out_b)
                n_nan = int(nan_t.sum().item())
                finite_ulp = ulp_t[~nan_t]

                if preset['require_bitwise']:
                    # Bit-exact: any nonzero ULP or any NaN-vs-number is a mismatch.
                    ok = (n_nan == 0) and bool((finite_ulp == 0).all().item()
                                               if finite_ulp.numel() else True)
                else:
                    tol = preset['max_ulp']
                    over = bool((finite_ulp > tol).any().item()
                                if finite_ulp.numel() else False)
                    ok = (n_nan == 0) and not over

                if finite_ulp.numel():
                    iter_max_ulp = int(finite_ulp.max().item())
                    max_ulp_observed = max(max_ulp_observed, iter_max_ulp)
                nan_observed += n_nan

                # Relative error, reusing one float diff (cheap vs the launch).
                a_cpu = out_a.float().reshape(-1)
                b_cpu = out_b.float().reshape(-1)
                rel = ((a_cpu - b_cpu).abs() / a_cpu.abs().clamp(min=1e-10))
                if rel.numel():
                    max_rel_observed = max(max_rel_observed, float(rel.max().item()))

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
                achieved_max_ulp=float(max_ulp_observed),
                achieved_max_rel=max_rel_observed,
                # consistency holds iff the largest finite ULP fits the budget
                # AND no NaN divergence was seen (unless the level is the
                # unconstrained sentinel level).
                consistency_ok=((max_ulp_observed <= preset['max_ulp'] and nan_observed == 0)
                                or preset['max_ulp'] >= 1e10),
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
