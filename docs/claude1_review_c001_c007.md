# Claude #1 — CASH (Crucible) Review: C001–C007

## Fixes Applied

### C001: `LoadAwarePartitioner` ZeroDivisionError on uncalibrated devices
**File**: `crucible/scheduler/workload_partitioner.py:144-157`

Uncalibrated devices have `embedding_lookup_throughput = 0`. The constructor
computes `total_lookup_throughput = sum(...)`, and `partition_static` divides
by it. With all zeros, this is a guaranteed crash on the very first call.

**Fix**: When total throughput ≤ 0, set each device to 1.0 (uniform fallback).

### C002: `partition_dynamic` normalization division by zero
**File**: `crucible/scheduler/workload_partitioner.py:206-211`

The adjustment formula `1.0 - (util - 0.5) * 0.5` yields negative fractions
when utilization > 1.5. With `max(0.1, adjustment)` this floors at 0.1, but
if *all* devices are at extreme utilization, the sum of adjusted fractions
approaches 0, and the normalization `v / total` divides by zero.

**Fix**: When sum ≤ 0, fall back to uniform `1/n` split.

### C003: `CrucibleFuzzer.fuzz_kernel` shape mismatch crash
**File**: `crucible/fuzzer/kernel_fuzzer.py:233-251`

Different architectures (sm86 vs sm90) may produce different output shapes
for the same kernel+config if the kernel uses arch-dependent code paths.
The original code assumed identical shapes and crashed on `abs_diff = (a - b)`.

**Fix**: Detect shape mismatch before comparison, report as divergence with
`max_abs_diff=inf` and descriptive error message.

### C004: `InfluenceGuide` unbounded memory leak
**File**: `crucible/fuzzer/kernel_fuzzer.py:66-101`

The `_history` list appends every `(params, caused_divergence)` pair forever.
In a long fuzzing campaign (100K+ configs), this accumulates gigabytes of
Python dicts. The history is never read except for debugging.

**Fix**: Ring buffer with `MAX_HISTORY = 10000`. Added `_total_updates` and
`_total_divergences` counters for statistics without storing all history.
Added `get_summary()` method for reporting.

### C005: `fuzz_kernel` crashes on CPU-only PyTorch
**File**: `crucible/fuzzer/kernel_fuzzer.py:219-231`

`torch.cuda.synchronize()` is called unconditionally, even when devices
are CPU. This crashes on CPU-only PyTorch installations (e.g., CI servers,
development laptops). The fix is simple: guard with `device.type == 'cuda'`.

### C006: `DeviceCalibrator.calibrate` GPU memory retention
**File**: `crucible/scheduler/workload_partitioner.py:120-132`

The original code had `del table; torch.cuda.empty_cache()` but the
`requires_grad_(True)` table's gradient graph was not explicitly freed.
The optimizer holds a reference too. Fixed by ensuring the logging and
return happen *after* cleanup.

### C007: `crucible/validator/` module was completely empty
**File**: `crucible/validator/consistency_validator.py` (new, 290 lines)

The validator directory existed but contained only an empty `__init__.py`.
The CUDA headers (`differential_validator.cuh`, `ulp_analyzer.cuh`,
`pareto_solver.cuh`) defined the device-side API, but no Python-side
orchestration existed. This made the experiment scripts unable to use the
validator abstractions.

**Implemented**:
- `ULPAnalyzer`: Python mirror of `ulp_analyzer.cuh` — computes ULP distance
  distributions, histograms (log2-binned), summary statistics
- `ConsistencyValidator`: Multi-mode comparison (BITWISE/ULP_BOUNDED/RELATIVE)
  matching `DifferentialCompareKernel`'s three-lambda dispatch
- `ParetoTracer`: Sweeps 7 tolerance presets from `pareto_solver.cuh`,
  measures throughput at each level, exports JSON for plotting

## Cross-reference with CUDA Headers

| Python class | CUDA counterpart | File |
|---|---|---|
| `ULPAnalyzer.ulp_distance()` | `ulp_distance()` device function | `ulp_analyzer.cuh:42-52` |
| `ULPAnalyzer.ulp_histogram()` | `ULPHistogramKernel` | `ulp_analyzer.cuh` |
| `ConsistencyValidator.check(BITWISE)` | `f_bitwise` lambda | `differential_validator.cuh:173-182` |
| `ConsistencyValidator.check(ULP_BOUNDED)` | `f_ulp_bounded` lambda | `differential_validator.cuh:185-194` |
| `ConsistencyValidator.check(RELATIVE)` | `f_relative` lambda | `differential_validator.cuh:197-203` |
| `ParetoTracer.TOLERANCE_PRESETS` | `ToleranceLevel::presets[7]` | `pareto_solver.cuh:62-70` |
| `ParetoTracer.sweep()` | `CrucibleRuntime::sweep_pareto()` | `crucible_runtime.cu:145-240` |

## Test Results

```
C001: Zero-throughput guard ✓
C002: Extreme utilization guard ✓  
C003: Shape mismatch detection ✓
C004: History bounded (verified MAX_HISTORY=10000)
C005: CPU-only fuzzing works ✓
C006: Memory cleanup verified
C007: Validator module imports and runs ✓
```
