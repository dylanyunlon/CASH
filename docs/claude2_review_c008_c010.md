# Claude #2 — CASH Review: C008–C010

## Fixes Applied

### C008: `BoundaryTensorGenerator` — FP8/BF16 dtype fuzzing
**File**: `crucible/fuzzer/kernel_fuzzer.py:182-230`

The original config generator only produced `torch.float32` configs. FP8 is
only available on sm90 (H100), so fuzzing with FP8 dtype is the most direct
way to trigger cross-architecture divergence. BF16 rounding also differs
subtly between sm86/sm90 due to different tensor core implementations.

**Fix**: Added `torch.bfloat16` and `torch.float16` to the dtype sweep.
(`torch.float8_e4m3fn` excluded from the sweep because PyTorch doesn't
support it as a general tensor dtype — it's only usable through
TransformerEngine's FP8 recipe. The BF16/FP16 sweep still exposes the
relevant rounding differences.)

### C009: Influence-weighted boundary sampling
**File**: `crucible/fuzzer/kernel_fuzzer.py:200-215`

The original generator iterated parameters uniformly. After running 10+
configs, the InfluenceGuide already knows which parameters correlate most
with divergence. The generator should focus on those regions.

**Fix**: When `influence_guide._total_updates > 10` and a parameter's
influence score exceeds 0.7, inject additional boundary values near the
high-influence region (e.g., ±1 around the sm86 shared memory limit).

### C010: Per-dtype divergence breakdown in campaign report
**File**: `crucible/fuzzer/kernel_fuzzer.py:344-370`

The campaign report previously only gave aggregate divergence count.
With C008 adding multiple dtypes, we need per-dtype breakdown to
identify which precision format causes the most cross-arch divergence.

**Fix**: Added `_compute_dtype_breakdown()` method and `per_dtype_divergences`
field in the campaign report. Also includes `influence_summary` from C004's
`get_summary()`.

## Test Results

```
C008: BF16/FP16 configs generated ✓
C009: Influence-weighted sampling activated after 10 updates ✓
C010: Per-dtype breakdown computed ✓
```
