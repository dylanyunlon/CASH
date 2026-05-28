# Claude #3 — CASH (Crucible) Review: C011–C016

Continuing from Claude #1 (C001–C007) and Claude #2 (C008–C010). This pass
focuses on correctness bugs in the validator and fuzzer that survive the
earlier fixes — mostly *silent* errors that don't crash but produce wrong
numbers in the paper figures.

## Fixes Applied

### C011: `DeviceCalibrator.calibrate` dead code after `return`
**File**: `crucible/scheduler/workload_partitioner.py`

The method had the final `logger.info(...)` + `return profile` block
duplicated. The second copy sat *after* the first `return`, so it was
unreachable. Harmless at runtime but misleading, and it tripped linters.

**Fix**: Removed the unreachable duplicate. One `return profile` remains.

### C012: `ParetoTracer.sweep` compared float deltas against ULP tolerances
**File**: `crucible/validator/consistency_validator.py`

`max_ulp_observed` was populated from `report.max_abs_diff` — a raw float
difference — and then compared against `preset['max_ulp']`, a tolerance
measured in **ULPs**. The two quantities are in different units, so
`achieved_max_ulp` in every Pareto point (and the `consistency_ok` flag
derived from it) was meaningless. For Exp3 — whose entire purpose is the
correctness×performance frontier — this is a result-invalidating bug.

**Fix**: Compute the genuine per-iteration max ULP via
`ulp_analyzer.ulp_distance(...).max()` and accumulate that into
`max_ulp_observed`. The float `max_rel_diff` is kept separately as before.

### C013: `ULPAnalyzer.ulp_distance` widened fp16/bf16 to fp32 before counting ULPs
**File**: `crucible/validator/consistency_validator.py`

`float_to_biased_int` did `x.view(torch.int32)` after the caller cast the
input to fp32. C008 added bf16/fp16 to the fuzzer's dtype sweep, so the
validator now routinely receives 16-bit tensors. Casting bf16→fp32 first
means two **adjacent** bf16 values (1 ULP apart in bf16) register as 65536
ULPs apart in fp32 — the ULP metric no longer measures ULPs in the format
under test, which is exactly the precision where cross-arch divergence lives.

**Fix**: Made the reinterpretation dtype-aware. fp16/bf16 are viewed as
`int16` (sign-flip mask `0x7FFF`, max ULP `2^15-1`); fp32 stays `int32`.
Arithmetic is promoted to int64 to avoid 16/32-bit overflow. Mixed-dtype
pairs fall back to a well-defined fp32 comparison.

### C014: `ULPAnalyzer.ulp_histogram` silently dropped large-ULP elements
**File**: `crucible/validator/consistency_validator.py`

The histogram used `bins = min(num_bins-1, max_log)` with
`range=(0, bins)`. When the real `log2(ULP)` range exceeded the truncated
bin count, every element past the last edge was discarded by
`np.histogram`, so `sum(counts) < nonzero_count`. With a small `num_bins`
this dropped **all** counts to zero while reporting thousands of nonzero
elements — a histogram that doesn't sum to the population it claims to plot.

**Fix**: Always let the histogram range span the full observed maximum
(`range=(0, ceil(max_log)+1)`), capping only the bin *count* at `num_bins`.
Counts are now conserved: `sum(histogram_counts) == nonzero_count`.

### C015: `CrucibleFuzzer` divergence indices collapsed multi-dim outputs to row IDs
**File**: `crucible/fuzzer/kernel_fuzzer.py`

`torch.where(abs_diff > 1e-6)[0]` on a 2-D `(rows, dim)` tensor returns only
the dim-0 (row) coordinate, discarding the column. Embedding outputs are
`(batch, dim)`, so a divergence at `[3, 7]` was reported as index `3` —
ambiguous and not reproducible.

**Fix**: Flatten `abs_diff` first and report flat element indices into the
contiguous buffer (e.g. `[3, 7]` in a width-128 tensor → `391`).

### C016: `consistency_ok` in Pareto points (follow-on from C012)
**File**: `crucible/validator/consistency_validator.py`

`consistency_ok = max_ulp_observed <= preset['max_ulp'] or ...` was only
meaningful once C012 made `max_ulp_observed` a true ULP count. With the C012
fix this flag now correctly reflects whether the level's ULP budget held.

## Test Results

```
C011: Single return in calibrate (no dead code)         ✓
C012: Pareto achieved_max_ulp is a true ULP count        ✓
C013: bf16 adjacent values report 1 ULP (was 0)          ✓
C014: sum(histogram_counts) == nonzero_count             ✓
C015: 2-D divergence at [3,7] reported as flat 391       ✓
C016: consistency_ok derived from real ULP budget        ✓

Full integration (--quick): regression C001-C003 ✓,
  Exp1 fuzzing ✓, Exp2 partition ✓ (1.67x), Exp3 pareto ✓
```

## C017–C018: Device ULP kernel bug + faithful Python mirror

I had not read the CUDA headers on the first pass — which was a mistake,
because the project's actual contribution lives in `crucible/include/`, and
the most serious bug in the repo is in the device code, not the Python.

### C017: `float_to_biased_int` broke monotonicity (silent, kernel-wide)
**File**: `crucible/include/crucible/ulp_analyzer.cuh`

The header's comment promises a representation "monotonic across ±0", but the
implementation was `if (ix < 0) ix = 0x80000000 - ix;`. `0x80000000` is a
signed `int32` literal equal to `INT_MIN`, so the subtraction overflows and
*fails to reverse* the descending order of the negative half. Concretely the
biased keys came out as `-2.0 → -1073741824`, `-1.0 → -1065353216`,
`-0.5 → -1056964608` — **increasing as the value increases toward zero**,
the opposite of what a total order requires. Consequences:

- `ulp_distance(-0.0, +0.0)` returned `0` instead of `1`.
- Every sign-crossing or large-magnitude comparison was garbage.
- Adjacent values *within* the negatives stayed correct (constant bias
  offset), so any unit test on nearby numbers passed — the bug hid exactly
  where casual testing looks.

This feeds `ULPDistanceKernel`, `ULPHistogramKernel`, and `ErrorHeatmapKernel`,
i.e. every Exp1/Exp3 numerical-divergence figure the paper produces.

**Fix**: the canonical radix-sort float key, computed entirely in `uint32`
(cannot overflow):

```
bits = __float_as_int(x);
mask = (bits >> 31) ? 0xFFFFFFFF : 0x80000000;   // negs: flip all; pos: flip sign
key  = bits ^ mask;
```

Verified monotonic from −2.0 … +2.0 and `−0.0 vs +0.0 == 1 ULP`, both via a
host C++ emulation (`g++ -O2`) and against the Python mirror.

### C018: rewrote `ULPAnalyzer` to actually match the header (retracts C013/C014)
**File**: `crucible/validator/consistency_validator.py`

Reading the header showed my own C013/C014 had drifted *away* from the
reference rather than toward it:

- **C013 retracted**: I had made the Python ULP "dtype-aware" (int16 for
  bf16/fp16). But the CUDA path is fp32-only (`__float_as_int`, `const
  float*`), and `ULP_BOUNDED` tolerances in `pareto_solver.cuh` are calibrated
  in fp32 ULPs. A 16-bit ULP count is not comparable to the device output, so
  the dtype-aware version produced numbers that could never line up with the
  kernel. Reverted to fp32 reinterpretation to match the kernel bit-for-bit.
- **C014 superseded**: my dynamic-range `np.histogram` invented a bin schema
  the kernel doesn't use. The header fixes `ULP_HIST_BINS = 34` with
  `ulp_to_bin` (bin 0 = exact, bin k∈1..32 = `[2^(k-1), 2^k)`, bin 33 =
  NaN/catastrophic). The Python now implements that exact schema, so the two
  histograms are the same figure. Count conservation (the legitimate part of
  C014) is preserved: `sum(counts) == total_elements`.
- Also corrected the stale `ULP_HIST_BINS = 64` constant (header says 34) that
  the earlier cross-reference table had wrong, and made `summary()` exclude the
  `UINT64_MAX` NaN sentinel so it can't dominate `max`/`mean`/percentiles.

**Tests**:
```
C017: device key monotonic −2.0…+2.0, −0.0 vs +0.0 = 1 ULP   ✓ (g++ host emu)
C018: Python ULP matches corrected key                        ✓
      histogram uses 34-bin ulp_to_bin schema, counts conserved ✓
      ulp_to_bin parity with header (0/1/2/2/3/3/4/…/32/33)    ✓
      summary excludes NaN sentinel                            ✓
      ULP_BOUNDED still flags NaN                               ✓
Full integration (--quick): all experiments + regression       ✓
```

## C019–C024: production audit (running the validator path adversarially)

After C011–C018 I ran the *validator* path directly (the integration suite never
exercises it) and probed boundaries the way the brief demanded — user angle
("does my fix break a contract?") and system angle ("invariants, overflow,
consistency across CUDA × Python"). Six defects surfaced, several introduced by
my own C012/C018. Full analysis in `docs/claude3_remediation_plan_c019_c024.md`.

### C019: one ULP definition across both kernels (was: three)
**File**: `crucible/include/crucible/differential_validator.cuh`

`f_ulp_bounded` carried a *third* biasing formula, `(ia<0)?0x7FFFFFFF-ia:ia`,
inconsistent with the C017-fixed `ulp_analyzer.cuh`: positives left unbiased,
negatives in a disjoint range (non-monotonic across sign), and the subtraction
done in **int32** — which overflows for `ia` near `INT_MIN` (demonstrated:
`0x7FFFFFFF - 0x80000001` wraps to −2 instead of 4294967294). For a
cross-architecture correctness validator, two contradictory ULP definitions is
the worst-case inconsistency. **Fix**: `f_ulp_bounded` now delegates to the
shared `Crucible::ulp_distance` (int64, monotonic) and compares in `double`
(exact for all finite ULP < 2^32, vs the old `float` cast that lost precision
above 2^24). Contract "mismatch iff ULP > tol" preserved; definition corrected.

### C020: int64 ULP + explicit sentinel + NaN mask (was: float64 sentinel)
**File**: `crucible/validator/consistency_validator.py`

The NaN sentinel `2^64−1` was stored through `float64`, which rounds it to
`2^64` — it did not round-trip, and any aggregate touching it was fragile. ULP
is integer-valued and the max finite fp32 ULP is 4278190081 < 2^32, so int64
holds every value exactly. **Fix**: `ulp_distance` returns **int64**; new
`ulp_distance_with_mask` returns `(int64, nan_mask)`; NaN is an in-range,
exactly-representable sentinel (`2^62`) plus the mask. `summary`/`histogram`/
sweep report `nan_count` and a finite-only `max`/`mean`, so a single NaN can no
longer poison the statistics (was D4). *Breaking internal API*: return dtype
float64→int64 and sentinel value changed; all in-repo callers updated in this
commit.

### C021: total functions on empty input
`ulp_distance(empty).max()` raised `RuntimeError`. **Fix**: every reduction is
guarded; empty input returns empty/zero results, never throws.

### C022: compute ULP once per Pareto iteration (was: twice)
My C012 called `check(ULP_BOUNDED)` (which computes ULP) and then recomputed
ULP a second time — doubling the dominant non-launch cost of a 7×N sweep.
**Fix**: one `ulp_distance_with_mask` per iteration drives the mismatch
decision, finite-max, and NaN count. Identical results, half the validation
work.

### C023: exact-integer histogram binning (was: float log2)
The kernel bins via an exact integer shift loop; my C018 Python used
`floor(log2(x))`. Empirically equal on this libm, but a faithful mirror must
not depend on `log2` rounding at power-of-two boundaries. **Fix**: binning via
`np.searchsorted` on integer power-of-two edges — exact, and verified identical
to the kernel's loop for every value.

### C024: differential test locking the contract
**File**: `tests/test_ulp_parity.py` (new, no GPU needed)

Compiles the CUDA primitives' logic into a host binary and asserts Python ==
C++ ULP and bin assignment across 2704 adversarial pairs (±0, ±inf, NaN,
denormals, power-of-two ULP gaps), histogram count conservation, empty/all-NaN
behaviour, and no sentinel leakage in the Pareto sweep. This encodes the
"one ULP definition across CUDA × Python" invariant as an executable gate.

### Verified-correct, left unchanged
- C017 device key: max finite ULP `ULP(−inf,+inf)=4278190081 < 2^32` → bin 32;
  bin 33 reserved for NaN; Python finite clip to 32 matches exactly (host emu).
- `ConsistencyValidator.check` is empty-safe; its NaN-in-`max_abs` behaviour is
  pre-existing and off the hot path — flagged, not changed, to avoid scope creep.

### Test results (all on CPU / host-compiled emulation)
```
C019: f_ulp_bounded overflow + monotonicity fixed (g++ host proof)   ✓
C020: int64 ULP, sentinel round-trips, NaN via mask                  ✓
C021: empty input returns zeros, no RuntimeError                     ✓
C022: single ULP pass per Pareto iteration                           ✓
C023: ulp_to_bin / histogram exact-integer, == CUDA on all values    ✓
C024: 2704-pair Python↔C++ parity + conservation + edge cases        ✓
Full integration (--quick): regression + exp1/2/3 unchanged          ✓
py_compile all modules + data generator synthetic path               ✓
```

## Notes for Claude #4

- C017 is unverified on real hardware — it's a host-emulated proof. A real
  sm86+sm90 run should confirm the kernel output now matches the Python mirror
  numerically.
- `ConsistencyValidator.check(RELATIVE)` uses `a.abs()` as the denominator,
  so the relative metric is asymmetric in `(a, b)`. This matches the CUDA
  `f_relative` lambda's reference convention, so left as-is — flagging only
  in case a symmetric variant is wanted later.
- All testing here is CPU-only (no GPU in this environment); behavioural
  fixes are unit-verified but cross-arch numerics still need a real
  sm86+sm90 run.
