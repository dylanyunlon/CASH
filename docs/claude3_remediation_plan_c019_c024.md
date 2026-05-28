# Remediation Plan — C019–C024 (post-audit of C011–C018)

Author stance: *The Art of Computer Programming*. Every claim is backed by an
exact invariant, a range argument, or a reproducing test. Each fix is examined
from two adversarial angles before it is written:

> **(U) User angle** — does this change silently break a contract someone
> already depends on (return type, value range, field name, exception
> behaviour, numerical result)?
> **(S) System angle** — correctness invariants, overflow, determinism,
> complexity, and consistency *across* the CUDA kernels and the Python mirror.

And each fix folds in the standard a large engineering org would require
("融合大厂"): single source of truth, exact-integer arithmetic over float
proxies, explicit sentinels, defined behaviour on empty input, and no
redundant work in hot loops.

---

## Findings from running the full pipeline

The integration suite (`run_hetero_crucible.py`) passes green on CPU. That is
precisely why these defects are dangerous — they are silent. Direct probing of
the validator path (which the integration test does **not** exercise) found:

| # | Severity | Where | Defect |
|---|---|---|---|
| D1 | **High** | `differential_validator.cuh` `f_ulp_bounded` | A *third* ULP biasing formula, inconsistent with the fixed `ulp_analyzer.cuh`; same monotonicity bug as C017 (negatives not reversed, positives unbiased) **and** computed in `int32` (overflow-prone). The cross-arch validator therefore has two contradictory definitions of "ULP". |
| D2 | **High** | `consistency_validator.py` NaN sentinel | Sentinel `2^64−1` is stored through `float64`, which rounds it to `2^64`. The "max representable" sentinel does not round-trip; equality with the original int is lost. |
| D3 | Medium | `ParetoTracer.sweep` (my C012) | `.max()` on the per-iteration ULP tensor crashes (`RuntimeError`) if a kernel ever yields an **empty** output. No defined behaviour on empty input. |
| D4 | Medium | `ulp_distance` / C012 | A single NaN element poisons the reported `achieved_max_ulp` to the `~1.8e19` sentinel, silently corrupting the Pareto figure's y-axis. NaN count and finite-max must be reported separately. |
| D5 | Medium | `ParetoTracer.sweep` (my C012) | ULP is computed **twice** per iteration — once inside `check(ULP_BOUNDED)` and again in my added line — doubling the dominant cost of the hot loop (7 presets × N iters). |
| D6 | Low/robustness | `ulp_histogram` (my C018) | Binning uses `floor(log2(x))` in floating point; the CUDA reference uses an **exact integer** shift loop. Empirically they agree on this libm, but the mirror must not depend on an undocumented `log2` rounding guarantee when an exact integer op exists. |

Verified-correct (no change needed):
- C017 device key: max finite ULP = `ULP(−inf,+inf)` = 4278190081 < `2^32`,
  so it lands in bin 32 and bin 33 is reserved for NaN. The Python finite clip
  to 32 matches the kernel exactly. Monotonic −2.0…+2.0 confirmed via `g++`.

---

## C019 — Single source of truth for the float→ordered-key map (fixes D1)

There must be **one** `float_to_biased_int`, used by every kernel and mirrored
by Python. Today `ulp_analyzer.cuh` (fixed in C017) and
`differential_validator.cuh::f_ulp_bounded` disagree.

**Action**: delete the bespoke biasing in `f_ulp_bounded`; have it call the
shared `Crucible::float_to_biased_int` and compute the distance in `int64`
(via the shared `ulp_distance`), exactly as `ulp_analyzer.cuh` does.

- **(U)** `f_ulp_bounded`'s public contract is "mismatch iff ULP > tolerance".
  The contract is unchanged; only the (previously wrong) numeric definition of
  ULP changes. Callers comparing against the *same* tolerance presets now get
  the corrected, monotonic distance — which is the whole point of the
  validator. No field/signature change.
- **(S)** Removes `int32` overflow in the negative branch
  (`0x7FFFFFFF - ia` overflows for `ia` near `INT_MIN`). Eliminates the
  two-definitions inconsistency: a single proof of monotonicity now covers
  both kernels. Big-org practice: *one* canonical primitive, no copy-paste of
  subtle bit math.

## C020 — Integer ULP type + explicit sentinel in Python (fixes D2, D4)

Stop routing ULP through `float64`. Keep it `int64` (max finite ULP < `2^32`
fits with 31 bits to spare) and represent NaN with an explicit, in-range
sentinel plus a **separate NaN mask**, never a value that must survive a float
cast.

**Action**:
- `ulp_distance` returns `int64`; NaN positions recorded in a returned/queryable
  mask. Internally use a sentinel that is exactly representable and clearly
  out of the finite range (e.g. `INT64 NAN_ULP = 1<<62`), but callers that
  need "is NaN" use the mask, not value comparison.
- `summary`, `ulp_histogram`, and the Pareto sweep report `nan_count` and a
  finite-only `max_ulp`/`mean_ulp`.

- **(U)** This *changes the dtype* of `ulp_distance`'s return (float64 →
  int64) and the sentinel value. Anyone downcasting to float or comparing
  against `2^64−1` would break. Mitigation: the only in-repo callers are the
  validator's own methods and `ParetoTracer`; all are updated in the same
  commit. The new `int64` return is strictly more correct (exact) and the
  `nan_count` field is additive. Documented in the review doc as a breaking
  internal API change with rationale.
- **(S)** Exactness restored: `int64` holds every finite ULP with no rounding;
  the sentinel round-trips. NaN no longer contaminates aggregate statistics.
  Big-org practice: explicit sentinels + companion validity mask (the "Option/
  NULL bitmap" pattern), never a magic float.

## C021 — Defined behaviour on empty input (fixes D3)

A correctness tool must not throw on a legal-but-empty comparison.

**Action**: guard reductions. `ulp_distance` on 0 elements returns an empty
`int64` tensor; `summary`/`histogram`/sweep treat empty as
`{max:0, mean:0, counts: zeros}` and `max_ulp_observed` stays at its identity
(0). No `.max()` on empty.

- **(U)** Previously this raised `RuntimeError`; now it returns a well-formed
  zero result. Strictly more forgiving — cannot break a working caller, fixes a
  latent crash. (If a caller *relied* on the exception, that would be perverse;
  none do.)
- **(S)** Removes an unhandled-exception path from a library function.
  Big-org practice: total functions — defined output for every input in the
  domain, including the empty set.

## C022 — Compute ULP once per iteration in the Pareto sweep (fixes D5)

**Action**: compute the per-iteration ULP tensor **once**, derive both the
mismatch decision and the max/NaN stats from it (or have `check` return the
distances it already computed). Remove the duplicate pass added in C012.

- **(U)** No observable change to results — same numbers, fewer cycles. The
  reported `consistency_rate`/`throughput` are unaffected because the
  measured region is the kernel launch, not the validation; this only trims
  validation overhead.
- **(S)** Halves the dominant non-launch cost of the sweep
  (O(presets·iters·N) → one ULP pass per iter). Big-org practice: no redundant
  O(N) work in a hot loop; compute-once, reuse.

## C023 — Exact-integer histogram binning in Python (fixes D6)

**Action**: replace `floor(log2(x))` with an exact integer `floor_log2`
(bit-length − 1 on `int64`), mirroring the CUDA `while(t>>=1)` loop bit-for-bit.

- **(U)** Output histogram is identical on this platform (verified) and now
  *provably* identical to the kernel on all platforms. No schema/field change.
- **(S)** Removes dependence on libm `log2` exactness at power-of-two
  boundaries. The Python mirror becomes a faithful spec of the kernel, not an
  approximation of it. Big-org practice: when an exact integer operation
  exists, never validate against a floating-point proxy.

## C024 — Tests that lock the cross-layer contract

**Action**: add a self-contained test (no GPU) that asserts, for a curated set
including ±0, ±inf, NaN, denormals, and power-of-two ULP gaps:
1. Python `ulp_distance` == host-C++ `ulp_distance` (compiled & run in-test).
2. Python `ulp_to_bin`/histogram == CUDA `ulp_to_bin` for all bins.
3. `f_ulp_bounded` (host emu of the C019 version) agrees with Python at each
   tolerance preset in `pareto_solver.cuh`.
4. Empty-input and all-NaN inputs return defined results.

- **(U)** Pure addition; gives users a regression gate so a future edit can't
  silently re-introduce D1/D2/D6.
- **(S)** Encodes the "one ULP definition across CUDA × Python" invariant as an
  executable check. Big-org practice: differential testing of the
  reference implementation against the production kernel.

---

## Order of operations

C019 (CUDA unify) → C020 (int64 + sentinel) → C021 (empty) → C022 (dedupe) →
C023 (integer bins) → C024 (lock with tests). Re-run full integration after
each, then regenerate the `git am` patch.
