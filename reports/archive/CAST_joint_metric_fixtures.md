# CAST-4B joint metric: synthetic fixture verification

This implements only the fixture-verification phase of
[CAST-4B](../../CAST_ROADMAP.md#cast-4b-test-feature-relationships-and-representation-coverage).
All inputs are fixed synthetic arrays. This is implementation evidence; it
establishes no source coverage, representation coverage or classification
transfer result. Existing CAST banks, runners, W1 definitions and thresholds
are unchanged. There is no real-data, archived-artifact or network access.

## Metric and freeze contract

The reusable implementation is [joint.py](../../src/abvid/cast/joint.py).
`gaussian_mmd_squared` returns the **biased empirical squared MMD**:

```text
k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
MMD^2 = sum(k(X_i,X_j))/n^2 + sum(k(Y_i,Y_j))/m^2
        - 2*sum(k(X_i,Y_j))/(n*m)
```

Diagonals are included. All pairs are evaluated, including for unequal sample
counts and singleton/degenerate inputs. Only negative floating-point roundoff
is clipped to zero. This is a descriptive discrepancy with finite-sample bias;
it provides no significance claim for correlated recording groups and no
universal MMD pass threshold. Time and memory are quadratic in sample count.

`descriptor_vector` concatenates the existing four descriptor families in
their declared order. For family f with d_f coordinates and frozen training
scales s_f, its block is `x_f / s_f / sqrt(4*d_f)`. Squared distance therefore
weights families equally and coordinates equally within each family. The
function accepts scales; it never estimates them from compared samples.
Subtracting one shared training mean would cancel in all pairwise distances.

The declared **data-derived-at-freeze-time bandwidth rule** is the median
strictly positive, unordered pairwise Euclidean training-vector distance.
`median_bandwidth(training, seed=..., max_samples=1024)` uses all training rows
up to that cap. Above the cap, it selects rows without replacement using a
local `numpy.random.Generator(PCG64(seed))` and retains input order. Duplicate
rows' zero distances are excluded explicitly. An entirely collapsed
calibration sample is rejected rather than assigned an arbitrary bandwidth.
The returned value is sigma, with the kernel convention shown above.

A future separately authorized experiment must freeze training identities and
row order, training-only scales/floors, family weights, estimator, seed,
sampling cap and numeric sigma before held scoring. Reuse that sigma and
preprocessing across comparison arms in the same role; never recalibrate
from generated or held samples. This module performs no storage or role
admission. Its callers must enforce those provenance restrictions.

Scoring itself uses no randomness. The fixtures calibrate scales and bandwidth
on a separate, fixed synthetic training sample (PCG64 seed 4217, calibration
seed 42). The determinism check also exercises seeded calibration subsampling.

## Verification fixtures

[test_joint.py](../../tests/cast/test_joint.py) contains these tests (12 cases):

- `test_mmd_detects_dependence_with_exactly_matched_family_marginals`: exact
  empirical marginal matches, with cross-family correlation reversed; every
  existing per-family W1 is zero while MMD squared exceeds 0.05.
- `test_mmd_detects_collapsed_samples_and_degenerate_mixtures`: a point mass
  and a two-point mixture each differ from a spread reference, with finite
  MMD squared above 0.02; each collapsed distribution still matches itself.
- `test_mmd_detects_excessive_spread_with_frozen_bandwidth`: eightfold spread
  produces MMD squared above 0.2 with the same frozen kernel.
- `test_identical_empirical_distributions_have_zero_mmd`: identical and
  jointly row-permuted samples give zero within absolute tolerance 1e-14.
- `test_seeded_bandwidth_and_mmd_repeat_bit_for_bit`: repeated seeded
  calibration and scoring have identical float hex encodings; different
  seeds exercise different calibration subsets.
- `test_marginal_w1_misses_missing_joint_modes_that_mmd_catches`: four equally
  weighted corners versus two diagonal corners have zero W1, unit spread
  ratios and complete marginal coverage, but MMD squared exceeds 0.01.
- `test_biased_estimator_matches_analytic_gaussian_with_unequal_counts`:
  independent closed-form point-mass check, including a singleton.
- `test_descriptor_vector_uses_frozen_scales_and_equal_family_weights`:
  unequal family dimensions, fixed coordinate order and unchanged inputs.
- `test_median_bandwidth_uses_only_positive_unordered_training_distances`:
  hand-computed bandwidth, collapsed calibration and invalid options.
- `test_mmd_rejects_invalid_matrices_dimensions_and_bandwidths`.
- `test_descriptor_vector_rejects_missing_misaligned_or_invalid_scales`.

These numeric separation margins are fixture assertions, not scientific
acceptance criteria. The matched-marginal controls use the existing
`coverage.compare` implementation, preserving its historical definition.

## Graded selection

```sh
uv run --extra cast pytest -m 'not archive' --ignore=tests/test_h1.py --ignore=tests/test_simulation.py --ignore=tests/test_benchmark_prototype.py -q
```

The run uses the existing pinned CAST environment through
`UV_PROJECT_ENVIRONMENT=/Users/magi/workspace/ABVID/.venv`, with `UV_OFFLINE=1`,
`UV_NO_SYNC=1` and a temporary writable `UV_CACHE_DIR`. Data, artifact and
archive roots point to unused temporary fixture locations. A temporary JUnit
report records the exact outcome. No lockfile update or dependency download
is needed. Harness-baseline maintenance remains human-only; test changes
require line-by-line human review before merge under [AGENTS.md](../../AGENTS.md).

Result: **733 passed, 1 failed, 3 skipped**, exit status 1, in 68.372 seconds
(JUnit wall time). The sole failure is
`tests/test_harness_integrity.py::test_collected_test_count_matches_baseline`:
**735 selected tests versus the committed CAST baseline of 678**. All 12 new
fixture cases pass. The baseline-schema and frozen-file hash canaries pass.
The three skips comprise two module collection skips (baseline-only `sklearn`
and `numba` unavailable in the CAST environment) and the existing local CAST
artifact-bundle check. JUnit records 737 entries because it includes those two
collection skips alongside the 735 selected test items. The baseline and all
other protected files remain unchanged.

## Commit delivery

The three files were committed directly as `c60a682` on
`cast4b/joint-metric-fixtures`; the `/tmp` bundle route was not used. On
2026-11-08 that branch was merged with `cast4a/serialization-revision` into
`cast/diagnosis-20261102` (merge `1153aea`). The user ran the harness baseline
refresh in the merge commit: CAST 723 → 748, baseline 756 → 781 for both
branches together.
