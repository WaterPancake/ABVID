# H2 backend: ground-angle indexing repair

2026-10-04. This is the working H2 backend copy of pyroadacoustics commit
`642f95ea83417b662906712c5902d7b7dd665092`. The original GPLv3 [licence](LICENSE)
and [upstream file hashes](UPSTREAM.json) are retained. The immutable reference
remains under `../evidence/pyroadacoustics`; the original H2 design lock is unchanged.

**Indexing is repaired and checked. Full H2 execution remains blocked by the
other physical and implementation gates.** This backend has not generated an H2
vehicle corpus or been used to train/evaluate a classifier.

## What changed

Only two methods in [simulatorManager.py](pyroadacoustics/simulatorManager.py)
have changed behavior:

- `_precompute_complex_angle_reflection_filter_table` stores `w_tmp` and `Rp`
  as complex128 arrays with shape `(179 angles, 512 frequencies)`, filling each
  angle's row instead of overwriting one frequency vector.
- `_get_asphalt_reflection_filter` selects the same angle row from **both**
  arrays. Each selected row retains all frequency bins. Existing rounding and
  clamping to ±89° are preserved.

Spreading, the hardcoded 343 m/s in the ground formula, filter length, material
priors, atmosphere, trajectory construction and all other methods are unchanged.
This is a partial implementation of the frozen H2-R1 repair contract.

## Verification

Use [verification v2](../results/ground_indexing_20261004_v2/verification.json),
its [applicable patch](../results/ground_indexing_20261004_v2/ground_indexing.patch),
and the [individual equation checks](../results/ground_indexing_20261004_v2/filter_equation_checks.json).

- Nine targeted regression tests pass: complex table shape/normal-incidence
  limits; all 179 angle rows versus direct equations; 32 complete filter checks
  across eight angles and four distances; rounding/grazing boundaries; exclusion
  of unselected angle rows; preservation of individual frequency bins; detection
  of the original defect; unchanged Material-object branch; unchanged direct-only
  artificial signal output.
- Nine unchanged upstream SimulatorManager tests pass against this exact working
  package, with its import path checked.
- Maximum absolute FIR-coefficient difference in the 32 equation checks is about
  `1.11e-16`. This checks the retained equations and indexing, not physical accuracy
  of the entire reflection model.
- The patch is applied to a fresh reference copy and reproduces the modified file's
  bytes exactly. AST comparison limits method changes to the two named methods.
- The H2 design/manifests and all 8,257 protected historical/encoder artifacts still
  pass verification.

Recomputed equivalent filters can differ at roughly `1e-17` in this numerical
environment; equivalence tests use rtol `1e-13`, atol `1e-14`. Material-branch and
direct-only reference comparisons remain exact. This does not discharge the
future full-render waveform-hash determinism gate. The initial v1 numerical run
passed but produced a malformed diff at the original file's missing trailing
newline; v2 repairs and checks the diff writer without changing the backend fix.

Tests use isolated, pinned NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.11.1 with
Python 3.11.13; the full resolved check environment is saved with the result. The
existing H1 and root environments were not modified. Reproduce from repository root
with a new output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --no-project --python experiments/reproduction/.venv/bin/python --with-requirements experiments/h2/backend-check-requirements.txt python experiments/h2/verify_ground_indexing.py --output /tmp/abvid-ground-indexing-check
```

## Reflected-path spreading: work remaining

No operator download, permission request, field recording, equipment purchase or
manual code change is needed to perform this correction. It is local software and
analytic verification work. The user requested the indexing fix and an explanation
of this remaining work; spreading has therefore been left unchanged in this pass.

The next bounded implementation should:

1. Use one geometric pressure-spreading factor for the total reflected distance,
   `1/(a+b)`, replacing the current product `1/a * 1/b`. Keep reflection and air
   absorption as separately controlled effects; preserve the direct path.
2. Disable air absorption and substitute a unit reflection response in an analytic
   fixture. Compare the delayed reflected waveform against the mirrored point-source
   solution for a rigid flat plane. Sweep source/receiver heights and distances;
   check amplitude and delay **before** receiver RMS normalization. A sequential
   implementation of spreading could equivalently use `1/a` then `a/(a+b)`; it must
   not restart spreading with another `1/b` at the ground.
3. Check the reflection/direct amplitude ratio and combined interference, then
   restore the actual ground/air filters and repeat the frozen physical checks.
   Record the patch and reference discrepancies before admitting the backend.

This verifies the declared idealized model. Calibrating it to a particular real
road would be separate empirical work; it is not a prerequisite for repairing this
spreading law. Common speed of sound, exact trajectory/source adapters, full
physical/manipulation checks and execution/corpus locks also remain outstanding.
