# H2 backend repair status

2026-10-04. The working H2 backend derives from pyroadacoustics commit
`642f95ea83417b662906712c5902d7b7dd665092`. The GPLv3 [licence](LICENSE) and
[original file hashes](UPSTREAM.json) are retained. The upstream evidence and
original H2 design lock are unchanged.

**The pre-target numerical validation now passes, and execution r1 is frozen.**
The full paired corpus, all 40 heads and target scoring are also complete and
[independently verified](../../reports/ARTIFACT_INDEX.md#artifact-70e853936c31).
See the [H2 result report](../../reports/H2_source_path_results.md). Numerical
checks and transfer results do not establish calibration to a real road or vehicle.

## Ground-angle indexing: complete

In [simulatorManager.py](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/h2/backend/pyroadacoustics/simulatorManager.py),
`_precompute_complex_angle_reflection_filter_table` now retains both `w_tmp` and
`Rp` as complex128 arrays of shape `(179 angles, 512 frequencies)`. The old loop
overwrote one frequency vector at every angle. `_get_asphalt_reflection_filter`
now selects the same angle row from both tables, retaining all frequency bins.
Rounding and clamping to ±89° are preserved.

The exact indexing-only revision is retained under
[`backend_revisions/indexing_20261004_v1`](../../reports/ARTIFACT_INDEX.md#artifact-6fe26fdf29cd).
Its `simulatorManager.py` SHA256 is
`294e40c7db03ec472952d44ecd2b7f377025d130ecde2591812f706a5ceefeed`.
Only two methods differ from upstream in that revision. The current working copy
also contains the subsequent propagation edits listed below; the indexing-only
parity claims must not be applied to those additional edits.

Verification:

- [v2 report](../../reports/ARTIFACT_INDEX.md#artifact-4f8ff924317a) and its
  [applicable patch](../../reports/ARTIFACT_INDEX.md#artifact-4e726558a1ca)
  preserve the original indexing-only check.
- [v3 report](../../reports/ARTIFACT_INDEX.md#artifact-d01a915317e5) rechecks
  that retained revision and separately tests the indexing contract in the
  current working copy. It records both sets of code hashes.
- Nine indexing-only regressions cover all 179 angle rows, complex values,
  normal-incidence limits, 32 full filters across eight angles and four distances,
  rounding/clamping, isolation from unselected rows, frequency-bin preservation,
  detection of the original defect, and unchanged Material/direct-only behavior.
- Nine unchanged upstream manager tests run against that indexing-only revision.
- Three additional tests cover every working-copy table row, the 32 filters with
  its environment's speed of sound, and isolation/clamping at grazing incidence.
- The patch must reproduce the indexing-only file byte for byte; AST comparison
  must find only the two intended method changes. The metadata verifier also
  checks the H2 design and all 8,257 protected historical/encoder artifacts.

The equation checks establish table selection and implementation agreement,
not field accuracy. Recomputed filters can differ at roughly `1e-17`; equivalence
checks have explicit floating-point tolerances. Full-render hash determinism was
checked separately and passed for all 9,600 H2 renders. The initial v1 numerical checks passed but its patch
was malformed at the original file's missing trailing newline; v2 corrected and
round-trip checked the patch writer. v3 keeps the check reproducible after later
working-copy changes, without overwriting either historical report.

Run from the repository root with a new output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --no-project --python experiments/reproduction/.venv/bin/python --with-requirements experiments/h2/backend-check-requirements.txt python experiments/h2/verify_ground_indexing.py --output /tmp/abvid-ground-indexing-check
```

The isolated check uses NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.11.1 with
Python 3.11.13. Its resolved environment is saved with each result. H1 and root
environments are unchanged. The verifier defaults to the retained indexing-only
revision for historical parity, and explicitly loads the working copy for its
additional indexing checks; it does not silently select an installed package.

## Reflected path: numerical checks pass; no operator action needed

No new download, access approval, recording, equipment purchase or manual code
change is required from the operator. This is local software and analytic
verification work under the declared idealized point-source/flat-ground model.
Calibrating a particular real road would be separate empirical work.

For source-to-ground length `a` and ground-to-microphone length `b`, the H2 model
requires one geometric pressure-spreading factor `1/(a+b)`. The upstream product
`1/a * 1/b` effectively restarts spreading at the bounce point. The difference
changes the reflected/direct balance and thus their interference; normalizing
the final recording's RMS does not generally remove it.

The **current working copy has validated propagation corrections** beyond
the validated indexing revision:

1. Remove first-leg geometric attenuation and apply `1/(a+b)` at the receiver.
2. Initialize the second-leg delay from microphone height, not source height.
3. Use the environment's `c` in the ground formula and detect motion in all axes.
4. Correct Sinc table interpolation weights and the air-filter frequency scale;
   use 31 Sinc taps consistently across all paths.

The first version did not pass admission. The retained
[static physics probe](../../reports/ARTIFACT_INDEX.md#artifact-f43177175ab8) passes
its static impulse amplitude/delay checks at 5/20/50 m and air-filter checks, but
**fails four 3 kHz tone-amplitude checks**. The largest error is about 1.112 dB
for the unit-reflected component at 20 m, beyond the frozen 0.5 dB tolerance.
That probe exercises an ideal unit ground response; it does not validate the
actual ground response, moving propagation or real-world acoustics.

The subsequent [static probe](../../reports/ARTIFACT_INDEX.md#artifact-94ca43ba5ce7)
passes with a maximum tone error of 0.00882 dB. The
[complete renderer check](../../reports/ARTIFACT_INDEX.md#artifact-c84297060566)
passes 151 cases: independent image-path amplitude/delay fixtures including
unequal heights, actual static ground/air transfer (maximum magnitude error
0.0143 dB), all prescribed moving range/speed/direction corners (maximum frequency
error 0.539%), scalar/batched equivalence and exact replay. The fixed limits remain
0.5 dB and 1%. Checks operate **before RMS normalization** where appropriate.

All 480 source cases also pass. Fresh-process replay reproduces all 240 planned
validation observations byte for byte, with identical hashes in sequential and
four-worker generation. Fixed-order complex arithmetic resolves the initial
last-bit replay failures without quantizing waveforms. The inherited spherical
field coefficient's gains, and its distinction from a local passive plane-wave
coefficient, are disclosed in the [implementation amendments](../../experiments/h2_simulation/notes/IMPLEMENTATION_AMENDMENTS.md).

[Execution r1](../../reports/ARTIFACT_INDEX.md#artifact-570c30d67290) records the complete
patch, code, environment and evidence. Full-corpus verification, fixed feature
extraction and the 40-head lock were completed before H2 target scoring. The
separate final audit reconstructed predictions, metrics and paired intervals;
original upstream and historical experimental artifacts remain unchanged.
