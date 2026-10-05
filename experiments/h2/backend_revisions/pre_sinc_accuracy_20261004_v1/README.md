# H2 backend repair status

2026-10-04. The working H2 backend derives from pyroadacoustics commit
`642f95ea83417b662906712c5902d7b7dd665092`. The GPLv3 [licence](LICENSE) and
[original file hashes](UPSTREAM.json) are retained. The upstream evidence and
original H2 design lock are unchanged.

**The ground-angle indexing repair is verified. The subsequent propagation work
is not fully validated.** No H2 vehicle corpus, fitted head or target score has
been produced by this backend. Analytic test signals are not a vehicle corpus.

## Ground-angle indexing: complete

In [simulatorManager.py](pyroadacoustics/simulatorManager.py),
`_precompute_complex_angle_reflection_filter_table` now retains both `w_tmp` and
`Rp` as complex128 arrays of shape `(179 angles, 512 frequencies)`. The old loop
overwrote one frequency vector at every angle. `_get_asphalt_reflection_filter`
now selects the same angle row from both tables, retaining all frequency bins.
Rounding and clamping to ±89° are preserved.

The exact indexing-only revision is retained under
[`backend_revisions/indexing_20261004_v1`](../backend_revisions/indexing_20261004_v1/).
Its `simulatorManager.py` SHA256 is
`294e40c7db03ec472952d44ecd2b7f377025d130ecde2591812f706a5ceefeed`.
Only two methods differ from upstream in that revision. The current working copy
also contains the subsequent propagation edits listed below; the indexing-only
parity claims must not be applied to those additional edits.

Verification:

- [v2 report](../results/ground_indexing_20261004_v2/verification.json) and its
  [applicable patch](../results/ground_indexing_20261004_v2/ground_indexing.patch)
  preserve the original indexing-only check.
- [v3 report](../results/ground_indexing_20261004_v3/verification.json) rechecks
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
checks have explicit floating-point tolerances. Full-render hash determinism is
a separate outstanding gate. The initial v1 numerical checks passed but its patch
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

## Reflected path: no operator action needed; validation unfinished

No new download, access approval, recording, equipment purchase or manual code
change is required from the operator. This is local software and analytic
verification work under the declared idealized point-source/flat-ground model.
Calibrating a particular real road would be separate empirical work.

For source-to-ground length `a` and ground-to-microphone length `b`, the H2 model
requires one geometric pressure-spreading factor `1/(a+b)`. The upstream product
`1/a * 1/b` effectively restarts spreading at the bounce point. The difference
changes the reflected/direct balance and thus their interference; normalizing
the final recording's RMS does not generally remove it.

The **current working copy already has draft propagation corrections** beyond
the validated indexing revision:

1. Remove first-leg geometric attenuation and apply `1/(a+b)` at the receiver.
2. Initialize the second-leg delay from microphone height, not source height.
3. Use the environment's `c` in the ground formula and detect motion in all axes.
4. Correct Sinc table interpolation weights and the air-filter frequency scale.

These edits are not an admitted H2 execution backend. The retained
[static physics probe](../results/physics_probe_20261004_v1/report.json) passes
its static impulse amplitude/delay checks at 5/20/50 m and air-filter checks, but
**fails four 3 kHz tone-amplitude checks**. The largest error is about 1.112 dB
for the unit-reflected component at 20 m, beyond the frozen 0.5 dB tolerance.
That probe exercises an ideal unit ground response; it does not validate the
actual ground response, moving propagation or real-world acoustics.

Remaining work is to correct and independently validate the interpolation
accuracy, retain the failed evidence, and repeat the frozen static/moving checks.
The reflected path must be checked against an independent mirrored-source
solution with air off and a unit ground response, including unequal heights,
amplitude, delay, direct/reflected ratio and combined interference **before RMS
normalization**. Then restore the ground/air filters, check motion, all prescribed
range/speed corners, deterministic replay and the frozen paired validation slots.
Record all corrections and freeze the executable code/environment before corpus
generation. No change to a tolerance or experimental factor is justified by this
indexing check. H2 training and target scoring remain behind those gates.
