# CAST-4A fitter and cache readiness

This additive step packages the historical smooth8 fitter, checks synthetic
numerical parity, and audits the 380 named saved parent computations. It does
not fit new real recordings, construct new fold populations, or run coverage or
classification experiments. The original grouped design and its hash lock stay
unchanged.

Reusable modules:

- `abvid.cast.smooth_fit.fit`: historical smooth8 optimization with the same
  four starts, loss, 300 steps, checking streams and winner rule.
- `abvid.cast.roles`: exact training roles, complete whole-site exclusion,
  same-parent fit ancestry and full-population calibration checks.
- `abvid.cast.reuse_chain`: existing source-file, fit, calibration, recipe,
  environment and historical-verification audit.
- `abvid.cast.readiness`: validate the canonical design, parity evidence and
  audit inputs, then save a reproducible per-parent/per-fold reuse decision.

The [audit contract](audit.json) pins the existing canonical experiment
configuration and [chain contract](chain_contract.json). It is an additional
implementation/provenance record, not a replacement scientific protocol or a
change to the historical replay command.

Use a separate CAST environment because the shared `.venv` may serve the
incompatible baseline profile. Resolve its external root through `abvid.paths`:

```sh
CAST_ENV="$(.venv/bin/python -c 'from abvid.paths import roots; print(roots()["artifact"] / "environments/cast-readiness-20261007")')"
UV_PROJECT_ENVIRONMENT="$CAST_ENV" uv sync --frozen --extra cast
"$CAST_ENV/bin/python" -m abvid.cast.readiness audit --config repo:experiments/cast_coverage/grouped_coverage/readiness/audit.json
"$CAST_ENV/bin/python" -m abvid.cast.readiness verify --config repo:experiments/cast_coverage/grouped_coverage/readiness/audit.json
```

Both audit and verification reread existing candidate artifacts and validate the
full declared content/role chain. They do not execute historical audit runners,
which also generate and evaluate populations. Existing output must replay
identically; it is never overwritten. Input or environment changes fail closed.

Reuse decisions cover anchored fitted/calibrated parameter records and original
smooth8 artifacts only. Later v9/v11 reconstructed waveforms lack independently
anchored historical byte hashes and are explicitly excluded. Every new fold
must recompute its population statistics, scales, children and prototypes.

Evidence remains external under `artifact:cast/grouped_coverage/`:

- `cast4a_smooth_fit_parity_20261007_v1`: three full synthetic parity records.
- `cast4a_readiness_validation_20261007_v1`: focused tests and the retained
  first attempt that correctly rejected the baseline package versions.
- `cast4a_readiness_20261007_v1`: per-parent checks, fold eligibility, source/test
  snapshots, input references and a commit/environment/hash lock.

See [the readiness report](../../../../reports/CAST_fit_readiness.md) for results
and limits. Historical replay certificates are checked as existing evidence;
the saved real-data optimizations and waveforms are not regenerated in this step.
