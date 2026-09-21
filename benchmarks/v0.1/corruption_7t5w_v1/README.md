# Controlled-corruption track: 7 tracked / 5 wheeled sessions

This track reuses the immutable native-real manifest, grouped outer-fold models,
and decision thresholds. It evaluates recorded-background addition and simulated
microphone responses; it does not train a new model or consume locked recordings.

## Reproduce

From the repository root, with the native-real benchmark artifacts, original
reviewed source WAVs, four background WAVs/sidecars, and the validated PANNs
checkpoint installed locally:

```bash
uv sync --extra dev --extra inspection --extra pretrained
uv run python scripts/evaluate_controlled_corruption.py
uv run python scripts/verify_corruption_benchmark.py
uv run python scripts/summarize_corruption_benchmark.py
```

The runner can resume completed conditions only when all recorded inputs,
configuration, protocol and implementation hashes still match. For changed inputs
use a new `--output benchmarks/v0.1/NEW_VERSION` for all three commands. Keep output
directory basenames unique, since local tensors use `runs/OUTPUT_BASENAME/`.
Checkpoint/media files are local dependencies, not bundled redistributable assets.

## Evidence

- [`run_lock.json`](run_lock.json): preregistered configuration, code/manifest and
  checkpoint hashes, native-window checksums, background source/rights records.
- [`experiment.json`](experiment.json): run status, seed, commit, environment,
  runtime, split/model references and evidence limits.
- [`results.json`](results.json): all per-fold, per-class, per-session and
  worst-session metrics, confusion matrices, macro precision/recall/F1, and
  descriptive session-bootstrap intervals.
- [`report.md`](report.md): compact per-condition comparison.
- [`aggregate_summary.json`](aggregate_summary.json): equal-background curves,
  changes relative to native performance, paired descriptive session intervals.
- [`accuracy_vs_added_background_snr.png`](accuracy_vs_added_background_snr.png):
  environmental backgrounds and competing traffic shown separately.
- [`verification.json`](verification.json): exact source-window and corruption
  reconstruction, prediction replay and held-out membership checks.
- `conditions/*.json`: every deterministic crop, SNR, common gain and waveform
  hash; local prediction/feature tensors are identified by SHA-256.

SNR means native-window-to-added-background ratio, not isolated engine SNR. The
whole native window already includes its original ambience. All four backgrounds
are excluded from fitting the current probes, but are not newly sealed assets
relative to historical ABVID work or external encoder pretraining. Traffic can
contain another vehicle class. Microphone curves are explicitly simulated.

See the [protocol](../../../docs/benchmark_v0_1_corruption_protocol.md) and
[interpretation](../../../docs/benchmark_v0_1_corruption_results.md).
