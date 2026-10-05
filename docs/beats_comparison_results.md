# Frozen BEATs comparison — completed 2026-09-20

**Domain: real → real nested unseen recording-session development.** This is not
synthetic transfer, an untouched test, or field detection. All three representations
use the same 785 non-startup two-second windows, 7 tracked/5 wheeled sessions, and
35 outer/24 inner session-pair splits. No newly found candidate was admitted.

## Primary results

Train-only weighted scaler/logistic regression; C selected in inner folds; fixed 0.5
threshold. Encoders frozen. No preprocessing search in this comparison.

| Representation | Balanced accuracy | Tracked recall | Wheeled recall | Worst session/context |
|---|---:|---:|---:|---:|
| PANNs semantic35 control | 53.73% | 78.35% | 29.10% | 0% |
| PANNs full2048 control | 48.02% | 69.09% | 26.95% | 0% |
| BEATs mean-pooled768 | **57.05%** | **79.01%** | **35.08%** | **0%** |

BEATs minus semantic control: **+3.32 percentage points**, paired session-bootstrap
95% interval **−17.56 to +23.03 points**. This does not establish a dependable gain.
BEATs overall descriptive BA interval is 41.61–73.47%. Only twelve independent
sessions exist; overlapping windows and fold contexts are not independent samples.
Bootstrap resamples session-mean recalls within class, not windows or folds.

Against the full PANNs embedding, the paired difference is +9.03 points, interval
+0.26 to +17.94 points. That is the secondary comparison, not a reason to disregard
the inconclusive comparison against the stronger semantic control.

The preregistered secondary five-C ensemble scored 55.66% for BEATs, 54.15% for
PANNs semantic35 and 48.53% for full2048. No post-hoc rule was promoted.

## Where performance moved

Mean recall across each session's held-out training contexts; final column shows the
minimum context for BEATs. These are not independent new evaluation sessions.

| Development session | PANNs semantic | BEATs | BEATs minimum |
|---|---:|---:|---:|
| Goodwood/Abarth | 30.61% | 83.67% | 71.43% |
| StuG | 82.50% | 92.50% | 87.50% |
| T72/BMP3 | 92.17% | 43.19% | 31.88% |
| T80/MT-LB | 74.67% | 71.67% | 56.67% |
| AMX30 | 72.41% | 93.28% | 83.62% |
| Ford Model T | 11.56% | 2.04% | 0% |
| Abrams Bright Star | 87.83% | 100% | 100% |
| Stryker Pinon | 6.29% | 10.34% | 3.73% |
| HMMWV M1151 | 16.67% | 61.17% | 44.87% |
| Bradley/Abrams | 65.80% | 99.60% | 99% |
| Maserati | 80.39% | 18.16% | 10.17% |
| MPF Tankodrome | 73.10% | 52.87% | 25.29% |

BEATs helps HMMWV substantially, but trades away Maserati and T72/BMP3 performance;
Stryker remains poor. The evidence still points to session-dependent generalization,
not an encoder-capacity fix. Do not delete difficult sessions to improve the score.

## Reproducibility and limitations

- [Preregistered protocol](beats_comparison_protocol.md) and
  [configuration](../configs/benchmark_beats_v1.yaml).
- [Versioned machine-readable results](../benchmarks/v0.1/beats_comparison_v2/results.json),
  [experiment metadata](../benchmarks/v0.1/beats_comparison_v2/experiment.json), and
  [artifact hashes](../benchmarks/v0.1/beats_comparison_v2/artifact_sha256.json).
- Local feature cache and 35-fold model states: `runs/beats_comparison_v2/`.
  Full metric dictionaries, confusion matrices, F1, class/session results and splits
  are saved for every representation; large tensors/weights are not redistributed.
- 210 saved head/fold replays passed (3 representations × 35 folds × 2 head rules).
  Splits matched exactly and the PANNs semantic control reproduced exactly.
- Repeated BEATs extraction maximum error 0; single-versus-batch maximum absolute
  difference 7.75e-7. All 785 input WAV hashes checked; no startup/protected overlap.
- Apple M3 Pro CPU, four Torch threads/one BLAS thread: feature extraction 18.99 s;
  experiment 60.24 s, excluding downloads/model loading and final packaging.
  These are offline batch timings, not streaming inference latency.
- BEATs has 90,311,792 parameters, all frozen. Only the small logistic head is fit.
  Code comes from pinned official Microsoft source; checkpoint comes from a pinned
  public mirror because the official download returned 403. Its mirror SHA256 is
  verified, but equivalence to an official checksum was not independently verified.
  AudioSet pretraining/source overlap is unknown for both encoders.

Reproduce without replacing this run:

```bash
uv sync --extra dev --extra pretrained
uv run python scripts/fetch_beats.py
uv run python scripts/run_beats_comparison.py --output runs/beats_comparison_repeat --package benchmarks/v0.1/beats_comparison_repeat
uv run pytest -q
```

Requires the existing reviewed local source corpus and preprocessing-grid control
artifacts. The scripts stop on mismatched hashes/roles rather than silently changing
the experiment. Frozen preprocessing-grid/native benchmark artifacts are unchanged.
The first output directory (`beats_comparison_v1`) is retained for traceability:
its artifact index mistakenly included mutable macOS `.DS_Store` metadata. Packaging
now excludes that file. The full same-protocol run was repeated into `v2`, reproducing
all model metrics exactly; no model, selection rule, data or hyperparameter changed
in response to scores. All 45 run/package artifact hashes passed verification.

**M7 gates 2–4 still fail.** Confirmation was not run. T90M/JLTV remain untouched.
No new deployment default, hierarchical/open-set model or presence detector was built.
Next evidence: independent heavy-wheeled recordings and real human responses from the
[blind listening pilot](blind_listening_pilot.md), not another post-hoc grid.
