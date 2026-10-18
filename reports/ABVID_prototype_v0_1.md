# ABVID real-data benchmark prototype v0.1

Completed 2026-10-05. A runnable car/truck model is saved at
`artifact:benchmarks/abvid_real_beats_v0_1`, with WAV inference and an evaluation
report. [Commands and fixed protocol](../experiments/benchmark_prototype/README.md).

The build fitted 30 StandardScaler/logistic-regression heads on the existing
hash-verified BEATs feature cache: five fixed seeds × six IDMT held-out groups,
190 cars and 190 trucks per training selection. BEATs remains frozen. This
reproduces the existing H1 real-data arm through the current shared library;
the scientific recipe, populations and splits are unchanged.

| IDMT grouped development evaluation | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Frozen BEATs + trained linear head | 53.08% | 65.87% | 66.56% |
| Always-car control | 47.27% | 50.00% | 89.74% |

These scores average the six group scores and five training-selection seeds
equally. All 4,413 admitted IDMT car/truck files are evaluated once per seed.
The macro-F1 95% interval is **47.97–58.31%**, inherited from the existing H1
group/selection bootstrap after exact prediction reproduction. It is conditional
on these fits and six conservative provenance groups. Original recording
sessions and independent physical vehicle identities remain unknown.

The class imbalance explains why always-car has high accuracy. The classifier
improves macro-F1 and balanced accuracy, with substantial classification error
remaining. These data have already been exposed during development.

## Verification

- **30 focused tests passed** across the prototype, H1, BEATs and canonical
  configuration suites.
- All 30 classifier probability sets exactly match archived H1 held-out IDMT
  predictions; maximum absolute difference: **0**.
- All saved numeric heads reload with exactly the same predictions.
- Twelve predeclared raw WAVs (one car and truck per group, selected by file-ID
  order) reproduce processed waveform hashes, embeddings and probabilities
  exactly; maximum embedding and probability differences: **0**.
- The standalone `predict` command successfully loads the saved bundle and
  matches cached predictions on the fixed fold-0 car/truck pair. The car is
  correctly predicted as car; the truck is also predicted as car. That error
  is retained in the saved checks and benchmark.
- Build, fitting and verification took **7.5 seconds** using the existing cache
  on the development macOS arm64 machine. Python 3.11.13, Torch 2.11.0,
  NumPy 1.26.4 and scikit-learn 1.4.2; CPU inference uses four Torch threads,
  and classifier fitting uses one BLAS thread. Peak RAM was not measured.

The default WAV model is `BEATs768__real__seed42__fold0`, fixed by declaration
order. Its individual fold score is not the headline benchmark. The CLI uses a
centered two-second observation and always selects car or truck; vehicle
detection, rejection of other sounds and probability calibration are not
established.

## Saved evidence

All large inputs remain outside the checkout. Under the artifact directory:

| File | Contents |
|---|---|
| `heads.npz` | Thirty numeric scaler/linear heads; no pickle |
| `model.json` | Training/evaluation IDs, classes, dataset/version, split, seeds, configuration, checkpoint provenance, package versions, code hashes, commit and working-tree state |
| `evaluation.json` | Scores, uncertainty provenance, per-example raw parity checks, runtime and model-manifest hash |
| `EVALUATION.md` | Generated readable benchmark report |
| `CLI_VERIFICATION.json` | Standalone WAV prediction outputs and parity checks |

The BEATs checkpoint stays in `artifact:models/beats/`; its mirror and code
hashes are verified. Identity against an official upstream checkpoint checksum
remains unverified. The bundle relies on the configured local checkpoint and
baseline environment and is not a standalone redistributed dataset/model package.

The exact frozen H1 preprocessing is retained: channel mean, 8 kHz intermediate
rate, 16 kHz model rate, centered two seconds, no gain normalization or padding.
Scaler/classifier fitting uses training data only. No validation-based search,
augmentation, threshold tuning, encoder fine-tuning, new MELAUDIS scoring or
CAST transfer experiment was performed. Historical locks and runners were not
modified. Source-model dominance remains an untested hypothesis in this prototype.
