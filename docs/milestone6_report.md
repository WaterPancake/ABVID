# Milestone 6 — Noise-Invariant Representation Learning

Date: 2026-08-21

Status: first controlled, reproducible experiment complete

Performance claim: **limited to the exact held-out procedural sessions, corruption
partitions, and two native-real sessions recorded here**

## Experiment identity

| Field | Value |
|---|---|
| Run | `runs/m6_invariance_seed42` |
| Git commit | `88062f6c83eb028dbcada963b6c48834785499f0` |
| Seed | 42 |
| Device | CPU |
| Synthetic manifest SHA-256 | `5d8950eaae54a13aabd1f18a5d792ef747de332675943aad5dcb1b113714d18a` |
| Real manifest SHA-256 | `84fb1ef0e9f44df7f8d5c48400ea3f9b3bac3de2b3bdf5ae63004b073883c1dd` |
| Combined dataset version | `a0a38e6d900e6ea85c5eb395583ad2f44a376f1c82abb0553fe61036289b03d5` |
| Model | Small log-Mel CNN, 64-dimensional pooled embedding |
| Training | 10 epochs, batch 32, Adam, learning rate 0.001 |

The complete machine-readable configuration, grouped sample IDs, checkpoint, and
metrics are in `experiment.json`, `splits.json`, `models.pt`, and `metrics.json`
inside the run directory.

## Protocol

The comparison holds architecture, initialization seed, batch-order seed, grouped
split, class weighting, optimizer, and corrupted validation set fixed. The methods
differ only in training input and objective:

| Method | Training input and objective |
|---|---|
| Standard supervised | Clean input and weighted tracked/wheeled cross entropy |
| Augmentation only | Corrupted input and the same cross entropy |
| Representation invariance | Clean and paired-corrupted cross entropy, plus clean/corrupted consistency, same-vehicle hard-positive consistency, and a cosine triplet-margin hard-negative term |

The invariance weights are 0.5 for the same-event clean/corrupted cosine loss, 0.1
for same-vehicle hard-positive cosine loss, and 0.1 for the established triplet-margin
loss with margin 0.2. Hard positives use the same vehicle under the most different
available operating/session/nuisance metadata. Hard negatives use the opposite coarse
class while matching operating condition, background, geometry, microphone-response
state, and then SNR as closely as possible. All 111 eligible training anchors have a
hard positive and hard negative.

Splits remain grouped by `recording_session`:

| Partition | Base samples | Samples used by controlled protocol |
|---|---:|---:|
| Synthetic train | 524 | 111 in-distribution |
| Synthetic validation | 157 | 21 in-distribution |
| Synthetic test | 159 | 159 all-corruptions |
| Native-real train | 95 | Not used for Milestone 6 training |
| Native-real validation | 52 | Not used for model selection |
| Native-real test | 124 | 124 |

Training and validation exclude `road traffic`, `far_offset`, and all applied
microphone-response transforms. The isolated test partitions are:

| Condition | Support | Tracked / wheeled |
|---|---:|---:|
| Seen corruption | 28 | 22 / 6 |
| Unseen noise: road traffic | 11 | 10 / 1 |
| Unseen microphone response | 31 | 18 / 13 |
| Unseen environment proxy: far geometry | 27 | 4 / 23 |
| All held-out synthetic corruptions | 159 | 84 / 75 |
| Native real | 124 | 64 / 60 |

The environment partition is a propagation-geometry proxy, not an unseen-room test;
the generated corpus contains no impulse-response observations.

## Results

Balanced accuracy is used because several controlled nuisance partitions are class
imbalanced.

| Test domain | Standard supervised | Augmentation only | Representation invariance |
|---|---:|---:|---:|
| Seen synthetic corruption | 70.5% | 66.7% | **72.7%** |
| Unseen synthetic noise | 40.0% | **50.0%** | 35.0% |
| Unseen synthetic microphone | 70.3% | 65.4% | **76.9%** |
| Unseen synthetic environment proxy | **91.3%** | 73.9% | 89.1% |
| All synthetic corruptions | 77.7% | 73.4% | **78.9%** |
| Native-real held-out sessions | 29.0% | **48.4%** | 12.5% |

On all held-out synthetic corruptions, the invariance objective improves balanced
accuracy by 1.3 percentage points over clean-only supervision and 5.5 points over
augmentation-only training. It also increases mean paired clean/corrupted embedding
cosine similarity:

| Method | Mean paired cosine similarity | Mean normalized L2 distance |
|---|---:|---:|
| Standard supervised | 0.845 | 0.443 |
| Augmentation only | 0.890 | 0.356 |
| Representation invariance | **0.916** | **0.333** |

This is evidence that the objective made paired embeddings more similar on this
synthetic test set. It is not, by itself, evidence of a generally useful or
domain-invariant representation.

### Synthetic robustness by SNR

| SNR | Standard supervised | Augmentation only | Representation invariance |
|---:|---:|---:|---:|
| 30 dB | 93.8% | 75.0% | **100.0%** |
| 20 dB | 81.7% | 85.0% | **85.8%** |
| 10 dB | 81.2% | 79.2% | **84.4%** |
| 5 dB | **79.2%** | 66.7% | **79.2%** |
| 0 dB | 66.2% | **67.9%** | 66.8% |
| -5 dB | **87.5%** | 75.0% | 77.5% |
| -10 dB | **69.4%** | 67.2% | 64.4% |

The non-monotonic curve, especially at -5 dB, reflects different events and small
class supports in each SNR slice. This corpus is not a paired base-event SNR sweep,
so the table must not be interpreted as a clean causal SNR response.

### Native-real transfer

The fixed real test sessions are the same as Milestone 5: AMX-30 tracked audio and
Maserati GranTurismo wheeled audio. Confusion matrices use rows as true classes and
columns as predictions in `[tracked, wheeled]` order:

| Method | Confusion matrix | Tracked recall | Wheeled recall |
|---|---|---:|---:|
| Standard supervised | `[[36, 28], [59, 1]]` | 56.2% | 1.7% |
| Augmentation only | `[[62, 2], [60, 0]]` | 96.9% | 0.0% |
| Representation invariance | `[[16, 48], [60, 0]]` | 25.0% | 0.0% |

The invariance model is the worst real-transfer model in this comparison. It improves
synthetic embedding consistency without closing the sim-to-real gap, and every method
still effectively fails the held-out wheeled session. The 48.4% balanced accuracy of
augmentation-only training is also not a useful transfer result: it is produced by
predicting almost everything as tracked.

## Reproducibility check

The complete CPU run was repeated from the manifest-hash-validated feature cache in
`.artifacts/m6_repeat.g1WXZE`. After excluding only `created_at_utc`, `metrics.json`
was identical. Every model checkpoint tensor was also exactly equal. The full test
suite passes with 51 tests.

## Conclusions and technical debt

Milestone 6 succeeds as an implemented, controlled experiment but not as a transfer
method. The paired objective creates the intended synthetic representation behavior
and gives a small aggregate synthetic improvement, yet it substantially worsens the
critical synthetic-to-real result.

The primary limitations are:

- only one random seed was evaluated;
- strict nuisance isolation leaves just 111 training and 21 validation examples;
- the unseen-noise partition has only one wheeled example, and the environment proxy
  has only four tracked examples;
- the SNR slices do not reuse identical base events;
- geometry is only a proxy for environment because no impulse-response corpus was
  present;
- synthetic train and test groups can contain the same vehicle identities, although
  never the same recording session;
- the native-real test still contains only one source session per class.

The next valid improvement is not a Milestone 7 hierarchy. It is to expand independent
real sessions, generate paired base-event nuisance/SNR sweeps, run multiple fixed
seeds, and repeat the same Milestone 6 comparison without selecting weights on the
real test set.
