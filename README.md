# ABVID passive vehicle-acoustics benchmark

This repository builds deterministic paired audio for the first research question:
do wheeled and tracked vehicle signatures remain identifiable after realistic
environmental and microphone corruption?

Milestones 1-5 provide an offline augmentation engine, leakage-safe classical/CNN
baselines, a separate controlled procedural source corpus, deterministic microphone-
array baselines, and a first grouped synthetic-to-real experiment. That transfer run
is a negative result on a small reviewed corpus and is not a general tracked-versus-
wheeled performance claim. Milestone 6 adds a controlled paired-representation
comparison against clean-only and augmentation-only training.

The current deliverable is **ABVID Benchmark v0.1 plus an inspectable offline replay
demo**. The benchmark contract, source roles, current gate, and release procedure are
documented in [`docs/benchmark_v0_1.md`](docs/benchmark_v0_1.md); the demo design is in
[`docs/replay_demo_plan.md`](docs/replay_demo_plan.md). Milestone 7 remains blocked.

The 7/5 native benchmark now has a
[failure diagnostic report](docs/benchmark_v0_1_failure_diagnostics.md) and a
[completed controlled-corruption track](docs/benchmark_v0_1_corruption_results.md):
30 conditions using real recorded backgrounds and separately simulated microphone
responses, with fixed held-out-session models. These are development stress tests,
not evidence of field-ready classification. The
[recording audit reel](docs/recording_audit_reel.md) supports human review.
The [current architecture and log-Mel diagnostics](docs/current_model_and_logmel_diagnostics.md)
explain the two-branch classifier and link to selected held-out examples with audio.
The [embedding/context follow-up](docs/embedding_context_results.md) compares full
PANNs embeddings, matched 2/4/8-second context, and direct 32 kHz extraction;
[preprocessing and model research](docs/audio_preprocessing_model_review.md)
documents the next controlled options. None of these follow-ups passes the gate.
Startup is now excluded from active training/evaluation with recordings and review
history preserved. The [completed 24-way preprocessing grid](docs/preprocessing_grid_results.md)
uses all 785 eligible windows without a session cap. Nested selection scored 51.86%
balanced accuracy for PANNs and 47.26% for classical features; the gate remains unmet.

Audit the live catalog roles without training a model:

```bash
uv run python scripts/audit_benchmark.py \
  --config configs/benchmark_v0_1.yaml \
  --targets data/targets
```

## Setup

The project requires Python 3.11 or newer. The checked-in `.python-version` selects
3.11 for `uv`:

```bash
uv sync --extra dev --extra inspection
```

An equivalent standard-library environment is:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,inspection]'
```

## Input layout and metadata

Audio discovery is recursive and accepts WAV, FLAC, OGG, AIFF, and AIF files. A
recommended target layout is:

```text
data/targets/
├── tracked/session_001/clip_001.wav
└── wheeled/session_002/clip_002.wav
```

The class is inferred from the first directory and the recording session from the
audio file's parent directory. For an explicit, future-proof identity, place a JSON
sidecar next to each target (for example, `clip_001.json`):

```json
{
  "vehicle_class": "tracked",
  "vehicle_model": null,
  "recording_session": "session_001",
  "operating_condition": "steady_speed"
}
```

Supported operating conditions are `idle`, `accelerating`, `decelerating`,
`steady_speed`, `startup`, `mixed`, and `unknown`. Mixed recordings can expose
reviewed time intervals instead of assigning one label to the entire file:

```json
{
  "vehicle_class": "tracked",
  "recording_session": "session_001",
  "operating_condition": "mixed",
  "condition_segments": [
    {
      "operating_condition": "idle",
      "start_seconds": 12.5,
      "end_seconds": 24.0,
      "notes": "Post-start stationary interval verified from the source video."
    }
  ]
}
```

When segments are present, each interval is a selectable target source and random
crops are constrained to its boundaries. The manifest records both the condition
and segment times.

Background categories are inferred from the first directory below
`data/backgrounds/`, or specified by a sibling sidecar:

```json
{
  "category": "road traffic"
}
```

Impulse responses are optional and are discovered recursively below
`data/impulse_responses/`. The generator never downloads any source data.

### Vetted source collection

`configs/audio_sources.yaml` is a reviewable source catalog. As of 2026-09-20 it
contains 29 vehicle records and 10 approved environmental recordings. Admission is
stricter than catalog inclusion: local development discovery currently contains
7 tracked and 5 wheeled reviewed sessions, while consumed and future locked sources
remain explicitly excluded. The first refreshed 7/5 native-real benchmark scored
39.70% mean balanced accuracy for the primary fusion and failed development gates
2-4. The catalog records source pages, recording sessions,
operating conditions, expected licenses, attribution, and output placement. Audio
files remain local and are ignored by Git.

The separate collector resolves current provider metadata, checks the expected
license exactly, downloads only explicitly selected sources, extracts audio from
video with `ffmpeg`, writes WAV files plus provenance sidecars, and maintains a
JSONL collection manifest. It is the only networked command in the project:

```bash
# Inspect records and current Commons/Internet Archive rights without downloading
uv run python scripts/collect_audio.py --list
uv run python scripts/collect_audio.py \
  --catalog configs/audio_sources.yaml \
  --all-approved --dry-run --output .artifacts/collection_preview

# Download the approved set into data/ (explicit network action)
uv run python scripts/collect_audio.py \
  --catalog configs/audio_sources.yaml \
  --all-approved --output data
```

The collector includes a review gate for sources with ambiguous rights. Do not
use `--include-review-required` for redistribution without resolving the license.
The Internet Archive tank recording is intentionally review-gated; the approved
tracked sample currently comes from the DVIDS page, which states PUBLIC DOMAIN.
Wikimedia may temporarily return HTTP 429 for large or repeated downloads; retry
later rather than bypassing the source/license checks.

After adding or refining catalog condition labels, synchronize already-collected
sidecars without redownloading audio:

```bash
uv run python scripts/sync_catalog_metadata.py \
  --catalog configs/audio_sources.yaml --output data
```

All waveform APIs use `[num_channels, num_samples]`. Target channels are always
preserved. Mono background and mono impulse responses may be explicitly broadcast
over target channels; incompatible multichannel sources raise an error instead of
being silently downmixed. For the one-microphone experiment,
`audio.background_channel: 0` explicitly selects and records channel 0 from every
background. Set it to `null` to require exact channel compatibility.

## Generate a dataset

Populate the three input directories, then run:

```bash
uv run python -m vehicle_audio.cli generate \
  --config configs/default.yaml \
  --targets data/targets \
  --backgrounds data/backgrounds \
  --impulse-responses data/impulse_responses \
  --output data/generated \
  --num-samples 1000 \
  --seed 42
```

The installed `vehicle-audio generate` command and `scripts/generate_dataset.py`
expose the same interface. Relative input source identifiers, recording sessions,
original and output sample rates, crop offsets, the sample seed, every sampled
augmentation parameter, and a complete configuration snapshot are stored in each
record.

Long recordings are seeked before decoding: only the selected crop plus a small
resampling margin is read, rather than loading an entire field recording for every
sample. Crop offsets are recorded both as original-rate frames and output-rate
samples.

Output is organized as paired events:

```text
data/generated/
├── manifest.jsonl
└── sample_000000_<seed>/
    ├── clean.wav
    ├── corrupted.wav
    └── metadata.json
```

`clean.wav` is the explicitly resampled and temporally cropped target before
corruption. `corrupted.wav` has the same sample rate, channel count, and length.
Both are float WAVs. The root JSONL record and the sample's `metadata.json` are
identical.

The corruption pipeline samples each enabled augmentation independently. It
supports random gain, controlled-SNR background mixing, low/high/band-pass filters,
random EQ, round-trip resampling, clipping, dynamic-range compression, impulse
response convolution, random cropping, and a microphone frequency response. Edit
`configs/default.yaml` to change ranges or set an augmentation's `enabled` flag or
`probability`.

Target selection defaults to `class_condition_balanced`: select tracked/wheeled
first, then an operating condition available for that class, then a source. This
prevents a class with more collected recordings from silently dominating generated
examples. Set `sampling.target_strategy` to `uniform_source` to recover plain
source-uniform sampling, or `class_balanced` to balance only the top-level class.

The `snr_db` field is the SNR at the controlled mixing stage. Later nonlinear or
frequency-selective microphone corruptions may change an SNR measured from the final
waveform.

## Tests

Tests use generated waveforms and do not require an audio corpus:

```bash
uv run pytest
```

They verify requested SNR, sample counts/rates, multichannel shape preservation,
finite clipping, required manifest fields, and same-seed byte reproducibility.

## Tiny end-to-end smoke test

Create deterministic toy vehicle, background, and impulse-response recordings:

```bash
uv run python scripts/create_toy_audio.py --output .artifacts/toy_inputs
```

Generate six paired examples through the real CLI:

```bash
uv run python -m vehicle_audio.cli generate \
  --config configs/default.yaml \
  --targets .artifacts/toy_inputs/targets \
  --backgrounds .artifacts/toy_inputs/backgrounds \
  --impulse-responses .artifacts/toy_inputs/impulse_responses \
  --output .artifacts/toy_generated \
  --num-samples 6 \
  --seed 42
```

Render a waveform and log-Mel comparison for one pair:

```bash
SAMPLE_DIR="$(find .artifacts/toy_generated -maxdepth 1 -type d -name 'sample_*' | sort | head -n 1)"
uv run python scripts/inspect_sample.py "$SAMPLE_DIR" \
  --output .artifacts/toy_sample_inspection.png
```

For multichannel audio, add `--channel N`; inspection selects that channel
explicitly and never averages channels together.

## Milestone 2: leakage-safe baselines

The training CLI supports two modular one-channel baselines:

- `classical`: MFCC, spectral centroid/bandwidth/rolloff, spectral flatness,
  harmonic peak concentration, RMS, zero-crossing rate, and crest factor followed
  by logistic regression;
- `cnn`: standardized log-Mel spectrogram, a small convolutional encoder, temporal
  pooling, and a linear classifier.

The default grouped split assigns complete `recording_session` values to exactly
one partition and refuses corpora with fewer than three sessions per class. The
old sample-level run under `runs/simple_cnn_channel0_seed42` is retained only as a
known-leaked historical artifact and must not be reported as performance evidence.

Train both baselines on the same grouped split:

```bash
uv run python scripts/train_baseline.py \
  --manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --output runs/m2_classical_grouped_seed42 \
  --model classical --channel 0 --epochs 30 --batch-size 64 \
  --learning-rate 0.01 --seed 42 --device cpu \
  --split-strategy grouped --group-field recording_session \
  --test-domain 'synthetic -> synthetic'

uv run python scripts/train_baseline.py \
  --manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --output runs/m2_cnn_grouped_seed42 \
  --model cnn --channel 0 --epochs 10 --batch-size 32 \
  --learning-rate 0.001 --seed 42 --device cpu \
  --split-strategy grouped --group-field recording_session \
  --test-domain 'synthetic -> synthetic'
```

Every run stores `experiment.json`, `splits.json`, `model.pt`, `metrics.json`, a
versioned feature cache, and `accuracy_by_snr.csv`. Metrics include accuracy,
balanced accuracy, per-class and macro precision/recall/F1, confusion matrices,
expected calibration error, negative log likelihood, Brier score, and SNR slices.

## Milestone 3: controlled procedural source corpus

The source generator is intentionally separate from augmentation and labels every
output `procedural_synthetic`; it is not real-world audio or a high-fidelity vehicle
simulator. The checked configuration crosses 4 tracked and 4 wheeled identities
with idle, acceleration, steady-speed, and deceleration states, two simulation runs,
and the same two geometries for every identity:

```bash
uv run python -m vehicle_audio.cli synthesize-sources \
  --config configs/procedural_vehicles.yaml \
  --output data/synthetic_targets \
  --seed 20260819

uv run python -m vehicle_audio.cli generate \
  --config configs/default.yaml \
  --targets data/synthetic_targets \
  --backgrounds data/backgrounds \
  --impulse-responses data/impulse_responses \
  --output data/generated/m3_controlled_seed42 \
  --num-samples 840 --seed 42
```

Each source sidecar and generated manifest preserves vehicle ID/class, engine state,
RPM and throttle ranges, speed range, acceleration, load, simulation run, and
source/listener geometry. Controlled holdouts use `--split-strategy holdout` and
select complete values of `operating_condition`, `background_category`, or
`vehicle_id`; exact commands and results are in
[`docs/milestones_1_3_report.md`](docs/milestones_1_3_report.md).

## Milestone 4: multichannel simulation and baselines

The array simulator writes synchronized tensors in `[num_channels, num_samples]`
form. It applies a known source direction and distance, per-microphone delay and
attenuation, independent impulse and frequency responses, real recorded background
crops, independent sensor noise, and exact controlled SNR. Azimuth is measured from
array broadside: `0` degrees is positive y and positive angles rotate toward positive
x. The checked configuration is a four-microphone uniform linear array, so evaluation
is intentionally restricted to the unambiguous front half-plane `(-90, 90)`.

Generate the checked, paired-SNR experiment corpus:

```bash
uv run python -m vehicle_audio.cli generate-array \
  --config configs/multichannel.yaml \
  --source-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --backgrounds data/backgrounds \
  --output data/generated/m4_array_paired_sessions_seed42 \
  --num-samples 896 --seed 42
```

With `paired_snr_sweep: true`, the sample count must be divisible by the number of
configured SNRs. Each base event reuses the same source crop, background crops,
geometry, array responses, and random seed at every SNR; only the controlled noise
scale changes. Source selection cycles through every recording session in each class
before reusing one, and every observation records its `base_event_id` and complete
array realization.

Run localization, beamforming, and the controlled classifier comparison:

```bash
uv run python scripts/evaluate_multichannel.py \
  --manifest data/generated/m4_array_paired_sessions_seed42/array_manifest.jsonl \
  --output runs/m4_multichannel_paired_sessions_seed42 \
  --seed 42 --epochs 30 --batch-size 64 --learning-rate 0.01 \
  --group-field recording_session \
  --test-domain 'procedural synthetic -> simulated multichannel synthetic (paired SNR, session-balanced)'
```

The comparison keeps the handcrafted feature extractor, linear classifier, split,
and training settings fixed while changing only the audio representation:

- one microphone;
- two/four microphones without beamforming, using mean feature fusion;
- two/four microphones with GCC-PHAT- or SRP-PHAT-steered delay-and-sum beamforming.

The evaluator also measures GCC-PHAT and SRP-PHAT angular error against the known
source azimuth. It saves machine-readable metrics, split groups, checkpoints, CSVs,
a versioned feature cache, and the three required plots. The checked experiment and
its limitations are documented in
[`docs/milestone4_report.md`](docs/milestone4_report.md).

## Milestone 5: real-world transfer protocol

Native real recordings are prepared separately from synthetic augmentation. The
preparer selects one explicit channel, resamples, and emits deterministic fixed
windows; it does not add corruption or infer an unavailable SNR:

```bash
uv run python -m vehicle_audio.cli prepare-real \
  --config configs/real_corpus.yaml \
  --targets data/targets \
  --output data/real_eval_v2
```

The resulting `real_manifest.jsonl` preserves recording session, source URL, license,
source hashes, window boundaries, content-review status, and normalized audio
provenance. `corpus_audit.json` fails closed unless there are at least three complete
recording sessions per class, complete provenance, and reviewed vehicle-audio segment
boundaries.

Once that audit passes, run the fixed protocol:

```bash
uv run python scripts/evaluate_transfer.py \
  --synthetic-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --output runs/m5_transfer_seed42 \
  --seed 42 --device cpu
```

The evaluator compares real-only training, synthetic-only training, and synthetic
pretraining followed by 1%, 5%, 10%, and 25% of the fixed real training partition.
All methods use the same held-out real recording sessions. It records requested and
actual adaptation fractions, complete group/sample splits, checkpoints, calibration,
a learning curve, and a PCA view of pooled encoder embeddings. PCA is labeled as
visualization only and is not treated as proof of representation quality.

The checked corpus contains 271 reviewed windows from four tracked and three wheeled
sessions. On the fixed two-session real test set, the real-only model reaches 10.2%
balanced accuracy and the synthetic-only model reaches 32.8%; limited-real adaptation
does not improve the latter. Every model misses the held-out wheeled session, so this
is evidence of severe source/session overfitting and sim-to-real failure, not a useful
category classifier. See [`docs/milestone5_status.md`](docs/milestone5_status.md) for
the exact split, hashes, metrics, calibration, and limitations.

## Milestone 6: paired noise-invariance evaluation

The invariance evaluator uses Milestone 1's aligned `clean_path`/`corrupted_path`
pairs and holds complete `recording_session` groups out of training. It compares the
same CNN architecture, initialization seed, batch order, split, and validation rule
across:

- standard supervised training on clean inputs;
- augmentation-only supervised training on corrupted inputs;
- paired clean/corrupted classification without a consistency term;
- paired training with class-weighted cross entropy, clean/corrupted cosine
  consistency, same-vehicle hard positives, and metadata-matched cross-class hard
  negatives in an established cosine triplet-margin objective;
- the same invariance objective in a projection head, leaving the classifier encoder
  free to retain class information.

Run the checked comparison with:

```bash
uv run python scripts/evaluate_invariance.py \
  --synthetic-manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --output runs/m6_invariance_seed42 \
  --heldout-noise 'road traffic' \
  --heldout-geometry far_offset \
  --consistency-weight 0.5 \
  --hard-positive-weight 0.1 \
  --hard-negative-weight 0.1 \
  --triplet-margin 0.2 \
  --epochs 10 --batch-size 32 --learning-rate 0.001 \
  --seed 42 --device cpu
```

Training excludes the selected noise category, microphone-response augmentation,
and held-out geometry. Test partitions isolate each nuisance where possible, plus an
all-corruptions view and the same grouped native-real test sessions used by Milestone
5. Geometry is explicitly recorded as an environment proxy because the checked
corpus has no impulse-response observations; this is not a true unseen-room test.
Each run saves machine-readable configuration, manifest hashes, complete grouped
splits, model checkpoints, per-condition/per-SNR metrics, embedding consistency, and
comparison plots. Because Milestone 6 never trains or selects on native-real audio,
the evaluator also writes a clearly marked all-session diagnostic with per-session
metrics and a class-balanced mean of session recall. The fixed held-out real split
remains the primary transfer result.

Aggregate controlled training-seed repeats only after keeping `--split-seed` fixed:

```bash
uv run python scripts/aggregate_invariance.py \
  runs/m6_factorial_2s_ablation/seed_*/metrics.json \
  --output runs/m6_factorial_2s_ablation/aggregate
```

The aggregator rejects mismatched commits, datasets, split/configuration fields,
method sets, evaluation conditions, and supports. It writes JSON and CSV summaries
with mean, median, sample standard deviation, min/max, and per-class recall, plus a
condition comparison plot with error bars.

In the checked seed-42 run, invariance training raises all-corruption synthetic
balanced accuracy from 77.7% to 78.9% relative to clean-only training and increases
paired embedding cosine similarity from 0.845 to 0.916. It does **not** transfer: its
balanced accuracy on the fixed native-real sessions is 12.5%, versus 29.0% for
clean-only and 48.4% for augmentation-only, and it has 0% recall on the held-out
wheeled session. See [`docs/milestone6_report.md`](docs/milestone6_report.md) for the
exact protocol, condition and SNR tables, confusion matrices, reproducibility check,
and limitations.

The subsequent balanced-factorial, five-seed checkpoint adds paired-supervised and
projection-head ablations plus session-balanced diagnostics. It confirms that none of
the invariance variants fixes the held-out real wheeled failure. See
[`docs/milestone6_improvement_checkpoint.md`](docs/milestone6_improvement_checkpoint.md)
for the controlled aggregate and the remaining Milestone 7 gate.

### Pretrained environmental-audio baseline

The optional frozen-encoder experiment uses the official MIT-licensed PANNs Cnn14
AudioSet model. It compares standardized linear tracked/wheeled heads on both the
2,048-dimensional embedding and the 527 AudioSet outputs, with corrupted-only and
paired clean/corrupted training for each representation. The encoder is never
updated. A second stage adapts only the paired AudioSet-output head using nested
1%/5%/10%/25%/100% subsets of the grouped real training partition, selects epochs on
the separate real validation sessions, and evaluates once on the fixed real test
sessions. The same factorial split and nuisance partitions are retained.

Because PANNs was pretrained on real AudioSet material, this is an external-pretraining
baseline, not a strict synthetic-only model even when its ABVID linear head sees only
synthetic examples.

Install the optional inference dependency and place the official checkpoint under the
ignored artifact directory:

```bash
uv sync --extra dev --extra inspection --extra pretrained
mkdir -p .artifacts/models/panns
wget -O .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  'https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth'
shasum -a 256 .artifacts/models/panns/Cnn14_mAP=0.431.pth
```

The required SHA-256 is
`0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`.
The evaluator refuses any other checkpoint. The upstream `panns-inference` package
also creates `~/panns_data/class_labels_indices.csv` on first import; this 14 KB
AudioSet label table is an upstream behavior, not an ABVID experiment artifact.

Run one controlled seed with a shared, manifest- and checkpoint-hash-validated
feature cache:

```bash
uv run python scripts/evaluate_pretrained.py \
  --synthetic-manifest data/generated/m6_factorial_2s_seed42/manifest.jsonl \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --embedding-cache runs/m6_panns_audioset_transfer/feature_cache.pt \
  --output runs/m6_panns_audioset_transfer/seed_42 \
  --epochs 50 --batch-size 32 --extraction-batch-size 16 \
  --learning-rate 0.001 --weight-decay 0.0001 \
  --real-fraction 0.01 --real-fraction 0.05 \
  --real-fraction 0.10 --real-fraction 0.25 --real-fraction 1.0 \
  --adaptation-epochs 100 --adaptation-learning-rate 0.0001 \
  --seed 42 --split-seed 42 --device cpu
```

Each run records the external checkpoint URL, hash, size, repository, license,
pretraining dataset, split, seed, probe states, fixed-test metrics, all-session
diagnostics, and real-adaptation learning curve. Aggregate repeated seeds with the
same `scripts/aggregate_invariance.py` command used above; when adaptation results are
present it also writes an uncertainty plot for the learning curve. The PANNs
checkpoint and feature cache remain local and are not committed. See
[`docs/milestone6_pretrained_transfer_report.md`](docs/milestone6_pretrained_transfer_report.md)
for the controlled five-seed result and its Milestone 7 gate assessment.

For a stricter real-only category diagnostic, run nested leave-one-tracked-session
and one-wheeled-session-out evaluation on a fixed, auditable subset of 35
vehicle/engine AudioSet outputs:

```bash
uv run --extra pretrained --extra inspection python \
  scripts/evaluate_semantic_sessions.py \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --feature-cache runs/m6_panns_audioset_transfer/feature_cache.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/m6_semantic_nested_sessions_ensemble \
  --regularization-c 0.001 --regularization-c 0.01 \
  --regularization-c 0.1 --regularization-c 1.0 \
  --regularization-c 10.0 --seed 42
```

Every outer fold excludes one complete tracked and one complete wheeled recording
session. Regularization is selected only through pair-wise inner folds among the
remaining sessions. Training weights give each class equal mass, each session within
a class equal mass, and each window within a session equal mass. The evaluator reports
both the inner-selected probe and an equal-probability ensemble of probes fitted at
every declared regularization value. No ensemble member is selected using an outer
test pair. The evaluator saves all inner/outer groups and sample IDs, both model
families, window metrics, CSV, and a comparative fold plot. See
[`docs/milestone6_semantic_session_report.md`](docs/milestone6_semantic_session_report.md).

The strongest current real-session diagnostic fuses that semantic ensemble with an
equal-weight ensemble of the 53 classical MFCC/spectral features. Its wheeled
decision threshold is selected separately in every outer fold using only inner
held-out session pairs:

```bash
uv run --extra pretrained --extra inspection python \
  scripts/evaluate_fusion_sessions.py \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --panns-feature-cache runs/m6_panns_audioset_transfer/feature_cache.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/m6_fusion_nested_sessions \
  --regularization-c 0.001 --regularization-c 0.01 \
  --regularization-c 0.1 --regularization-c 1.0 \
  --regularization-c 10.0 \
  --wheeled-threshold 0.2 --wheeled-threshold 0.25 \
  --wheeled-threshold 0.3 --wheeled-threshold 0.35 \
  --wheeled-threshold 0.4 --wheeled-threshold 0.45 \
  --wheeled-threshold 0.5 --wheeled-threshold 0.55 \
  --wheeled-threshold 0.6 --channel 0 --seed 42
```

The evaluator caches the classical features, records every inner and outer sample
assignment, saves both five-member probe ensembles for every outer fold, and reports
thresholded metrics without claiming probability calibration. See
[`docs/milestone6_fusion_session_report.md`](docs/milestone6_fusion_session_report.md).

The historical seven-session development rule was frozen with:

```bash
uv run --extra pretrained --extra inspection python \
  scripts/freeze_fusion_model.py \
  --real-manifest data/real_eval_v2/real_manifest.jsonl \
  --panns-feature-cache runs/m6_panns_audioset_transfer/feature_cache.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/m6_fusion_frozen_development_v2 --channel 0 --seed 42
```

That model-selection artifact was subsequently evaluated exactly once on the
Sherman/PDSounds pair. It achieved 49.69% balanced accuracy and did not confirm the
development result. The command below identifies that historical mechanism; do not
rerun it against Sherman/PDSounds or overwrite its completed artifacts:

```bash
uv run --extra pretrained --extra inspection python \
  scripts/evaluate_locked_fusion.py \
  --locked-manifest .artifacts/real_locked_pair_sherman_pdsounds_v1/real_manifest.jsonl \
  --frozen-model runs/m6_fusion_frozen_development_v2/frozen_fusion_model.pt \
  --checkpoint .artifacts/models/panns/Cnn14_mAP=0.431.pth \
  --output runs/m6_locked_pair_sherman_pdsounds_v1 --channel 0
```

The locked evaluator requires exactly one unseen session per class, rejects overlap
in session/source/media/hash provenance, validates the frozen feature definitions and
checkpoint, and refuses to overwrite a completed evaluation. For Benchmark v0.1,
first pass the refreshed development gates, freeze a new rule, then evaluate the
reserved T90M/JLTV pair once. See the benchmark contract for the required ordering.

## Reproducibility boundary

For identical source files, configuration, package versions, command seed, and CPU
platform, generation is deterministic. Float-WAV timestamps are normalized so
separate CLI processes produce byte-identical audio. Each single-channel sample uses
a derived `augmentation_seed`; each array base event uses a derived `array_seed`.
The seeds and all realized parameters are recorded. Training runs record the manifest
hash as the dataset version, the feature implementation version, seed, complete split
groups, model configuration, checkpoint, metrics, and Git commit when the checkout is
a Git repository.

## License

Project-authored code and documentation are available under the [MIT License](LICENSE).
Third-party recordings, videos, pretrained model weights, publications, and other
externally sourced assets retain their own licenses; the MIT license does not
relicense them. Source attribution and audio-license metadata are recorded in
`configs/audio_sources.yaml` and the versioned benchmark manifests. Downloaded
media and model checkpoints are not included in this repository.
