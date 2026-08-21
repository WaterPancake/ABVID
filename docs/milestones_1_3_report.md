# Milestones 1-3 Reproducible Report

Date: 2026-08-19  
Runtime: Python 3.11, CPU  
Repository state: this workspace is not a Git checkout, so experiment metadata
records `git_commit: unavailable_not_git_checkout`.

## Scope and domain

The experiments in this report are **synthetic → synthetic**. The vehicle sources
are controlled procedural signals; their background recordings are real field
recordings processed by the Milestone 1 augmentation engine. These results do not
measure transfer to real vehicle recordings.

## Milestone 1 — augmentation engine

The engine implements all required gain, filtering/EQ, SNR mixing, resampling,
clipping, compression, impulse-response, cropping, and microphone-response stages.
Every public waveform uses `[channels, samples]`. `audio.background_channel: 0`
is an explicit one-microphone background selection and is written to each manifest
record; setting it to `null` restores strict channel matching.

Validation command:

```bash
uv run pytest -q
uv run python scripts/create_toy_audio.py \
  --output .artifacts/milestone1_final_inputs
uv run python -m vehicle_audio.cli generate \
  --config configs/default.yaml \
  --targets .artifacts/milestone1_final_inputs/targets \
  --backgrounds .artifacts/milestone1_final_inputs/backgrounds \
  --impulse-responses .artifacts/milestone1_final_inputs/impulse_responses \
  --output .artifacts/milestone1_final_a \
  --num-samples 14 --seed 42
```

Result:

- full suite: 34 passed;
- 14 paired examples generated twice in separate processes;
- both manifests and all 28 WAV files were byte-identical;
- toy manifest SHA-256:
  `dee5bfdc35c5fd6fc1aaa67949d6d1e4ab661a1831409ad7632c8da4433086ef`;
- inspection image: `.artifacts/milestone1_final_inspection.png`.

The byte check uncovered and fixed libsndfile's wall-clock value in float-WAV
`PEAK` chunks. Only that informational timestamp is normalized; samples are not
modified.

## Milestone 2 — grouped classical and neural baselines

Both models use explicit channel 0. The classical model uses standardized MFCC,
spectral centroid/bandwidth/rolloff, flatness, harmonic peak concentration, RMS,
zero-crossing rate, and crest factor features with logistic regression. MFCCs use
absolute log-Mel input so a clip does not depend on its feature-extraction batch.
The neural model uses a log-Mel CNN with a separate encoder and classifier.

Both runs use the same seed-42 grouped assignment:

- group field: `recording_session`;
- train: 40 sessions / 524 observations;
- validation: 12 sessions / 157 observations;
- test: 12 sessions / 159 observations;
- no session appears in more than one partition.

Dataset manifest:
`data/generated/m3_controlled_seed42/manifest.jsonl`  
Dataset SHA-256:
`5d8950eaae54a13aabd1f18a5d792ef747de332675943aad5dcb1b113714d18a`

### Aggregate test results

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | ECE | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Classical | 0.8868 | 0.8886 | 0.8875 | 0.8886 | 0.8868 | 0.0381 | `[[72,12],[6,69]]` |
| CNN | 0.8491 | 0.8536 | 0.8575 | 0.8536 | 0.8489 | 0.0404 | `[[65,19],[5,70]]` |

Rows are true classes and columns are predicted classes in
`[tracked, wheeled]` order.

### Accuracy versus SNR

| SNR dB | Classical | CNN |
|---:|---:|---:|
| 30 | 0.9545 | 0.9091 |
| 20 | 1.0000 | 0.9091 |
| 10 | 1.0000 | 0.9286 |
| 5 | 0.8000 | 1.0000 |
| 0 | 0.8387 | 0.7742 |
| -5 | 0.8824 | 0.7647 |
| -10 | 0.6842 | 0.6316 |

The curve is not expected to be perfectly monotonic: each SNR slice also contains
independently sampled filtering, EQ, clipping, compression, background crop, and
microphone response. Exact supports and complete calibration metrics are stored in
each run's `accuracy_by_snr.csv` and `metrics.json`.

Artifacts:

- `runs/m2_classical_grouped_seed42/`
- `runs/m2_cnn_grouped_seed42/`

An independent repeat at `runs/m2_classical_grouped_seed42_repeat/` produced
identical splits, epoch history, validation metrics, and test metrics.

## Milestone 3 — controlled procedural vehicle corpus

Generation command:

```bash
uv run python -m vehicle_audio.cli synthesize-sources \
  --config configs/procedural_vehicles.yaml \
  --output data/synthetic_targets --seed 20260819
```

The source corpus contains:

- 4 tracked and 4 wheeled identities;
- idle, accelerating, steady-speed, and decelerating states for every identity;
- 2 runs and the same 2-geometry grid for every identity/state;
- 64 independent `simulation_run`/`recording_session` groups;
- explicit RPM, throttle, speed, acceleration, load, and geometry metadata;
- source manifest SHA-256:
  `8da4354449bba4b22ac2456738f61f42f7dfdf0a071bcb885a9619a2272ac3d1`.

The augmented corpus contains 840 observations: 424 tracked and 416 wheeled,
201 accelerating, 213 decelerating, 205 idle, and 221 steady-speed examples. All
seven requested SNRs and the rain, road-traffic, and animals/insects backgrounds
are represented.

### Controlled holdout results

Commands (the CNN vehicle run changes `--model classical --epochs 30
--batch-size 64 --learning-rate 0.01` to `--model cnn --epochs 10
--batch-size 32 --learning-rate 0.001`):

```bash
# Unseen operating state
uv run python scripts/train_baseline.py \
  --manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --output runs/m3_classical_unseen_state_seed42 \
  --model classical --epochs 30 --batch-size 64 --learning-rate 0.01 \
  --channel 0 --seed 42 --device cpu --split-strategy holdout \
  --group-field operating_condition \
  --validation-group idle --test-group decelerating \
  --test-domain 'synthetic unseen state -> synthetic'

# Unseen environment category
uv run python scripts/train_baseline.py \
  --manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --output runs/m3_classical_unseen_environment_seed42 \
  --model classical --epochs 30 --batch-size 64 --learning-rate 0.01 \
  --channel 0 --seed 42 --device cpu --split-strategy holdout \
  --group-field background_category \
  --validation-group 'road traffic' --test-group 'animals/insects' \
  --test-domain 'synthetic unseen environment -> synthetic'

# Unseen complete vehicle identities
uv run python scripts/train_baseline.py \
  --manifest data/generated/m3_controlled_seed42/manifest.jsonl \
  --output runs/m3_classical_unseen_vehicle_seed42 \
  --model classical --epochs 30 --batch-size 64 --learning-rate 0.01 \
  --channel 0 --seed 42 --device cpu --split-strategy holdout \
  --group-field vehicle_id \
  --validation-group tracked_charlie --validation-group wheeled_charlie \
  --test-group tracked_delta --test-group wheeled_delta \
  --test-domain 'synthetic unseen vehicle -> synthetic'
```

| Experiment | Model | Validation holdout | Test holdout | Accuracy | Balanced accuracy | Macro F1 | ECE |
|---|---|---|---|---:|---:|---:|---:|
| Unseen state | Classical | idle | decelerating | 0.9765 | 0.9773 | 0.9765 | 0.0167 |
| Unseen environment | Classical | road traffic | animals/insects | 0.8777 | 0.8679 | 0.8728 | 0.1083 |
| Unseen vehicle | Classical | tracked/wheeled Charlie | tracked/wheeled Delta | 0.7630 | 0.7622 | 0.7624 | 0.1425 |
| Unseen vehicle | CNN | tracked/wheeled Charlie | tracked/wheeled Delta | 0.7962 | 0.7951 | 0.7953 | 0.0502 |

The unseen-vehicle training partition contains Alpha and Bravo from each class;
Charlie is validation-only and Delta is test-only. The drop from grouped-session
performance to unseen-vehicle performance is the most informative result here:
source identity remains an important shortcut even in the controlled corpus.

Artifacts:

- `runs/m3_classical_unseen_state_seed42/`
- `runs/m3_classical_unseen_environment_seed42/`
- `runs/m3_classical_unseen_vehicle_seed42/`
- `runs/m3_cnn_unseen_vehicle_seed42/`

## Limitations and next technical debt

1. Procedural sources encode simplified harmonic engine and mobility textures.
   Strong synthetic results may reflect generator structure, not real vehicle
   acoustics.
2. The unseen-state split holds out the complete decelerating trajectory, but its
   RPM and speed interval overlaps the accelerating trajectory. It tests a novel
   dynamic state, not strict extrapolation to a disjoint continuous operating range.
3. The unseen-environment split has only three categories and one or two source
   recordings per category, so environment category is confounded with source
   recording identity.
4. No measured impulse responses are locally available; the full augmentation
   path is tested with a toy IR but reverb is absent from the 840-example corpus.
5. Wikimedia returned HTTP 429 on approved real vehicle downloads. The collector
   respected the provider limit; no alternate mirror or bypass was used.
6. A small CNN was used instead of downloading a pretrained encoder. Its modular
   encoder/classifier boundary allows a pretrained environmental-audio encoder to
   be added later without changing split or reporting contracts.
7. This report contains no synthetic→real result. Milestone 5 is required before
   drawing conclusions about field performance.
