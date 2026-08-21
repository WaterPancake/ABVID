# Legacy Real-Source Pilot Rundown

This document describes the earlier 1,000-example pilot under `data/generated/`.
Its sample-level classifier split is known to leak recording sessions and its model
score is not valid research evidence. See `milestones_1_3_report.md` for the current
grouped synthetic→synthetic experiments.

The current dataset contains **1,000 synthetic four-second observations**. Each
observation pairs a real vehicle excerpt with a corrupted version of the same
excerpt.

## Vehicle sources

Three source recordings were used:

| Class | Vehicle/source | Duration | Original rate | License |
|---|---|---:|---:|---|
| Tracked | TR-85M1 Romanian tank-range recording from [DVIDS](https://www.dvidshub.net/video/343966/romanian-tanks) | 163.6 s | 48 kHz stereo | Public domain |
| Wheeled | Abarth 205 Monza at the 2009 Goodwood Festival of Speed from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Abarth_205_Monza_(1950).ogg) | 15.3 s | 44.1 kHz stereo | CC BY-SA 3.0 |
| Wheeled | 1926 Ford Model T starting from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ford_Model_T_%22Tin_Lizzy%22_-_Starting_the_Engine.webm) | 38.3 s | 48 kHz stereo | CC BY 3.0 |

Local copies and their provenance sidecars are stored under `data/targets/`.

The generator sampled the three recordings uniformly, producing:

- 344 tracked examples
- 656 wheeled examples

The class imbalance results from having one tracked recording and two wheeled
recordings.

## Background source

Every corrupted example currently uses a segment from one background recording:

- Heavy rain recorded in Malvern Wells by david.j.pitt
- Approximately 25 minutes long
- 44.1 kHz stereo
- Obtained through [Internet Archive/radio aporee](https://archive.org/details/aporee_52865_60398)
- Recorded in the local provenance metadata as public domain

The local background recording and provenance sidecar are stored under
`data/backgrounds/`.

No impulse responses have been added yet. The current generated dataset therefore
contains no simulated room or environmental reverberation.

## Generation process

For each sample, using a deterministic seed:

1. One of the three vehicle recordings was selected.
2. A random four-second segment was extracted.
3. A random four-second segment of rain was extracted.
4. Both segments were explicitly resampled to 16 kHz.
5. The original two channels were preserved.
6. The resampled vehicle segment was saved as `clean.wav`.
7. Independently selected corruptions were applied.
8. The result was saved as `corrupted.wav`.
9. Every source, crop offset, seed, and augmentation parameter was written to
   `metadata.json` and the root `manifest.jsonl`.

The corruption order is approximately:

```text
vehicle crop
    ↓
random gain
    ↓
reverberation, if an impulse response exists
    ↓
low/high/band-pass filtering and random EQ
    ↓
rain mixed at controlled SNR
    ↓
microphone frequency-response approximation
    ↓
round-trip resampling degradation
    ↓
dynamic-range compression
    ↓
mild clipping
```

## Randomized conditions

Every example receives:

- A random gain from -12 to +3 dB
- Rain mixed at one of 30, 20, 10, 5, 0, -5, or -10 dB SNR

Other corruptions are independently selected:

| Corruption | Probability | Range |
|---|---:|---|
| Low-pass filter | 35% | 2.5-7.5 kHz |
| High-pass filter | 25% | 30-300 Hz |
| Band-pass filter | 15% | Random low/high cutoffs |
| Random EQ | 35% | -5 to +5 dB |
| Microphone response | 50% | -6 to +3 dB frequency curve |
| Resampling degradation | 25% | Round-trip through 4, 8, or 12 kHz |
| Compression | 25% | Ratio 2:1-6:1 |
| Clipping | 15% | Threshold 0.65-0.95 |
| Reverberation | 0% currently | No impulse-response recordings available |

The recorded `snr_db` value is the SNR at the controlled mixing stage. Later
frequency-selective or nonlinear corruptions can change an SNR measured directly
from the final waveform.

## Output structure

The generated dataset is stored under `data/generated/`:

```text
data/generated/
├── manifest.jsonl
└── sample_000000_<seed>/
    ├── clean.wav
    ├── corrupted.wav
    └── metadata.json
```

Both audio files are four seconds long, stereo, and sampled at 16 kHz. The
one-microphone baseline classifier explicitly selects channel 0; the generator
itself does not downmix the stereo recordings.

## Current limitations

This is still a narrow source collection:

- Only one tracked recording session is represented.
- Only two wheeled recording sessions are represented.
- Every corrupted sample uses the same rain recording.
- No impulse responses are available, so reverberation is absent.
- Vehicle identity, recording session, and class are strongly confounded.
- Random train/test clip splits contain excerpts from the same source sessions.
- Other corruptions vary within each SNR group, so accuracy by SNR is not a pure
  SNR-only controlled comparison.

The dataset is sufficient for exercising the generation and training pipeline,
but it cannot yet demonstrate generalization to unseen vehicles, recording
sessions, microphones, or environments. Multiple independent sessions for each
class, more background categories, and measured impulse responses should be added
before treating classifier accuracy as research evidence.
