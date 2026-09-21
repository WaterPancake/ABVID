# Current model architecture and log-Mel diagnostics

2026-09-20. Domain: **real -> real, nested unseen-session development**.
This describes the current 7/5 benchmark, not the older small-CNN experiments.

The plots and listening gallery preserve the frozen 593-window benchmark.
The active catalog subsequently excluded startup intervals on 2026-09-20;
historical startup examples here remain archival, not current training examples.
The uncapped non-startup study is specified in the
[preprocessing grid protocol](preprocessing_grid_protocol.md).

## Current architecture

```text
Reviewed real source -> channel 0 -> 16 kHz -> 2 s windows, 1 s hop
    |
    +-> 53 handcrafted features -> training-only standardization
    |       -> five regularized logistic classifiers -> mean probabilities
    |
    +-> upsample 16 -> 32 kHz -> frozen PANNs Cnn14
            -> 527 AudioSet scores -> select 35 vehicle/mechanical scores
            -> training-only standardization
            -> five regularized logistic classifiers -> mean probabilities
                        |
        equal average of the two branch probabilities
                        |
        inner-session-selected wheeled threshold -> tracked / wheeled
```

There are 35 held-out-session-pair fits, not one deployed final checkpoint. Each
outer fit holds out an entire tracked and wheeled recording session. The five
regularization values are 0.001, 0.01, 0.1, 1 and 10. Each branch's five probabilities
are averaged. Ten binary linear probes together contain 450 coefficient/intercept
parameters per outer fit; scaler statistics are fitted on training data too.

**Classical branch.** Twenty MFCC means and twenty standard deviations; mean/std
of spectral centroid, bandwidth, rolloff, flatness and top-bin concentration;
RMS, zero-crossing rate and crest factor. Its frontend uses a 1,024-sample FFT,
256-sample hop, 64 Mel bands spanning 20-8,000 Hz at 16 kHz. MFCCs use natural-log
Mel power. Much temporal information is reduced to summary statistics.

**Semantic branch.** Cnn14 has six two-convolution blocks (64, 128, 256, 512, 1024,
2048 channels), frequency averaging, temporal maximum-plus-mean pooling, a
2,048-dimensional embedding, and a 527-way AudioSet output layer. The installed
model has 80,753,615 neural-network parameters excluding its fixed spectral frontend.
ABVID does not update those parameters. The current head uses **35 selected
AudioSet output scores, not the available 2,048-dimensional embedding**. This is
a restrictive task representation even though the encoder itself is large.

The PANNs frontend uses 32 kHz, FFT 1,024, hop 320, 64 Slaney Mel bands, 50-14,000 Hz,
power-to-dB and the pretrained batch-normalization layers. Code references:
[native feature loader](../src/vehicle_audio/real_panns_features.py),
[semantic feature selection](../src/vehicle_audio/semantic_session_evaluation.py),
[classical features](../src/vehicle_audio/baseline.py),
[fusion](../src/vehicle_audio/fusion_session_evaluation.py).
Upstream background: [PANNs authors' repository](https://github.com/qiuqiangkong/audioset_tagging_cnn)
and [paper](https://arxiv.org/abs/1912.10211).

## What preprocessing currently does—and does not do

- Native observations preserve gain and select channel 0; there is no native
  denoising, wind suppression, per-clip loudness normalization or audibility gate.
- The current 16 kHz corpus is subsequently upsampled for the 32 kHz PANNs model.
  Content above 8 kHz has already been removed; the upsampling cannot restore it.
  This is a bandwidth mismatch worth testing, not a demonstrated explanation of
  the failures. The authors also provide a 16 kHz checkpoint, but using it would
  change the model and require a new evaluation.
- Inputs are only two seconds long. A listener hearing a longer event can use
  context and changes over time that a single benchmark window lacks. There has
  been no blinded human evaluation, so hearing separability is encouraging evidence
  of possible cues, not a quantified model-versus-human comparison.

## Generated plots and exact listening clips

Open the [local gallery](../.artifacts/logmel_diagnostics_7t5w_v2/index.html).
It contains eight deliberately selected examples and their exact two-second model
input WAVs, original timestamps, annotations, license/attribution and held-out scores.

| Comparison | Examples | What the saved predictions show |
|---|---|---|
| [Better-classified](../.artifacts/logmel_diagnostics_7t5w_v2/correct_examples.png) | Abrams 25-27 s; Maserati 0-2 s | Semantic correct in all held-out contexts for both; fusion correct 5/5 and 5/7, respectively |
| [Spectral resemblance](../.artifacts/logmel_diagnostics_7t5w_v2/spectral_overlap.png) | Abrams 155-157 s; Stryker 154-156 s | Both accelerating; both branches call the Stryker tracked in all 7 contexts |
| [Wheeled failures](../.artifacts/logmel_diagnostics_7t5w_v2/wheeled_failures.png) | HMMWV 44-46 s; Ford 12-14 s | Both branches and fusion wrong in all 7 contexts for each |
| [Branch disagreement](../.artifacts/logmel_diagnostics_7t5w_v2/branch_disagreement.png) | T-72 38-40 s; Abarth 3-5 s | T-72 semantic correct 5/5 but fusion 0/5; Abarth classical correct 7/7 but fusion 0/7 |

All percentages concern these selected windows, not whole-session or overall
performance. Every contributing model excluded the entire displayed source session.
Different contexts have different original thresholds. An average displayed score
does not define a new prediction or an operationally calibrated confidence.

Each figure uses three columns:

1. Diagnostic 16 kHz log-Mel with a shared power reference across all eight examples.
2. The same diagnostic Mel with each clip's peak aligned to 0 dB, solely to inspect
   structure without gain obscuring it; this normalization was **not** applied to
   the model. Frequency is Mel-spaced, with Hz tick labels.
3. The actual PANNs log-Mel output before pretrained batch normalization. Its
   separate shared frontend reference cannot be directly compared in absolute
   magnitude to the other filterbank. The cyan line marks the upstream 8 kHz limit.

All columns use a 70 dB display range. Brightness is relative spectral power, not
sound pressure or measured engine SNR. Plot arrays and selection details are saved
in `plot_arrays.npz` and `examples.json` beside the gallery.

## Interpretation, without overclaiming

- The better-classified Abrams example shows sustained narrow-band structures;
  the selected Maserati startup has a distinct time-varying onset. They are also
  different operating states, so this is not proof of a general class signature.
- The Stryker example has substantial low-frequency concentration and a broadly
  similar mean spectral shape to the selected accelerating Abrams. They were
  deliberately matched by normalized time-mean log-Mel profile (similarity 0.934).
  Temporal details still differ; this matching is not evidence that they are
  perceptually indistinguishable or inseparable by another model.
- The T-72 example contains much less very-low-frequency power, and the semantic
  branch already classifies it correctly. A denoising explanation cannot account
  for fusion reversing that correct branch decision by itself.
- Low-frequency power is not automatically wind. The Stryker example has 64.6%
  of its Hann-windowed spectral power below 100 Hz, while the failed HMMWV example
  has only 2.4%. A universal high-pass filter cannot be assumed to solve both;
  it can remove useful vehicle energy as well as wind.

## Recommended next experiments—not applied here

**Follow-up:** recommendations 1 and 3 have now been evaluated separately in
[the embedding/context/bandwidth report](embedding_context_results.md). The
original model and plots described on this page are unchanged. The proposals
below are retained as historical rationale, not a current uncompleted checklist.

1. Compare the existing 35-score semantic head against a regularized head on the
   already cached **2,048-dimensional frozen embeddings**, with training-only
   normalization and inner-only regularization selection. Keep classical and
   semantic-only controls; do not assume equal fusion helps.
2. Test modest gain normalization and mild high-pass options as separate ablations,
   with any cutoff chosen only on inner training/validation sessions. Include the
   unchanged input. Do not begin with aggressive speech-oriented denoising.
3. Test direct original-to-32 kHz extraction (or a matched 16 kHz pretrained model)
   and 4-8 second context using matched reviewed events. Keep bandwidth and context
   changes separate so their effects can be identified.

These are hypotheses to preregister, not promised improvements. Only 12 independent
development sessions exist despite 593 overlapping windows; a larger freely trained
network can overfit those sessions. The current frozen benchmark and reserved
recordings remain unchanged.

Reproduce into a new local output directory:

```bash
uv run python scripts/plot_native_logmel_diagnostics.py \
  --output .artifacts/logmel_diagnostics_7t5w_v2
```
