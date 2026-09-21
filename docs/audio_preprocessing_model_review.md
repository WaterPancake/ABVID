# Audio classification preprocessing: evidence and ABVID recommendations

2026-09-20. Primary-source review; recommendations below are hypotheses for
**native real → real unseen-session development**, not measured improvements.
See [the embedding/context protocol](embedding_context_protocol.md) for the
experiments actually run. No new pretrained model was downloaded in this review.

## What other models actually do

| Model | Documented input/frontend | Relevant lesson for ABVID |
|---|---|---|
| PANNs Cnn14 | 32 kHz, 64 log-Mel bands, FFT 1,024 / hop 320, 50–14,000 Hz; separate 16 kHz checkpoint with FFT 512 / hop 160 and upper limit 8 kHz | Use the frontend belonging to the checkpoint. Compare direct 32 kHz extraction separately from changing encoder. Full 2,048-dimensional embeddings are available. [Authors' implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| YAMNet | Mono float waveform at 16 kHz in the digital range −1 to +1; MobileNet-based encoder; 0.96 s analysis patches every 0.48 s; 1,024-dimensional embeddings and temporal score aggregation | A useful lightweight frozen-embedding control. The tutorial explicitly preserves source folds when expanding recordings into frames. Digital float scaling is **not** per-recording peak normalization. [TensorFlow tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio) |
| AST | 16 kHz, 128-bin filterbanks; waveform mean subtraction; dataset-statistic normalization toward mean 0 / standard deviation 0.5; training-time time/frequency masks and mixup | Normalization is part of the model contract. Use published pretrained constants first; estimate any replacement statistics from training folds only. Its mask/mixup recipe is augmentation, not test-time denoising. [Authors' recommendations](https://github.com/YuanGongND/ast#use-pretrained-model-for-downstream-tasks), [loader](https://raw.githubusercontent.com/YuanGongND/ast/master/src/dataloader.py) |
| BEATs | 16 kHz, 128-bin Kaldi filterbanks, 25 ms frames / 10 ms hop; implementation scales waveform by 2¹⁵ and uses fixed filterbank mean 15.41663 and standard deviation 6.55582 (denominator twice the standard deviation) | A transformer representation comparator with an exact frontend to reproduce. Do not feed our existing PANNs log-Mel images into it or normalize twice. Handle padding masks when pooling variable-length embeddings. [Microsoft implementation](https://github.com/microsoft/unilm/blob/master/beats/BEATs.py) |
| PaSST | Example pretrained wrapper expects 32 kHz; documented configuration uses 128 Mel bins, FFT 1,024 / hop 320; training uses spectrogram masks and drops patches with Patchout | Another frozen environmental-audio comparator. Patchout is training regularization, not a wind-removal algorithm. Use the supplied frontend and evaluation mode. [Authors' repository](https://github.com/kkoutini/PaSST), [paper](https://arxiv.org/abs/2110.05069) |

These methods are candidates, not a ranked vehicle-classification leaderboard.
Their AudioSet/ESC-50 results do not establish performance on tracked/wheeled
recordings. Pretraining overlap with web-sourced vehicle audio is unknown and
must remain an explicit caveat, even with leakage-free ABVID session splits.

## Preprocessing worth testing next

The following order is an ABVID-specific proposal, not an upstream claim that
these settings work for vehicles. Keep the unchanged waveform as a control.

1. **DC removal and conservative level normalization, separately.** Subtracting
   the waveform mean is used by AST. Separately test a bounded RMS gain (for
   example, target −20 dBFS with a ±12 dB gain cap and a non-clipping limiter on
   the applied gain, not waveform clipping). Define silence handling first.
   Gain changes can reduce recording-level differences but cannot improve SNR;
   they amplify engine and noise together. Avoid per-window peak normalization
   as the default: one transient can determine the scale. Do not interpret
   arbitrary digital gain as physical sound pressure. Apply identical rules to
   training and held-out observations and fit any parameters only inside training.
2. **Mild high-pass filtering as a controlled ablation.** Preregister no filter
   versus, for example, 30/50/80 Hz second-order high-pass options. Select a cutoff
   only inside inner folds. Low-frequency wind is a plausible nuisance in some
   recordings, but engine fundamentals also live at low frequencies, and some
   failed HMMWV examples have very little sub-100-Hz energy. Track both recalls
   and worst-session behavior; do not manually select a filter for each test clip.
   Specify filter initialization and causal versus offline filtering explicitly.
3. **Noise/channel robustness through training examples.** Reuse our real
   backgrounds, controlled SNR, gain, mild EQ and microphone-response variation
   on training sessions. Reserve entire background recordings for unseen-noise
   testing. Compare augmentation alone with the existing invariance objective;
   more corruptions of a recording do not create more independent sessions.
   SpecAugment/mixup are documented in AST, but require a training setup that
   understands masked inputs/soft labels—not an inference-time cleaning switch.
4. **PCEN as a separate frontend experiment.** Per-channel energy normalization
   combines adaptive gain control with compression instead of static log
   compression. Its original evidence is noisy/far-field keyword spotting, not
   vehicles. It may suppress stable background levels, but can also suppress
   sustained engine cues. Test with a separately trained small model or a
   deliberately adapted frontend, **not** by substituting PCEN for log-Mel in a
   frozen PANNs checkpoint. Specify smoother state/reset behavior across windows.
   [Original PCEN paper](https://arxiv.org/abs/1607.05666)
5. **Temporal detail without inventing new data.** Compare mean versus
   mean-plus-standard-deviation pooling of frame embeddings, or aggregate
   consecutive short-window probabilities within a reviewed event. This is
   distinct from feeding one longer waveform to Cnn14. Keep training-only
   aggregation selection and event boundaries fixed. Any use of future audio
   must be labeled offline; causal buffering adds latency.

## What I would not enable by default

- Speech-enhancement or speech-isolation models: their desired output is speech,
  not vehicle machinery. Whether they destroy useful engine/track cues needs an
  explicit benchmark, not a listening-quality assumption.
- Aggressive spectral subtraction, fixed broad notches, or broad high-pass
  filtering without nuisance measurements. These may create artifacts or remove
  class cues. A noise estimate from an entire test recording also changes the
  inference contract and can use future information.
- Per-clip spectral whitening/cepstral normalization without an ablation. It may
  remove microphone coloration but also the stable spectral envelope we classify.
- Time stretching or pitch shifting indiscriminately: engine speed is relevant
  signal, and these operations may create physically implausible examples.
- A more powerful train-from-scratch classifier before frozen environmental
  encoders and preprocessing controls. Twelve independent sessions are still a
  very small dataset, regardless of overlapping-window count.

## Bounded next model comparison

After the current representation/context/bandwidth experiments, preregister one
small comparison: current PANNs controls, YAMNet for a lightweight alternative,
and **one** transformer (BEATs or AST) with frozen embeddings and the same
regularized heads. Keep model-specific preprocessing correct, use identical
reviewed centers and session folds, and report runtime/memory with accuracy.
Do not try all architectures and claim the best development score as confirmation.
The reserved T90M/JLTV sources remain untouched until a development rule genuinely
passes and a new confirmation protocol is frozen.
