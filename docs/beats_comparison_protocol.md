# Frozen BEATs comparison v1

Preregistered 2026-09-20, before checkpoint inference or fitting. Domain: real → real
nested unseen-session development. Same 785 non-startup windows, channel 0, 16 kHz,
two seconds, seven tracked/five wheeled sessions as preprocessing-grid v1. Manifest
SHA is pinned in `configs/benchmark_beats_v1.yaml`. Fail closed on changed files,
unreviewed intervals, startup, consumed locked sources or future reserved sessions.

Use frozen **BEATs iter3+ AS2M pretrained**, not its AudioSet-finetuned classifier.
The AS2M-trained tokenizer informed its pretraining; it is not a representation with
no prior label exposure. Follow the [official implementation](https://github.com/microsoft/unilm/tree/732d834db70ee0fc3886b4bcbcfb4ce7fb829be2/beats):
128-bin Kaldi filterbank, 25 ms/10 ms, waveform × 32768, supplied fixed normalization.
Mean-pool final patch tokens to 768 dimensions. No fitted frontend statistics,
denoising, extra gain/filtering, context expansion, augmentation, encoder tuning,
layer search, or threshold selection. Exact-length inputs; no added padding.

Use the existing weighted train-only StandardScaler + logistic regression and all
35 outer session pairs / 24 inner pairs. C = .001, .01, .1, 1, 10; seed 42;
inner mean BA, then median BA, then smaller C. Primary: selected-C head at 0.5.
Fixed five-C probability ensemble is secondary, not an alternative headline picked
after seeing scores. Both PANNs semantic35 and full2048 control representations use
identical waveform windows and splits. Primary paired comparator is semantic35;
the embedding comparator helps distinguish dimensionality from representation.

Report BA, both class recalls, every session, worst session/context, descriptive
stratified session-bootstrap 95% intervals and paired differences. Save all standard
metric dictionaries, splits, heads, model/config/code/data hashes and actual runtime.
Independently replay every saved head/fold. Check repeated and single-versus-batch
embedding extraction for numerical agreement. Frozen encoders may have unknown
AudioSet source overlap; no claim of pretraining-clean evaluation is made.

The official OneDrive download returned HTTP 403. Use the publicly accessible
[lpepino mirror](https://huggingface.co/lpepino/beats_ckpts) at the pinned revision,
and verify its published LFS SHA256 before `torch.load(weights_only=True)`.
This verifies mirror artifact integrity, **not identity against an official checksum**
(none independently obtained). Keep that provenance limitation visible. Source code
is downloaded separately from the pinned official Microsoft commit with its license.
No checkpoint/source redistribution in benchmark metadata package.

No human-listening responses are required or used. New candidate videos are not
downloaded/admitted by this experiment. Confirmation is not run, even if development
looks better; passing development would require the separate checkpoint/decision-rule
freeze and preregistered confirmation criteria before opening reserved audio.
