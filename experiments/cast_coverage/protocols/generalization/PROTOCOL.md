> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../README.md).

# CAST held-group acoustic coverage v0

This is the separately authorized response to **“test generalization”**, following
the completed CAST-0–2 pilot. It implements the minimal CAST-3 sampler controls
and first CAST-4 coverage diagnostic. The pilot remains unchanged. Its 43/50
near-equal-start disagreements prevent interpreting this empirical bank as an
identified physical population. This experiment tests the consequences of
sampling the existing effective controls; it does not certify the prerequisite
of stable physical parameters or claim classifier transfer.

## Freeze and data boundary

Before opening any held audio, verify the pilot lock and its 50 fitted artifacts,
validate the source-only manifest and H1 fold-0 lock, freeze this document, code,
configuration, all 570 held IDs, training bank, training descriptor scales and
sampling schedule. Run tests, generate **all** samples, and hash their artifacts
before creating a held-audio access receipt. No held loss influences sampling,
scaling, parameters, sample selection, output selection or thresholds.

The only new audio access is H1 fold 0: `connected_4001f06f57cfeee7`,
Schleusinger-Allee, 2019-11-12, 491 cars and 79 trucks. Require exact metadata,
allowlisted paths, source byte hashes, native audio shape/rate and cross-split
ancestry checks before decoding. This is a CAST-held **H1-exposed development**
group, at a site also represented in training. It is neither a new final test
nor an unseen-site test. MELAUDIS, mixed caches, supplied synthesis banks,
military/reserved recordings and classifiers remain outside this experiment.

All 50 real calibration parents count as real-data access: 25 per class, five
per group/class across five groups. Keep ambiguity and boundary flags. No fit
is dropped for a poor score. Unknown physical quantities remain unknown.

## Sampling

For each class, seeds `[42,123,456,789,1024]`, and 50 indices, generate one clip
per arm (1,500 total; 250 per class/arm). Each draw uses a deterministic private
NumPy generator. Groups have equal probability, then parents within a group
have equal probability; draws are with replacement, not independent sessions.

* **Joint:** draw one complete fitted vector. This is empirical fitted-exemplar
  replay with new phase/noise, not a learned continuous population model.
* **Prototype:** use the class mean, first averaging within groups, then across
  groups. Normalize the two mean weight vectors to sum to one.
* **Marginals:** independently draw a parent by the same group/parent rule for
  each of the 25 scalar coordinates; then renormalize the harmonic and noise
  weight vectors. These simplex constraints reintroduce dependence. Record
  each coordinate's donor and every contributing parent/group.

Use the unchanged renderer, with separately derived `sampling` phase and noise
streams and exactly one noise realization. Matched class/seed/index uses the
same phase/noise across arms (common random numbers). Sampling input IDs are
namespaced separately from all real fitting/checking IDs. Every output records
the full vector, bank/config hashes, parent ancestry, donor mapping, flags,
stream seeds, waveform hash and origin `real_calibrated_observation_synthesis`.
Save unit-RMS float waves and separately scaled, peak-safe playback; descriptor
metrics use the shape wave. Playback gain is recorded and never affects scores.

## Frozen descriptors and scores

Apply the pilot's fixed stereo mean, native→8→16 kHz resampling, centered two
seconds, DC removal and AC-RMS normalization to real observations. No padding.
Use the existing Welch/50-ms-envelope implementation for all arms. Four primary
families receive equal weight:

1. Natural log of 64 equal-width Welch-power band proportions over 0–4 kHz,
   floored at 1e-8 (62.5-Hz bins; 2048-point Welch, 1024 overlap).
2. The renderer's eight broad-band energy proportions.
3. Forty successive 50-ms RMS envelope values.
4. Twenty envelope modulation magnitudes at 0.5–10 Hz, divided by 40.

For each class and coordinate, scale by **training-real** population standard
deviation, floored by family at 0.1, 0.01, 0.05 and 0.02, respectively. Freeze
these scales before held access; generated and held data never set the scales.
Compute empirical univariate Wasserstein-1 distance per coordinate divided by
its scale, mean within family, then mean across four families. Lower is better.
This is a marginal descriptor distance, not waveform similarity, joint support,
perceptual quality, a classifier metric or calibrated probability.

Compare each arm/seed with all held clips of that class. Also report 25 actual
training-real clips versus held real clips as an observational reference. It
has a smaller sample count and is **not** a matched generated control. Report
seed means and full min/max ranges, not session confidence intervals. Compare
the two classes by an unweighted macro mean, never the class-imbalanced pool.

Coverage is the fraction of held values inside each generated coordinate's
5th–95th percentile interval, averaged within family and across families. Report
lower/upper tail misses and p05/median/p95 for each coordinate. Spread is mean
generated coordinate SD divided by mean held coordinate SD after the frozen
scaling, separately by family. Also record unique complete vectors, unique
parents and source groups; stochastic noise does not increase real ancestry.
Plots of long-term spectrum, envelope and modulation accompany the scores.

Before evaluation, declare a provisional acoustic adequacy rule: joint sampling
must reduce mean distance by at least 5% against **each** generated control in
**each** class, cover at least 80% of marginal held values in each class, and
retain spread ratios between 0.5 and 2 in every class/family (seed means).
These engineering cutoffs are explicit, provisional and not validated research
standards. Preserve any failure; do not tune or change them after held access.
Passing would support only this descriptor-level development check; it cannot
establish real-world or classification generalization.

Only one held group is available in this check: group and worst-group results
are consequently identical, with no population confidence claim. Correlated
clips and generated samples cannot substitute for independent groups. No held
conditional fitting is performed. Held IDs and generated inspection examples
are selected lexicographically/first index before audio access, never by score.

## Reproducibility and stopping

Use the parent's pinned environment read-only and CPU with one Torch thread.
Technical acceptance requires unit/provenance tests, exact deterministic replay
of every generated waveform and all parent/donor choices, hashes of every held
input/artifact, recomputed held preprocessing/descriptors and coverage scores,
and unchanged parent artifacts/source plus unchanged tracked ABVID diff.
Save failure records, exact counts, timing, peak RSS, protocol and code copies.
New runs refuse overwrites. On any defect, preserve the failed run and repair
under a new freeze; record whether held audio had already been exposed.

Stop at this CAST-4 diagnostic. No classifier, distribution tuning, additional
fit, new dataset, further fold or military milestone is authorized by this
protocol. Derived audio remains local under the source's recorded licence.
