# CAST-5A precision feasibility before protocol freeze

2026-10-11. **Six source groups can support a useful paired comparison, but the
current design cannot be described as adequately powered to establish a true
three-point gain exceeds a three-point margin.** Those are the same boundary.
H1 supports conditional planning widths around **3.6–5.1 macro-F1 percentage
points** for several relevant source perturbations. At that variability,
normal planning approximations require a true gain around **5.6–6.6 points**
for 80% probability that the lower two-sided 95% bound exceeds three. These
are sensitivity scenarios, not a measured CAST-versus-augmentation variance.

The three-point margin and the proposed [CAST-5A design](../CAST_ROADMAP.md#cast-5a-test-classification-utility-within-idmt)
remain unchanged and **unfrozen**. This analysis used existing saved H1 scalar
scores and resampling draws only. It performed no new fitting, inference,
feature extraction, audio generation or held-recording evaluation.

## The quoted H1 result and its limits

The saved BEATs real-plus-procedural minus real contrast is **2.710900 points,
95% CI [1.237780, 4.740657]**, width **3.502877 points**. The paired bootstrap
estimator SD is **0.888871 points**, obtained from the saved 10,000 paired draws,
not inferred solely from the rounded width. The two arm estimators correlate
**0.9681**. Pairing explains how their difference can be much more precise than
either absolute score. These figures reproduce the [H1 mixing result](H1_baseline_results.md#paired-gap-and-mixing-estimates).

That contrast evaluated **MELAUDIS**, resampling four uneven target groups,
six source folds and five selections. It compared procedural mixing with
real-only training. CAST-5A instead evaluates held-out **IDMT** and compares
**RC: real + CAST** with **RA: real + conventional augmentation**. H1 did not
run RA, RC or their pairing. CAST-5A also proposes a new common DC-removal and
unit-RMS profile. Thus **the actual RC−RA covariance, variance and effect remain
unknown**; no single expected interval width can be estimated directly for it.
The target mixing result is a useful optimistic contextual scenario, not a
substitute for source-domain precision evidence.

If its empirical asymmetric interval shape transferred unchanged, an observed
RC−RA estimate would need to exceed **4.473 points** for the lower bound to
exceed three. A symmetric half-width approximation gives **4.751 points**.
The normal SD-based 80%-support planning effect is **5.490 points**. None of
these predicts that CAST will achieve the required gain.

## Source-domain paired variance evidence

The [H1 source diagnostics](H1_source_diagnostics.md) use the same six IDMT
group holdouts and five selection seeds proposed for CAST-5A. I reconstructed
all saved scalar macro-F1 values from their existing confusion matrices,
replayed the shared group/selection and site/selection indices, and reproduced
the paired intervals. Site resampling carries both date groups together; it
does not refit site-held models unless the contrast itself is `held_site`.

BEATs is the proposed primary representation. All values below are percentage
points. “SD” is the SD of the paired bootstrap mean estimator, not the SD of
individual recordings or 30 supposedly independent fits.

| Existing source contrast | Group SD | Group CI width | Site SD | Site CI width | Group-based true effect for 80% support of >3* |
|---|---:|---:|---:|---:|---:|
| Nested regularization − fixed C | 0.469 | 1.925 | 0.346 | 1.401 | 4.315 |
| Native16 − core8 bandwidth/resampling | 1.186 | 4.753 | 1.104 | 4.371 | 6.323 |
| One-second − two-second context | 0.919 | 3.627 | 1.005 | 4.033 | 5.575 |
| 359 −190 parents/class, fixed C | 1.266 | 4.954 | 1.132 | 4.188 | 6.547 |
| 359 −190, penalty-scaled C | 1.299 | 5.129 | 1.181 | 4.372 | 6.639 |

*Normal location/scale approximation, conditional on the historical paired SD.
The bandwidth/context comparisons are waveform-profile perturbations; the
budget comparisons change training support. They form a useful planning range,
not interchangeable replicas of RA or RC. Nested C is a low-variance reference;
choosing it alone would be optimistic. Group arm-estimator correlations for
bandwidth, context and the two 359−190 contrasts are 0.930, 0.951, 0.876 and
0.873 respectively; their paired variances are **1.407, 0.844, 1.603 and
1.687 points²**. The saved record retains both variances and covariance.

The other declared contrasts remain visible as stress cases:

| Existing source contrast | Group CI width | Site CI width | Interpretation |
|---|---:|---:|---|
| Nested threshold − fixed threshold | 4.888 | 5.518 | Decision-rule change; substantial truck-recall cost |
| Entire-site holdout − date-group holdout | 14.226 | 18.830 | Different training-domain support; not ordinary augmentation |
| 359 −25 parents/class, fixed C | 11.608 | 14.012 | Large budget change |
| 359 −25, penalty-scaled C | 11.527 | 14.176 | Large budget change |

Consequently, **3.6–5.1 is not an upper bound on RC−RA width**. If its benefit
varies strongly across locations, the few-site limitation could dominate.
Site-bootstrap intervals need not always be wider: averaging the two observed
date effects can reduce variance. A narrow conditional interval from three
sites does not establish calibrated population coverage.

## Detecting a gain versus establishing a practical margin

The roadmap currently requires the lower paired two-sided 95% confidence bound
to exceed **3**, rather than merely exceed zero. For a normal planning model
with paired-estimator SD \(s\),

\[
\text{width}\approx 2(1.960)s,\qquad
P(L>3\mid\Delta)\approx\Phi\!\left((\Delta-3)/s-1.960\right),
\]

\[
\Delta_{80}\approx3+(1.960+0.842)s.
\]

At a true effect exactly three, the approximate probability of the lower bound
exceeding three is **2.5%**, regardless of precision. Approximately **95%** of
intervals cross the margin at that boundary. This is the behavior of the chosen
superiority rule, not a defect that more recordings can eliminate at the exact
boundary. Detecting a positive three-point effect is a different question.

| Conditional variance scenario | True +3: probability L>0 | True +5: probability L>3 | True +6: probability L>3 | True +7: probability L>3 |
|---|---:|---:|---:|---:|
| H1 target procedural-mixing SD | 92.1% | 61.4% | 92.1% | 99.4% |
| Source context-change SD | 90.4% | 58.6% | 90.4% | 99.2% |
| Source bandwidth-change SD | 71.6% | 39.2% | 71.6% | 92.1% |
| Source 359−190 fixed-C SD | 65.9% | 35.2% | 65.9% | 88.5% |
| Source 359−190 scaled-C SD | 63.7% | 33.7% | 63.7% | 86.9% |

These are not prospective frequentist power guarantees. The original bootstrap
is conditional on fixed overlapping fits, finite observed groups and selection
seeds. It does not include retraining on newly sampled sites, unidentified
recording sessions or all CAST/augmentation stochastic variability. Normal
approximations are especially uncertain with only three sites.

A further variance sensitivity is retained for every contrast. With the
bandwidth proxy, multiplying paired SD by 0.5, 1, 1.5 or 2 gives normal interval
widths **2.32, 4.65, 6.97 or 9.30 points**. At a true five-point benefit, the
corresponding probabilities of clearing the three-point margin are **92.1%,
39.2%, 20.2% or 13.2%**. Actual RC−RA pairing therefore matters materially.

## Can additional seeds resolve the problem?

For the existing difference matrix \(D_{sf}\), decompose each cell into its
grand mean, group effect \(a_f\), selection effect \(b_s\), and double-centered
residual \(e_{sf}\). The variance of the exact empirical crossed-bootstrap mean is

\[
V=\frac{\operatorname{mean}(a_f^2)}{G}
 +\frac{\operatorname{mean}(b_s^2)}{S}
 +\frac{\operatorname{mean}(e_{sf}^2)}{GS}.
\]

These are empirical population second moments, not variance-component estimates
for new real-world groups. For site resampling, first average both dates within
each site to form a five-selection × three-site matrix. Independent exhaustive
multinomial enumeration verified this identity: 58,212 weighted selection/group
count-vector combinations and 1,260 selection/site combinations, with maximum
variance disagreement **3.55e−15 points²** across the BEATs contrasts.

For the bandwidth contrast, group, selection and interaction terms are
**0.736, 0.440 and 0.204 points²**. The normal width from this exact empirical
variance is **4.60 points** at five selections, **4.03** at ten, **3.71** at
twenty, and **3.36** even with infinitely many selections. The corresponding
site-level floor is **3.27 points**. The fixed-budget contrast's group-level
floor is **4.21 points**. These are conditional extrapolations under unchanged
heterogeneity, not a recommendation to treat resampled/repeated seeds as data.

More selection or synthesis seeds can reduce Monte Carlo/training variation.
They cannot turn six conservative date groups at three locations into more
independent recording domains. Additional independent sites/recording sessions
would address that limitation; their required number cannot be determined from
unobserved RC−RA variance or these three sites alone.

## Recommendation for the freeze

Retain the three-point margin as the scientific practical-effect threshold.
**Do not freeze a claim of 80% power for an effect of three points under the
current lower-bound-above-three rule.** H1 supports proceeding as a bounded
source-domain estimation/development comparison, with a material chance of an
inconclusive margin result if the real benefit is only three to five points.
It is not automatically inconclusive: effects near zero can rule out the
practical margin, and sufficiently large consistent gains can support it.

Before executable CAST-5A freeze:

1. Record these conditional precision scenarios alongside the unchanged margin.
   Use roughly 6–6.6 points as a planning alternative for 80% margin support
   under the central source proxies; retain the wider-variance sensitivities.
   This is an assumed design alternative, not an expected CAST effect.
2. Freeze the paired RC/RA randomness, all outer roles and equal-group estimand,
   and both six-group and three-site uncertainty reports. Decide beforehand how
   discordant group/site conclusions affect advancement. Do not select the
   narrower interval after seeing results or count 30 fits as independent units.
3. If a high probability of a decisive result is required, obtain a separately
   scoped, training-only matched RC/RA variance pilot or more independent
   source groups before fixing the budget. A pilot on these same sites remains
   development evidence; it cannot certify new-site uncertainty. No such pilot
   was run here.
4. If the intended scientific target is instead *detecting a positive effect
   of about three points*, explicitly revise the decision question before the
   experiment. An interval above zero plus a three-point estimate does not
   establish that the true benefit exceeds three. This analysis does not make
   that change or lower the margin.

The whole-site sensitivity, class-recall constraints, multiplicity decisions
and evaluation of other arms still need explicit freeze choices. No CAST-5A
protocol, classifier run or transfer experiment is authorized by this planning
calculation.

## Reproducibility and complete results

[Analysis instructions](../experiments/cast_coverage/precision_planning/README.md)
point to the thin reader and reusable numerical helpers. The retained revision
is `artifact:cast/precision/cast5a_h1_precision_20261011_v2`.
`precision.json` includes **44 contrast/resampling rows**: both BEATs and MFCC,
nine source contrasts under both group and site resampling, and four target
mixing contrasts per representation. It retains paired score matrices,
covariances, correlations, bootstrap SDs, empirical variance decompositions and
all normal location/scale scenarios. No representation was selected by its
result. MFCC remains secondary in the proposed design.

Each input is hashed and checked again after analysis. Saved result-file hashes,
the H1 canonical wrapper and frozen diagnostic configuration/lock are validated.
All source bootstrap samples replay, all **36 source interval rows and eight
target interval rows** agree with their original summaries, and paired-variance
identities and closed-form fixtures pass. The executed analysis code and a
payload hash lock are stored beside the result. Source code changes, outputs
and documentation are additive; original H1 artifacts, grouped CAST contracts,
harness tests and baseline remain untouched.
