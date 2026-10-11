# CAST-4A grouped source coverage protocol

Prepared 2026-10-07. This freezes a source-only design and metadata inventory;
**scientific execution is not authorized by this preparation**. No audio has been
decoded, fits or observations generated, descriptors or embeddings extracted, or
coverage metrics computed for this study. Existing IDMT exposure remains
development exposure. This is neither classification transfer nor independent
confirmation.

The normative machine-readable choices and exact input lists are in
[design.json](design.json), with the frozen
[numerical contract](numerical_contract.json) and
[canonical configuration](../../../configs/experiments/cast_grouped_coverage.json).
Follow [CAST_ROADMAP.md](../../../CAST_ROADMAP.md#cast-4a-broaden-grouped-coverage-evaluation).
The [preparation report](../../../reports/CAST_grouped_coverage_design.md)
records this metadata-only freeze at
`artifact:cast/grouped_coverage/cast4a_design_20261007_v1`.
The future results belong in
[`reports/CAST_grouped_coverage.md`](../../../reports/CAST_grouped_coverage.md);
that report is a planned deliverable, not an existing result.

## Question and fixed scope

Does the fixed v15 population satisfy the existing marginal descriptor criteria
across the six admitted IDMT conservative groups? Compare its coverage/distance
trade-off with v14 and matched controls. Add a separately reported three-site
holdout sensitivity analysis. Neither analysis selects a new width or tunes the
renderer. All current car/truck class rules and existing exposure records remain.

Use only admitted IDMT metadata and, after separate execution authorization, the
exact source observations allowed by each frozen role. No MELAUDIS, mixed-domain
feature cache, released synthetic bank, military/reserved recording, new encoder,
classifier, CAST-4B metric, or target scoring enters this experiment. Historical
protocols, locks, environments and results remain unchanged.

## Input inventory and roles

Resolve every reference through `abvid.paths`. The historical references are:

| Input | Root-qualified reference | SHA256 |
|---|---|---|
| H1 exact selections | `archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json` | `3e427a2f4ee084ecdd4102dd3a9863a55cc06496abdb9688258ec26cbaaf901e` |
| Admitted metadata | `archive:experiments/h1/frozen/h1_common_budget_v1.3/admitted.jsonl` | `ee8b67b002a4528b036e6d2a3ab6f8f6015cd59fffbbc27214ad2786a7d31f96` |
| Frozen H1 configuration | `archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json` | `81783da5dcc135e92c80862d92a4a8da3cf4ae8af36704cd53e51c7291159191` |

The IDMT subset contains 4,413 observations: 3,902 cars and 511 trucks. Each has
recorded admission, class, integrity and metadata passes. Preserve the two
admitted `near_full_scale_not_proof_of_clipping` flags; do not exclude their rows.
The metadata inventory does not constitute a new waveform-integrity check.

### Primary: six conservative connected-group holdouts

Use each literal `lock["folds"][i]["train_ids"]["42"]` list, in its recorded order:
380 unique observations, 190/class, per fold. Held IDs are the corresponding
literal `test_ids`; use all of them. Do not redraw training selections.

| Fold | Held group | Site and date | Held car/truck | Available training car/truck |
|---:|---|---|---:|---:|
| 0 | `connected_4001f06f57cfeee7` | Schleusinger-Allee, 2019-11-12 | 491/79 | 3411/432 |
| 1 | `connected_654b5df57ec64387` | Schleusinger-Allee, 2019-11-13 | 666/110 | 3236/401 |
| 2 | `connected_667018ac7661cccb` | Fraunhofer-IDMT, 2019-10-23 | 254/14 | 3648/497 |
| 3 | `connected_72766641e1daf436` | Langewiesener-Strasse, 2019-11-19 | 1365/152 | 2537/359 |
| 4 | `connected_b00d69ddbb66d0fe` | Fraunhofer-IDMT, 2019-10-22 | 302/10 | 3600/501 |
| 5 | `connected_c6148cf0cafb5753` | Langewiesener-Strasse, 2019-11-18 | 824/146 | 3078/365 |

The held lists partition all 4,413 rows. The 2,280 training-role assignments use
1,462 distinct IDs (997 car, 465 truck), not 2,280 independent recordings. Each
fold uses five conservative training groups. No held-group ancestor may enter
its bank, including through population centers, scales or calibration records.

### Sensitivity: three whole-site holdouts

Keep both dates from each site together. The metadata-only feasibility check
supports the same **190/class** budget in all three folds; no reduced-budget
fallback is invoked.

| Held site | Held car/truck | Available training car/truck | Training groups |
|---|---:|---:|---:|
| Fraunhofer-IDMT | 556/24 | 3346/487 | 4 |
| Langewiesener-Strasse | 2189/298 | 1713/213 | 4 |
| Schleusinger-Allee | 1157/189 | 2745/322 | 4 |

For each exact `site_id`, form `trainpool` from all admitted IDMT rows outside
that site, and freeze:

```python
abvid.data.splits.select(trainpool, 190, 42, f"cast4a_site_v1|{site}")
```

This existing selector permutes sorted IDs separately by class and conservative
group using its deterministic stream seed, then draws in sorted-group
round-robin order, skipping exhausted pools. Its 380 ordered output IDs are
frozen in the design manifest and shared by every arm. Use all held-site IDs.
Do not filter and top up an H1 fold selection, borrow held-site observations, or
substitute another selection after inspecting outcomes. Report the site analysis
separately from the primary six folds.

### Limits and metadata checks

Check within each role pair that file IDs, source-file hashes, duplicate groups,
conservative parent proxies, provenance groups, paired-event IDs and source time
axes do not cross training/held roles. Preserve all known ancestry links. All
4,413 rows have unknown `recording_session`, `original_media_id` and measured
`vehicle_speed`; physical vehicle identities are not established. Retain those
unknowns. Eighteen candidate recording/time-axis tokens do not certify eighteen
independent sessions. The six groups cover only three locations.

The retained device token `SE` and channel pair `CH34` are constant. Site is
associated with speed-limit tokens 30/50/70 Kmh for Fraunhofer/Langewiesener/
Schleusinger respectively; these are not measured vehicle speeds. Small truck
supports, shared locations and overlapping training populations limit inference.

## Fixed populations, arms and streams

For every primary and sensitivity fold, use six arms: `joint`, `prototype` and
`marginals`, each at T=1 and T=1.1. T=1 is the unchanged v14
`spectrotemporal_context` reference; T=1.1 is fixed v15. Apply the historical
width transform only to noise-spectrum, mixture and envelope controls, retaining
spacing and harmonic weights exactly. T=1 must avoid a transform/inverse round
trip. Each control is constructed from its own width's identical population,
with equal group/parent/child weighting and all contributing ancestry retained.

Freeze the full existing renderer/fitter/calibration/metric settings in the
numerical contract. For each bank, recompute training-only group and class centers,
effects, virtual children, prototypes, width centers and descriptor scales.
Instantiate input allowlists and held-group/site exclusion masks from each bank's
actual roles; do not copy historical fold-0 masks into different populations.
Record each bank's statistics separately. Reuse a saved per-recording fit only
after exact settings, source content, complete ancestry and role
compatibility have been audited. A metadata match identifies a candidate for
reuse, not an approved reusable fit.

Use synthesis seeds `[42,123,456,789,1024]`, 50 observations/class/seed/arm,
indices 0–49. Seed 42 for training selection and the five synthesis seeds have
separate roles. Preserve the historical outer donor/child/phase/noise schedules
across folds and widths, without adding a fold/site/width to their RNG identity.
Use the historical `cast_improvement_sampling_v1:outer` namespace and
`WidthPrior(..., fold="outer", ...)` RNG convention while applying the actual
held-role masks. Keep fold/analysis/output identity separate from RNG identity,
with independent ancestry and statistics for every bank. Matching streams does
not imply matching donors or children when the available population changes.

The complete schedule is 3,000 generated observations/fold: **18,000 primary**
and **9,000 site sensitivity**, 27,000 total. All six arms, both classes and every
seed are mandatory. Preserve parent ambiguity, alternative fits, boundary flags,
upstream and width-specific projections, and all generation records.

## Metrics, acceptance and paired summaries

Retain historical descriptor definitions: `log_spectrum`, `bands`, `envelope`
and `modulation`, equally weighted after the frozen training-only scaling rule.
Coverage is the average fraction of held values inside generated 5th–95th
percentile intervals. W1 averages scaled one-dimensional coordinate distances;
it is not multivariate transport. Family spread compares generated and held
scaled descriptor standard deviations under the existing implementation.

For each analysis/fold/class/width, average each metric over **all five** fixed
synthesis seeds. Compute relative gain against each same-width control as
`1 - mean_seed_W1_joint / mean_seed_W1_control`, not the mean of seedwise ratios.
The joint arm passes only when all of these hold:

- Mean marginal coverage is at least 0.80.
- Relative W1 gain is at least 0.025 against **each** of prototype and marginals.
- Each descriptor family's mean spread ratio lies inclusively in [0.5, 2].
- All input, completeness, numerical, ancestry and replay gates pass.

The historical
`metric_config.criteria.relative_W1_gain_over_each_control_each_class = 0.05`
field is retained only as inert historical metadata in the numerical contract;
**0.025 is the effective frozen acceptance value**. The historical outer runner
also set the effective margin to 0.025. Do not treat the legacy field as an
active threshold or silently rewrite the archived contract.
Undefined/nonfinite spread or gain, including a nonpositive control W1
denominator, fails the applicable gate. Do not replace it with a favorable value.

The primary claim concerns fixed T=1.1. Every primary fold/class must pass on
its five-seed mean to claim coverage criteria hold across the evaluated groups.
Apply and report the same criteria to T=1 and separately to each site fold. A
pooled mean cannot rescue a failed fold/class. Keep the historical single-group
v15 result unchanged whatever the new outcome. Report every individual seed,
each fold/class, worst observed fold and the coverage/W1/spread trade-off.

For each class and for the equally weighted two-class summary, report paired
equal-fold mean contrasts: T=1.1 minus T=1 joint coverage and family spread;
T=1 joint W1 minus T=1.1 joint W1; and, within each width, control W1 minus joint
W1 plus the corresponding relative gains. Positive W1 contrasts favor the
candidate joint arm. Do not weight larger held groups more heavily.

Use 10,000 bootstrap draws from `numpy.random.Generator(PCG64(314159))`,
initialized separately for each analysis, resampling six whole folds with
replacement for the primary analysis and three
whole sites for sensitivity. Resample the matched fold vector jointly, retaining
both classes, all arms and all five synthesis seeds; derive its fixed-seed means
before each equal-fold summary. Report 2.5th/97.5th percentile intervals with
linear quantile interpolation. Do not independently resample arms, clips or
seeds. Show synthesis-seed variation separately. These are descriptive
conditional intervals: training folds overlap, dates share sites, and especially
three-site intervals have fragile support. They do not establish independent
physical replicates, fresh confirmation, or a new significance-based pass gate.

## Execution gates, failures and deliverables

1. **Preparation gate:** validate canonical JSON and exact design/input hashes;
   verify literal primary selections, deterministic site selections, budgets and
   metadata separation. Freeze protocol, source-only manifests and numerical
   settings. This authorized preparation ends here, with no research metrics.
2. **Before scientific execution:** obtain separate execution authorization and
   validate the new runner against the frozen design using focused synthetic
   numerical, determinism, ancestry and reference-parity checks. Resolve any
   configuration mismatch before audio access. Do not modify a historical lock
   or launch a scientific run merely to check packaging.
3. **Bank gate:** fit only the selected training IDs under frozen settings, or
   approve exactly compatible saved computations through the recorded audit.
   Freeze each complete 380-parent bank, derived statistics, scales, masks,
   configuration hashes and ancestry before held-role evaluation. No descriptor
   or parameter fitted to a held observation enters that fold's bank.
4. **Generation gate:** finish and inventory the full 3,000-observation schedule
   for a fold before scoring its held-role descriptors. Validate finite bounded
   rendering and exact expected identities; reject missing, duplicated or extra
   records. An observation may train another declared fold, but its training
   role never permits leakage into the fold where it is held.
5. **Evaluation and replay gate:** evaluate the entire frozen held population,
   preserve all per-seed/per-class/per-family outcomes, and verify reproducible
   generation and held-descriptor replay under the recorded environment. Only
   then compute complete-schedule summaries and acceptance decisions.

Record every failed fit, seed and fold without replacing parents, retrying with
new seeds, relaxing grouping, changing settings, or dropping difficult examples.
A bank that cannot be formed is a failed fold. Retain legitimate ambiguous or
boundary fits and their flags; do not discard them by outcome. A technical
failure blocks the relevant downstream stage and any all-groups success claim;
unaffected folds may complete. Mark available partial results explicitly and do
not compute the planned complete-schedule bootstrap interval from a selected
subset. Exact reruns to recover an interrupted operation require unchanged
inputs/settings and a retained interruption/recovery record.

The frozen site selections use 946 distinct IDs (534 car, 412 truck). Across
both analyses, 3,420 training-role assignments use 1,864 distinct IDs. The
historical v11 protocol metadata identifies 380 candidate saved computations;
1,484 selected IDs have no candidate in that named inventory. These are
metadata workload counts, not permission to open or reuse fits, nor a count of
independent sessions. Runtime remains unknown until a separately authorized
fixed-adapter timing pass; do not extrapolate the old pilot timing to this work.

If fixed v15 fails, report the failure and its dominant source-side residual.
One later bounded revision or a separately authorized utility falsification
experiment requires a new contract; neither occurs inside this comparison.

Deliver the planned report, design and role manifests, all metric/failure rows,
population/projection and ancestry records, configuration/checkpoint hashes
where applicable, commit and dirty-source snapshot, exact environment,
hardware/precision, stage-specific runtime/memory, and verified replay commands.
Store audio, banks, descriptors and full runs outside the checkout through
`abvid.paths`. Keep reusable implementation in `src/abvid/cast` and experiments
as thin protocol/configuration layers. This study can establish broader
source-domain coverage only; classifier utility, transfer, clean physical source
recovery and independent vehicle diversity remain untested.
