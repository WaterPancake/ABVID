# CAST gap diagnosis and data decision

Follow-up: the separately authorized [bounded stage diagnosis](CAST_stage_diagnosis_20261025.md)
is now complete. It localizes an early reconstruction mismatch and an additional
population gap; final calibration helps overall in the fixed role. The saved-cache
audit below remains unchanged as the earlier evidence.

2026-10-25. **The clearest measured CAST gap is between its generated observations
and useful real-data class structure. Collecting real recordings from new contexts
is justified; increasing the number of draws from the current CAST population is
not yet a supported remedy.** A bounded reconstruction-versus-sampling diagnostic
should precede another renderer revision. New data can improve diversity and
evaluation credibility, but an improvement in performance is not guaranteed.

This report combines existing results with a new, post hoc audit of saved CAST-5A
features and frozen heads. No new model was fitted, no audio or encoder checkpoint
was opened, and no target or sealed-confirmation data was accessed. These are
source-domain development diagnostics, not independent confirmation or a new
cross-dataset CAST result.

## What is established

| Gap | Evidence | Conclusion and limit |
|---|---|---|
| Added classification utility | CAST-5A BEATs RC − RA macro-F1 **+0.47 points [−0.97, +2.08]**; AUROC **−1.78 [−4.75, +0.70]** | The declared ≥3-point benefit is ruled out under this protocol. A smaller benefit is not established. This is conditional on v15, 190 real + 190 synthetic examples/class, the fixed encoder/head and shape-only 2-second profile. |
| Synthetic-to-real discrimination inside IDMT | CAST-only BEATs AUROC **60.55%**, versus real-only **71.31%**; CAST-only truck recall **29.56%**, versus **65.57%** | There is a 10.76-point difference between mean AUROCs and a substantial truck-recall deficit. This difference is descriptive; no new paired confidence interval is claimed. |
| Acoustic coverage versus recognition | v15 meets its marginal-descriptor objective on one repeatedly exposed outer group; 5A adds no practical utility across six IDMT groups | Successful descriptor coverage did not establish useful classifier variation. The coverage metrics and the classifier inspect different structure. |
| Real-data generalization | H1 whole-site holdout costs BEATs **10.21 F1 points**; 190→359 real examples/class gives **+0.76 [−1.88, +3.07]** | Existing location/context vulnerability is much better evidenced than a benefit from more excerpts within the same groups. New-context data is a priority, not a proven cure. |
| Broader coverage evidence | CAST-4A prepared 1,864 parents and 27,000 samples, but all nine replay gates stopped before held scoring | This is a WAV-container timestamp defect. It supplies no positive or negative CAST-4A coverage result. |

Sources: [CAST-5A](CAST_source_classification.md), [v15 development record](../CAST_improvement.md),
[H1 source diagnostics](H1_source_diagnostics.md), [CAST-4A technical failure](CAST4A_run_20261011.md).
RC means real + CAST; RA means real + conventional augmentation. Historical intervals
retain their original conditional group/selection bootstrap interpretation.

## New finding: CAST's class structure is displaced in feature space

The [audit specification](../../experiments/cast_utility/GAP_AUDIT.md) was written before
calculating these new metrics. The [complete JSON](CAST_gap_audit_20261025.json)
retains every role and both representations. The audit used 30 role populations,
120 already fitted R/C heads, and 130 hash-checked input files, including all four
feature caches, head records and the original results. Historical artifacts were
rechecked after use. Every metric below is an equal mean over 30 roles, with no
independence or confidence-interval claim.

R is the saved real-only head; C is the saved CAST-only head. “Real parents” means
each role's selected training parents, not its held group. C samples are unconditional
population draws, not reconstructions paired one-to-one with those parents.

| Frozen head → input | BEATs AUROC | BEATs balanced accuracy | MFCC AUROC | MFCC balanced accuracy |
|---|---:|---:|---:|---:|
| R → its real training parents | 100.00% | 100.00% | 85.71% | 77.69% |
| R → conventional augmented copies | 97.22% | 91.89% | 80.69% | 73.12% |
| R → CAST population samples | **67.64%** | **59.96%** | 76.91% | 68.64% |
| C → its CAST training samples | **100.00%** | **100.00%** | 84.63% | 77.11% |
| C → the real training parents | **67.63%** | **57.62%** | **58.89%** | **55.02%** |

The generated BEATs training problem is perfectly fitable by its saved linear head,
yet the learned rule carries poorly even to the real parents behind that population.
Its truck recall on those real parents is only **33.30%**. Ordinary augmentation
retains much more of the original real head's ranking. Conversely, MFCC's real head
still ranks CAST samples reasonably well while its CAST-trained head ranks the real
parents poorly. Thus the mismatch is directional and representation dependent;
it cannot be reduced to one universal “no class signal” diagnosis.

Training accuracy is optimistic by construction. In particular, the BEATs R head
also perfectly fits real training data and generalizes poorly to held groups. The
new probes are descriptive evidence about alignment, not new performance estimates.

### Geometry using real-training scales only

Apply each R head's existing StandardScaler to all inputs. Let `d` be the vector
between real truck and car centroids and `d_s` its generated counterpart. The
direction cosine is `dot(d,d_s)/(norm(d)*norm(d_s))`. Class displacement is the
same-class centroid offset divided by `norm(d)`. These are diagnostic coordinate
distances, not probabilities, physical measurements or a new acceptance test.

| Diagnostic | BEATs conventional aug. | BEATs CAST | MFCC conventional aug. | MFCC CAST |
|---|---:|---:|---:|---:|
| Real/synthetic class-direction cosine | **0.977** | **0.294** | **0.986** | **0.514** |
| Generated / real centroid separation | 0.981 | **1.344** | 0.968 | 0.908 |
| Car centroid displacement / real class separation | 0.370 | **5.932** | 0.262 | **3.636** |
| Truck centroid displacement / real class separation | 0.410 | **6.014** | 0.261 | **3.540** |
| Car within-class RMS spread / real | 1.032 | 1.329 | 1.057 | 0.984 |
| Truck within-class RMS spread / real | 1.035 | 1.345 | 1.063 | 1.037 |

All 30 BEATs CAST role cosines lie between **0.159 and 0.445**; ordinary augmentation
lies between **0.960 and 0.987**. This is not driven by one selected role. CAST can
separate its synthetic class centers more than real centers while separating them
in a poorly aligned direction. It is therefore inaccurate to describe the BEATs
result simply as collapsed car/truck centroids. There is a large domain offset and
distorted class geometry despite nonzero synthetic class separation.

High-dimensional centroid distances are scale dependent and do not characterize
full distributions. The normalizer is real class separation, not real within-class
spread. A sixfold value does **not** mean a six-standard-deviation displacement.
Roles overlap, share synthesis streams, and repeat parents. Neither their count nor
the number of generated samples creates independent real recording contexts.

## Where the loss could arise

The present evidence locates a failure at the **sampled population → representation
→ real class decision** interface. It does not yet identify the upstream cause.

1. **The fitting/coverage objectives can miss class-relevant structure.** The
   four-resolution spectral fitting loss and marginal descriptor metrics reward
   acoustic matching, rather than preservation of a frozen encoder's class geometry.
   v15 widening raised coverage while worsening outer class-balanced W1 by 4.460%.
   This mismatch is observed; proving which missing structure causes the utility
   failure needs an ablation. Changing the objective to BEATs loss would create a
   new method, and should not turn this diagnostic representation into its own
   unqualified success criterion.
2. **Fitted controls are nonunique.** CAST-4A retained disagreeing near-equivalent
   starts in 1,714/1,864 parents (**92.0%**), plus noise-dominant and harmonic-order
   flags. Numerically successful fitting does not establish physical identifiability.
   These flags support inspecting uncertainty and winner selection; they do not
   demonstrate that alternative-start sampling would fix utility. The historical
   v4 alternative-mixture attempt also failed its source criteria.
3. **Calibration/sampling may disrupt useful dependencies.** Reconstructed parents,
   calibrated parents and unconditional population draws have not been compared in
   classifier feature space here. Their separation is the missing diagnostic.
   More draws from an unchanged population do not repair a systematic offset.
4. **Real-data context sensitivity remains a separate bottleneck.** BEATs can fit
   the real labels, while location holdouts remain poor; RA also adds essentially
   no utility. Frozen-feature/head limitations, preprocessing and context shift are
   candidates. The H1 bandwidth result (~3 F1 points) warrants a matched sensitivity,
   but cannot be silently substituted into the fixed CAST comparison.

No result here identifies source mismatch over propagation, environment or sensor
mismatch. CAST fits recorded observations containing all these contributions.
Location also bundles fleet, road, source state and acquisition context. The earlier
H2 path effect was largely reproduced by filter latency, so it does not resolve
that physical attribution either ([latency control](../H2_air_latency_control.md)).

## Should we gather more data?

**Yes: prioritize independently documented real contexts, stronger truck support,
and a small controlled mechanism dataset.** This recommendation follows the observed
learning-curve and site-holdout results; it is not an estimated return on data
collection. The two needs are different:

| Need | Data that addresses it | What it would establish |
|---|---|---|
| Better classification development and credible confirmation | Camera-labelled car/truck pass-bys across new sites, dates, fleets and recording sessions; both classes in each context | Whether generalization improves with new contexts and whether CAST adds value over matched extra real data |
| Explain the simulator's physical mismatch | Repeated known vehicles/states with synchronized audio and RPM, measured speed, geometry, sensor/gain metadata; multiple positions and backgrounds | Controlled tests of state/path/sensor effects; not automatic clean-source recovery |

For a feasible field pilot, retain the existing [roadside plan](target_panel.md#step-4-own-roadside-collection-planned-not-started):
at least **8 capture sessions, 4 sites, 2 dates, 300 trucks overall and 20 trucks per
session**, with car support in every session. These are acquisition targets, not a
power calculation or sufficient independent-confirmation sample size. Assign
development and sealed confirmation roles at capture time; hold out complete
sites/sessions, retain original-recording links, and log unknown vehicle identities
as unknown. Repeated pass-bys of one known vehicle retain the same vehicle link.

Use a reference microphone and a simultaneous secondary sensor, retain full-band
recordings and full pass-bys, and record background-only intervals. Record actual
speed when available; speed limits are not speed measurements and road speed is
not RPM. Define car/truck/bus/van labels before capture, retain ambiguous labels,
and exclude motorcycles from this study. A small RPM-labelled controlled subset
can be separate from opportunistic roadside capture. RPM alone is insufficient to
identify torque, load or source/path separation.

### Public-data options checked on 2026-10-25

| Candidate | Current evidence and role | Remaining admission issue |
|---|---|---|
| **MVD** | Best public candidate found for near-balanced car/truck development: the authors report **1,005 cars and 1,077 trucks**, four urban locations and four MEMS microphones; clips padded to 3 s. [MVD, arXiv:2309.03544, PDF p. 2, §III-A](https://arxiv.org/pdf/2309.03544). | The [current repository](https://github.com/Ashhad785/MVD) links original MVD and augmented MVDA separately and displays MIT, but a dataset-specific grant and session mapping were not established by the reviewed pages. Recording-number + microphone IDs identify samples, not independent sessions. Audit release hashes, overlap and grouping; use original MVD only. |
| **FSD50K** | Secondary broad-context, weak-label check. The [provider release](https://zenodo.org/records/4060432) supplies uploader metadata, uploader-disjoint dev/eval sets and per-clip licences. | The earlier [metadata scan](dataset_scan_20261012.md) counted only 154 Truck clips. That class count was not re-downloaded/recounted today. Multi-label mixtures require an explicit ambiguity rule; uploader grouping does not establish session independence or absence of pretraining overlap. |
| **DCASE 2024 Task 10** | Retain its existing sealed confirmation role. The [official task page](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring) describes six sites and minute-level passenger/commercial-vehicle counts. | Commercial vehicle is a proxy, not truck. Existing metadata has only about four sites with both candidate classes, unknown original sessions, and a separate [seal contract](dataset_admission_dcase2024_t10.md). It cannot be repurposed as CAST debugging data. |
| **Real engine idle data / synthetic RPM data** | A new [real engine release](https://zenodo.org/records/18777405) describes 350 cars and 320 retained idle samples; the [Procedural Engine Sounds provider](https://rdoerfler.github.io/procedural-engine-sounds-page/) describes synthetic audio from three engine timbres and separately pooled controls. | The former does not establish truck pass-by or synchronized RPM support; the latter is useful synthetic development material, not independent real-engine validation. Neither supplies the missing multi-context car/truck test. |

The MVD paper was checked in its text representation first and then against the
authoritative PDF. Its sample-rate unit contains an apparent typo; no header rate
is asserted here without file admission. No candidate audio was downloaded, no
authors were contacted, and no new dataset was admitted in this audit.

## Recommended next work, in order

1. **Run a small training-only reconstruction-versus-population diagnosis before
   changing CAST.** Freeze the comparison first: original training observation →
   smooth8 reconstruction → calibrated same-parent reconstruction → existing
   population sample, under identical preprocessing and frozen BEATs/MFCC heads.
   Retain multiple checking noise realizations and existing ambiguity flags. If
   class alignment is lost at reconstruction, target renderer/objective capacity;
   if it survives reconstruction and fails after calibration/sampling, target that
   stage. Conditional reconstructions never enter a generative evaluation bank.
   This comparison was **not run** by the saved-cache audit.
2. **Begin the new-context data workstream.** First resolve MVD's licence/session
   mapping and plan the documented roadside pilot. Evaluate real-data gains from
   new contexts separately from gains from more clips in the existing contexts;
   keep budgets, representations and fitting rules matched. This is more informative
   than merely extending the current synthesis run.
3. **Keep the negative CAST-5A result and its advancement rule.** CAST-5B's utility
   precondition is unmet. Any later low-real-data, new-context or revised-CAST test
   needs its own protocol and matched R/RR/RA/RC controls. A low-budget learning curve
   could test label efficiency, but that potential benefit is currently unknown.
4. **Repair CAST-4A serialization only as a separate, pinned implementation revision.**
   Preserve the failed run and exact sample arrays. This would complete an open
   coverage measurement; it would not resolve the utility failure. No pinned runner,
   frozen config, test, baseline or lock was edited here.

## Reproduction and verification

Implementation: [`abvid.cast.utility_gap_audit`](../../src/abvid/cast/utility_gap_audit.py).
Run the [audit command](../../experiments/cast_utility/GAP_AUDIT.md) with a new output
filename; existing outputs are not overwritten. Use `.venv-baseline` (Python
3.11.13, NumPy 1.26.4, scikit-learn 1.4.2). The frozen checkpoint identity in the
original feature provenance is `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`;
the audit never loads that checkpoint. Dataset: admitted IDMT_V1 car/truck;
source selections 42/123/456/789/1024, six connected-group roles per selection;
all source controls retain their recorded provenance and unknown sessions.

The canonical H1/grouped JSON configurations and their referenced hashes passed
validation. Geometry identity, translated-centroid and reversed-label controls
passed; independent scikit-learn AUROC/BA calculations matched the first role
for both representations. A second audit reproduced every numerical result,
role row, input hash, historical statistic and design identity exactly; only
working-tree provenance differs. The replay is at
`artifact:cast/utility/gap_audit_20261025_replay.json` (SHA256
`3d329ab529fa4e4521e3e93635dfe5ff2ee3f4285541d0a882fb351f7b2d8de5`).
All **35 existing utility/storage regression tests passed**; no tests or harness
baselines were edited. Local report links resolve. Numerical diagnostics carry no new group uncertainty
claim; original 5A group/site intervals are retained in the JSON.

The original CAST-5A results hash is
`ce2df9b35e58e19b5a4326c8d9e636f3b5b9ac197cc07b4988ddaf7e9d4f2bf3`.
The saved design bundle identity is
`bceaf26b983c0ef948fc492ea25045d77d29c5e0c23cd57a60c48f63f88832fc`;
the execution context's digest of that manifest is
`986708c811e0653b7f705ce351a1a920c3f8170e077d9452c3183185ac93cd6f`.
Both are retained under distinct fields in the new audit. Commit and dirty-tree
status, script/specification hashes and every checked input hash are included.
