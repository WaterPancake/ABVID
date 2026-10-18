# Military acoustic signature P0 research and preparation

2026-10-11; protocol `military_signature_p0_v1`; base commit
`1937c8992da93f33dba7f4a228e4c2812d0535c5`, with a dirty working tree.

P0 produced a frozen draft protocol, explicit manifest schema, offline caption-hidden
trimming/import/export tools, conservative provenance/reuse grouping, and artificial
validation evidence. The tooling passed its focused checks. This establishes
preparation mechanics only. No real audio/video was accessed, no representations
were extracted and no classifier or CAST model was fitted. The retired military
benchmark and protected-source roles were not resumed or changed.

The scientific question remains open: can the proposed descriptors retain vehicle
model information across independent sessions while reducing nuisance information?
Source-model dominance over propagation mismatch remains a hypothesis. Existing
papers motivate components but do not validate this proposed study population.

## Reviewable deliverables

- [Frozen protocol draft](../experiments/military_signature/PROTOCOL_DRAFT.md),
  [canonical configuration](../experiments/military_signature/p0_protocol.json),
  [manifest schema](../experiments/military_signature/manifest_schema.json), and
  [freeze ledger](../experiments/military_signature/p0_freeze.json).
- [Reusable implementation](../src/abvid/signature/__main__.py): synthetic fixtures,
  strict manifest/config admission, blind packet construction, interval import,
  native-frame export, label-blind waveform reuse audit and group/readiness checks.
- [Source evidence](../experiments/military_signature/source_evidence.json): paper
  IDs, PDF byte hashes/page/section locators, original Markdown references, author
  statements, interpretations, unknowns and failed source access.
- [Public candidate metadata](../experiments/military_signature/candidate_metadata.json):
  seven HTML-only leads, retrieval times, snapshot hashes, conflicts and rights
  quarantine. This is deliberately not an admitted media/segment manifest.
- [Synthetic regression checks](../tests/test_signature_p0.py) and the existing
  overlap-audit regression. The new test file requires human review before merge.

## What the literature supports

Read the four local Markdown representations first, then verified the selected
PDF pages, including rendered tables. These are a focused feasibility review,
not an exhaustive literature search.

| Source and verified locator | Evidence | Implication for this project |
|---|---|---|
| P04, Altmann 2004, PDF p16 / printed p728, §5.1; PDF p21 / p733, Table 5 | Engine and track periodicities; diesel Leopard 1/2 examples; spectral peak/order ambiguity. | Periodic descriptors are plausible. Dominant peaks alone do not identify RPM or clean source structure. |
| P05, Altmann et al. 2002, PDF p20 / p1104, Table 3 footnote; p21–22 / p1105–1106, conclusions | Training/testing vectors share passes; harmonic powers depend on state/angle; engine types can unite different vehicle types. | Published recognition is not evidence for this project's independent-session model claim. |
| P11, Göksu 2018, PDF p2–3 / p95–96, §II | Four physical cars, with one speed-sweep episode training and another testing. | Speed variation was studied, but independent-session/model-population transfer remains unestablished. |
| P12, Guo et al., journal issue 2012, PDF p9–11 / p37–39, §6/Table 1/Fig. 3 | Five anonymous classes; six runs/class; three runs training, three testing. | Run holdout is useful precedent; named models, individual identities and independent sessions remain unknown. |
| HW_M1_AGT1500, Honeywell brochure, PDF p1, product description | M1 Abrams uses the AGT1500 gas turbine. | Verifies a coarse powerplant association, not a recording-specific acoustic cue. |

Full claim locators and PDF hashes are in the source ledger. The proposal's
T-72/T-80/T-90/T-80UD engine table remains incompletely verified. Rostec pages
failed retrieval, ODIN returned no readable content, and the official ARMOR PDF
was inaccessible through the web tool. Search excerpts were retained as leads,
not upgraded into verified facts. Variant-specific candidate powerplants and L3
engine-order labels remain unknown where unsupported.

## Public metadata feasibility

HTML snapshots were fetched without loading scripts or associated media. The
seven leads cover captions for M1, Leopard, Bradley and Stryker, not the six-model
pilot quota. They provide **zero verified independent recording groups** and no
usable-audio/background evidence.

| Candidate | Metadata lead | Unresolved issue |
|---|---|---|
| [Abrams 1754](https://www.dvidshub.net/video/1754/abrams-tank-b-roll) | 2005; AFN Iraq; 83 s | Location, variant, recording identity and audio quality unknown. |
| [Abrams 572786](https://www.dvidshub.net/video/572786/taa-abrams-set-up-b-roll-package) | 2017; 1st Marine Division; 505 s; M1A1 tag | Tag is not verified interval identity; backgrounds and audibility unknown. |
| [Leopard 855729](https://www.dvidshub.net/video/855729/us-soldiers-explore-finnish-leopard-battle-tank) | 2022; 5th MPAD; 38 s; caption 2A4 | Cultural-exchange scene may lack clean running audio; credits differ between byline/caption. |
| [Bradley 776583](https://www.dvidshub.net/video/776583/b-roll-stallions-conduct-bradley-live-fire) | 2020; 319th MPAD; 43 s | Firing described; contamination not auditioned. |
| [Stryker 386317](https://www.dvidshub.net/video/386317/m1126-stryker-combat-vehicle-convoy-ramstein-air-base) | 2015; DIMOC; 126 s; M1126 | Convoy caption does not establish one audible vehicle. |
| [Stryker 937004](https://www.dvidshub.net/video/937004/pinon-canyon-maneuver-site-b-roll-package) | 2024; 4th Infantry Division PAO; 263 s | Caption Sept 3 versus page Sept 4; recording/session continuity unknown. |
| [Leopard/NATO 879491](https://www.dvidshub.net/video/879491/nato-allies-train-ukrainian-soldiers-leopard-tanks-poland-b-roll) | 2023; Natochannel; 283 s | Date taken/posting conflict; editorial-use restriction despite public-domain banner. Rights quarantine. |

The [DVIDS FAQ](https://www.dvidshub.net/about/faq) describes exceptions to general
public-domain availability. Per-asset restrictions must be resolved before any
future collection; hosting on DVIDS alone is not a rights clearance or an outlet
control. Audio suitability, permissions for the intended research purpose and
independence were not inferred from public metadata.

## Design corrections and negative findings

1. **Background does not identify a channel inverse.** In `x = Hs + n`, a
   source-free interval constrains noise `n`; it does not identify `H` or separate
   `H` from `s`. This is algebraic reasoning. The draft permits a noise-reference
   ablation, not a claim of microphone/propagation removal or clean CAST source
   recovery. Calibration and physical identification remain new study inputs.
2. **Eight groups is feasibility; ten is only a minimum.** The pilot target is
   six models × eight independently verified components, not eight asset IDs per
   model. Ten groups/model is a proposal threshold, not a power calculation.
   New unexposed groups and evidence across physical vehicles are needed for a
   model-wide claim. Adding two groups does not make exposed pilot results blind.
3. **DVIDS-only support limits outlet conclusions.** A constant outlet probe is
   undefined. Originator/producer/camera confounds remain. With only M1 turbines,
   a turbine/diesel contrast is also model-associated. Russian models appearing
   only in new outlets do not separate outlet and model effects.
4. **Labels and effective fits have limits.** Unknown engine orders yield no L3
   AP. Fine separation without coarse hierarchy/null support does not pass the
   claim gate. CAST fits to recorded observations remain effective descriptors;
   their stability is not recovered physical source truth. Representation learning
   and hyperparameter selection do not use vehicle labels. Development phase
   gates expose labels/results and must be disclosed as development evidence.
5. **No-match is not independence.** The reuse audit catches the tested crop,
   gain/polarity and resampling cases, but leaves short/silent inputs unresolved.
   Codec/time-stretch/edit/mixture behavior, real false matches and aggregate
   pilot-scale memory/runtime remain unverified. All known provenance, suspected
   reuse and repeated known individuals are conservatively grouped before roles.

These are limitations of the current evidence and study design, not evidence that
the eventual experiment will succeed or fail.

## Validation evidence

The final artificial workflow is at
`artifact:military_signature/p0/validation-20261011-final/validation.json`.
It records Python/library versions, the base commit, dirty-tree status,
implementation/config/schema hashes, seeds 17/29 and every raw fixture hash.

| Check | Observed result | What it establishes |
|---|---|---|
| Focused signature + existing overlap tests | 37 passed: 32 new signature cases, 5 overlap cases | Authorization/type/provenance guards, contamination/overlap handling, grouping, packet commitments, native rounding and tested reuse cases. |
| Artificial end-to-end workflow | 6 assets; 8 native clips; 2 logged rejections; 3 components | Fixture generation → blind packet → import → grouping → native export, with exact sample comparisons. |
| External JSON Schema validator | Draft 2020-12 schema and resulting manifest valid | Interoperable structural schema; runtime additionally checks scientific constraints. |
| Offline viewer DOM events (`jsdom@26.1.0`) | 6 cards; 7 saved intervals; import/export produced 2 clips and retained 5 rejections | Add/remove, invalid bounds/overlap/contamination guards, reviewer/completeness checks and JSON serialization. Audio/play/download APIs were mocked. |
| Native Firefox access | Computer-use approval rejected access | Actual decoding, audible playback, layout and native download remain unverified. Human browser handoff required. |
| CAST harness canary | 707 collected vs 467 expected; 1 failed, 2 passed | Count drift remains visible; baseline-file/frozen-file checks passed. |

DOM evidence is at
`artifact:military_signature/p0/validation-20261011-r2/dom-check/validation.json`.
The reproducible DOM script and package lock are retained there. It is a functional
JavaScript check, not a real browser or blind scientific review. The final
artificial workflow uses the same viewer source bytes. No generative vehicle
simulation, trained representation or scientific metric was tested by this work.

The harness already collected 661 against 467 before P0. This work legitimately
adds 32 cases; other work also changed the shared suite. Do not attribute the
entire 707 count to this project. The human-only baseline and existing tests,
lockfiles and frozen experiment configs were not edited to make the canary pass.
Four `experiments/joint_confirmation/*DRAFT.md`/`PLAN.md` files changed elsewhere
during this work; they were not edited or reverted by P0.

## Remaining decisions and exact handoffs

P0 preparation can be reviewed using the freeze check and reproduction commands
in the protocol. Prior versions remain in external storage; reruns require fresh
paths. No commit or merge was performed.

- **Browser handoff:** open the final `blind/index.html` locally, play an artificial
  recording, review all cards, save JSON and exercise import. Supply approved
  Firefox access or perform this human check; current DOM evidence cannot prove
  decoding, rendering or audible playback.
- **Harness handoff:** review the new test file line by line; account for all
  legitimate suite changes in both environments; have a human refresh the baseline
  in the same commit before merge. No agent baseline refresh is authorized.
- **Research handoff:** provide explicit bounded P1 collection/processing/fitting
  authorization, exact frozen checkpoint/extraction/CAST budgets, confirmed
  per-asset rights, independently supported session groups and a six-model roster.
  Missing or ambiguous powerplants stay unknown unless an accessible authoritative
  PDF or primary specification resolves them.
- **Confirmation design:** before model-wide claims, freeze new unexposed groups,
  physical-vehicle support, final allocation/power analysis, noise-reference
  assumptions, and synthetic recovery/negative-control criteria. Preserve all
  failed gates and inconclusive/null results; do not collect until a result turns
  positive.

The next scientific action is a separately authorized P1 admission/pilot protocol.
The P0 artifacts do not authorize that action or provide its data.
