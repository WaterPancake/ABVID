# Joint confirmation plan: CAST and H2 replication on one sealed set

Decided 2026-10-11 (user): use the sealed DCASE 2024 Task 10 development set for
**both** the CAST transfer question and a corrected H2 replication, frozen together
and scored in **one** pass. The set is not split between studies, and neither study
opens it first. This plan is not a frozen protocol; it orders the work and lists the
open decisions.

Basis: [proposed experiments](../../reports/proposed_experiments.md) allows one
confirmation batch for several comparisons if all are fixed before access. The seal
and its rules are in [`dcase2024_t10`](../admission/dcase2024_t10/README.md); the
sealed `PROTOCOL.md` is not edited.

## Why one joint batch

- **Opening it twice:** after the first look, the set is no longer untouched.
- **Splitting by site:** each study would get about two informative sites. cv support
  is 96/93/35/4 minutes at loc1/loc3/loc6/loc5.
- **Splitting by time within a site:** not independent; the site, array and traffic
  are shared.

## Tracks (develop separately on IDMT/MELAUDIS; nothing touches sealed audio)

| Track | Steps | Status 2026-10-11 (evening) |
|---|---|---|
| **C: CAST** | CAST-4A grouped coverage → CAST-4B joint metric → CAST-5A within IDMT → CAST-5B on exposed MELAUDIS → freeze one CAST procedure | **4A:** parents complete; no coverage result (WAV `PEAK`-timestamp replay defect, [report](../../reports/CAST4A_run_20261011.md)). **5A complete 2026-10-12:** BEATs RC − RA macro-F1 +0.47 [−0.97, 2.08], so a practical gain is **ruled out** ([report](../../reports/CAST_source_classification.md)). The CAST-5B precondition is not met |
| **H: H2 replication (H2R)** | renderer v2 with the H2 filter latency declared and validated → H2R design → development run on exposed MELAUDIS → freeze | Renderer v2 done and tested, including bit-exact archive parity. The corrected 2×2 assembled from existing heads shows **no usable classifier** once latency is removed ([H2R draft](H2R_DRAFT.md)); design decision pending |
| **A: access** | label-blind cross-corpus overlap audit tool → DCASE feature route (four-capsule channel mean, the comparison's resampling route, the sealed window rule) → precision check from DCASE support counts (metadata only) | Overlap tool done (`abvid.data.overlap`; real-audio check passed). Audit-access (A1) and scoring (A2) authorization checks are implemented. Rough precision noted in the [protocol draft](PROTOCOL_DRAFT.md) |
| **J: joint freeze** | one document naming both studies' heads, preprocessing, metrics, primary designation and multiplicity rule → overlap audit run → human authorization → single scoring pass → seal marked consumed | [Protocol draft](PROTOCOL_DRAFT.md) unfrozen; waiting on tracks C/H and on user decisions |

DCASE metadata already admitted may inform the design without opening audio:
- each site's geometry (`meta.json`);
- class support;
- the provider's synthetic per-site pass-bys, after their own admission as
  development material.

Any such use is declared in the frozen document.

## Rules for the joint freeze

1. **Primaries and multiplicity.** Each study keeps one primary contrast:
   - CAST: RC − RA (real + CAST vs real + ordinary augmentation).
   - H2R: the declared source-versus-path contrast.

   One study is named primary, or a Holm correction across the two is declared
   before access.
2. **Metrics:** macro-F1, balanced accuracy and AUROC, plus AP against prevalence,
   with an AUROC margin declared in advance. The task is reported as **car/cv**,
   never as car/truck.
3. **Uncertainty:** whole-site resampling, with draws lacking a class rejected and
   counted. Per-site rows are reported, flagging sites with fewer than about 10 cv.
4. **Failed tracks.** A track that fails development is either dropped before the
   freeze or included as a pre-declared secondary arm. It is never added after access.
5. **Single pass.** Features are extracted once and every frozen head is scored once.
   The seal then becomes `consumed`, and later redesigns need new confirmation data.

## Decisions

Made by the user on 2026-10-11:
- **Primary:** CAST RC − RA is the single primary. H2R is a pre-declared secondary
  without multiplicity adjustment.
- **H2R:** the existing hashed latency-free heads only, with no new training.
- **AUROC margin:** 3 points, a stated convention.
- **CAST-5A:** defaults accepted (RA, RU, a 5-point truck-recall regression limit),
  and a likely inconclusive outcome accepted.

**Consequence of CAST-5A (2026-10-12):** the CAST primary was conditional on passing
5A, and it did not pass. H2R's corrected development evidence is also null. **No
candidate with positive development evidence currently exists for the one-shot DCASE
pass.** The user must decide between:
- keeping DCASE sealed until such a candidate exists;
- re-scoping the pass to another frozen question (for example, replication of the
  real-only IDMT → new-dataset loss, using the already frozen real-only heads), with CAST
  and H2R as secondaries.

**Decision (user, 2026-10-12):** keep DCASE sealed for later. No confirmation pass is
planned until a candidate has positive development evidence.

Still open:
- What the eventual DCASE pass will test.
- Authorization A1, the audit download, at the joint freeze.
- Authorization A2, the single scoring pass.

Long term, the planned roadside collection ([target panel](../../reports/target_panel.md))
removes the need for studies to share one confirmation set.
