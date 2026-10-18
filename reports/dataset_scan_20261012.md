# Dataset scan: truck-rich, multi-context vehicle audio (2026-10-12)

**Goal:** find more trucks from more independent recording contexts than IDMT (six
groups at three sites) and MELAUDIS (exposed; 3% trucks) provide. Every count below
comes from published label or metadata files only, read as untrusted data. **No audio
was downloaded** and no model was applied. Nothing is admitted yet: each candidate
still needs provenance checks, a licence check and the cross-corpus overlap audit
([`abvid.data.overlap`](../src/abvid/data/overlap.py)) before any role is assigned.

## Candidates

| Dataset | Truck support | Car support | Independent contexts | Labels | Licence | Exposure / caveats | Proposed role |
|---|---:|---:|---|---|---|---|---|
| **MVD** (Delhi-NCR; [paper](https://arxiv.org/abs/2309.03544), [repo](https://github.com/Ashhad785/MVD)) | **1,077 clips** | 1,005 | 4 MEMS microphones at 4 urban locations; sessions unknown | Clip-level single class (car / truck / motorbike / none), 3 s at 22.05 kHz, padded | Repo is MIT (code). **The data licence is not stated.** Data is on Google Drive with no checksums | Never used here. Different fleet and country. Augmented MVDA copies must be excluded | **Training / development source**, after the data licence and recording grouping are confirmed with the authors |
| **FSD50K** ([Zenodo 4060432](https://doi.org/10.5281/zenodo.4060432)) | 154 clips (dev 110, eval 44) | 999 | **~80 uploaders for Truck**, ~450 for Car | Weak clip-level multi-label (Truck & Car co-occur in 15 clips) | Per clip: CC0 / BY / BY-NC / Sampling+ | Never used. Freesound, not AudioSet, but overlap is unknown. Mixed, uncontrolled content | **Many-context ranking test** (car vs truck), grouped by uploader |
| **SONYC-UST v2** ([Zenodo 3966543](https://doi.org/10.5281/zenodo.3966543); [paper](https://arxiv.org/abs/2009.05188)) | 406 verified "large-sounding engine" clips (~1,574 by crowd majority) | 248 verified "medium-sounding engine" | **48 of 56 NYC sensors** in the verified positives | Weak 10 s presence tags; crowd agreement α = 0.36; verified for validate/test | CC BY 4.0 | Never used. A **proxy** task: large engine includes buses, trucks and machinery | **Multi-sensor proxy test or weak training data** |
| TUT Sound Events 2017 ([Zenodo 814831](https://doi.org/10.5281/zenodo.814831)) | 61 "large vehicle" events (buses + trucks) in 15 of 24 recordings | 304 "car" events | 24 dev + 8 eval street recordings (Finland) | Strong onset/offset labels (verified v2) | Non-commercial (TUT) | Never used. Small | Low priority. Possibly a strongly labelled event-timing check |
| MAVD-traffic ([Zenodo 3338727](https://doi.org/10.5281/zenodo.3338727)) | **25 truck events** total | ~600 car | 47 annotated recordings (Montevideo) | Strong events with vehicle-part ontology | see record | **Exposed through the P02 test material** (R0); mostly cars and buses | **Excluded** for trucks |
| Sound of Traffic / DCASE 2024 T10 | 228 single-cv minutes | 1,361 | 6 sites (about 4 informative) | Per-minute counts | CC BY-NC-SA 4.0 | Same dataset ([paper](https://www.isca-archive.org/interspeech_2024/ghaffarzadegan24_interspeech.pdf)); the public subset of 415 h | **Sealed** for later confirmation |
| AudioSet / DCASE 2017 Task 4 | thousands (Truck class) | thousands | many YouTube uploaders | Weak clip labels | YouTube terms | **BEATs and CNN14 were pretrained on AudioSet**, so any evaluation is contaminated | Training-only, if at all |

## Recommendations

1. **MVD is the only truck-rich, near-balanced, field-recorded candidate.** It has about
   1,000 clips per class from 4 locations. Before admitting it:
   - ask the authors for the data licence and per-recording session and date grouping;
   - pin a snapshot of the Drive files with SHA256 values;
   - run the overlap audit against IDMT, MELAUDIS and the AI4TEN banks.

   It would add four new groups in a different country and fleet. That is useful for
   training diversity, and as a cross-country development target.
2. **FSD50K gives many independent contexts but few trucks.** A car-vs-truck ranking test
   grouped by uploader (about 80 truck sources) would answer "does it generalize across
   recording contexts?" better than any six-group split, at the cost of messy, weak
   labels. Report it per dataset, never pooled.
3. **SONYC-UST v2 is the broadest sensor network** (48 sensors with verified
   large-engine positives). It is a proxy task. Use it as a secondary multi-site check,
   not as car/truck evidence.
4. **Exclude MAVD for trucks. Treat TUT 2017 as optional. Keep AudioSet out of
   evaluation.**
5. The planned **roadside collection** remains the only fully controlled option for
   independent sessions with camera-verified truck labels.

## Evidence

The label and metadata files examined are listed below. Each was fetched to an
untrusted scratch directory and parsed with isolated Python.
- MVD: paper text and repository README (naming scheme: recording number, microphone
  `O`/`A`/`M`/`H`, class).
- MAVD: `annotations_{train,validate,test}.zip`.
- FSD50K: `FSD50K.ground_truth.zip` and `FSD50K.metadata.zip` (uploader and licence per
  clip).
- TUT 2017: `TUT-sound-events-2017-development.meta.zip` (verified `meta/street`).
- SONYC-UST v2: `annotations.csv` (annotator 0 = verified).
- Sound of Traffic: the Interspeech 2024 paper text.
