# Bounded P1 media-admission draft

2026-10-11 local date; **draft for a separate user decision, not authorization**.
P0 authorization explicitly excludes real-media collection and processing.
The [P0 protocol](PROTOCOL_DRAFT.md) and original 55-file freeze remain unchanged.
This stage prepares a reviewable admitted/rejected media population before a
separate Tier 0 pilot decision. It does not execute the six-model × eight-group
pilot or establish any model-recognition result.

## Proposed action and bounds

If explicitly authorized, implement a separately versioned admission path and
inspect rights/provenance for the ten named DVIDS assets below. Collect and
process an asset only after its per-asset restrictions and intended research
use pass a recorded admission decision. Unclear or conflicting rights remain
quarantined and require a specific resolution; a public-domain banner alone
does not clear them.

| Asset ID | Public metadata URL |
|---|---|
| 1754 | [Abrams b-roll](https://www.dvidshub.net/video/1754/abrams-tank-b-roll) |
| 572786 | [Abrams package](https://www.dvidshub.net/video/572786/taa-abrams-set-up-b-roll-package) |
| 855729 | [Finnish Leopard](https://www.dvidshub.net/video/855729/us-soldiers-explore-finnish-leopard-battle-tank) |
| 776583 | [Bradley b-roll](https://www.dvidshub.net/video/776583/b-roll-stallions-conduct-bradley-live-fire) |
| 386317 | [Stryker convoy](https://www.dvidshub.net/video/386317/m1126-stryker-combat-vehicle-convoy-ramstein-air-base) |
| 937004 | [Pinon Canyon package](https://www.dvidshub.net/video/937004/pinon-canyon-maneuver-site-b-roll-package) |
| 329602 | [M1151 training](https://www.dvidshub.net/video/329602/humvee-training-b-roll) |
| 607339 | [M109A6 firing](https://www.dvidshub.net/video/607339/arkansas-field-artillery-unit-fires-m109a6-paladins-during-operation-western-strike) |
| 967175 | [Army birthday parade](https://www.dvidshub.net/video/967175/army-250-parade-celebration-b-roll) |
| 966208 | [Staged vehicles](https://www.dvidshub.net/video/966208/us-army-vehicles-staged-west-potomac-park-waterfront-b-roll) |

Bounds are ten asset IDs; one selected media rendition per admitted asset; at
most 3 GB total downloaded media; at most 20 minutes per asset and 45 minutes
total decoded source duration. The current page durations sum to 42 minutes
11 seconds, but actual media must satisfy the bounds. Failed, unavailable or
rejected assets are retained as outcomes. No unlisted substitute is collected.
The restricted NATO Leopard asset 879491 is excluded from this draft.

Collection, original-media hashes, native audio extraction, blind review packets,
label-blind reuse checks, provenance grouping and reviewed interval import/export
are the proposed processing scope. Representation extraction, classifier fits,
CAST fits, contrastive learning, scientific retrieval/probe metrics and protected
source access remain outside this stage. All new media/results remain external,
resolved through `abvid.paths`; no existing raw data, protected roles, historical
manifests or frozen P0 files are changed. Human-only harness baseline changes
and merge approval are not part of this authorization.

## Required admission sequence

1. Preserve the source page, rights statement, date/unit/location conflicts,
   producer attribution, rendition metadata and retrieval failures. Record the
   intended use as local acoustic research without publication or
   redistribution of the media. Resolve restrictions before accessing media.
2. Enforce byte/duration bounds and hash the chosen original rendition. Retain
   its native audio rate/channels and extraction provenance. No quality label,
   engine identity or session independence is inferred from captions.
3. Use a new admission configuration/tool version. The current P0 configuration
   permits synthetic fixtures only and must continue to reject real media.
   Verify the new processing mechanics before using admitted files.
4. Build randomized caption-hidden audio/waveform/spectrogram review packets.
   A human reviews audible vehicles, foreground/background intervals,
   contaminants, motion and rejection reasons. Caption shot-log intervals are
   not used to preselect or accept audio. If no reviewer is available, stop at
   the packet handoff and report that admission is incomplete.
5. Audit reuse and retain all known or suspected shoot/population links. Session
   and individual IDs remain `unknown` unless evidence establishes them.
   Assets with unresolved independence cannot fill a verified-group quota or
   be divided across experimental roles. No train/development/test allocation
   is performed in this admission stage.
6. Import human annotations, retain every rejection and export accepted clips
   with native-frame provenance. Attach caption labels separately after blind
   trimming, with confidence and evidence. Mixed-model event material remains
   diagnostic unless a new protocol establishes valid model/interval identity.

## Outputs and stopping rule

Deliver a rights/provenance admission ledger, native-media/extraction hashes,
caption-hidden packets, imported human annotations where available, conservative
group/reuse audit, rejected/unavailable asset list and an admission report. The
report must state admitted foreground/background support, verified independent
groups, known physical-vehicle support and remaining unknowns, including zero
support where appropriate. Store them under a fresh external P1-admission run.

Stop after these ten assets, at a budget bound, or at a required rights/reviewer
handoff. This draft does not permit collecting until a positive gate appears.
The six-class identity definition, eight-group pilot coverage, exact frozen
representation/checkpoint, fit budgets and Tier 0 evaluation allocation must
be resolved in a separate pilot protocol before fitting or metric evaluation.
