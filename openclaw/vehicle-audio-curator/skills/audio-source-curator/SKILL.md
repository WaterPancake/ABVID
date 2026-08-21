---
name: audio-source-curator
description: Find, verify, and propose openly licensed vehicle and environmental audio without downloading until the operator approves a source ID.
---

# Audio source curation skill

## Evidence standard

A candidate is usable only when all of these are recorded:

- authoritative source page and stable direct download URL;
- creator/recordist and title;
- exact license or public-domain statement, plus license URL;
- source/session identity and a defensible vehicle class or background category;
- media format and approximate duration;
- attribution and share-alike obligations.

Do not infer a license from a search result, a platform's default behavior, or the
fact that a file can be downloaded. CC BY-NC, unclear Internet Archive rights, and
unlicensed sound-effect libraries are not approved by this skill.

## Proposal schema

Create `proposals/YYYY-MM-DD-<slug>.yaml` only after evidence review:

```yaml
source_id: candidate-...
status: review_required
kind: target  # target or background
provider: wikimedia_commons  # or internet_archive/direct when supported by catalog
source_page: https://...
direct_url: https://...
title: ...
creator: ...
license: ...
license_url: https://...
vehicle_class: tracked  # target only
vehicle_model: null
recording_session: ...
category: road traffic  # background only
format: ...
duration_seconds: null
attribution: ...
evidence:
  - https://...
reason_for_review: ...
```

Use `status: approved` only when the source meets the project's allowlist and a
human has already approved adding it to `configs/audio_sources.yaml`. A discovery
proposal should normally remain `review_required`.

## Existing collector

The project collector is the single permitted downloader. First preview:

```bash
cd "$VEHICLE_AUDIO_PROJECT_ROOT"
uv run python scripts/collect_audio.py \
  --catalog configs/audio_sources.yaml \
  --source-id <source-id> --dry-run --output .artifacts/collection_preview
```

Only after an explicit operator approval, run the same command without `--dry-run`.
The collector checks current provider metadata, normalizes media to WAV with ffmpeg,
creates a provenance JSON sidecar, and merges results into
`data/collection_manifest.jsonl`.
