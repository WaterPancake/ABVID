# P1 admission authorization and current limits

The user authorized more real-media collection after reviewing the bounded
P1 admission request: “Get more data,” retaining the per-asset time limit.
The initial 20 GB allocation was then replaced by “lets start with 5 GB.”
The [current contract](p1_admission_5gb.json) therefore uses **5,000,000,000 bytes
total across the new data and artifact roots**, with **1,200 seconds per asset**.
It expands the original ten-asset discovery list; the old 45-minute aggregate
draft limit no longer defines this expanded batch. Storage is a ceiling, not a
requirement to fill the disk. The first collection target is 64 playable assets
across the six candidate caption classes, subject to the budget and availability.

Collection and preparation are authorized. Rights/provenance review, original
rendition retention, native audio preparation, caption-hidden human review and
label-blind reuse auditing are within scope. Fitting, representation extraction,
protected-source evaluation and media publication remain outside it.

All original P0 files and their 55-file freeze remain unchanged. A separately
versioned P1 implementation and external run are used; real files are never
passed through P0's synthetic-only admission guards. Existing archival source
catalogs are read as metadata solely to exclude known historical assets/filenames.
Protected media are not opened or used for waveform comparisons.

Metadata diversity across dates, units and sites helps candidate selection but
does not establish independent sessions or physical vehicles. All collected
material is pending blind review and conservative provenance admission. Captions
are retained privately and do not preselect accepted audio intervals.
