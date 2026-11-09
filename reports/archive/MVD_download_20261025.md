# Original MVD acquisition

**Complete:** 4,229/4,229 original WAVs downloaded; zero failed transfers.
Total audio bytes: **559,682,776** (559.68 MB / approximately 533.76 MiB).
All **4,229 Google CRC32C checks passed**, and every local file was rehashed
against its acquisition SHA-256. Completion: **2026-10-26 02:14:02 UTC**
(2026-10-25 local time). Transfer plus final rehash took 660.03 seconds.

All files have the same inspected header profile: 22,050 Hz, mono PCM16,
66,150 frames / 3 seconds. There are **4,176 distinct SHA-256 values** among
the 4,229 filenames: 53 additional byte-identical copies. Every original is
retained; no deduplication, relabelling or session inference was performed.
The duplicates form 53 pairs; none crosses the provider's filename class labels.
This does not replace a later annotation or cross-corpus overlap audit.

2026-10-25. Download of the original Moving Vehicle Detection audio release,
requested by the user. Source: [Ashhad785/MVD](https://github.com/Ashhad785/MVD),
repository commit `bd797a0049b120e43d90ceb9da30ac3aee4e7979`.

Published [MVD Dataset folder](https://drive.google.com/drive/folders/1VhPQAoAWs4TFDOEXauBQIZrCCM4_0qt9)
contains the [original MVD audio folder](https://drive.google.com/drive/folders/1kB-rEWlXDwuIPmEKp_qszpTqWRy_Gtmh).
The separate augmented MVDA release was not selected. All four original classes
are preserved as acquisition material; this does not add motorcycles to the
current car/truck experiment.

## Storage and frozen inventory

Destination: `data:MVD/original_20261025_v1/`, resolved through `abvid.paths` to
`/Users/magi/workspace/abvid-data/MVD/original_20261025_v1/`.

The public embedded folder view listed **4,229 distinct filenames and Drive IDs**.
The complete page is preserved as `folder_inventory.html`, and `inventory.json`
records filenames, file IDs, source URLs, provider recording/microphone tokens
and filename labels. The connected Drive folder listing stopped at 1,000 entries
without a continuation token; the complete public folder view was therefore used
for bulk acquisition. The Drive connector also successfully fetched a probe file.

| Filename label | Files in frozen inventory |
|---|---:|
| Car (`C`) | 1,005 |
| Truck (`T`) | 1,077 |
| Motorbike (`M`) | 1,122 |
| No vehicle (`N`) | 1,025 |
| Total | 4,229 |

These are provider filename labels, not a new manual annotation audit. Five
recording tokens contain letters; they are retained exactly, without treating
the letters as proof of augmentation or independence. All `recording_session`,
physical vehicle and site identities remain `unknown`. Microphone tokens do not
by themselves establish independent recording sites or sessions.

## Download integrity

[`abvid.data.mvd_download`](../../src/abvid/data/mvd_download.py) validates the
canonical admission configuration, freezes the folder inventory and downloads
from Google's public file endpoint with eight bounded workers. It does not
execute remote code or download the separate vulnerability dataset with the
same MVD abbreviation.

Each successful file retains:

- Original filename and bytes; existing files are never replaced.
- HTTP byte-count checks, complete RIFF/WAVE length check, and WAV header
  metadata without decoding audio samples.
- Verification of Google's `x-goog-hash` CRC32C and MD5 when supplied, plus
  a locally computed SHA-256.
- Provider URL/file ID, checksum headers, acquisition time and attempt record.

The CRC32C implementation passed the `123456789` known-answer check and agreed
with Google's checksum for the probe. Invalid HTML and truncated RIFF responses
were rejected in local checks. The final downloader rehashes every published WAV
and compares the file inventory to successful transfers. Complete per-file
records and failures, if any, are retained in the external acquisition directory.

These checks establish transport/local-file integrity. Google checksums are not
an independently published author release manifest. No acoustic features,
listening review, classification, fitting or cross-corpus overlap computation
is part of this download.

## Admission status

**Download only; pending dataset admission.** A dataset-specific audio licence,
recording/session mapping, annotation audit and overlap checks remain unresolved.
The repository's MIT licence is retained as repository provenance and is not
silently applied to the separately hosted WAV files. No split, training role or
confirmation role is assigned, and existing experiments remain unchanged.

The external directory preserves the download plan, implementation snapshot,
frozen source inventory, transfer log and timestamped checksum manifest/receipt.
The final receipt determines whether acquisition is complete; download completion
does not change the admission status above.

| Final evidence | SHA256 |
|---|---|
| [Receipt](../../../abvid-data/MVD/original_20261025_v1/receipt_20261026T021402194353Z.json) | `87c799e1adb123e579b4dc3008d574f102e088f7af88bf09df42b709c40364e2` |
| [Per-file checksum manifest](../../../abvid-data/MVD/original_20261025_v1/files_20261026T021402194353Z.jsonl) | `ed4b0b58772b5ff3791274acc12c2026f0cf207bc1f90ce8351f59f6f2b1bc1a` |
| Frozen filename/Drive-ID inventory | `cb4e04dd558e313b302b4e066921f6cd4d1259dd159c80084d92dbc9d403d0d2` |

`repository_snapshot/` also preserves the pinned repository README and MIT
licence text with their own provenance hashes. No existing test, experiment
configuration, lock, model, source dataset or result was modified.

```sh
.venv-baseline/bin/python -B -m abvid.data.mvd_download \
  --output data:MVD/original_20261025_v1 \
  --inventory-html /private/tmp/abvid_mvd_folder_20261025.html \
  --workers 8
```

For a new acquisition use a new destination and a newly saved complete folder
view. `--resume` requires the same frozen inventory/implementation and verifies
previously downloaded files against their recorded hashes.
