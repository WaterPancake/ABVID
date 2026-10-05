# Repository refactor, 2026-10-05

The active checkout now follows the civilian acoustic-transfer research plan.
Reusable code is under `src/abvid`; CAST is explicitly `src/abvid/cast`.
Protocols/configuration, results interpretation and large local artifacts have
separate locations. No scientific experiment, training run or target-domain
model evaluation was launched as part of this restructuring.

## What changed

- Retired project-local OpenClaw and the earlier military benchmark code/data
  from the active checkout. Original tracked history remains available.
- Preserved the entire local working tree, including untracked research work,
  in `../abvid-artifacts/archive/pre-restructure-20261005/`.
- Added a source/protocol snapshot to Git before the refactor:
  [77daa1b3](https://github.com/WaterPancake/ABVID/commit/77daa1b339906d35b6fa0a59a166ecf2eac01174).
  This retains original R0/H1/H2/CAST runners without maintaining duplicate active
  implementations. Large artifacts and downloaded data are local only.
- Moved actual shared admission, grouping, preprocessing, MFCC/BEATs interfaces,
  linear head, metrics/bootstrap, source synthesis, propagation and CAST numerical
  functions into importable packages. The [symbol map](refactor_symbol_map.json)
  identifies original paths, hashes and extracted definitions.
- Organized experiment instructions under `admission`, `r0_reproduction`,
  `h1_baselines`, `h2_simulation` and `cast_coverage`.
- Added one validated JSON wrapper format, storage-root configuration and a small
  `abvid` CLI. Wrapper defaults run regression/help entry points; full historical
  study sequences retain their original scripts and environments in the archive.
- Replaced stale benchmark-first README/agent instructions. Preserved the dated
  scientific roadmap/protocol text while updating navigation.
- Retained the repaired pyroadacoustics backend, original GPL license and repair
  revisions. There is no new propagation correction in this refactor.

## Numerical and packaging validation

| Check | Result |
|---|---|
| CAST and canonical-config regression suite | 215 passed |
| Admission, baseline, BEATs interface, simulation and config suite | 39 passed |
| Extracted numerical definitions | 160 unchanged after ignoring import relocation |
| CAST rendered fixtures | 18 waveform pairs exactly equal |
| v15 population generation | 48 draws and population statistics exactly equal |
| H1 preprocessing and MFCC | Exact equality on an artificial stereo fixture |
| H2 source factors | Both source waveforms and metadata exactly equal |
| H2 path components | Four component arrays exactly equal |
| H2 metric dictionary | Exact equality on artificial predictions |
| Archived source/config/document hashes | 704 unchanged |
| Admitted audio references | 22,861 resolve; zero missing |
| Original CAST source allowlist | 50 files pass original guard and hash checks |
| Archive Git state | Original commit and binary tracked diff preserved |
| Installation | Both uv profiles install; sdist/wheel build; wheel imports outside checkout |
| Documentation navigation | 559 local file links resolve |

The two pytest totals include the shared configuration suite in each environment.
Exact parity here checks fixtures and extracted algorithms, not an independent
repeat of every historical study. Full original results/locks were preserved and
were not regenerated or relabelled. Evidence: [CAST tests](refactor_cast_tests.xml),
[baseline tests](refactor_baseline_tests.xml), [CAST parity](refactor_cast_parity.json),
[baseline parity](refactor_baseline_parity.json), [storage check](refactor_storage_check.json).

## Storage and remaining boundaries

The checkout now contains about 4–5 MiB of project files, excluding Git history,
virtual environments and build caches, compared with approximately 78 GiB before
reorganization. The two installed environments together occupy about 1.4 GiB.
Large data were relocated on the same disk, so their storage was **not reclaimed**.
Nothing in the raw corpora or completed results was purged.

The original archive `dataset/` remains a real directory because frozen CAST
source validation rejects symlinked source roots. `../abvid-data/civilian` is a
convenient view of it. The admitted MELAUDIS extraction and model directories
have compatibility links from their historical paths. Never weaken historical
provenance guards to accommodate a storage layout.

The [artifact index](ARTIFACT_INDEX.md) locates full local evidence; small code and
protocol links resolve to the source snapshot in Git. The literature matrix is
byte-for-byte unchanged. The [structure guide](../docs/STRUCTURE.md) describes
module ownership and archive relocation constraints.

CAST descriptor coverage remains source-domain development evidence. The source
versus propagation hypothesis and any new CAST-to-real classifier study retain
their existing scientific status; this refactor supplies no new transfer result.
