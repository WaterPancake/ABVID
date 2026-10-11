# CAST-4A grouped coverage design freeze

Completed 2026-10-07. **Protocol and metadata preparation passed; the scientific experiment is unrun.** This pass read pinned metadata and source code only. It did not decode audio, open fitted banks or feature arrays, fit models, generate observations, or compute research metrics.

The [protocol](../experiments/cast_coverage/grouped_coverage/PROTOCOL.md), [machine-readable design](../experiments/cast_coverage/grouped_coverage/design.json), [numerical recipe](../experiments/cast_coverage/grouped_coverage/numerical_contract.json) and [canonical entry](../configs/experiments/cast_grouped_coverage.json) now fix the comparison. [CAST_ROADMAP.md](../CAST_ROADMAP.md#cast-4a-broaden-grouped-coverage-evaluation) records this preparation checkpoint.

## Frozen comparison

- Six primary folds retain the literal H1 seed-42 training and held lists. Each has 190 cars and 190 trucks from the five remaining conservative groups.
- Three separately reported sensitivity folds hold out whole sites, including both dates. Every fold supports the same 190/class budget. The tightest training pool contains 213 trucks; there is no need to reduce the budget or borrow held examples.
- Each fixed width, v14 T=1 and v15 T=1.1, gets its own joint, prototype and independent-marginal arm: six arms total. Five fixed synthesis seeds and 50 observations/class/seed/arm give 18,000 planned primary and 9,000 planned site observations.
- The acceptance rule remains mean coverage at least 80%, at least 2.5% W1 reduction against both same-width controls, and every family mean spread in [0.5,2]. Every primary fold/class must pass for the across-groups claim. There is no width search, held-data tuning or complete-case rescue.
- Paired equal-fold contrasts and 10,000 whole-fold bootstrap draws are fixed in advance. Seed variation is separate. These are descriptive intervals given overlapping training sets and only three sites.

The archived v15 protocol has an obsolete nested 5% margin. The effective outer evaluator and stated rule used 2.5%; the new contract identifies the obsolete field as inactive and freezes 2.5% explicitly. Historical files were not edited.

## Checked roles and workload

The source-only export preserves all 4,413 admitted IDMT rows (3,902 cars, 511 trucks) exactly, including unknown fields and the two admitted near-full-scale flags. The checks cover file and decoded hashes, duplicate groups, recording/time-axis and event proxies, groups, and every known session/media/vehicle link. Neither selected training rows nor the complete eligible training pool overlaps the held role on a known ancestor. Audio hashes were carried from the pinned admission record, not recomputed from audio.

| Partition | Held role | Held car/truck | Selected training car/truck | Candidate cached parents within training |
|---|---|---:|---:|---:|
| group_0 | `connected_4001f06f57cfeee7` | 491/79 | 190/190 | 380 |
| group_1 | `connected_654b5df57ec64387` | 666/110 | 190/190 | 85 |
| group_2 | `connected_667018ac7661cccb` | 254/14 | 190/190 | 73 |
| group_3 | `connected_72766641e1daf436` | 1365/152 | 190/190 | 91 |
| group_4 | `connected_b00d69ddbb66d0fe` | 302/10 | 190/190 | 84 |
| group_5 | `connected_c6148cf0cafb5753` | 824/146 | 190/190 | 84 |
| site_0 | `Fraunhofer-IDMT` | 556/24 | 190/190 | 66 |
| site_1 | `Langewiesener-Strasse` | 2189/298 | 190/190 | 86 |
| site_2 | `Schleusinger-Allee` | 1157/189 | 190/190 | 103 |

The primary selections use 1,462 distinct file IDs (997 car, 465 truck). Site selections use 946 (534 car, 412 truck). Together, 3,420 training-role assignments use **1,864 distinct recordings**, not 3,420 independent observations.

The named historical v11 metadata identifies 380 possible cached parent fits. All 380 enter at least one new training role; 1,484 selected IDs have no candidate in that inventory. **No saved fit has been approved for reuse by this pass.** Each candidate still needs its full same-parent input/component/settings/environment ancestry checked. Every fold must recompute population centers, effects, virtual children, prototypes, width centers and descriptor scales using only its allowed training IDs.

All 4,413 rows retain unknown original sessions, original media IDs, physical vehicle identities and measured speeds. Eighteen candidate recording/time-axis tokens are not verified sessions. The six conservative groups cover three sites. Site and speed-limit token are associated; 30/50/70 Kmh are not measured vehicle speeds. All groups have prior development exposure, so the new design cannot make them fresh confirmation data.

## Frozen artifacts and verification

External output: `artifact:cast/grouped_coverage/cast4a_design_20261007_v1`.

| Artifact | Purpose | SHA256 |
|---|---|---|
| `lock.json` | Commit, dirty status, environment and hash lock | `392222ec36f88bf9012655a788005d2ae72422336c8ed786e9ec04f5978b2de4` |
| `sources.jsonl` | Exact source-only admission rows | `c73bef9c1200f29880d34769fe73f6b402ac632db97d458b802d390acb73f02c` |
| `folds.json` | All nine ordered training and held ID lists | `d24e46b1303733170d25b5e09aa1963a479110ffd2566ae5bf6353b2e24b1376` |
| `planned_generation.jsonl` | 27,000 planned identities and paired seeds; no audio | `2cba6c3cb8346cfd4d3c7a3d4d401a59b3d3208b1e05cf0a8be94024ef458ab8` |

Payload identity: `ba520e38f2f7d9e804b7b10db2ca034eabb7b7c8bda1b6a8d1bfe0d5f8288043`. The bundle contains 58 payload files plus its lock, approximately 30.4 MB, outside the checkout. It includes candidate-reuse metadata, input copies and the relevant source snapshot. The snapshot records commit `25cb13a151a4ecf458e3da1d8ba7651351e52c42` plus the dirty working-tree state; it does not claim these additions are committed or pushed.

Validation completed:

- All 122 input/implementation/numerical-reference hash checks passed, plus the canonical contract and entry-point hash checks.
- All 88 focused protocol and metadata tests passed, including alias leakage, exact group/site selections, null preservation, paired schedules, forbidden file reads, changed dependencies, altered module paths, incomplete artifacts and tampering even when local hashes are rewritten.
- Fresh external freeze and a separate verification invocation reconstructed all 58 payloads byte-for-byte from the pinned inputs. Metadata regeneration leaves the scientific metric field null.
- The five historical canonical configurations remain byte-identical. No frozen archive runner, lock, environment or result was edited.

Commands:

```sh
.venv/bin/python -m abvid.cli validate configs/experiments/cast_grouped_coverage.json --check-files
.venv/bin/python -m abvid.cli prepare configs/experiments/cast_grouped_coverage.json --execute
.venv/bin/python -m abvid.cast.grouped_design verify --config repo:configs/experiments/cast_grouped_coverage.json
.venv/bin/python -m pytest -q tests/test_protocol.py tests/cast/test_grouped_design.py
```

Without `--execute`, `prepare` only prints the metadata command. An existing output is verified rather than overwritten. `replay` rejects the metadata-only configuration. The new metadata path uses the current repository environment and never launches an archived scientific runner.

## Next prerequisite and limits

The next implementation step is a packaged smooth8 fitting/role adapter with focused synthetic and historical parity checks. The current packaged `abvid.cast.fit.fit` uses the original rectangular renderer and cannot silently substitute for the historical smooth8 fitter. Historical wrappers also hard-code the old 380-parent fold-0 role. They must remain unchanged; the new adapter must enforce each frozen group/site allowlist. This preparation does not implement that adapter or execute the comparison.

After that prerequisite and an explicitly scoped scientific run, audit candidate reuse, fit the remaining selected parents, freeze every training-only bank, generate its complete declared schedule, then score held-source descriptors. The numerical recipe records the CPU environment and inherited optimization settings. Historical smooth8 fitting took 3,057.92 seconds for 380 parents on four CPU workers; the exact hardware model is unknown. A simple linear estimate is about 3.3 hours of base fitting for 1,484 cache misses, or 4.2 hours for all 1,864 parents. These are planning estimates only, excluding calibration, cache auditing, generation, scoring and replay; total runtime remains unknown.

CAST-4A can establish how broadly the fixed observation population meets its source-descriptor criteria. It does not test classifier improvement, cross-dataset transfer, or source-model dominance over propagation mismatch. CAST-4B and CAST-5 retain their separate protocols and evidence requirements.
