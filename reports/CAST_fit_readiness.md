# CAST-4A fitter and saved-fit readiness

Completed 2026-10-07. The historical smooth8 fitter is packaged, exact synthetic parity passed, and **380/380 saved calibrated parameter records are eligible for reuse within their declared training roles**. This is an implementation and provenance result. No new real recordings were fitted, no study population was generated, and no coverage or classification result was computed.

The [grouped design freeze](CAST_grouped_coverage_design.md) remains unchanged. This step adds the [audit contract](../experiments/cast_coverage/grouped_coverage/readiness/audit.json), [chain contract](../experiments/cast_coverage/grouped_coverage/readiness/chain_contract.json) and [replay instructions](../experiments/cast_coverage/grouped_coverage/readiness/README.md), without editing historical runners, locks, environments or results.

## Individual checks

| Check | Finding | Scope and limit |
|---|---|---|
| Packaged fitter | `abvid.cast.smooth_fit.fit` has the same function-body AST as the archived smooth8 fitter. | Uses the smooth renderer and preserves four starts, 300 steps, fit/check streams, alternatives and the fit-loss winner rule. It does not open audio or select training roles. |
| Exact numerical parity | Three synthetic cases passed in isolated original-layout and packaged processes: a short mixture, a short fixed-harmonic case, and the full 300-step mixture. | Full-budget comparison matched all 301 history states across four starts, parameters, winner, seeds, target, checking waveform and component hashes. Only elapsed time was excluded. No real observations were used. |
| Fold-role adapter | Exact 380-parent banks and 190-parent class populations enforced across all nine frozen partitions. | Rejects unselected parents, held ancestry, incomplete class calibration populations, cross-parent fit ancestry and old population statistics. Site roles exclude both date groups. Unknown metadata remains unchanged. |
| Source identity and preprocessing provenance | All 380 source files matched recorded SHA256, byte count and native header shape/rate. Parent metadata matched the frozen source rows exactly. | Original samples were not decoded again. The pinned historical preprocessing replay and original saved observation hashes supply preprocessing evidence. |
| Smooth8 optimizer records | Every saved parent retained all four starts and 301 loss states, deterministic namespaces, first-minimum best states and correct winner/baseline records. | Fresh state/content checks; no new real-data optimization replay. |
| Calibration ancestry | Same-parent chain verified through v7 mixture, v8 envelope, v9 spectrum, v10 temporal and v11 spectral-resolution calibration. | Checks immutable settings, environments, upstream locks, observations, protected controls, bank/row hashes and 151-state v9/v11 histories. Recorded numerical replay certificates remain historical evidence. |
| Original saved components | Original smooth8 `hashes.json` files are anchored by per-parent historical verification records. | Original fit alternatives, components and observations passed their anchored closures. |
| Later reconstructed waveforms | V9/v11 saved waveforms have self-consistent current closures but lack independently anchored historical byte hashes. | **Excluded from reuse.** Their parameter banks and full calibration records are anchored separately, so this does not prevent parameter-record reuse. |
| Failure and path controls | Shared or per-parent failures prevent eligibility; exact closure filename sets, direct paths and symlink rejection enforced before payload reads. | No silent replacement, missing-parent omission or relaxed role rule. No historical population-generating audit command was invoked. |
| Immutable replay | A second audit reconstructed the frozen output byte-for-byte. | 18 payload files plus a lock; output cannot be silently overwritten. |

The audit checked 192 shared files, 8,740 per-parent file-hash entries and 1,900 canonical stage-row hashes. There were zero shared or per-parent integrity/recipe failures in the required CAST environment. The source observations audited are the original 380 fold-0 training parents; some are held in other proposed folds. Their cached values are usable only in the folds where they are selected for training. No held-evaluation descriptor asset or target-domain feature cache was opened or scored.

## Preserved scientific limitations

Reuse eligibility is not physical identifiability. Existing smooth8 flags remain: 345 records with disagreeing near-equivalent starts, nine harmonic-order ambiguity flags, 19 noise-dominant spacing flags, six unresolved-salience flags and five large temporal residual flags; categories overlap. Twenty-one records have boundary hits. Later calibration flags are retained too: v11 records include 67 noise-dominant spacing flags and 14 mixture-boundary flags. No thresholds were relaxed to admit these records, and none were discarded because their acoustic fit is ambiguous.

Every fold still rebuilds its own group/class effects, children, prototypes, width centers and descriptor scales. Even group 0 cannot reuse old population statistics through this adapter. The additional 27,000 planned observations have not been generated, and neither the source-versus-propagation hypothesis nor classifier utility has been tested by this audit.

## Reuse by frozen role

| Partition | Eligible cached parents | Parents still requiring fits or another audited cache |
|---|---:|---:|
| group_0 | 380 | 0 |
| group_1 | 85 | 295 |
| group_2 | 73 | 307 |
| group_3 | 91 | 289 |
| group_4 | 84 | 296 |
| group_5 | 84 | 296 |
| site_0 | 66 | 314 |
| site_1 | 86 | 294 |
| site_2 | 103 | 277 |

Across the combined selections, 380 of 1,864 distinct parent recordings have an eligible named cache entry; **1,484 distinct parents still need fits**. The table counts fold-role assignments and must not be summed as independent recordings.

## Validation and environment

**262 combined checks passed**, with zero failures or skips, plus the three separately completed archived-versus-packaged parity comparisons. The combined suite covers protocol/metadata replay, role boundaries, fitter smoke checks, saved-chain negative cases, audit tampering, renderer numerics and validation-evidence integrity. The full 300-step fit result digest is `dd5bdb6958bcec8b3be2dfcce3a4ee65beca56b98d650af28eb61f6e25b0df85`; checking waveform digest is `88b2bc3e4e75179d1090ddc5dbbd19c19465a7b83a197a8233829b1b9d8a0b84`.

The shared `.venv` changed to the incompatible baseline profile during work. The first integration attempt correctly failed the frozen NumPy-version check; its log/XML were retained. A transient lower-level audit in that profile also failed exact numerical lift checks and was not used as eligibility evidence. No baseline environment was overwritten. Final validation and audit used a separate environment at `artifact:environments/cast-readiness-20261007`, created with `uv sync --frozen --extra cast`, preserving the existing project lock:

- Python 3.11.13; NumPy 2.4.6; SciPy 1.17.1; Torch 2.13.0; SoundFile 0.14.0.
- Synthetic fitter parity used one Torch thread, one inter-op thread and deterministic algorithms in separate processes. The historical implementation used its preserved archive interpreter.
- Full synthetic fit timings were 18.374 seconds historical and 17.033 seconds packaged; these are validation timings, not a performance claim or real-data runtime forecast.

The audit entry point validates the existing canonical grouped experiment configuration, reconstructs its metadata freeze, checks current implementation/evidence hashes, and requires the frozen numerical environment before reading candidate artifacts.

## Evidence and next work

Audit output: `artifact:cast/grouped_coverage/cast4a_readiness_20261007_v1`.

| Artifact | SHA256 |
|---|---|
| `lock.json` | `2dca3f510bbba025445803798571d0fa88ceb82c737e1775e19dbf70fddd8514` |
| `summary.json` | `f5022f82cf9efc0d112b0987ed883b53df63ef8633a60d78d617f66488f0cfed` |
| `parents.jsonl` | `ab403e60e4e929ecac76c61a222427bec2acc5d67c808fc36bf97a58d29fb3db` |
| `fold_reuse.json` | `9ec621e14962a0b384728a9ff3cbecaf87441842a5100efe76ea8364b4c8a943` |

Payload identity: `bcbdca5de863a1bd0166271ffe5f76c9cdc7615d182d30be8f059eb6de01f59e`. The audit took 13.96 seconds. It records commit `25cb13a151a4ecf458e3da1d8ba7651351e52c42` plus dirty status and exact source/test/evidence snapshots. This report does not claim these additions are committed or pushed.

Per-parent records retain every checked hash, failure list, stage flag and reuse exclusion. `fold_reuse.json` lists exact eligible and missing IDs for each role. Parity evidence and focused-test logs are stored alongside the audit in the external artifact root; the [README](../experiments/cast_coverage/grouped_coverage/readiness/README.md) gives version-pinned setup and replay commands.

The requested fitter/parity and saved-fit admission work is complete. The next implementation is the thin grouped execution runner that connects the existing calibration stages and role guards to the frozen schedule. Its later scientific run must fit the 1,484 missing selected parents, rebuild each fold's population statistics, generate the declared arms and evaluate coverage with all failures retained. No new method, data budget, width, class rule or acceptance criterion is introduced here.
