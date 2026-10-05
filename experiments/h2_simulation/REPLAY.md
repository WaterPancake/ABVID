> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../README.md).

# H2 source × propagation experiment

**H2 is complete and independently verified.** Ground-angle indexing, reflected spreading and the subsequent numerical corrections passed the pre-target checks. All 480 dry sources and 9,600 observations passed exact replay; all 40 fixed heads were locked before scoring. This is the first controlled physical comparison, roadmap E4–E5. See the [full results](../../reports/H2_source_path_results.md) and [independent audit](../../reports/ARTIFACT_INDEX.md#artifact-70e853936c31).

On previously exposed MELAUDIS development data, BEATs source-minus-path D is −34.90 macro-F1 points (conditional 95% CI −43.56 to −29.17): the declared propagation intervention outperformed the declared source intervention. This contradicts the operational source-dominance hypothesis, not every possible source model. All cells retain zero-recall group/class cases; the best BEATs cell reaches 38.19% macro-F1 / 62.99% balanced accuracy. This is neither independent confirmation nor real-road calibration.

Current execution: [`source_path_20261004_r1`](../../reports/ARTIFACT_INDEX.md#artifact-570c30d67290), SHA256 `ef935e5f29940d9691ccac681c4661fb21b4fd366dd3a61cf5ede8c8ef461cd1`. See [implementation progress](notes/IMPLEMENTATION_PROGRESS.md), [pre-target implementation amendments](notes/IMPLEMENTATION_AMENDMENTS.md) and [execution commands](notes/RUN_H2.md). The design and execution locks retain their original gate states. The subsequent [corpus verification](../../reports/ARTIFACT_INDEX.md#artifact-56ef1ae53545), [feature lock](../../reports/ARTIFACT_INDEX.md#artifact-baada073f5db), [40-head lock](../../reports/ARTIFACT_INDEX.md#artifact-e2027d0769cb) and final audit establish completion without rewriting those historical locks.

- [Protocol](notes/PROTOCOL.md): source/path ranges, matched controls, dataset roles, models, uncertainty and falsification rules.
- [Canonical configuration](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/experiments/h2/config.json): four cells, five seeds, fixed C=1, frozen BEATs768 + MFCC26.
- [Backend audit](notes/BACKEND_AUDIT.md): source provenance and required renderer repairs.
- [Working backend and verification](../../vendor/pyroadacoustics/README.md): retained indexing history and completed pre-target propagation checks.
- [Frozen lock](../../reports/ARTIFACT_INDEX.md#artifact-9ec4cd50dcc9): SHA256 `fc73981fe6350cb3384079634677887b4a1ca7b78558fb2806c02515ee5ef8d0`.
- [Verification](../../reports/ARTIFACT_INDEX.md#artifact-ee1645f76b00): pairing/roles/population/determinism and historical-file checks.

Per seed/cell:190 cars +190 trucks for training, from19 templates/class ×10 paths;50/class from five disjoint validation templates. The completed experiment contains240 base templates,480 S-level dry sources,1,200 class-shared geometry slots,2,400 base events,9,600 rendered observations and40 fitted heads. These are verified generated quantities, not independent real vehicles. Test domain is synthetic → the exact8,066 previously exposed H1 MELAUDIS development excerpts. There is no untouched confirmation set.

The metadata verification passed; eight failure-mode tests cover leakage, class-specific geometry, source/path pairing, target population, tuning and false completion. All8,257 protected H1/diagnostic/transfer/encoder artifacts remained unchanged. Waveform, source-adapter, repaired-backend, environment-lock and physical-validation hashes are null on purpose. The configured upstream ground branch has an angle-table indexing defect and a reflected-spreading discrepancy; it cannot be used unchanged to infer a physical ranking.

The historical audit and frozen gate above describe the pre-repair state. The failed 11-tap static probe and first replay failures are retained. Subsequent source and renderer checks pass after the documented repairs: 480 source cases, 151 renderer cases and exact fresh-process replay of the 240 prescribed validation observations. Four-worker and sequential generation produce identical waveform hashes. The subsequent full bank also passed exact replay of all9,600 waveforms. This establishes the declared numerical checks, not real-road calibration. The result report includes all cells, seeds, representations, class/group failures and paired uncertainty; no target-selected revision was made.

## Inspect the manifests

All under [`frozen/source_path_20261004_v1/`](../../reports/ARTIFACT_INDEX.md#artifact-a28dc2ef95f1):

| File | Contents |
|---|---|
| `templates.jsonl`, `dry_source_plan.jsonl` | Actual source parameters, phases, noise seeds, disjoint parent groups, planned S0/S1 identities |
| `geometries.jsonl`, `events.jsonl` | Exact class-independent trajectories/crop indices and source joins |
| `render_plan.jsonl` | Four planned cell observations per paired event; audio hashes null |
| `fit_plan.jsonl` | Exact train/validation job IDs for40 future fixed heads |
| `target_manifest.jsonl` | Exact H1 target IDs/labels/groups/source hashes; evaluation-only role |
| `timing_slots.jsonl` |30 prespecified validation slots for later synthetic timing/checks |
| `execution_gate.json` | Unresolved implementation/physics gates; ready=false |
| `protected_artifacts.json`, `reference_files.json` | Prior evidence and code snapshots, with hashes |

Markdown snapshots in the frozen folder preserve their original text/link base (`experiments/h2`); use the canonical protocol/audit links above for navigation. Snapshot scripts are provenance copies; invoke the working metadata-only entry points below from the repository root.

## Verify without running an experiment

The metadata tools import NumPy only; they do not decode waveforms, import the renderer/encoder, fit or infer.

```bash
PYTHONPATH=experiments/h2 experiments/reproduction/.venv/bin/python -m unittest discover -s experiments/h2 -p test_design.py -v
experiments/reproduction/.venv/bin/python experiments/h2/verify_design.py --freeze experiments/h2/frozen/source_path_20261004_v1 --output /tmp/h2-verification.json
```

The output path must not exist. The original freeze was produced by:

```bash
experiments/reproduction/.venv/bin/python experiments/h2/freeze_design.py --output experiments/h2/frozen/source_path_20261004_v1
```

That directory is immutable; the command refuses to overwrite it. Any design revision needs a new ID/configuration and a new destination. The fixed upstream evidence is a read-only audit snapshot, with original notices; do not treat it as an installed, validated execution dependency.
