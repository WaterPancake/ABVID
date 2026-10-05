# CAST balanced finite sampling, v13

This experiment changes only how the fixed 50 examples are allocated across
the unchanged v12 empirical population. It balances direct source groups,
parent use inside each group and virtual children. The marginal control gets
the same treatment independently per coordinate; the prototype and all
waveform streams remain unchanged. See the [frozen protocol](PROTOCOL.md).

The completed source comparison retains the exact v12 reference. Balanced
allocation passes the source thresholds but worsens joint W1 in both classes:
0.567073 car / 0.541720 truck versus 0.561161 / 0.524022. Class-balanced
W1 worsens 2.18%; coverage is 84.619% / 87.805%. All 259 tests and complete
15,000-record/300-score replay pass, with all 150 v12 reference scores exact.
No new outer comparison is permitted for this result. See the
[source scores](evaluations/group_balanced_v13_full_source/summary.json),
[replay audit](evaluations/group_balanced_v13_full_source/audit_verification.json)
and [progression decision](evaluations/group_balanced_v13_full_source/progression_decision.json).
The best verified outer coverage therefore remains v12's 79.359% / 78.997%.

Reproduce the source comparison and conditional new outer check from the
ABVID root with fresh IDs:

```sh
bash CAST/improvement/balanced_v13/reproduce_outer.sh fresh_v13_source fresh_v13_outer
```

For source work only:

```sh
bash CAST/improvement/balanced_v13/reproduce.sh fresh_v13_source
```

The existing pinned `.venv`, original local inputs and audited v12 source
artifacts are prerequisites. No installation, download or dataset mutation
occurs. The workflow verifies all source ancestors, runs the full numerical
and provenance suite, freezes every allocation plan, generates 15,000 source
examples and replays all 300 scores. It also checks all 150 v12 reference
scores exactly and creates source plots and fixed audio examples.

Only a new source-selected candidate that passes every criterion and full
replay proceeds to outer generation. All 1,500 outer outputs precede held
descriptor access, and every output plus all 570 held raw descriptors is
verified. A retained v12 reference is not reevaluated on the outer group.
Exit 3 means scientific progression or the outer criteria failed; all
artifacts are preserved. Existing paths cannot be overwritten.

Individual stages:

```sh
sh CAST/improvement/balanced_v13/run.sh source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/balanced_v13/run.sh audit-source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/balanced_v13/outer.sh run --source-eval-id FRESH_SOURCE_ID --eval-id FRESH_OUTER_ID
sh CAST/improvement/balanced_v13/outer.sh verify --eval-id FRESH_OUTER_ID
```

This is a finite-sampling hypothesis. It does not expand the data, change
parameter bounds, widen coverage intervals, reduce control budgets or turn
the previously exposed outer group into independent confirmation. Descriptor
coverage is not classification accuracy or real-world vehicle generalization.
All parameters remain effective observation controls with complete original
recording/group ancestry and research-use audio restrictions.

Completed review artifacts: [source comparison](diagnostics/group_balanced_v13_figures/source_comparison.png),
[group/family diagnostics](diagnostics/group_balanced_v13_figures/source_families_groups.png),
and [ten fixed audio examples](diagnostics/group_balanced_v13_audio/index.html).
All 73 audio-review links and 71 artifact hashes were checked. The complete
source replay took 442.26 s; source generation/scoring took 185.02 s after
83.26 s of upstream verification. No raw outer input was opened by v13.
