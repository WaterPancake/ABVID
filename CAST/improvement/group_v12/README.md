# CAST group-effect prior, v12

This additive experiment tests two fixed ways to combine within-group
parameter variation with shifts between recording groups. The class-agnostic
v11 renderer and all 380 fitted parents remain unchanged. The
[protocol](PROTOCOL.md) defines the three candidates, matched controls,
complete ancestry and unchanged acceptance criteria.

The completed source comparison selects `group_predictive`: 84.745% car /
87.134% truck coverage, and 9.564% / 16.014% gains versus matched marginals.
Its class-balanced W1 improves 1.23%, with worse car distance and better truck
distance. All 241 tests and the complete 22,500-record/450-score source audit
pass; all 150 v11 reference scores reproduce exactly. The new outer comparison
is fully audited: **79.359% car / 78.997% truck coverage**, with both control
margins and all spreads passing. Coverage still fails; the goal is not achieved.
All 1,500 waveforms and 570 raw held descriptors replay exactly. See the
[outer scores](evaluations/group_v12_outer_development/scores.json),
[verification](evaluations/group_v12_outer_development/verification.json),
[audio review](diagnostics/group_v12_outer_review/index.html) and
[full development report](../../../reports/CAST_improvement.md).

Source evidence: [summary](evaluations/spectrum16_group_v12_full_source/summary.json),
[replay audit](evaluations/spectrum16_group_v12_full_source/audit_verification.json),
[all frozen group statistics](evaluations/spectrum16_group_v12_full_source/prior_tables.json).

Reproduce the source comparison and, conditionally, the selected new outer
comparison from the ABVID root:

```sh
bash CAST/improvement/group_v12/reproduce_outer.sh fresh_v12_source fresh_v12_outer
```

Use fresh IDs. The command verifies the existing pinned v11 source bank and
all upstream artifacts, runs the full numerical/provenance suite, freezes
fold-specific statistics, generates 22,500 source examples and replays them
exactly. It writes plots and a fixed-example audio review. Only a newly
selected passing candidate proceeds to 1,500 outer examples, followed by
complete generated-audio and 570-record raw held-descriptor verification.
Exit 3 means the scientific progression or outer criteria were not met;
all evidence remains saved. A retained v11 reference is not reevaluated on
the outer group.

For source work only:

```sh
bash CAST/improvement/group_v12/reproduce.sh fresh_v12_source
```

The existing `.venv` and immutable v11 source evaluation at
`../resolution_v11/evaluations/smooth16_temporal41_v11_full_source` are
prerequisites. No package installation, dataset modification or download
occurs. CPU determinism uses one Torch thread. Source fitting was already
completed and audited; this version changes only the sampling distribution.

Stage commands for inspecting the completed artifacts are:

```sh
sh CAST/improvement/group_v12/run.sh source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/group_v12/run.sh audit-source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/group_v12/outer.sh run --source-eval-id FRESH_SOURCE_ID --eval-id FRESH_OUTER_ID
sh CAST/improvement/group_v12/outer.sh verify --eval-id FRESH_OUTER_ID
```

The outer group is already exposed development data. Descriptor coverage
and W1 gains are not classifier accuracy or independent real-world
generalization. Source groups are conservative site/date proxies, not
known independent vehicles. Virtual children add no independent parents.
The independent-marginal control depends on the 69-coordinate representation.
All local audio derivatives retain their recorded research-use restrictions.
