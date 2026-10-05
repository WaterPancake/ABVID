# Sixteen spectral controls with temporal41

This source-only revision addresses the remaining source spectral residuals.
It reuses the exact 380 audited v10 parents and fits sixteen spectral controls
and mixture with the unchanged expected-spectrum objective. See
[PROTOCOL.md](PROTOCOL.md) for the fixed method and all access restrictions.

From the ABVID root, with the pinned existing environment and frozen source
parent artifacts present:

```sh
bash CAST/improvement/resolution_v11/reproduce.sh NEW_ID
```

This runs tests, freezes and calibrates 380 parents, replays all optimizations
and 760 reconstruction WAVs, compares source folds with the exact v10 reference,
replays 15,000 samples and 300 scores, then creates plots and systematic audio
review. It returns code 3 for a scientific failure or if the unchanged reference
is retained. It never opens outer inputs. Use a fresh ID; no overwrite occurs.

Individual stages allow inspection between completed artifacts:

```sh
bash CAST/improvement/resolution_v11/run.sh calibrate --run-id NEW_ID --workers 4
bash CAST/improvement/resolution_v11/run.sh audit-bank --run-id NEW_ID --workers 4
bash CAST/improvement/resolution_v11/run.sh source --run-id NEW_ID --eval-id NEW_SOURCE_ID
bash CAST/improvement/resolution_v11/run.sh audit-source --eval-id NEW_SOURCE_ID
```

Only a newly selected spectrum16 candidate passing the original source criteria
and exact audit may run the separate conditional outer development comparison:

```sh
bash CAST/improvement/resolution_v11/outer.sh run --source-eval-id NEW_SOURCE_ID --eval-id NEW_OUTER_ID
bash CAST/improvement/resolution_v11/outer.sh verify --eval-id NEW_OUTER_ID
```

The outer group has already been exposed by H1, CAST v0 and temporal41. A future
result is continued development evidence. The scalar marginal control has 69
coordinates; its comparison depends on this representation and does not establish
universal generator superiority or classification performance. No environment,
dataset, split, baseline artifact, source weight or scoring criterion changes.

The complete conditional workflow is:

```sh
bash CAST/improvement/resolution_v11/reproduce_outer.sh NEW_RUN_ID NEW_OUTER_ID
```

It runs calibration, source comparison, all audits, figures/audio review and
conditional outer evaluation. A verified scientific failure exits 3. The
current experiment executed these stages individually; both wrappers passed
shell syntax and invalid-ID checks.

The completed v11 source comparison selects spectrum16 and passes. The verified
outer result reaches 77.984% car / 76.054% truck coverage; both margins and
family spreads pass, but the 80% coverage goal remains unmet. All 224 tests,
380 calibration replays, 760 reconstruction WAVs, 15,000 source samples,
1,500 outer samples and 570 raw held descriptor replays pass. See the
[development report](../../../reports/CAST_improvement.md) for results,
measured runtime, failure records and limitations.
