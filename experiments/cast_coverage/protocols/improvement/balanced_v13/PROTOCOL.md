> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../../../README.md).

# Balanced finite population sampling, v13

V12's audited source-only diagnostic reports mean within-fold coverage seed
SD of 3.175 pp car / 2.532 pp truck. In its fixed 50-draw inner-fold batches,
source-group counts span 7–20 / 7–22 despite an expected 12.5 per group.
Distinct direct parents average 42.84/39.32. Waveform and child variation also
contribute; these diagnostics do not isolate the causal role of imbalance.

Freeze one new sampling strategy against exact `group_predictive` v12.
`group_balanced` changes finite categorical allocation only. The original
380 source parents, v11 renderer, v12 transformations, group-count factor,
virtual children, projections, prototypes and population weights are identical.
No real refitting, descriptor-derived ordering, extra samples, new seeds,
temperature setting, class-specific choice or acceptance change is allowed.

## Fixed allocation rule

For a categorical variable with K equally likely categories and N draws,
allocate floor(N/K) instances of every category. Allocate the remainder to
a uniformly permuted subset of categories, then randomly permute all N
assigned indices. This keeps each category's count within floor/ceil(N/K),
and its marginal probability remains 1/K under the declared randomization.

For each class, seed and arm:

1. Balance N=50 direct group indices.
2. Within each assigned group's positions, balance its parent indices. This
   uses each original parent as evenly as possible within that group; a
   distinct parent count is still not a new independent recording session.
3. Independently balance the v12 virtual-child indices over all N positions.
   There are 2G virtual children per parent, with the same group/sign ordering
   and parameters as v12. No virtual parameter is edited.

Joint sampling uses one such plan and selects complete 69-coordinate vectors.
The marginal arm gets a separately randomized complete plan per scalar
coordinate, followed by the existing simplex normalization. All arms retain
the same empirical population. The prototype is exactly v12's population
mean with the same phase/noise streams. Phase/noise input IDs for every
class/seed/index also remain unchanged in joint and marginal arms.

Seed each class/seed/arm plan by the declared SHA-256 namespace in
`balanced_prior.py`. Freeze all 50-index parent/child plans and their hashes
before generation, along with the complete v12 center and parent ancestry.
No plan is selected by observed loss, audio or feature values. Preserve all
bounded projections and original fitted-parameter ambiguity flags.

## Unchanged gates and data restrictions

Use the same five training groups with validation-group exclusion before any
statistics or plans. Outer group `connected_4001f06f57cfeee7`, MELAUDIS,
mixed caches, supplied synthesis banks and military/reserved data remain
excluded from model fitting, plan construction and source selection.
The exposed v12 outer coverage result (79.359%/78.997%) is known human
development context; it supplies no calibration statistic or selection input.

Keep 50/class/seed/arm, seeds [42,123,456,789,1024], every original parent,
four equally weighted families, train-only scales, 5th–95th intervals,
family spread [0.5,2] and the unchanged selection function. Both classes
must reach coverage >=80% and >=2.5% relative W1 gain over each matched
control. The scalar marginal control remains representation dependent.

Tests must establish the categorical/group/conditional-parent/child balancing
invariants, exact old reference and prototype waveforms, deterministic plans
and rendering, correct unchanged population/stream ancestry, and rejection
of invalid schedules or excluded/tampered parents. Preserve every earlier
numerical and provenance test. Replay all 15,000 source records and 300
scores; the 150 v12 reference scores must remain exact.

Only a new selected source-passing, fully audited candidate may enter a new
outer development comparison. Retaining v12 does not authorize repeating its
outer result. For a new candidate, freeze all statistics and draw plans,
generate all 1,500 waveforms before held-descriptor access, and replay every
output plus all 570 raw held descriptors. Retain scientific failures without
altering the criteria. No classifier training or independent confirmation is
claimed, and earlier code, runs, datasets, splits and environments are immutable.
