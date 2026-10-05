# CAST scoped group effects, v14

This preregistered comparison retains the v12 sampler and all 380 fitted
parents. It tests whether group transforms should leave original harmonic
spacing and weights intact, modifying noise/mix alone or noise/mix/envelope.
The exact v12 full-scope prior is the reference. Controls share each scoped
population, with prototypes recomputed under identical group/parent/child
weights. No new fitting or change to the renderer occurs.

From the ABVID root, reproduce the complete conditional workflow with fresh IDs:

```sh
bash CAST/improvement/context_v14/reproduce_outer.sh fresh_v14_source fresh_v14_outer
```

For the source comparison only:

```sh
bash CAST/improvement/context_v14/reproduce.sh fresh_v14_source
```

The existing `.venv`, local original inputs, and all frozen source artifacts
through v13 are prerequisites. The workflow makes no package installation,
download, dataset/split/environment change or classifier update. It verifies
source ancestry, runs all numerical/provenance tests, freezes scoped population
statistics, and generates 22,500 source samples. It replays all 450 scores
and requires all 150 exact v12 reference scores to match the preceding audit.
Source plots and ten fixed matched audio cards expose the outcomes.

A retained reference or source failure returns code 3 and preserves all
results without another outer comparison. Only a new selected passing
candidate with a complete source audit can generate 1,500 outer outputs;
all outputs precede held-descriptor access. Every generated waveform and
all 570 raw held descriptors are subsequently replayed. A verified outer
scientific failure returns code 3 with its results intact.

Individual stages:

```sh
sh CAST/improvement/context_v14/run.sh source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/context_v14/run.sh audit-source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/context_v14/outer.sh run --source-eval-id FRESH_SOURCE_ID --eval-id FRESH_OUTER_ID
sh CAST/improvement/context_v14/outer.sh verify --eval-id FRESH_OUTER_ID
```

The [protocol](PROTOCOL.md) fixes the hypotheses, sample budget, controls,
scores and unchanged 80%/2.5% criteria. Its original pre-implementation text
is preserved in [preregistration_original.md](preregistration_original.md).
Restored blocks retain their original parent boundary/ambiguity metadata;
discarded group-transform projections are distinguished from applied ones.

These remain effective observation parameters. No block is claimed to be a
recovered physical source or nuisance component. The exposed outer group
provides development evidence; it does not establish independent vehicle
generalization, field performance or classification transfer. Audio exports
retain their local research-use restrictions.

The completed run selected `spectrotemporal_context_T1` on source folds.
Its class-balanced source W1 improved by only 0.181% over v12. Outer coverage
was 78.849% car / 78.725% truck, so the goal failed despite both margins and
all spread checks passing. The 278-test gate, 22,500-source/450-score replay,
150 exact reference scores, 1,500 outer waveforms and 570 raw held descriptors
all passed verification. See the [development report](../../../reports/CAST_improvement.md),
[source audio](diagnostics/group_context_v14_audio/index.html) and
[outer review](diagnostics/context_v14_outer_review/index.html).
No data, environment, baseline or acceptance criterion changed.
