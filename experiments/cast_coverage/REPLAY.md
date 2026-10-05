> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../README.md).

# CAST source-fold width calibration, v15

This source-only comparison tests the four preregistered widths
`[1, 1.1, 1.25, 1.5]` of v14's selected noise/mix/envelope population.
Harmonic controls, donor/child/phase/noise schedules, renderer, fitted parents,
50-sample budgets, metrics, five seeds and actual outer criteria stay fixed.
All arms share the same width-specific population, with matched prototypes.

The original source criteria remain required. The added internal gate
requires >=80% mean coverage in every source validation group, for each
class. Only eligible widths compete by the original W1 ranking. Source
failure or a retained T=1 reference stops before outer access. The
[protocol](protocols/improvement/width_v15/PROTOCOL.md) explains the evidence, exact transform and limitations.

From the ABVID root, with the existing environment, original local inputs,
v11 fitted bank and completed frozen ancestors through v14:

```sh
bash CAST/improvement/width_v15/reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID
```

For source evaluation, replay, plots and fixed audio only:

```sh
bash CAST/improvement/width_v15/reproduce.sh FRESH_SOURCE_ID
```

Individual scientific stages:

```sh
sh CAST/improvement/width_v15/run.sh source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/width_v15/run.sh audit-source --eval-id FRESH_SOURCE_ID
sh CAST/improvement/width_v15/outer.sh run --source-eval-id FRESH_SOURCE_ID --eval-id FRESH_OUTER_ID
sh CAST/improvement/width_v15/outer.sh verify --eval-id FRESH_OUTER_ID
```

The workflow runs all existing tests, freezes source statistics before
generation, replays all 30,000 samples and 600 scores, and requires 150 exact
T=1 v14 reference scores. Ten fixed source cards contain original/fitted
parent audio plus all four generated widths with a common playback gain.
Only a new source-selected eligible width with full replay can generate
1,500 outer samples before held-descriptor access. The outer audit replays
all generated waveforms and all 570 raw held descriptors; plots and fixed
audio review expose successes and failures.

Code 3 indicates a retained scientific failure or unchanged reference,
with all results preserved. No package installation, data download, dataset,
split, environment, original baseline or classifier change is performed.
Audio derivatives retain their local research restrictions. The repeatedly
exposed outer group provides development evidence, not fresh confirmation,
classification transfer or independent physical-vehicle generalization.

The completed run `context_width_v15_full_source` selects **T=1.1**. Its
minimum source-fold coverage is 83.280% car / 88.814% truck, with all original
criteria also passing. The audited `width_v15_outer_development` result has
**82.019% / 81.286% coverage**, **12.295% / 17.207% W1 gain over marginals**,
and **20.946% / 24.048% gain over prototypes**. All spread checks pass.
All 306 tests, 30,000 source records, 600 scores, 150 exact reference scores,
1,500 outer waveforms and 570 raw held descriptors pass verification.

This achieves the specified mean-coverage/control-margin goal. W1 is 4.460%
worse than narrower v14 in the class-balanced outer mean, and two car seeds
remain below 80%; all five seeds are retained. See the [full report](../../reports/CAST_improvement.md),
[source audio](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/improvement/width_v15/diagnostics/context_width_v15_audio/index.html),
[source plots](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/improvement/width_v15/diagnostics/context_width_v15_figures_r1/source_comparison.png)
and [outer audio/plots](https://github.com/WaterPancake/ABVID/blob/77daa1b339906d35b6fa0a59a166ecf2eac01174/CAST/improvement/width_v15/diagnostics/width_v15_outer_review/index.html).

Measured source preparation took 87.46 s, generation/scoring including the
test gate 407.36 s, and complete source replay 675.56 s. Outer generation and
scoring took 413.78 s, including 31.79 s for waveforms; full outer replay took
355.23 s. These are observed local timings with presentation work overlapping
part of outer preparation/replay. The actual run executed the documented
stages individually; wrapper syntax and invalid-ID rejection were checked.
