# CAST coverage improvement work

This separate development experiment pursues the user's **80% per-class held
coverage and 2.5% relative distance improvement against each control**. The old
CAST pilot and generalization failure remain unchanged. See [PROTOCOL.md](PROTOCOL.md)
and [RESEARCH_LOG.md](RESEARCH_LOG.md) for the contract, evidence and current
status. An 80% source-development result does not count as achieving the outer
held-group objective.

The completed v15 revision uses a source-selected width **T=1.1** for the
scoped noise/mix/envelope population. Verified outer coverage is **82.019%
car / 81.286% truck**, with W1 gains of **12.295% / 17.207%** over matched
marginals and **20.946% / 24.048%** over prototypes. All original spread limits
pass. The numerical goal is achieved; this remains exposed development,
not independent confirmation or classification transfer.

All **306 tests pass**. Full replay verifies 30,000 source samples, 600 scores,
150 exact reference scores, 1,500 outer waveforms and 570 raw held descriptors.
Reproduce the conditional workflow with original local inputs, the pinned
environment and frozen parent artifacts available:

```sh
bash CAST/improvement/width_v15/reproduce_outer.sh NEW_SOURCE_ID NEW_OUTER_ID
```

See [v15 instructions](width_v15/README.md), the [development report](../../reports/CAST_improvement.md)
and the [fixed outer audio review](width_v15/diagnostics/width_v15_outer_review/index.html).
The stricter source gate requires >=80% mean coverage in every source fold
and retains all original source criteria. It chooses T=1.1, whose minimum
source-fold coverage is 83.280% car / 88.814% truck. The outer result retains
all five fixed seeds, including two car seeds below 80%; the declared mean
passes. Broader coverage costs 4.460% higher mean W1 than the narrower v14
reference, while both matched-control margins still pass.

Historical v12 coverage was 79.359%/78.997%. The [v13 balanced-sampling comparison](balanced_v13/README.md)
retained v12 without another outer run. The [v14 scoped-prior comparison](context_v14/README.md)
was fully audited but lowered outer coverage to 78.849%/78.725%. All of these
results and their original scientific failures remain preserved.

The original smooth8 workflow and subsequent experiments below are retained
as reproducible historical stages.

The initial renderer is smooth8: the same eight effective noise controls, now
interpolated as a smooth log-power frequency envelope. Harmonics, optimizer
budget, fitting objective, preprocessing and synthetic acceptance are retained.
No dependency installation, classifier, MELAUDIS, protected/military audio or
physical-source interpretation is introduced.

Fresh-run workflow, with the existing local IDMT files, frozen H1/CAST parent
artifacts and pinned environment present:

```sh
bash CAST/improvement/reproduce.sh smooth8_reproduce 4
```

This runs the stages below, retains unsuccessful results, and refuses an outer
comparison if the source-selected candidate fails the declared criteria. A
source failure exits nonzero and is a scientific result, not an instruction to
change the acceptance rule. Use a fresh ID; the script does not overwrite or
silently resume an existing experiment. The individual commands below support
resuming a known unfinished stage from its verified artifacts.

From the ABVID root, using a new run ID and fresh evaluation IDs:

```sh
bash CAST/improvement/run.sh prepare --run-id smooth8_reproduce
bash CAST/improvement/run.sh synthetic --run-id smooth8_reproduce --workers 4
bash CAST/improvement/run.sh fit --run-id smooth8_reproduce --scope pilot --workers 4
bash CAST/improvement/analyze.sh audit-fits --run-id smooth8_reproduce --scope pilot
bash CAST/improvement/analyze.sh source --run-id smooth8_reproduce --eval-id smooth8_pilot_reproduce --scope pilot --temperatures 1 --include-v0
bash CAST/improvement/analyze.sh audit-source --eval-id smooth8_pilot_reproduce
bash CAST/improvement/run.sh fit --run-id smooth8_reproduce --scope full --workers 4
bash CAST/improvement/analyze.sh audit-fits --run-id smooth8_reproduce --scope full
bash CAST/improvement/analyze.sh source --run-id smooth8_reproduce --eval-id smooth8_full_reproduce --scope full --temperatures 1 1.1 1.25 1.5
bash CAST/improvement/analyze.sh audit-source --eval-id smooth8_full_reproduce
```

The full stage reuses verified completed pilot fits, then fits the remaining
330 parents in H1 outer fold 0 / seed 42's exact 380-recording training set.
Individual completed fits are retained. Overwrites, changed source code,
changed configuration and modified source bytes fail closed.

To queue the remaining analysis behind an already running full fit:

```sh
.venv/bin/python CAST/improvement/continue_full.py --run-id EXISTING_RUN_ID --source-eval-id NEW_FULL_SOURCE_ID
```

This command never starts or restarts fitting. It waits up to two hours for a
successful full-fit summary, audits every fit, evaluates and audits the fixed
source grid, builds the figures/audio review, and runs the frozen outer check
only if the source-selected candidate passes. It saves an atomic continuation
receipt in the run folder. A failed source or outer criterion exits with code 3.
If waiting expires, inspect the original fitting process; timeout does not mean
that process stopped. Do not also launch the same analysis manually while its
continuation process is live.

Outer development evaluation is a separate action after inspecting and freezing
the training-only selection. It refuses candidates that fail the source criteria
or lack a matching successful replay audit, before accessing held descriptors:

```sh
bash CAST/improvement/analyze.sh outer run --source-eval-id smooth8_full_reproduce --eval-id smooth8_outer_reproduce
bash CAST/improvement/analyze.sh outer verify --eval-id smooth8_outer_reproduce
```

All generated outputs are complete before the outer evaluator reads its held
descriptor cache. It retains the original outer scales, scoring definition,
percentile interval, seeds and per-arm sample counts. Verification regenerates
every waveform/parameter choice and reprocesses each raw held recording.
This is the already exposed IDMT fold-0 **development** group; no fresh
confirmation or classification-transfer claim is supported.

Source-only diagnostic figures can be reproduced after the relevant stages:

```sh
bash CAST/improvement/analyze.sh diagnostics source --id smooth8_full_reproduce --output-id smooth8_full_source_figures
bash CAST/improvement/analyze.sh diagnostics fits --id smooth8_reproduce --scope full --output-id smooth8_full_fit_figures
bash CAST/improvement/analyze.sh gallery --run-id smooth8_reproduce --scope full --output-id smooth8_full_audio_review
```

The local audio review contains systematic best/median/worst cases, the first
input in each class/group, and links to all fits. Open its `index.html` directly,
or serve CAST locally from the ABVID root:

```sh
.venv/bin/python -m http.server --bind 127.0.0.1 --directory CAST 8765
```

The generated page uses no external assets.

Fit verification reloads the allowed raw audio, repeats preprocessing, replays
all fitted and unoptimized checking starts, and checks every saved waveform,
component, descriptor diagnostic and failure flag. Source verification replays
every generated sample and independently recomputes fold scales and scores.

The root environment is used read-only with the original dependency versions.
Set `CAST_PYTHON` to an identically pinned isolated Python if needed. The fitter
uses up to four workers, one Torch thread each. Analysis uses one CPU thread.
Exact sample-array replay is required; FLOAT WAV container timestamps can differ
across runs. All source ancestry, hashes, flags, parameters, audio, checking losses,
diagnostic plots and measured runtime are saved per fit.

The initial incomplete synthetic run `cast_smooth8_v1_20261004` is preserved.
Its repaired, accepted successor is `cast_smooth8_v1_20261004_r1`. The research
log records the normalization/replay correction and unchanged thresholds.

The completed full-bank v1 grid failed the required margin against marginals.
The next source-only prior revision is frozen in
[prior_v2/PROTOCOL.md](prior_v2/PROTOCOL.md). It reuses all audited fits and
separates spectral from envelope spread; it does not change the renderer or
acceptance rules. Reproduce its twelve-candidate source experiment with:

```sh
bash CAST/improvement/analyze.sh block source --run-id cast_smooth8_v1_20261004_r1 --eval-id NEW_BLOCK_SOURCE_ID
bash CAST/improvement/analyze.sh block audit --eval-id NEW_BLOCK_SOURCE_ID
```

The source command runs the numerical/provenance test gate before generation.
Every output includes its direct donors and all class-center calibration
parents. All twelve prior-v2 candidates failed the combined criteria; all
90,000 sample replays and 1,800 recomputed scores passed. No new outer
comparison ran. Reproduce its figure after a successful audit with:

```sh
bash CAST/improvement/analyze.sh block-figures --eval-id NEW_BLOCK_SOURCE_ID --output-id NEW_BLOCK_FIGURES_ID
```

The fixed capacity revision is documented in
[capacity_v3/PROTOCOL.md](capacity_v3/PROTOCOL.md); synthetic recovery must pass
before new real fitting. The evolving evidence is in
[CAST_improvement.md](../../reports/CAST_improvement.md).


The historical audited source comparison [direct-envelope v8](envelope_v8/README.md),
following [per-parent mixture v7](parent_mixture_v7/README.md). Its complete
workflow is `bash CAST/improvement/envelope_v8/reproduce.sh NEW_SOURCE_ID`.
V8 source coverage reaches 80.626% car / 80.448% truck, but gains against
independent marginals were -1.254% / -1.267%; that result left the combined goal unmet.
All 176 tests and 15,000 exact sample replays pass. No new outer evaluation ran.

The [temporal-v10 workflow](temporal_v10/README.md) is the first revision to
pass all source criteria. Its audited outer coverage is 77.212%/75.571%, while
matched margins and spread passed. That result left the numeric goal unmet.
All previous results are preserved. The subsequent [spectral-resolution protocol](resolution_v11/PROTOCOL.md)
uses only source parents, preserving all acceptance criteria.

The [spectral-resolution v11 workflow](resolution_v11/README.md) is also fully
completed and audited. Its source-selected candidate gives 77.984%/76.054%
outer coverage with both matched margins above 9%, still below the 80% goal.
Use its conditional reproduction command for calibration through held-group
replay and review; unsuccessful results remain preserved.
