# Reproduce the frozen H2 execution

Run from the repository root. Outputs are immutable; each command requires a new
destination. Do not restart a live job solely because an observation timed out.
The canonical execution is `source_path_20261004_r1`; the code/environment check
fails if a pinned executable or numerical dependency changes. The complete
resolved environment and code copies are saved under `execution/`.

All commands below use this prefix (shown once to avoid obscuring the stages):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --no-project --python experiments/reproduction/.venv/bin/python --with-requirements experiments/h2/backend-check-requirements.txt python
```

Append each command's script and arguments to that prefix. All stages below
completed for this execution, including the independent final audit and report.
The existing destinations are immutable and the commands refuse to overwrite
them. To reproduce, use new output destinations consistently through the chain;
do not delete or relabel the original run. No process is still running.

```text
experiments/h2/generate_corpus.py --stage full --workers 4 --execution-lock experiments/h2/execution/source_path_20261004_r1/lock.json --output experiments/h2/corpus/source_path_20261004_r1

experiments/h2/verify_corpus.py --corpus experiments/h2/corpus/source_path_20261004_r1 --workers 4 --replay-geometries all --output experiments/h2/corpus/source_path_20261004_r1/verification.json

experiments/h2/learning.py extract --execution-lock experiments/h2/execution/source_path_20261004_r1/lock.json --corpus experiments/h2/corpus/source_path_20261004_r1 --corpus-verification experiments/h2/corpus/source_path_20261004_r1/verification.json --output experiments/h2/cache/source_path_20261004_r1

experiments/h2/learning.py fit --execution-lock experiments/h2/execution/source_path_20261004_r1/lock.json --features experiments/h2/cache/source_path_20261004_r1 --output experiments/h2/models/source_path_20261004_r1

experiments/h2/learning.py evaluate --execution-lock experiments/h2/execution/source_path_20261004_r1/lock.json --models experiments/h2/models/source_path_20261004_r1 --output experiments/h2/results/source_path_20261004_r1

experiments/h2/verify_results.py --execution-lock experiments/h2/execution/source_path_20261004_r1/lock.json --corpus experiments/h2/corpus/source_path_20261004_r1 --features experiments/h2/cache/source_path_20261004_r1 --models experiments/h2/models/source_path_20261004_r1 --results experiments/h2/results/source_path_20261004_r1 --output experiments/h2/results/source_path_20261004_r1/verification.json

experiments/h2/report_results.py --corpus experiments/h2/corpus/source_path_20261004_r1 --models experiments/h2/models/source_path_20261004_r1 --results experiments/h2/results/source_path_20261004_r1 --output reports/H2_source_path_results.md
```

Before fitting: all 480 sources and 9,600 observations must pass count, file/data
hash, role, pairing, shape, normalization and peak checks. The command above also
regenerates every full waveform in fresh workers. Features preserve the H1
encoder/MFCC definitions; real target cache compatibility is checked through
hashes/IDs/configuration, without fitting target statistics.

Before evaluation: `model_lock.json` must contain all 40 fresh training-only
scaler/heads. Synthetic validation stays diagnostic. Evaluation loads the exact
8,066 H1 MELAUDIS observations only after checking that lock, then saves all
cells/seeds, group metrics, 10,000 paired bootstrap draws and the prespecified
decision. This target remains previously exposed development data.

The final audit checks fitted scaler moments, train/validation parents, saved
probabilities, metrics/intervals and protected historical hashes. It additionally
reconstructs all macro-F1 bootstrap draws and contrasts directly from integer
confusion counts, separately from the evaluation helpers. These checks passed.
The [report](../../reports/H2_source_path_results.md) includes all four cells,
representations, class/group failures, figures, runtime and limitations. Its
figures were visually inspected; the roadmap update records the result without
changing later experiment definitions. The audit and report utilities are
post-execution readers, not changes to the frozen generator or learning code.
