> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H2 ground/air and collapse follow-up

Active goal: separate ground reflection from air absorption, and investigate
the direct-path heads' truck bias. [Protocol](PROTOCOL.md) and [config](config.json)
declare the complete factorial comparison and all gain/distribution diagnostics
before new evaluation. The parent H2 experiment remains unchanged.

- [x] Freeze exact populations, parent hashes and new execution.
- [x] Validate new toggles and render/replay the complete paired bank.
- [x] Extract fixed features and freeze all 40 additional heads.
- [x] Evaluate all eight cells and the declared collapse checks.
- [x] Independently audit results, report every check and reconcile the roadmap.

The 72 renderer checks, 240-observation paired pilot, complete pilot replay and
five control tests passed before bulk generation. All 9,600 new observations now
pass complete fresh-process replay; all 9,600 original endpoints match exactly.
All 80 heads are locked (40 new and 40 original), and the complete evaluation and
independent audit passed. [Results](../../../reports/H2_ground_air_collapse.md): the
native BEATs interaction is +31.80 F1 points; neither ground nor air alone recovers
the joint gain. Fixed target RMS matching and synthetic ±12 dB controls contradict
a gain-only explanation of the direct-path truck bias. A read-only inspection
quantifies the air branch's differential FIR delay; attenuation versus latency
must be separated before assigning the interaction to physical absorption.
That implementation control is the next part of the active investigation.
MELAUDIS remains previously exposed development data. No user action or new
dataset is required.

Run each stage sequentially from the repository root in the existing pinned
overlay. Outputs fail closed when their destination already exists; these are
provenance records, not commands for overwriting a completed run.

```bash
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

Prefix each command below with:

```bash
uv run --no-project --python experiments/reproduction/.venv/bin/python \
  --with-requirements experiments/h2/backend-check-requirements.txt
```

```text
python experiments/h2_mechanisms/corpus.py generate --stage full
python experiments/h2_mechanisms/corpus.py verify
python experiments/h2_mechanisms/features.py synthetic
python experiments/h2_mechanisms/fit_evaluate.py fit
python experiments/h2_mechanisms/features.py target
python experiments/h2_mechanisms/fit_evaluate.py evaluate
python experiments/h2_mechanisms/verify_results.py
python experiments/h2_mechanisms/inspect_air_phase.py
python experiments/h2_mechanisms/report_results.py
```

The last three scripts are post-freeze, read-only audit/report tools. They are
hashed in their outputs and cannot change fitting or evaluation settings. The
independent confusion-count verifier also passed a deterministic artificial-data
check of all 504 F1, balanced-accuracy and predicted-truck intervals before the
new target results were available.
