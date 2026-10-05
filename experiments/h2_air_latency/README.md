# Air-filter attenuation versus implementation latency

The [verified ground/air result](../../reports/H2_ground_air_collapse.md) finds a
joint benefit but cannot attribute it to absorption: its finite filters also
change relative path latency. This narrow continuation separates those effects
with ground fixed. No original result or renderer is replaced.

See [protocol](PROTOCOL.md) and [configuration](config.json). Use the same pinned
environment as `experiments/h2_mechanisms/README.md`. No new data, encoder or tuning.

- [x] Freeze the exact populations and parent artifacts.
- [x] Validate amplitude-only and delay-only operators, then freeze execution.
- [x] Generate and exactly replay every complete new waveform; retain observations.
- [x] Extract fixed features and lock all new heads before target evaluation.
- [x] Evaluate, independently audit and report the complete four-cell comparison.

Completed 2026-10-05. The [full report](../../reports/H2_air_latency_control.md)
records each check, source/representation/group results, uncertainty and limits.
With ground on, native BEATs macro-F1 is 3.65% for neither air component, 3.68%
for attenuation only, 35.43% for latency only and 35.46% for both. Latency minus
attenuation is +31.76 points [24.77, 41.06]. Joint minus latency-only is +0.030
[−0.138, 0.206], within the declared ±3-point margin. The numerical latency
reproduces the gain; this is not evidence of physical air-absorption dominance.

The 74 synthetic-only renderer checks passed. The paired pilot generated and
exactly replayed 240 new complete waveforms/observations and reproduced 240
existing endpoints. Three control tests passed, including 504 independently
reconstructed uncertainty intervals on artificial predictions. The ten-file
execution freeze was admitted before full generation. All 9,600 new complete
waveforms/observations replay exactly and all 9,600 reused endpoints match.
The independent audit reproduced 80 heads, 160 target evaluations and 320
cross-cell checks, reconstructed 504 integer-count intervals and verified 240
source-interaction intervals. Prior artifacts remain unchanged. All target
results are exposed-development diagnostics; every cell retains a zero-recall
head/group/class case. No target tuning or independent confirmation occurred.

From the repository root, set `PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`.
Prefix each command with:

```bash
uv run --no-project --python experiments/reproduction/.venv/bin/python \
  --with-requirements experiments/h2/backend-check-requirements.txt
```

Run sequentially; existing destinations are preserved and cannot be overwritten:

```text
python experiments/h2_air_latency/corpus.py generate
python experiments/h2_air_latency/corpus.py verify
python experiments/h2_air_latency/phase_features.py
python experiments/h2_air_latency/phase_learning.py fit
python experiments/h2_air_latency/phase_learning.py evaluate
python experiments/h2_air_latency/verify_control.py
python experiments/h2_air_latency/report_control.py
```

The reporting script generates factual tables, JSON summaries and PNG/SVG figures;
the report narrative is reviewed separately. `complete_stage.py` checks the final
report, figure-review record, artifact links and roadmap references. All prior H2
and ground/air artifacts, including the first follow-up report and its figures,
are protected by the new design's hash inventory. The larger physical source,
background and sensor decomposition remains unresolved; no later experiment starts
as part of this completed control.
