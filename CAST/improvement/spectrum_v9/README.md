# Joint expected-spectrum calibration

Use the existing pinned ABVID environment and completed, audited v8 source
experiment. From the ABVID root, the complete source workflow is:

```sh
bash CAST/improvement/spectrum_v9/reproduce.sh NEW_ID
```

It runs the stages below, makes plots and a ten-example audio review, and
returns code 3 for a retained scientific failure. It never opens outer data.
The current experiment runs these identical stages individually. Use fresh IDs:


```sh
bash CAST/improvement/spectrum_v9/run.sh calibrate --run-id NEW_BANK_ID --workers 4
bash CAST/improvement/spectrum_v9/run.sh audit-bank --run-id NEW_BANK_ID --workers 4
bash CAST/improvement/spectrum_v9/run.sh source --run-id NEW_BANK_ID --eval-id NEW_SOURCE_ID
bash CAST/improvement/spectrum_v9/run.sh audit-source --eval-id NEW_SOURCE_ID
```

The calibration stage runs numerical/provenance tests, freezes code and all
380 source parents, then saves each parent's complete optimizer history,
initial/final parameters, failure flags, ancestry, runtime and two calibration
reconstruction WAVs. It preserves original fits. The bank audit repeats all
380 optimizations and verifies exact saved waveform samples. Source generation
requires that audit and compares the new bank with v8 using the same matched
controls. Source replay recomputes every draw, waveform, descriptor and score;
all 150 v8 reference scores must be identical.

Commands preserve existing outputs and refuse changed code/inputs. Inspect
`scientific_status.json`: successful numerical execution does not mean the
coverage/margin criteria pass. There is no automatic outer evaluation.

The [protocol](PROTOCOL.md) specifies the objective, fixed budget, data
restrictions and retained criteria. The checking streams have become calibration
inputs for the derived vectors; their reconstructions are not independent
validation. Source selection remains exploratory development on five groups.

The separate outer runner is conditional: it refuses failed candidates and
missing/mismatched source audits before accessing any outer inputs. A passing
audited source candidate permits:

```sh
bash CAST/improvement/spectrum_v9/outer.sh run --source-eval-id NEW_SOURCE_ID --eval-id NEW_OUTER_ID
bash CAST/improvement/spectrum_v9/outer.sh verify --eval-id NEW_OUTER_ID
```

It freezes the selected bank and all 1,500 samples before reading the original
570 outer descriptors, then verifies waveform replay and all 570 raw descriptor
reconstructions. It retains the original scales, split, counts, metrics and
class-specific target. This remains exposed outer **development**, because H1
and CAST v0 already used that group. These instructions do not imply an outer
run has occurred; consult the improvement report and access receipt.
