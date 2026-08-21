# Milestone 6 pretrained transfer report

Date: 2026-08-21

Status: frozen PANNs comparison and limited-real adaptation complete; Milestone 7
readiness not met

Performance claim: **development evidence limited to the exact factorial corpus and
seven native-real recording sessions below**

## Experiment identity

| Field | Value |
|---|---|
| Runs | `runs/m6_panns_audioset_transfer/seed_{7,19,42,73,101}` |
| Aggregate | `runs/m6_panns_audioset_transfer/aggregate` |
| Git commit | `84195a105157ff7722e4804b65d35d9158dc867c` |
| Training seeds | 7, 19, 42, 73, 101 |
| Grouped split seed | 42 for every run |
| Combined dataset version | `09ff29d1732fde110a4986052d9c8a19c3d18da3a06a1a76f126a1fbdcf03633` |
| Synthetic corpus | 2,688 fully crossed, paired two-second observations |
| Native-real corpus | 271 two-second windows; 4 tracked and 3 wheeled sessions |
| External encoder | PANNs Cnn14 pretrained on AudioSet; encoder frozen |
| Encoder checkpoint SHA-256 | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| Representations | 2,048-dimensional embedding and 527 AudioSet outputs |

Every seed records its complete configuration, manifest hashes, grouped splits,
PANNs provenance and checksum, trained probe states, metrics, and code commit. The
aggregate rejects mismatched commits, datasets, configurations, methods, conditions,
supports, adaptation fractions, or real-training sample IDs. Dispersion below is the
sample standard deviation across five training seeds.

PANNs was pretrained on real-world AudioSet material. Consequently, these experiments
measure transfer from an external environmental-audio representation; they are not a
strict `synthetic only -> real` result even when the ABVID probe is trained only on
synthetic examples.

## Frozen representation results

The fixed test contains the AMX-30 tracked session (64 windows) and Maserati
GranTurismo wheeled session (60 windows). Neither original recording appears in the
train or validation partitions.

| Representation and probe input | Synthetic all-corruption BA | Fixed native-real BA | Tracked recall | Wheeled recall |
|---|---:|---:|---:|---:|
| 2,048 embedding, corrupted | 79.91% +/- 0.89% | 31.50% +/- 6.18% | 40.00% | 23.00% |
| 2,048 embedding, clean + corrupted | 81.28% +/- 2.22% | 36.96% +/- 9.04% | 31.25% | 42.67% |
| 527 AudioSet outputs, corrupted | **86.40% +/- 1.24%** | 40.00% +/- 8.17% | 25.00% | 55.00% |
| 527 AudioSet outputs, clean + corrupted | 83.57% +/- 3.25% | **47.90% +/- 8.76%** | 23.13% | 72.67% |

The AudioSet-output probe improves fixed real balanced accuracy by 9.11 percentage
points over the prior small-CNN augmentation baseline (38.79%). It remains strongly
class-skewed: the best frozen probe recognizes wheeled examples but misses most
tracked examples.

Across all seven real sessions, the paired AudioSet-output probe reaches 46.85% +/-
7.26% session-balanced accuracy. This diagnostic is lower than the prior small-CNN
standard-supervised diagnostic (62.58% +/- 11.44%) and confirms that the apparent
fixed-test gain is not uniform across sources.

## Limited-real adaptation

Only the linear head of the paired 527-output model is updated. Real subsets are
nested and drawn exclusively from the grouped real training partition. Epoch
selection uses the separate real validation sessions. The fixed test remains the
same AMX-30/Maserati pair.

| Requested real fraction | Actual fraction | Training windows (tracked/wheeled) | Fixed native-real BA | Tracked recall | Wheeled recall |
|---:|---:|---:|---:|---:|---:|
| 0% | 0% | 0 | 47.90% +/- 8.76% | 23.13% | 72.67% |
| 1% | 2.11% | 2 (1/1) | 49.68% +/- 8.16% | 24.69% | 74.67% |
| 5% | 5.26% | 5 (4/1) | 48.57% +/- 8.77% | 27.81% | 69.33% |
| 10% | 10.53% | 10 (9/1) | 47.29% +/- 9.99% | 26.25% | 68.33% |
| 25% | 25.26% | 24 (22/2) | 51.45% +/- 4.89% | 36.56% | 66.33% |
| 100% | 100% | 95 (88/7) | **65.50% +/- 13.85%** | 65.00% | 66.00% |

The full real-development partition improves mean balanced accuracy by 17.60 points
over the frozen head and restores approximate class balance. However, seed dispersion
is large and the low-data learning curve is non-monotonic. The training partition's
seven wheeled windows all come from one Goodwood recording; the 88 tracked windows
come from two recordings. Class-weighted loss cannot manufacture missing session
diversity.

## Gate assessment

This is a real improvement, but it does not clear Milestone 7:

- 65.50% mean balanced accuracy with 13.85% seed deviation is not a reliable binary
  category baseline;
- only one wheeled recording session contributes to real adaptation;
- each class has only one fixed test session;
- low-data adaptation does not consistently improve either class;
- the fixed test has been examined repeatedly during development, so it cannot serve
  as a fresh final confirmation for the next milestone.

The next high-value action is data, not another loss coefficient. Admit at least two
additional reviewed moving wheeled sessions and two independent tracked sessions,
keep original recordings grouped, establish a fresh untouched test pair, and rerun
the frozen, adaptation-only, and best invariance controls. Milestone 7 should begin
only after balanced performance repeats across unseen session or vehicle identities.
