# Native-real failure diagnostics

Post-hoc development analysis; no retraining or selection. Protected recordings were not loaded.

All three model branches reproduce the frozen per-class recalls in all 35 outer folds.

| Model | Balanced accuracy | Tracked recall | Wheeled recall |
|---|---:|---:|---:|
| classical | 45.24% | 68.58% | 21.89% |
| semantic | 53.40% | 81.04% | 25.77% |
| fusion | 39.70% | 51.98% | 27.42% |
| fusion_fixed_0.5 | 41.66% | 71.68% | 11.64% |

Selected wheeled thresholds (35 folds): `{'0.6': 12, '0.3': 4, '0.25': 8, '0.4': 5, '0.35': 3, '0.2': 3}`.

## Session audit

Recall is the mean across that session's pairing contexts. RMS is digital level, not source sound pressure or SNR.

| Session | Class | Windows | RMS median dBFS | Bandwidth median Hz | Classical recall | Semantic recall | Fusion recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Abarth 205 Monza | wheeled | 7 | -27.0 | 1500 | 77.6% | 30.6% | 16.3% |
| Sturmgeschütz III Ausf. G | tracked | 8 | -14.2 | 750 | 75.0% | 75.0% | 47.5% |
| T-72B3 mod. 2016 and BMP-3 | tracked | 64 | -26.5 | 1293 | 7.2% | 95.0% | 16.9% |
| T-80BVM and MT-LBVMK | tracked | 60 | -28.6 | 828 | 68.3% | 81.7% | 47.3% |
| AMX 30 | tracked | 64 | -17.5 | 947 | 88.1% | 73.1% | 50.9% |
| Ford Model T (1926) | wheeled | 28 | -18.5 | 1615 | 17.3% | 4.1% | 38.3% |
| M1 Abrams | tracked | 46 | -18.0 | 323 | 99.6% | 94.3% | 82.2% |
| M1126 Stryker Infantry Carrier Vehicle | wheeled | 64 | -25.1 | 541 | 0.2% | 3.1% | 7.6% |
| M1151 Up-Armored HMMWV | wheeled | 64 | -30.4 | 476 | 13.6% | 9.6% | 12.3% |
| M2 Bradley and M1 Abrams | tracked | 64 | -13.9 | 195 | 77.5% | 65.3% | 54.7% |
| Maserati GranTurismo S | wheeled | 60 | -15.3 | 226 | 0.7% | 81.4% | 62.6% |
| Mobile Protected Firepower testbed | tracked | 64 | -14.8 | 694 | 64.4% | 82.8% | 64.4% |

## Operating states

| Session / state | Windows | Classical recall | Semantic recall | Fusion recall |
|---|---:|---:|---:|---:|
| Abarth 205 Monza / accelerating | 7 | 77.6% | 30.6% | 16.3% |
| Sturmgeschütz III Ausf. G / decelerating | 2 | 80.0% | 100.0% | 50.0% |
| Sturmgeschütz III Ausf. G / steady_speed | 6 | 73.3% | 66.7% | 46.7% |
| T-72B3 mod. 2016 and BMP-3 / idle | 7 | 0.0% | 100.0% | 8.6% |
| T-72B3 mod. 2016 and BMP-3 / steady_speed | 57 | 8.1% | 94.4% | 17.9% |
| T-80BVM and MT-LBVMK / accelerating | 23 | 88.7% | 79.1% | 48.7% |
| T-80BVM and MT-LBVMK / decelerating | 1 | 60.0% | 20.0% | 40.0% |
| T-80BVM and MT-LBVMK / mixed | 16 | 20.0% | 88.8% | 26.2% |
| T-80BVM and MT-LBVMK / steady_speed | 20 | 84.0% | 82.0% | 63.0% |
| AMX 30 / accelerating | 6 | 93.3% | 76.7% | 73.3% |
| AMX 30 / decelerating | 1 | 80.0% | 100.0% | 60.0% |
| AMX 30 / idle | 16 | 90.0% | 71.2% | 53.8% |
| AMX 30 / startup | 13 | 98.5% | 90.8% | 46.2% |
| AMX 30 / steady_speed | 28 | 81.4% | 64.3% | 46.4% |
| Ford Model T (1926) / idle | 21 | 22.4% | 4.8% | 44.2% |
| Ford Model T (1926) / startup | 7 | 2.0% | 2.0% | 20.4% |
| M1 Abrams / accelerating | 4 | 100.0% | 100.0% | 80.0% |
| M1 Abrams / decelerating | 6 | 100.0% | 100.0% | 76.7% |
| M1 Abrams / steady_speed | 36 | 99.4% | 92.8% | 83.3% |
| M1126 Stryker Infantry Carrier Vehicle / accelerating | 6 | 0.0% | 0.0% | 0.0% |
| M1126 Stryker Infantry Carrier Vehicle / idle | 13 | 0.0% | 0.0% | 4.4% |
| M1126 Stryker Infantry Carrier Vehicle / steady_speed | 40 | 0.4% | 2.9% | 6.4% |
| M1126 Stryker Infantry Carrier Vehicle / unknown | 5 | 0.0% | 17.1% | 34.3% |
| M1151 Up-Armored HMMWV / accelerating | 8 | 39.3% | 0.0% | 17.9% |
| M1151 Up-Armored HMMWV / idle | 19 | 1.5% | 27.8% | 13.5% |
| M1151 Up-Armored HMMWV / steady_speed | 37 | 14.3% | 2.3% | 10.4% |
| M2 Bradley and M1 Abrams / accelerating | 2 | 100.0% | 100.0% | 100.0% |
| M2 Bradley and M1 Abrams / steady_speed | 62 | 76.8% | 64.2% | 53.2% |
| Maserati GranTurismo S / idle | 14 | 2.0% | 53.1% | 50.0% |
| Maserati GranTurismo S / mixed | 45 | 0.3% | 89.8% | 66.3% |
| Maserati GranTurismo S / startup | 1 | 0.0% | 100.0% | 71.4% |
| Mobile Protected Firepower testbed / steady_speed | 64 | 64.4% | 82.8% | 64.4% |

## Limits

- No causal attribution from these observational associations.
- Overlapping windows and pairing contexts are dependent; no window-level significance tests.
- Device, distance and simultaneous vehicle counts are not systematically annotated.
- Mean logit contributions describe individual linear probes, not calibrated probabilities or causal sound components.
- Fixed-0.5 fusion is a post-hoc diagnostic, not a replacement selected model.
