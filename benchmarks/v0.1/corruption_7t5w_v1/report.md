# Controlled-corruption 7/5 v1

Fixed native-trained models; held-out real sessions with added real recordings or simulated microphone response. No retraining or threshold tuning.

SNR below is native-window-to-added-background ratio, not engine-only SNR. Traffic is competing-vehicle interference.

| Condition | Classical BA | Semantic BA | Fusion BA | Fusion tracked recall | Fusion wheeled recall | Worst fusion session-context |
|---|---:|---:|---:|---:|---:|---:|
| native | 45.24% | 53.40% | 39.70% | 51.98% | 27.42% | 0.00% |
| background-rain-heavy-aporee_snr_30 | 45.69% | 52.85% | 39.90% | 51.87% | 27.92% | 0.00% |
| background-rain-heavy-aporee_snr_20 | 45.62% | 53.35% | 41.02% | 52.60% | 29.45% | 0.00% |
| background-rain-heavy-aporee_snr_10 | 47.82% | 52.23% | 44.24% | 55.78% | 32.69% | 0.00% |
| background-rain-heavy-aporee_snr_5 | 50.45% | 51.24% | 45.88% | 55.94% | 35.82% | 0.00% |
| background-rain-heavy-aporee_snr_0 | 50.05% | 51.19% | 46.82% | 53.69% | 39.95% | 0.00% |
| background-rain-heavy-aporee_snr_-5 | 47.96% | 49.26% | 50.30% | 52.43% | 48.18% | 0.00% |
| background-rain-heavy-aporee_snr_-10 | 50.06% | 49.04% | 48.39% | 49.89% | 46.89% | 0.00% |
| background-rain-thunder_snr_30 | 45.44% | 53.72% | 40.23% | 52.77% | 27.69% | 0.00% |
| background-rain-thunder_snr_20 | 46.90% | 52.96% | 42.20% | 55.16% | 29.23% | 0.00% |
| background-rain-thunder_snr_10 | 50.51% | 49.31% | 43.89% | 61.94% | 25.85% | 0.00% |
| background-rain-thunder_snr_5 | 53.02% | 49.06% | 45.60% | 65.36% | 25.84% | 0.00% |
| background-rain-thunder_snr_0 | 51.13% | 50.63% | 47.23% | 69.47% | 25.00% | 0.00% |
| background-rain-thunder_snr_-5 | 50.23% | 51.96% | 47.94% | 72.18% | 23.70% | 0.00% |
| background-rain-thunder_snr_-10 | 50.96% | 50.92% | 49.22% | 74.24% | 24.20% | 0.00% |
| background-birdsong-sunny-day_snr_30 | 45.87% | 54.76% | 41.68% | 55.08% | 28.28% | 0.00% |
| background-birdsong-sunny-day_snr_20 | 47.43% | 54.86% | 43.10% | 58.64% | 27.55% | 0.00% |
| background-birdsong-sunny-day_snr_10 | 54.62% | 56.23% | 48.66% | 66.44% | 30.88% | 0.00% |
| background-birdsong-sunny-day_snr_5 | 55.37% | 54.69% | 49.76% | 69.15% | 30.36% | 0.00% |
| background-birdsong-sunny-day_snr_0 | 53.62% | 55.03% | 49.74% | 72.04% | 27.45% | 0.00% |
| background-birdsong-sunny-day_snr_-5 | 49.65% | 54.63% | 48.87% | 77.65% | 20.10% | 0.00% |
| background-birdsong-sunny-day_snr_-10 | 50.23% | 52.33% | 49.39% | 81.76% | 17.01% | 0.00% |
| background-road-traffic-keelung-aporee_snr_30 | 45.19% | 53.52% | 40.27% | 52.14% | 28.40% | 0.00% |
| background-road-traffic-keelung-aporee_snr_20 | 46.19% | 52.72% | 40.58% | 52.57% | 28.59% | 0.00% |
| background-road-traffic-keelung-aporee_snr_10 | 47.83% | 52.56% | 43.14% | 56.29% | 29.99% | 0.00% |
| background-road-traffic-keelung-aporee_snr_5 | 50.77% | 52.60% | 46.38% | 60.93% | 31.84% | 0.00% |
| background-road-traffic-keelung-aporee_snr_0 | 49.62% | 51.24% | 47.07% | 64.54% | 29.61% | 0.00% |
| background-road-traffic-keelung-aporee_snr_-5 | 47.32% | 49.39% | 45.42% | 65.63% | 25.22% | 0.00% |
| background-road-traffic-keelung-aporee_snr_-10 | 46.69% | 49.63% | 45.75% | 68.73% | 22.77% | 0.00% |
| simulated_low_frequency_attenuation | 37.95% | 52.05% | 36.00% | 40.99% | 31.01% | 0.00% |
| simulated_high_frequency_rolloff | 37.57% | 51.58% | 39.01% | 43.09% | 34.93% | 0.00% |

Every condition retains per-session/per-fold metrics, confusion matrices, descriptive session-bootstrap intervals and transformations. See results.json and conditions/.

Intervals resample 12 sessions, not dependent windows/folds; omit refit and new-background uncertainty. No calibrated-confidence claim. Current-probe background holdout does not mean historical or external-pretraining independence.
