# Preprocessing grid v1

Native real → real unseen-session development; 785 non-startup windows, 7 tracked / 5 wheeled sessions.

Headline: preprocessing and C chosen exclusively inside inner session folds.

| Representation | Rule | Balanced accuracy | Tracked recall | Wheeled recall | Worst recall |
|---|---|---:|---:|---:|---:|
| semantic35 | control | 53.73% | 78.35% | 29.10% | 0.00% |
| semantic35 | nested_grid | 51.86% | 77.31% | 26.41% | 0.00% |
| classical53 | control | 43.23% | 67.86% | 18.61% | 0.00% |
| classical53 | nested_grid | 47.26% | 67.84% | 26.67% | 0.00% |

The maximum in fixed_pipeline_results.json is NOT an unbiased estimate for selecting that pipeline.
All per-fold/per-session metrics and selection records are adjacent. Local feature caches/checkpoints and hashes are under runs/preprocessing_grid_v1.
The unchanged control uses the same uncapped non-startup dataset; historical capped/startup-inclusive scores are not a matched comparison.
