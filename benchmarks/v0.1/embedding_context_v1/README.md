# Embedding and context follow-up: all results

Native real → real, nested unseen-session **development**. Primary fixed-C probability ensemble.

| Experiment | Windows | Balanced accuracy | Tracked recall | Wheeled recall | Worst recall | Descriptive 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| direct32_embedding2048 | 593 | 44.44% | 75.04% | 13.83% | 0.00% | 35.91–52.79% |
| direct32_semantic35 | 593 | 50.50% | 78.59% | 22.41% | 0.00% | 39.51–66.37% |
| matched4_2s_embedding2048 | 511 | 44.24% | 78.02% | 10.45% | 0.00% | 38.42–50.71% |
| matched4_2s_semantic35 | 511 | 54.11% | 79.51% | 28.72% | 0.00% | 42.50–69.07% |
| matched4_4s_embedding2048 | 511 | 43.19% | 75.71% | 10.67% | 0.00% | 37.01–50.02% |
| matched4_4s_semantic35 | 511 | 50.40% | 80.75% | 20.05% | 0.00% | 39.86–65.24% |
| matched8_2s_embedding2048 | 372 | 55.66% | 81.78% | 29.55% | 0.00% | 42.56–74.04% |
| matched8_2s_semantic35 | 372 | 51.91% | 84.23% | 19.58% | 0.00% | 42.68–65.10% |
| matched8_4s_embedding2048 | 372 | 39.66% | 75.42% | 3.90% | 0.00% | 36.37–43.29% |
| matched8_4s_semantic35 | 372 | 56.00% | 80.24% | 31.76% | 0.00% | 42.39–70.30% |
| matched8_8s_embedding2048 | 372 | 35.35% | 68.52% | 2.18% | 0.00% | 27.99–43.01% |
| matched8_8s_semantic35 | 372 | 60.67% | 88.33% | 33.02% | 0.00% | 47.82–73.83% |
| native_embedding2048 | 593 | 44.99% | 74.29% | 15.68% | 0.00% | 38.72–51.57% |
| native_semantic35 | 593 | 53.40% | 81.04% | 25.77% | 0.00% | 42.08–68.35% |

Compare durations **within** a matched set only. Matched4 has 7/5 sessions; matched8 has 6/5.
Native/direct32 both use all 593 original windows and 7/5 sessions. Original fusion/classical controls remain in the frozen native benchmark.
Full selected-C secondary results, splits, per-fold metrics, per-session recalls, paired intervals and checkpoint hashes are in the adjacent JSON files.
Checkpoints and feature tensors remain under `runs/embedding_context_v1/`; no audio is redistributed here.
No condition passes the worst-session gate. No reserved-pair predictions were generated.
