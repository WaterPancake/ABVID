# CAST-5A precision planning

This analysis reads existing H1 scalar scores, confusion matrices and paired
bootstrap draws. It does not fit classifiers, decode audio/features, run
inference, generate CAST audio, or freeze CAST-5A. The three-point margin stays
unchanged. See [the findings](../../../reports/CAST5A_precision_planning.md).

Use the existing baseline environment from the repository root:

```sh
.venv/bin/python experiments/cast_coverage/precision_planning/analyze.py \
  --output artifact:cast/precision/cast5a_h1_precision_FRESH_ID
```

Choose a new output directory: existing results are refused. Numerical helpers
live in `src/abvid/evaluation/precision.py`; this script only loads and verifies
saved artifacts and writes the analysis. It validates the canonical H1 wrapper,
checks result-file hashes, reconstructs all saved source F1s from their existing
confusion matrices, replays paired resampling from the scalar score matrices,
and verifies every quoted interval against the saved summaries. No original
predictions, embeddings or fitted models are opened.

Outputs are `precision.json`, the exact executed scripts and a hash lock under
the external artifact root. Both representations, all nine declared source
contrasts under both resampling schemes, and four target mixing contrasts are
retained. Target results provide historical context, not RC−RA variance estimates.

Source resampling retains five selections and six whole groups, or three sites
carrying both dates together. The paired variance identity is checked directly.
The group/selection variance decomposition is exact for this empirical crossed
bootstrap. Normal location/scale probability scenarios are conditional planning
approximations, not validated prospective power with three sites. No nuisance,
hyperparameter or model is selected by this analysis.

The report uses revision `cast5a_h1_precision_20261011_v2`. The initial v1
calculation is retained; v2 adds a selectable new output directory without
changing numerical calculations. Repeating under a new output produces the same
contrast rows; code/provenance metadata can differ after documentation edits.
