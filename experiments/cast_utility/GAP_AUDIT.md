# CAST gap audit: saved evidence only

2026-10-25. This is a post hoc descriptive audit requested after CAST-5A's
negative result, not a new utility experiment or the proposed CAST-4B study.
The audit specification was written before calculating its new diagnostics.

Inputs are exclusively the existing CAST-5A design, real/RA/RU/RC feature
caches, saved R and C heads, and published results. Verify the design bundle,
protocol identity, feature hashes and head hashes before use. Resolve storage
through `abvid.paths`; validate the canonical grouped and H1 JSON configs.
No audio, encoder checkpoint, new fit, population generation, target cache or
sealed confirmation data is needed. Preserve historical inputs byte-for-byte.

For each of the 30 roles and both representations:

1. Apply the already fitted real-only (R) head to that role's real training
   parents, conventional augmentations (RA), CAST samples (C) and unfitted
   samples (U). Report AUROC, balanced accuracy and both recalls using the
   existing fixed decision rule. These are training-domain probes, not new
   held-out performance estimates.
2. Apply the saved CAST-only (C) head to its CAST training samples and the
   role's real training parents. This checks whether the generated training
   labels are fitable and whether their discriminant transfers back to the
   parents. It does not create new models.
3. Using only the R head's training-fitted StandardScaler, measure class-centroid
   separation, cosine alignment of the truck-minus-car centroid vectors,
   per-class centroid displacement divided by real class separation, and
   within-class RMS spread relative to real. Compare RA and C with the same
   definitions. These coordinate-sensitive diagnostics have no pass threshold.

Aggregate each metric as the equal mean over 30 roles; retain all role rows
and observed minima/maxima. Roles reuse parents and synthesis streams and are
not independent samples. Do not attach clip-bootstrap or 30-role confidence
intervals, select examples or methods from these results, or claim causal
attribution. Reproduce existing held-result summaries by hash, without using
new held predictions for model selection.

The cached generated samples are unconditional population draws, not paired
reconstructions. Thus this audit can locate a mismatch at the generated-feature
interface but cannot separate renderer capacity, fitting objective, calibration,
population sampling, or representation sensitivity. A training-parent
reconstruction comparison and controlled block ablations remain proposed work.

Outputs: an inspectable JSON audit and a gap/data decision report under
`reports/`. These are small summaries; existing audio/features/models remain
outside the checkout. The command is:

```sh
.venv-baseline/bin/python -B -m abvid.cast.utility_gap_audit \
  --output repo:reports/CAST_gap_audit_20261025.json
```
