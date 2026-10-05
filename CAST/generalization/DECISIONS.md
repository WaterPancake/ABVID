# Implementation decisions — 2026-10-04

* The follow-up “test generalization” authorizes a separate minimal CAST-3/4
  acoustic coverage diagnostic. The original CAST-2 stopping point and artifacts
  remain historical facts; nothing in the parent implementation is changed.
* Use the existing finite pilot bank, retaining every ambiguity flag. Do not
  interpret the experiment as having established a stable physical prior.
* Test generation with no held input. Conditional reconstruction and classifier
  training would answer different questions and are excluded from this run.
* Use all 570 metadata-allowed fold-0 observations, without fitting new training
  parents or changing the original 50-parent calibration budget. Group and site
  limitations are explicit. No outside dataset access is needed.
* Implement in `CAST/generalization/` so the parent source hash set, configuration,
  numerical tests and fail-closed held-data rejection remain unchanged.
* Use five seeds and 50 clips per class/seed/arm. Every output has complete donor
  ancestry. Generate all clips and freeze hashes before held access.
* Add a separate sampling stream by using the unchanged renderer with a copied
  seed configuration containing one `sampling` realization and overriding its
  phase draw with the same explicit sampling purpose. No parent config mutation.
* Use elementary marginal distribution distances, spread, tails and finite-bank
  diversity, with train-only scaling and predeclared provisional criteria. No
  generated-window confidence interval is presented as session uncertainty.
* Initial pre-access unit run: 66 passed, one exact permutation-invariance check
  failed because the SD reduction changed at floating-point roundoff under row
  reversal. Repair: sort each coordinate before all marginal score reductions.
  Keep the exact invariance test and all criteria unchanged. No held audio had
  been opened; no scientific score was available during this repair.
* Initial frozen run `cast_generalization_v0_20261004` generated all 1,500 clips
  before accessing all 570 held observations (zero decode/descriptor failures).
  Verification then stopped: Finder changed `generated/.DS_Store`. An exact
  inventory comparison showed no added/removed files and no changed experiment
  files; `.DS_Store` was the only changed hash. Preserve that entire run, including
  its source snapshot, frozen scores and failure record. Do not inspect scores
  until the engineering repair is complete.
* Repair only the artifact inventory to ignore files whose exact basename is
  `.DS_Store`, and add a regression check requiring changed audio and unexpected
  files still to be detected. No source audio, sampler, metric, scale, configuration,
  criterion or sampling schedule changes. Re-run under fresh ID
  `cast_generalization_v0_20261004_r1`. Verify the original and replayed schedules,
  scales and scores match exactly. This is not a new independent held-out test;
  the initial preregistration is the evidence of pre-access method commitment.
