# H2 execution progress

Completed objective: finish the remaining propagation correction and complete
the frozen H2 four-cell experiment. This checklist records the completed gates;
the linked manifests, locks, audits and result report supply the evidence.

- [x] Design/manifests frozen: `source_path_20261004_v1`.
- [x] Ground-angle indexing repair verified; original/indexing-only revisions retained.
- [x] Total reflected-path spreading and common-c repair independently validated.
- [x] Additional implementation defects documented and corrected before target access.
- [x] Dry-source adapter matches S0 reference and implements only frozen S1 changes.
- [x] Source manipulation, static/moving propagation, normalization and deterministic replay gates pass.
- [x] Thirty prespecified paired validation slots rendered and timing/resource estimate measured.
- [x] Executable code/environment frozen; pre-target clarifications documented.
- [x] All 480 dry sources / 9,600 observations generated, paired and verified without selective rejection.
- [x] Generated-corpus and feature locks complete; exact H1 target cache compatibility checked.
- [x] Forty fixed heads trained on their own synthetic training roles; validation diagnostic only.
- [x] Heads/preprocessing/decision rule locked before all-cell target scoring.
- [x] Full target/group/class/seed metrics and paired uncertainty saved and independently verified.
- [x] Hypothesis/falsification interpretation, limitations, runtime and artifacts reported; roadmap reconciled.

MELAUDIS remains exposed development data. No changes are selected from new target
scores. Historical R0/H1 and military milestone/evaluation boundaries remain intact.

## Current bounded repair evidence

The indexing-only backend is retained at
`backend_revisions/indexing_20261004_v1`; its v2 result remains unchanged. The v3
indexing verifier checks that revision and separately checks indexing in the
working renderer, so later propagation changes do not invalidate the historical
direct-path parity test or acquire a false validation claim from it.

The failed static probe and the first source/renderer replay failures remain
saved. The corrected 31-tap Sinc implementation, fixed-order complex arithmetic,
source adapter and batched renderer pass the subsequent checks. Details and
the spherical-field/passivity clarification are in `IMPLEMENTATION_AMENDMENTS.md`.

Execution `execution/source_path_20261004_r1/lock.json` is frozen, SHA256
`ef935e5f29940d9691ccac681c4661fb21b4fd366dd3a61cf5ede8c8ef461cd1`.
It records 30 code files, the resolved environment, 26 unit tests, all 480 source
cases, 151 renderer cases and fresh-process replay of all 240 prescribed paired
validation observations. Maximum moving-frequency error is 0.539% (limit 1%);
maximum actual static transfer magnitude error is 0.0143 dB (limit 0.5 dB).
Numerical validation does not establish real-road calibration.

Full generation completed into `corpus/source_path_20261004_r1`, with four
workers and unchanged source/path draws and budgets. There were zero failures
and zero replacement draws. The complete independent corpus check passed:
480 source replays, 9,600 full waveform replays and every saved hash, role,
pairing and preprocessing check. Generation took 16.55 minutes; full verification
took 15.34 minutes. Fixed feature extraction took 4.69 minutes; all 40 heads then
converged and were locked before target scoring. The final independent audit
passed, including every saved prediction, 10,000 paired bootstrap draws, a
separate integer-confusion calculation and all 8,257 protected historical hashes.

The [result report](../../reports/H2_source_path_results.md) records BEATs
source-minus-path D = −34.90 points, conditional 95% CI [−43.56, −29.17]. This
contradicts source dominance for the declared interventions; it does not rank
all physical mismatch or demonstrate reliable recognition. Every cell retains
a zero-recall group/class case. No operator action is needed.
