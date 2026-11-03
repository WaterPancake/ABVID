# Pending code review summaries

Prepared by the orchestrator (Hermes) on 2026-10-18 to speed the two owed
human reviews. Plain-language summaries of what each test asserts, with
anything questionable flagged. **These are review aids, not substitutes for
your line-by-line review** — the design question to keep in mind for every
test is: *does this test actually enforce the protocol rule it claims to, and
can anything it "verifies" be made to pass dishonestly?*

---

## Review set 1 — merged signature tests (overdue, flagged by CURRENT_STATUS)

Arrived via `988ff1c` through merge `bef43c9`. Both files are
**synthetic-only**: the P0 tests monkeypatch `ABVID_DATA_ROOT`/`ABVID_ARTIFACT_ROOT`
to temp dirs; the admission tests use mocked HTTP responses and seeded arrays.
Neither touches real media, the archive, or authorizes anything.

### `tests/test_signature_p0.py` (32 test items) — protocol-contract layer

Grouped by intent:

**Authorization cannot drift (5 items)**
- `test_canonical_config_and_schema_artifacts_match_runtime` — the committed
  `p0_protocol.json` / `manifest_schema.json` match the runtime `configuration()`
  / `schema()` exactly. Docs and code cannot diverge silently.
- `test_p0_cannot_expand_authorization` ×4 — flipping ANY of
  `real_media_collection`, `real_media_processing`, `model_fitting`,
  `protected_source_evaluation` to True fails validation ("Changed P0").
  **This is the anti-reward-hacking guard for the P0 scope. It works.**

**Session/independence integrity (6 items)**
- Extra manifest fields and wrong-typed fields are rejected.
- `unknown` sessions are preserved, NOT counted as independent groups, and
  readiness reports zero verified groups — the "unknown stays unknown" rule.
- Distinct asset IDs alone do not authorize independence (needs
  `session_evidence`).
- Transitive reuse links (a↔short, short↔b) block train/eval role splitting.
- Identical raw SHA256 groups assets even when claimed sessions differ —
  *detected waveform identity overrides claimed metadata*.
- Same `physical_vehicle_id` across roles → group-split error.

**Trim/segment rules (9 items)**
- Foreground requires exactly 1 audible vehicle; background requires 0;
  contaminated foreground, NaN times, out-of-range times, `blind_review=False`,
  and rejected-without-reasons all fail validation.
- Overlapping accepted foreground/background fails.
- Unverified-independence segments cannot enter evaluation.
- Missing same-asset background is a readiness failure (not
  collection-authorized, not model-claim-eligible).

**Dedup audit integrity (4 items)**
- The audit finds crop/gain/polarity/resample variants; short →
  `too_short_for_fingerprint`, silence → `silent_unresolved`; independence
  NOT certified; relabeling all titles changes nothing (label-blind).
- Stale audits (manifest changed since audit) are rejected.
- Conflicting session claims + detected reuse → role-split error (claims lose).
- **Tampering with a correlation score in the audit result breaks the
  acceptance rule — corrupt scores cannot create links.**

**Blind-review packet integrity (4 items)**
- The blind packet hides asset IDs, titles, and recording sessions from the
  HTML/JSON; private map lives outside the blind dir; import round-trips with
  rejections preserved.
- Private map cannot be inside the blind directory.
- Wrong batch hash fails; every recording must have an interval.
- Swapping two private-map entries is detected (commitment mismatch).

**Waveform handling (4 items)**
- Full preparation workflow: 8 clips exported, 2 rejections retained,
  `model_claim_eligible` False.
- Frame-exact trim: start/end frames asserted exactly (32000/96001), clip
  samples equal the expected slice, channels preserved, duplicate export
  raises.
- Raw-hash mismatch fails BEFORE audio is decoded/used.
- `media_kind="real"` rejected (synthetic-only); `repo:` outputs and path
  traversal (`data:../ABVID/...`) blocked.

**Flags for this file:**
- ⚠️ (minor) `test_blind_packet_hides_metadata_and_original_filenames` checks
  asset IDs, titles, and sessions — but not other identifying fields (unit,
  location, date). Verify the packet generator excludes those by construction;
  if not, extend the assertion.
- Nothing suspicious found. The negative-path density is appropriate for a
  contract layer.

### `tests/test_signature_admission.py` (13 tests) — P1 collector boundary

- **Budget/authorization pins:** committed P1 config must have exactly
  5 GB total / 1200 s per asset (matches `P1_AUTHORIZATION.md`); setting
  `model_fitting=True` fails ("Changed P1").
- **URL guard:** only the DVIDS CloudFront media host passes `media=True`;
  dvidshub.net page URLs, localhost, credential URLs, `file://`, and private
  CDN paths are refused.
- **Playlist integrity:** total duration > 1200 s is REJECTED, not silently
  trimmed; NaN durations, absolute-URL segments, AES-encrypted segments, and
  missing ENDLIST all fail.
- **Rendition choice:** picks 720p; avoids audio-only and 1080p.
- **Rights:** editorial-use captions quarantine even with a public-domain
  banner; a normal caption + real unit collects; removing the banner
  quarantines. The banner alone never overrides the caption.
- **Caption parsing is conservative:** "M1 heavy wrecker truck" → NO tank
  class (it's a wrecker); a general's name → no class; genuine "M1A2 + M109A6"
  → [abrams, m109]. No identity/session invention from captions.
- **Storage budget** counts data+artifacts, and symlinks are rejected.
- **Oversize download is rejected on Content-Length BEFORE any payload is
  read** (the mock raises if `.read()` is ever called — a "must not read"
  pattern).
- **octet-stream responses must still contain a real `#EXTM3U`**; HTML
  masquerading as a playlist fails; nothing written on failure.
- **Review card HTML leaks nothing:** no fixture names, vehicle names,
  "dvidshub", or template placeholders; keys must be opaque.
- **Human interval import:** every recording needs intervals; contaminated
  intervals rejected; overlaps rejected; non-finite rejected; and
  `experimental_role` in an annotation is REJECTED — roles cannot be assigned
  through the blind-annotation channel. *(Good leakage guard.)*
- **Fingerprint acceleration parity ×2:** packed landmark votes match
  `np.unique` brute force exactly (incl. negative offsets, empty input); the
  accelerated lag-correlation matches the original implementation exactly
  across sizes, offsets, 1e-12 quiet signals, and all-zeros/ones/sine cases.

**Flags for this file:**
- ⚠️ (minor) The URL guard is tested for `media=True` only; the metadata-page
  URL policy (DVIDS HTML pages) is not covered here. Confirm it's guarded
  elsewhere or accept the gap knowingly.
- The hardcoded CloudFront host is intentional (allowlist pinning).

**Verdict on set 1:** both files are consistent with the repo's guardrails
and the P0/P1 authorization documents. They are defensive, deterministic, and
do not touch real data. **Recommend approving** after your spot-check of the
two minor flags.

---

## Review set 2 — Codex's CAST-4B branch (`cast4b/joint-metric-fixtures`, commit `c60a682`)

Worktree: `~/.hermes/cache/scratch/abvid-cast4b`. Not yet merged.

### `src/abvid/cast/joint.py` (109 lines) — the estimator

- **Correct biased MMD²**: `mean k(X,X) + mean k(Y,Y) − 2·mean k(X,Y)`, fixed
  Gaussian kernel `exp(−‖x−y‖²/2σ²)`, diagonals included, no subsampling or
  fitting at scoring time, negative roundoff clipped to zero.
- **`descriptor_vector`**: concatenates the four CAST descriptor families in
  the fixed `FAMILIES` order (imported from `grouped_metrics` — mapping
  insertion order cannot change coordinates), each coordinate scaled by
  `1/(s_f·√(4·d_f))` → equal family weighting, exactly as its
  `FAMILY_WEIGHT_RULE` declares. Scales are caller-supplied; nothing is fit.
- **`median_bandwidth`**: median of strictly positive unordered pairwise
  distances; seeded PCG64 subsampling above `max_samples`, input order
  retained; an all-identical calibration sample is REJECTED rather than given
  an arbitrary bandwidth. Matches the declared `BANDWIDTH_RULE` text exactly.
- Input validation is thorough (finite, 2-D, positive bandwidth, matching
  dims, exact `int` types so `seed=True` is rejected).
- Docstrings are honest about the estimator's limits: finite-sample bias, no
  p-value, no universal threshold, quadratic cost — matching the roadmap's
  warning not to import significance claims.

**Flags:** none found. Style matches the repo (validation-first, clear
ValueError messages).

### `tests/cast/test_joint.py` (12 tests)

- Calibration uses a **separate** fixed synthetic training sample (PCG64
  4217); no compared distribution enters scale/bandwidth freezing — matches
  "freeze before scoring held groups."
- `matched_family_marginals` — the core fixture: dyadic values make every
  per-family marginal EXACTLY equal (asserted via the existing `compare`,
  W1 == 0), while cross-family correlation flips +1 → −1; MMD > 0.05. This is
  the roadmap's "matched marginals, different dependencies" control.
- `collapsed_samples` ×2 — point mass and two-point mixture: finite, > 0.02
  vs a spread reference; each still matches itself at 0.
- `excessive_spread` — 8× wider (spread_ratio == 8 asserted) with the same
  frozen kernel; MMD > 0.2.
- `identical_zero_mmd` — identical and row-permuted samples → 0 (1e-14).
- `seeded_determinism` — repeated calibration+scoring are **bit-for-bit**
  (float `.hex()` equality); a different seed exercises a different subset.
- `w1_misses_joint_modes` — 4 corners vs 2 corners: all 1-D marginals
  identical (W1 = 0, coverage = 1, spread = 1) but MMD > 0.01 — the exact
  motivation case for adding MMD.
- `analytic_gaussian` — closed form `2(1−e^{−9/8})` for point masses at 0 and
  3, including a singleton and swapped arguments.
- `frozen_scales_equal_weights` — per-family block weight 0.25, order
  invariance, inputs unmodified.
- `positive_distances_median` — hand-computed median (4), collapsed
  calibration rejected, invalid options rejected.
- 2 rejection tests for invalid matrices/dims/bandwidths/scales.

**Flags:** none found. The numeric thresholds (0.05/0.02/0.2/0.01) are labeled
as fixture margins, not scientific pass criteria — correctly, per the roadmap.

### `reports/CAST_joint_metric_fixtures.md` (120 lines)

- Scope framing is honest: fixture phase only; no coverage/transfer claim;
  existing definitions unchanged.
- The graded-selection numbers (733 passed / 1 failed canary / 3 skipped,
  735 vs 678) **match my independent reproduction exactly**, including the
  JUnit 737-vs-735 collection-skip discrepancy explanation.
- ⚠️ (stale, needs a one-line update at merge) The "Commit delivery" section
  describes delivery via a `/tmp` bundle because Codex's sandbox couldn't
  write the worktree's git index. That is now superseded: I committed the
  three files directly as `c60a682` on the branch. The section should be
  updated when merging so the report doesn't misstate the delivery path.

**Verdict on set 2:** implementation is correct and honestly documented;
tests cover every fixture the roadmap requires. **Recommend approving**,
with the stale report paragraph fixed at merge time.

---

## After both reviews

1. Approve set 1 → the owed `CURRENT_STATUS.md` review is satisfied (record it
   there).
2. Approve set 2 → merge `cast4b/joint-metric-fixtures` with the report
   paragraph fix, then the **human-only baseline refresh** (cast 723 → 735,
   baseline 756 → 768 — measured by the script, not hand-computed) in the same
   merge commit.
3. CI should be green on both matrix jobs after the merge.
