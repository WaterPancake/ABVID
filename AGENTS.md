# Agent instructions

**Read [`reports/STATUS.md`](reports/STATUS.md) first.** It is the current state
of the project: open actions, next steps, established results and known gaps.
When a result or decision changes, update it.

## The project

**Question:** which acoustic simulation components most strongly determine
synthetic-to-real car/truck classification with pretrained audio representations?
Source-model dominance over propagation mismatch is a hypothesis, not an assumption.
The plans are [`ABVID_ROADMAP.md`](ABVID_ROADMAP.md) and [`CAST_ROADMAP.md`](CAST_ROADMAP.md).

| Stage | What it is | Where it stands |
|---|---|---|
| Admission | Integrity, annotation and provenance checks | IDMT, MELAUDIS and AI4TEN banks admitted. DCASE 2024 T10 sealed. MVD downloaded, not admitted |
| R0 | Reproduce published papers with their own classes and splits | Partial; closed |
| H1 | Within-domain, cross-dataset and released synthetic-to-real gaps | Done. B1 later found a much stronger baseline recipe |
| H2 | Change controlled simulation components; keep populations, budgets, representation and evaluation fixed | Path effect was filter latency; source vs. path unresolved |
| CAST | Fit effective observations, then sample a population | Utility test (5A) negative. Transfer needs a separately authorized, frozen experiment; coverage is not transfer proof |

The earlier military tracked/wheeled benchmark, Milestone 7 gates and OpenClaw
workflow are retired from the active checkout. Their historical results and
protected-source status remain unchanged. Do not resume them implicitly. The
military signature side project (`experiments/military_signature`,
`src/abvid/signature`) is a separate track; its branch policy is an open decision
in `STATUS.md`.

## Where things go

- **Reusable code:** under `src/abvid`, including `src/abvid/cast`.
- **Experiments:** thin protocol/configuration layers. Do not import sibling
  experiment scripts through `sys.path`.
- **Reports:** `reports/STATUS.md` is the living summary. Superseded reports go
  in `reports/archive/`.
- **Large audio, models, features and generated results:** outside the checkout.
  Resolve storage through `abvid.paths`, and validate the canonical JSON config.
- **Third-party code:** keep pinned code, licenses and repair provenance in `vendor/`.

## Frozen: do not edit or move

- **Historical runners:** they use the external original-layout archive and exact
  recorded environments. Never edit their manifests or locks to accommodate a
  refactor.
- **Pinned files:** files fixed by path or SHA256 are `configs/experiments/*.json`,
  `uv.lock`, experiment `PROTOCOL.md` files, seals, locks and freezes. The DCASE
  seal, for example, binds its protocol's hash.
- **Pinned reports:** the reports left directly in `reports/` (not `archive/`)
  are pinned by those configs and locks. Keep them byte-identical in place.
  Stale links inside frozen documents are expected; leave them.

## Scientific rules

- **Grouping:**
  - Never randomly divide clips/windows from one original recording across roles.
    Group by original recording/session and retain all known provenance links.
  - Preserve `recording_session`; unknown metadata stays `unknown`. Do not invent
    independent sessions from filenames, clips or diagnostic subgroups.
- **Data roles:**
  - Fit preprocessing statistics/classifiers on training data only.
  - Do not tune on final target tests. Exposed target results remain exposed
    development evidence.
  - DCASE stays sealed until a human authorizes one scoring pass.
- **Tasks:** keep R0 original-paper classes distinct from the car/truck study.
  Motorcycle stays excluded from the latter unless a new protocol changes it.
- **Authorization:** scientific runs need a frozen protocol and the user's explicit
  authorization. Do not launch a new experiment merely to validate packaging; run
  focused regression/parity checks after refactoring.
- **Recording results:**
  - Record with every research result: domain, dataset version, split, seed,
    commit, checkpoint, configuration, metrics and group-level uncertainty.
  - Report AUROC, balanced accuracy and AP alongside macro-F1, per dataset (never
    pooled), with worst-group results.
- **Interpreting results:** distinguish recorded observations from clean physical
  source models, coverage from transfer, diagnostic evidence from independent
  generalization, and numerical implementation effects from physical mechanisms.
- **Literature:** Markdown first, PDF authoritative. Cite paper ID/page/section and
  distinguish author claims from interpretation. Use `unknown` when not established.

## Writing

The user learns as the work happens. Keep reports short and in plain language.
Lead with the result and what needs to happen next, and put hashes and settings
after that. Add new results and decisions to `reports/STATUS.md` instead of
creating long standalone reports.

## Running tests

CI runs both uv extras without archive-marker tests. The `cast` environment has no
librosa, so three test files are excluded there:

```sh
.venv/bin/python -m pytest -m "not archive" --ignore=tests/test_h1.py \
  --ignore=tests/test_simulation.py --ignore=tests/test_benchmark_prototype.py
.venv-baseline/bin/python -m pytest -m "not archive"
```

## Harness integrity (anti-reward-hacking)

The mechanical verification contract for coding agents (e.g., Codex):

- `tests/test_harness_integrity.py` is a canary: it fails when the collected
  test count drifts from `tests/harness_baseline.json` (silent test deletion
  or padding) or when a frozen file (uv.lock, the experiment configs) no
  longer hashes to its committed value.
- The two uv extras (`cast`, `baseline`) conflict by design and collect
  different test sets; the baseline records one expected count per
  environment. CI runs both. Archive-marker tests require the local artifact
  bundle and are excluded from CI.
- Baseline updates are HUMAN-ONLY: regenerate with
  `scripts/refresh_harness_baseline.py` in the SAME commit as a legitimate
  suite change, and call it out in the PR description. Never edit
  `tests/harness_baseline.json` by hand to make a red run green.
- Any diff touching `tests/`, `tests/harness_baseline.json`,
  `configs/experiments/*.json`, `uv.lock` or `.github/workflows/test.yml`
  requires line-by-line human review before merge. An agent must never edit
  tests, the baseline file, the canary, or this section to make a failing run
  pass — that is the exact failure mode this layer exists to catch.
