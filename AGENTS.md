# Active research scope

The active research question is: which acoustic simulation components most
strongly determine synthetic-to-real vehicle classification with pretrained audio
representations? Source-model dominance over propagation mismatch is a hypothesis,
not an assumption. Follow `ABVID_ROADMAP.md` and the current reports.

## Research sequence

1. Admit datasets through integrity, annotation and provenance checks.
2. R0: retain paper-specific classes and original splits for published reproduction.
3. H1: measure within-domain, cross-dataset and released synthetic-to-real gaps.
4. H2: change controlled simulation components while keeping populations, budgets,
   representation and evaluation fixed.
5. CAST: source-domain effective-observation fitting and coverage. Transfer needs
   a separately authorized and frozen experiment; coverage is not transfer proof.

The earlier military tracked/wheeled benchmark, Milestone 7 gates and OpenClaw
workflow are retired from the active checkout. Their historical results and
protected-source status remain unchanged. Do not resume them implicitly.

## Implementation boundaries

- Reusable code belongs under `src/abvid`, including `src/abvid/cast`.
- Experiments are thin protocol/configuration layers. Do not import sibling
  experiment scripts through `sys.path`.
- Frozen historical runners use the external original-layout archive and exact
  recorded environments. Never edit their manifests/locks to accommodate a refactor.
- Large audio, models, features and generated results stay outside the checkout.
- Validate the canonical JSON config and resolve storage through `abvid.paths`.
- Preserve pinned third-party code, licenses and repair provenance in `vendor/`.

## Scientific guardrails

- Never randomly divide clips/windows from one original recording across roles.
  Group by original recording/session and retain all known provenance links.
- Preserve `recording_session`; unknown metadata stays `unknown`. Do not invent
  independent sessions from filenames, clips or diagnostic subgroups.
- Fit preprocessing statistics/classifiers on training data only. Do not tune on
  final target tests. Exposed target results remain exposed development evidence.
- Keep R0 original-paper classes distinct from the current car/truck-only study.
  Motorcycle remains excluded from the latter unless a new protocol changes it.
- Record domain, dataset version, split, seed, commit, checkpoint, configuration,
  metrics and group-level uncertainty with every research result.
- Distinguish recorded observations from clean physical source models, source
  coverage from transfer, diagnostic evidence from independent generalization,
  and numerical implementation effects from physical mechanisms.
- Literature: Markdown first, PDF authoritative; cite paper ID/page/section and
  distinguish author claims from interpretation. Use `unknown` when not established.
- Run focused regression/parity checks after refactoring. Do not launch a new
  scientific experiment merely to validate packaging.

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
