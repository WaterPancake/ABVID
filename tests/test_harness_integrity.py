"""Harness-integrity canary tests (anti-reward-hacking layer).

These tests enforce the mechanical contract that a coding agent (Codex or any
other) cannot silently weaken the verification layer:

1. the collected test count matches the committed per-environment baseline
   (no silent test deletion or padding);
2. frozen files (lockfile, protocol registry, experiment configs) hash to
   their committed values — quietly editing them to dodge a failure is caught.

The two dependency environments (uv extras "cast" and "baseline") conflict by
design and collect different test sets, so the baseline records one expected
count per environment. The active environment is detected from installed
dependencies (librosa/sklearn exist only under the baseline extra) or from the
ABVID_TEST_ENV variable set explicitly by CI.

Maintenance rule (human-only): when a change legitimately adds or removes
tests or a frozen file legitimately changes, run
scripts/refresh_harness_baseline.py for BOTH environments and commit the
updated tests/harness_baseline.json in the SAME commit, calling it out in the
PR description. A red canary always means "suite shape changed" — a human
must review the tests/ diff before merging.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = REPO_ROOT / "tests" / "harness_baseline.json"

# Test modules that import baseline-extra-only dependencies (librosa/sklearn);
# they do not collect under the cast extra and are ignored there.
BASELINE_ONLY_MODULES = (
    "tests/test_h1.py",
    "tests/test_simulation.py",
    "tests/test_benchmark_prototype.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        pytest.fail(
            "tests/harness_baseline.json is missing — the integrity canary has "
            "no baseline. Restore it from git history or regenerate it with "
            "scripts/refresh_harness_baseline.py (human review required)."
        )
    return json.loads(BASELINE_FILE.read_text())


def detect_environment() -> str:
    """Which dependency environment is active: 'cast' or 'baseline'."""
    explicit = os.environ.get("ABVID_TEST_ENV")
    if explicit in ("cast", "baseline"):
        return explicit
    try:
        import librosa  # noqa: F401  (baseline extra only)
        import sklearn  # noqa: F401
    except ImportError:
        return "cast"
    return "baseline"


def selection_args(environment: str) -> list[str]:
    """The exact pytest selection CI grades — the canary counts the same set."""
    args = ["-m", "not archive"]
    if environment == "cast":
        args += [f"--ignore={m}" for m in BASELINE_ONLY_MODULES]
    return args


def _collected_test_count() -> int:
    """Run pytest --collect-only in a subprocess; return the number of tests."""
    environment = detect_environment()
    command = [
        sys.executable, "-m", "pytest",
        "--collect-only", "-q", "--no-header",
        *selection_args(environment),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=300,
    )
    # "-q" prints per-file summaries: "tests/foo.py: 12"
    per_file = re.findall(r"^(\S+\.py): (\d+)$", result.stdout, re.M)
    if per_file:
        return sum(int(count) for _, count in per_file)
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if match:
        return int(match.group(1))
    pytest.fail(
        f"Could not parse pytest --collect-only output (env={environment}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def test_harness_baseline_file_exists_and_is_valid():
    baseline = _load_baseline()
    assert isinstance(baseline, dict)
    assert {"environments", "file_hashes"} <= set(baseline.keys()), (
        "harness baseline must record environments and file_hashes"
    )
    environments = baseline["environments"]
    assert {"cast", "baseline"} <= set(environments.keys()), (
        "baseline must record a test_count for BOTH environments"
    )
    assert all(
        isinstance(environments[name].get("test_count"), int)
        for name in ("cast", "baseline")
    ), "each environment entry must have an integer test_count"
    assert isinstance(baseline["file_hashes"], dict) and baseline["file_hashes"]


def test_collected_test_count_matches_baseline():
    """No silent test deletion (count shrinks) or padding (count inflates)."""
    environment = detect_environment()
    expected = _load_baseline()["environments"][environment]["test_count"]
    collected = _collected_test_count()
    assert collected == expected, (
        f"Test count drifted from baseline in the {environment} environment: "
        f"got {collected}, expected {expected}. If tests were legitimately "
        "added/removed, regenerate tests/harness_baseline.json for BOTH "
        "environments in the same commit and call it out in the PR."
    )


def test_frozen_files_match_hashes():
    """Frozen files must hash to their committed baseline values.

    Covers the lockfile and the experiment protocol registry — the files a
    reward-hacking agent would most benefit from quietly editing.
    """
    file_hashes = _load_baseline()["file_hashes"]
    for rel_path, expected_hash in sorted(file_hashes.items()):
        path = REPO_ROOT / rel_path
        assert path.exists(), f"frozen file missing: {rel_path}"
        actual = _sha256(path)
        assert actual == expected_hash, (
            f"frozen file hash mismatch: {rel_path}\n"
            f"  expected {expected_hash}\n  got      {actual}\n"
            "If this change is legitimate, regenerate the baseline in the same "
            "commit and call it out in the PR."
        )
