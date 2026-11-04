"""Create-only run records: one directory per run, provenance written before any output."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from abvid.paths import repository_root, resolve
from abvid.provenance.hashing import sha256

PACKAGES = ("numpy", "scikit-learn", "torch", "soundfile")


def _git(*args):
    return subprocess.run(["git", *args], cwd=repository_root(), capture_output=True,
                          text=True, check=True).stdout.strip()


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def start(audit, *, config_path, features, allow_dirty=False):
    """Create artifact:audits/<audit>/<UTC stamp>-<commit>/ and write provenance.json.

    A dirty checkout is refused unless allowed; when allowed, every dirty path is recorded.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = [line for line in _git("status", "--porcelain").splitlines() if line]
    if dirty and not allow_dirty:
        raise RuntimeError(f"{len(dirty)} uncommitted paths; commit first or pass allow_dirty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = resolve(f"artifact:audits/{audit}/{stamp}-{commit[:8]}", must_exist=False)
    run.mkdir(parents=True, exist_ok=False)
    write_once(run, "provenance.json", dict(
        audit=audit, started_utc=stamp, commit=commit, dirty_paths=dirty, argv=sys.argv,
        python=sys.version.split()[0], platform=platform.platform(),
        packages={name: _version(name) for name in PACKAGES},
        config=dict(path=str(config_path), sha256=sha256(config_path)), features=features))
    return run


def write_once(run, name, payload):
    with open(run / name, "x") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
