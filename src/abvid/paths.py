"""Local data/artifact locations, independent of experiment numerical settings."""
import json
import os
from pathlib import Path


def repository_root():
    """Find the checkout from the current directory or an explicit override."""
    if os.environ.get("ABVID_REPO_ROOT"):
        return Path(os.environ["ABVID_REPO_ROOT"]).expanduser().resolve()
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / "ABVID_ROADMAP.md").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Run from the ABVID checkout or set ABVID_REPO_ROOT")


def roots():
    repo = repository_root()
    local = repo / "configs/local.json"
    settings = json.loads(local.read_text()) if local.is_file() else {}
    defaults = {"data_root": repo.parent / "abvid-data",
                "artifact_root": repo.parent / "abvid-artifacts"}
    result = {"repo": repo}
    for name, fallback in defaults.items():
        location = Path(
            os.environ.get("ABVID_" + name.upper(), settings.get(name, fallback))
        ).expanduser()
        result[name.removesuffix("_root")] = (location if location.is_absolute() else repo / location).resolve()
    location = Path(os.environ.get("ABVID_ARCHIVE_ROOT", settings.get(
        "archive_root", result["artifact"] / "archive/pre-restructure-20261005"
    ))).expanduser()
    result["archive"] = (location if location.is_absolute() else repo / location).resolve()
    return result


def archive_root():
    return roots()["archive"]


def resolve(reference, *, must_exist=True):
    """Resolve a root-qualified path, rejecting absolute paths and traversal.

    Root symlinks are intentional: archived manifests retain original paths to
    the externally stored civilian data. Escaping a root through a symlink is
    allowed only for one of the other explicitly configured storage roots.
    """
    if not isinstance(reference, str):
        raise ValueError("Path reference must be a string")
    scheme, separator, relative = reference.partition(":")
    if not separator or scheme not in ("repo", "data", "artifact", "archive"):
        raise ValueError("Expected repo:, data:, artifact: or archive: path")
    part = Path(relative)
    if not relative or part.is_absolute() or ".." in part.parts:
        raise ValueError("Path must be relative and cannot contain '..'")
    locations = roots()
    path = (locations[scheme] / part).resolve()
    allowed = [locations[scheme]]
    if scheme == "archive":
        allowed += [locations["data"], locations["artifact"]]
    elif scheme == "data":
        # The civilian view can point at the archived real source directory;
        # frozen CAST guards require that original directory to be nonsymlinked.
        allowed += [locations["artifact"]]
    if not any(path.is_relative_to(base) for base in allowed):
        raise ValueError("Path escapes its configured storage root")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path
