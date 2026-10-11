"""Canonical wrapper for frozen study commands and metadata preparation.

Scientific settings stay in the identified, hashed original contract. This
wrapper cannot override a split, a class mapping or a numerical parameter.
"""
import json
import re
from pathlib import Path

from .paths import resolve, roots
from .provenance.hashing import sha256

FIELDS = {"schema_version", "id", "phase", "description", "contract", "execution", "reports"}
PHASES = {"admission", "reproduction", "baseline", "simulation", "source_coverage"}
METADATA_SCRIPT = "repo:src/abvid/cast/grouped_design.py"
METADATA_MODULE = "abvid.cast.grouped_design"
METADATA_ARGUMENTS = ["freeze", "--config", "repo:configs/experiments/cast_grouped_coverage.json"]


def _validate_execution(execution):
    if set(execution) != {"mode", "environment", "script", "script_sha256", "module", "arguments"}:
        raise ValueError("Invalid execution contract")
    if not isinstance(execution["arguments"], list) or not all(isinstance(a, str) for a in execution["arguments"]):
        raise ValueError("Arguments must be an array of strings")
    if not re.fullmatch(r"[a-f0-9]{64}", execution["script_sha256"]):
        raise ValueError("Missing script SHA256")
    if execution["mode"] == "metadata_only":
        if (execution["environment"] != "cast" or execution["script"] != METADATA_SCRIPT or
            execution["module"] != METADATA_MODULE or execution["arguments"] != METADATA_ARGUMENTS):
            raise ValueError("Metadata preparation must identify the exact CAST grouped-design freeze command")
    elif execution["mode"] == "frozen_replay":
        if execution["environment"] not in {"cast", "reproduction"}:
            raise ValueError("Only explicitly identified frozen replay environments are supported")
        if execution["module"] not in (None, "cast_improvement.width_evaluation"):
            raise ValueError("Unsupported historical module entry point")
        if execution["module"] and (execution["environment"] != "cast" or
            execution["script"] != "archive:CAST/improvement/src/cast_improvement/width_evaluation.py"):
            raise ValueError("Module must identify its exact historical entry point")
        if not execution["script"].startswith("archive:"):
            raise ValueError("Frozen scripts must resolve within the original archive")
    else:
        raise ValueError("Only frozen_replay and metadata_only execution modes are supported")


def load(path, *, check_files=False):
    cfg = json.loads(Path(path).read_text())
    if set(cfg) != FIELDS or type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema or unexpected fields")
    if not isinstance(cfg["id"], str) or not re.fullmatch(r"[a-z0-9_]+", cfg["id"]):
        raise ValueError("Invalid experiment id")
    if cfg["phase"] not in PHASES or not isinstance(cfg["description"], str):
        raise ValueError("Invalid phase or description")
    contract, execution = cfg["contract"], cfg["execution"]
    if set(contract) != {"path", "sha256"} or not re.fullmatch(r"[a-f0-9]{64}", contract["sha256"]):
        raise ValueError("An exact contract path and SHA256 are required")
    _validate_execution(execution)
    if not isinstance(cfg["reports"], list) or not all(isinstance(p, str) for p in cfg["reports"]):
        raise ValueError("Reports must be an array of root-qualified paths")
    for reference in [contract["path"], execution["script"], *cfg["reports"]]:
        resolve(reference, must_exist=check_files)
    if check_files:
        for reference, expected in [(contract["path"], contract["sha256"]),
                                    (execution["script"], execution["script_sha256"])]:
            if sha256(resolve(reference)) != expected:
                raise ValueError(f"Changed frozen input: {reference}")
    return cfg


def command(cfg):
    _validate_execution(cfg["execution"])
    if cfg["execution"]["mode"] == "metadata_only":
        repo = roots()["repo"]
        return [str(repo / ".venv/bin/python"), "-m", METADATA_MODULE,
                *cfg["execution"]["arguments"]], repo
    archive = roots()["archive"]
    environment = ".venv" if cfg["execution"]["environment"] == "cast" else "experiments/reproduction/.venv"
    python = archive / environment / "bin/python"
    entry = ["-m", cfg["execution"]["module"]] if cfg["execution"]["module"] else [str(resolve(cfg["execution"]["script"]))]
    return [str(python), *entry, *cfg["execution"]["arguments"]], archive
