"""Audit the source roles and data gate for ABVID Benchmark v0.1."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vehicle_audio.collection import SourceSpec, load_catalog
from vehicle_audio.dataset import discover_target_recordings


def _load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("benchmark config must contain a mapping")
    required = {
        "benchmark_version",
        "benchmark_id",
        "source_catalog",
        "development",
        "consumed_locked_source_ids",
        "future_confirmation_source_ids",
        "tracks",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"benchmark config is missing fields: {sorted(missing)}")
    development = value["development"]
    if not isinstance(development, Mapping):
        raise TypeError("benchmark development configuration must be a mapping")
    if development.get("role_rule") != "admitted_to_corpus_true":
        raise ValueError("benchmark development.role_rule must be admitted_to_corpus_true")
    minimum = development.get("minimum_sessions_per_class")
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_sessions_per_class must be a positive integer")
    if list(development.get("classes", [])) != ["tracked", "wheeled"]:
        raise ValueError("benchmark classes must be [tracked, wheeled]")
    if development.get("group_field") != "recording_session":
        raise ValueError("benchmark group_field must be recording_session")
    for field in ("consumed_locked_source_ids", "future_confirmation_source_ids"):
        entries = value[field]
        if not isinstance(entries, list) or not entries or not all(
            isinstance(item, str) and item for item in entries
        ):
            raise ValueError(f"{field} must be a non-empty list of source IDs")
    return dict(value)


def _sessions_by_class(sources: Sequence[SourceSpec]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        if source.kind != "target" or source.admitted_to_corpus is not True:
            continue
        if source.vehicle_class not in {"tracked", "wheeled"}:
            continue
        if source.recording_session is None:
            raise ValueError(f"admitted target has no recording_session: {source.id}")
        grouped[source.vehicle_class].add(source.recording_session)
    return {name: sorted(grouped[name]) for name in ("tracked", "wheeled")}


def audit_benchmark(
    config_path: str | Path,
    *,
    targets_root: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_config(config_file)
    project_root = config_file.parent.parent
    catalog_path = project_root / str(config["source_catalog"])
    _, sources = load_catalog(catalog_path)
    by_id = {source.id: source for source in sources}

    consumed = set(config["consumed_locked_source_ids"])
    future = set(config["future_confirmation_source_ids"])
    overlap = sorted(consumed & future)
    unknown = sorted((consumed | future) - set(by_id))
    protected_admitted = sorted(
        source_id
        for source_id in consumed | future
        if source_id in by_id and by_id[source_id].admitted_to_corpus is not False
    )
    protected_classes = {
        role: sorted(by_id[source_id].vehicle_class for source_id in source_ids if source_id in by_id)
        for role, source_ids in (("consumed_locked", consumed), ("future_confirmation", future))
    }
    role_pairs_valid = all(
        classes == ["tracked", "wheeled"] for classes in protected_classes.values()
    )

    catalog_sessions = _sessions_by_class(sources)
    counts = {name: len(values) for name, values in catalog_sessions.items()}
    minimum = int(config["development"]["minimum_sessions_per_class"])
    data_gate_passed = all(counts[name] >= minimum for name in ("tracked", "wheeled"))

    local_check: dict[str, Any] = {
        "performed": False,
        "matches_catalog": None,
        "recording_sessions_by_class": None,
    }
    if targets_root is not None:
        target_path = Path(targets_root)
        if target_path.exists():
            local_sources = discover_target_recordings(target_path)
            local_grouped: dict[str, set[str]] = defaultdict(set)
            for source in local_sources:
                local_grouped[source.vehicle_class].add(source.recording_session)
            local_sessions = {
                name: sorted(local_grouped[name]) for name in ("tracked", "wheeled")
            }
            local_check = {
                "performed": True,
                "matches_catalog": local_sessions == catalog_sessions,
                "recording_sessions_by_class": local_sessions,
            }

    role_audit_passed = not overlap and not unknown and not protected_admitted and role_pairs_valid
    blocking_reasons: list[str] = []
    if not data_gate_passed:
        blocking_reasons.append(
            "development data gate requires at least "
            f"{minimum} independent sessions per class; found "
            f"tracked={counts['tracked']}, wheeled={counts['wheeled']}"
        )
    if not role_audit_passed:
        blocking_reasons.append("benchmark source roles overlap, are missing, or are not protected")
    if local_check["performed"] and not local_check["matches_catalog"]:
        blocking_reasons.append("local target discovery does not match catalog development roles")

    return {
        "benchmark_version": int(config["benchmark_version"]),
        "benchmark_id": str(config["benchmark_id"]),
        "source_catalog": str(catalog_path.relative_to(project_root)),
        "development": {
            "role_rule": "admitted_to_corpus_true",
            "minimum_sessions_per_class": minimum,
            "recording_sessions_by_class": catalog_sessions,
            "recording_session_counts": counts,
            "data_gate_passed": data_gate_passed,
        },
        "protected_roles": {
            "consumed_locked_source_ids": sorted(consumed),
            "future_confirmation_source_ids": sorted(future),
            "role_overlap": overlap,
            "unknown_source_ids": unknown,
            "protected_but_admitted_source_ids": protected_admitted,
            "classes_by_role": protected_classes,
            "role_audit_passed": role_audit_passed,
        },
        "local_collection": local_check,
        "tracks": config["tracks"],
        "ready_for_refreshed_development_evaluation": (
            data_gate_passed
            and role_audit_passed
            and (not local_check["performed"] or bool(local_check["matches_catalog"]))
        ),
        "blocking_reasons": blocking_reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v0_1.yaml"))
    parser.add_argument("--targets", type=Path, default=Path("data/targets"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_benchmark(args.config, targets_root=args.targets)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

