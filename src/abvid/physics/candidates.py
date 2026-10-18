"""Metadata-only RPM readiness audit of already admitted ABVID datasets.

Reads the pinned historical JSONL manifest, never audio or feature caches.
"""
import argparse
from collections import Counter
import json
import numpy as np

from abvid.paths import resolve
from .data import read_metadata, save, sha, verify
from .config import load

MANIFEST = "archive:experiments/h1/frozen/h1_common_budget_v1.3/admitted.jsonl"
MANIFEST_SHA256 = "ee8b67b002a4528b036e6d2a3ab6f8f6015cd59fffbbc27214ad2786a7d31f96"


def audit():
    path = resolve(MANIFEST)
    if sha(path) != MANIFEST_SHA256:
        raise ValueError("Historical admission metadata hash mismatch")
    datasets = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        group = datasets.setdefault(row["dataset_id"], {"rows": 0, "classes": Counter(),
            "releases": set(), "groups": set(), "explicit_rpm_fields": set(),
            "explicit_torque_fields": set(), "vehicle_speed_values": Counter(),
            "recording_session_values": Counter()})
        group["rows"] += 1
        group["classes"][row["canonical_class"]] += 1
        group["releases"].add(row["release_id"])
        group["groups"].add(row["provenance_group_id"])
        group["explicit_rpm_fields"].update(k for k in row if "rpm" in k.lower() or "engine_speed" in k.lower())
        group["explicit_torque_fields"].update(k for k in row if "torque" in k.lower())
        group["vehicle_speed_values"][str(row.get("vehicle_speed", "unknown"))] += 1
        group["recording_session_values"][str(row.get("recording_session", "unknown"))] += 1
    for group in datasets.values():
        for key, value in list(group.items()):
            if isinstance(value, set):
                group[key] = sorted(value)
            elif isinstance(value, Counter):
                group[key] = dict(value)
        group["disposition"] = "retain_existing_roles_no_RPM_calibration_from_this_manifest"
    cfg = load("repo:configs/physics/engine_v0.json")
    lock = json.loads(resolve(cfg["input_lock"]).read_text())
    root = resolve(cfg["data_root"])
    for entry in lock["metadata"]:
        verify(root/entry["path"], entry)
    released = list(read_metadata(root, lock).values())
    numeric = np.array([[float(v) for k, v in r.items() if k != "filename"] for r in released])
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite published metadata")
    release_summary = {"metadata_rows": len(released), "all_numeric_fields_finite": True,
        "inventory_bytes_including_metadata": lock["inventory_bytes"],
        "duration_hours_from_metadata": sum(float(r["duration_sec"]) for r in released)/3600,
        **{key: function(float(r[key]) for r in released)
           for key, function in (("rpm_min", min), ("rpm_max", max), ("torque_min", min), ("torque_max", max))}}
    return {"scope": "existing_admission_and_pinned_release_metadata_only", "manifest": MANIFEST,
            "manifest_sha256": MANIFEST_SHA256, "datasets": datasets,
            "procedural_engine_release": release_summary, "implementation_sha256": sha(__file__),
            "audio_or_features_opened": False,
            "limit": "Missing structured RPM/torque annotations are not proof that no external metadata exists; no estimates invented"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.output.startswith("artifact:physics_engine/"):
        raise ValueError("Use an additive artifact:physics_engine/ output")
    result = audit()
    save(resolve(args.output, must_exist=False), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
