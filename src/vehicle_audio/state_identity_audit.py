"""Read-only state/model inventory; never promotes review notes to training labels."""

import re
from collections import Counter, defaultdict

import yaml


def seconds(value):
    minutes, remainder = str(value).split(":")
    return 60 * int(minutes) + float(remainder)


def worksheet_segments(text, source_ids):
    """Read only requested source sections, retaining the exact reviewed model text."""
    result = {}
    for section in re.split(r"(?m)^### ", text):
        match = re.search(r"Source ID: `([^`]+)`", section)
        if not match or match[1] not in source_ids:
            continue
        source_id = match[1]
        if source_id in result:
            raise ValueError(f"duplicate worksheet source: {source_id}")
        blocks = re.findall(r"```yaml\n(.*?)```", section, re.S)
        if len(blocks) != 1:
            raise ValueError(f"expected one review block: {source_id}")
        result[source_id] = [
            {
                **segment,
                "start_seconds": seconds(segment["start"]),
                "end_seconds": seconds(segment["end"]),
            }
            for segment in yaml.safe_load(blocks[0])["segments"]
            if segment["decision"] == "keep"
        ]
    if set(result) != set(source_ids):
        raise ValueError("missing development source in worksheet")
    return result


def state_group(condition):
    # These are catalog-label groupings, not new visual or acoustic observations.
    if condition == "idle":
        return "idle_labeled"
    if condition in {"steady_speed", "accelerating", "decelerating"}:
        return "motion_labeled"
    if condition in {"mixed", "unknown"}:
        return "unresolved"
    raise ValueError(f"unsupported active condition: {condition}")


def audit_inventory(records, sources, reviews):
    """Requires provenance-validated development records from load_development."""
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
    intervals, inventory = [], []
    assigned = Counter()
    for source_id in sorted(by_source):
        source = sources[source_id]
        segments = sorted(source["condition_segments"], key=lambda s: s["start_seconds"])
        reviewed = reviews[source_id]
        if len(segments) != len(reviewed):
            raise ValueError(f"worksheet/catalog interval count mismatch: {source_id}")
        previous_end = -1
        model_groups = defaultdict(list)
        for segment in segments:
            start, end = segment["start_seconds"], segment["end_seconds"]
            if not 0 <= start < end or start < previous_end:
                raise ValueError("invalid/overlapping catalog intervals")
            previous_end = end
            matches = [
                r for r in reviewed
                if (r["start_seconds"], r["end_seconds"])
                == (start, end)
            ]
            if len(matches) != 1 or matches[0]["tag"] != segment["operating_condition"]:
                raise ValueError(f"worksheet/catalog interval mismatch: {source_id}")
            review = matches[0]
            model = review.get("vehicle")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"missing reviewed interval identity: {source_id}")
            windows = [
                r for r in by_source[source_id]
                if start <= r["window_start_seconds"] < r["window_end_seconds"] <= end
            ]
            for record in windows:
                if record["operating_condition"] != segment["operating_condition"]:
                    raise ValueError("manifest/catalog condition mismatch")
                assigned[record["sample_id"]] += 1
            row = {
                "source_id": source_id,
                "recording_session": source["recording_session"],
                "vehicle_class": source["vehicle_class"],
                "source_model_label": source["vehicle_model"],
                "reviewed_interval_model": model,
                "identity_evidence": "exact-boundary worksheet keep segment; catalog notes retained",
                "individual_vehicle_id": None,
                "engine_type": None,
                "rpm": None,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "operating_condition": segment["operating_condition"],
                "state_group": state_group(segment["operating_condition"]),
                "window_count": len(windows),
                "sample_ids": [r["sample_id"] for r in windows],
                "manifest_model_labels": sorted({str(r.get("vehicle_model")) for r in windows}),
                "catalog_notes": segment.get("notes", ""),
                "worksheet_notes": review.get("notes", ""),
            }
            intervals.append(row)
            model_groups[model].append(row)
        for model, rows in sorted(model_groups.items()):
            groups = {r["state_group"] for r in rows}
            inventory.append({
                "source_id": source_id,
                "recording_session": source["recording_session"],
                "vehicle_class": source["vehicle_class"],
                "reviewed_interval_model": model,
                "source_has_multiple_reviewed_models": len(model_groups) > 1,
                "same_model_state_pair_candidate": {"idle_labeled", "motion_labeled"} <= groups,
                "same_individual_pair_verified": False,
                "convoy_caveat": "convoy" in model.lower(),
                "seconds_by_group": {
                    group: round(sum(r["duration_seconds"] for r in rows if r["state_group"] == group), 6)
                    for group in ("idle_labeled", "motion_labeled", "unresolved")
                },
                "windows_by_condition": dict(sorted(Counter({
                    condition: sum(r["window_count"] for r in rows if r["operating_condition"] == condition)
                    for condition in {r["operating_condition"] for r in rows}
                }).items())),
            })
    if len(assigned) != len(records) or any(v != 1 for v in assigned.values()):
        raise ValueError("windows are duplicated or not assigned exactly once")
    pairs = [r for r in inventory if r["same_model_state_pair_candidate"]]
    return {
        "summary": {
            "windows": len(records),
            "approved_intervals": len(intervals),
            "approved_seconds": round(sum(r["duration_seconds"] for r in intervals), 6),
            "recording_sessions": len({r["recording_session"] for r in records}),
            "session_model_groups": len(inventory),
            "same_model_pair_candidate_sessions": len({r["recording_session"] for r in pairs}),
            "verified_same_individual_pairs": 0,
            "windows_in_multi_model_sources": sum(
                sum(r["windows_by_condition"].values()) for r in inventory
                if r["source_has_multiple_reviewed_models"]
            ),
            "windows_by_condition": dict(sorted(Counter(r["operating_condition"] for r in records).items())),
        },
        "inventory": inventory,
        "intervals": intervals,
    }


def render_report(result):
    lines = [
        "# Development state/identity inventory", "",
        "Metadata audit of native-real development audio, not model evaluation or new audiovisual review.",
        "Seconds count approved interval duration, not summed overlapping windows.",
        "Motion means a current steady_speed/accelerating/decelerating label; it is not a fresh observation.",
        "All individual-vehicle IDs and engine/RPM annotations remain unknown.", "",
        "| Class | Reviewed model / session | Idle seconds | Motion seconds | Unresolved seconds | Pair candidate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in result["inventory"]:
        times = row["seconds_by_group"]
        pair = "Same model only" if row["same_model_state_pair_candidate"] else "No"
        if row["convoy_caveat"]:
            pair += "; convoy"
        lines.append(
            f"| {row['vehicle_class']} | {row['reviewed_interval_model']} / `{row['recording_session']}` "
            f"| {times['idle_labeled']:g} | {times['motion_labeled']:g} | {times['unresolved']:g} | {pair} |"
        )
    lines += ["", "## Exact active intervals", "",
              "Source IDs, notes, original manifest model labels and sample IDs are retained in `audit.json`.", "",
              "| Source ID | Reviewed model | Start–end (seconds) | Condition | Windows |",
              "|---|---|---|---|---:|"]
    for row in result["intervals"]:
        lines.append(
            f"| `{row['source_id']}` | {row['reviewed_interval_model']} "
            f"| {row['start_seconds']:g}–{row['end_seconds']:g} "
            f"| {row['operating_condition']} | {row['window_count']} |"
        )
    return "\n".join(lines) + "\n"
