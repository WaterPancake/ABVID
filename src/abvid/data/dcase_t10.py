"""DCASE 2024 Task 10 (Bosch acoustic traffic monitoring) metadata admission.

Labels are per-minute pass-by counts for car and commercial vehicle (cv) by
direction. Only label tables, site metadata and archive directories are read;
no audio member is transferred. "cv" is a provider category, not the ABVID
"truck" class, and is kept under its own name.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .remote_zip import Volumes, central_directory, http_fetcher, read_member

DEVELOPMENT_RECORD = "10700792"
EVALUATION_RECORD = "11209838"
SITES = tuple(f"loc{i}" for i in range(1, 7))
COUNT_FIELDS = ("car_left", "car_right", "cv_left", "cv_right")
FIELDS = ("dow", "hour", "minute", *COUNT_FIELDS, "path", "split")
SPLITS = ("train", "val")
METADATA = r"loc[1-6]/(meta\.json|train\.csv|val\.csv)"
TRANSFER_LIMIT = 20_000_000
# Task page wording, retained verbatim: "car_left: number of passenger vehicles
# going left to right per minute". Direction is not used by the car/cv task.
LABEL_DEFINITION = ("Per-minute pass-by counts by provider category and travel direction; "
                    "label source (coil, radar, camera or human) is not disclosed per site.")


def record(record_id):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=60) as response:
        return json.load(response)


def archive_parts(files, stem):
    """Volume keys in disk order: stem.z01 ... stem.zNN, then stem.zip."""
    parts = sorted((k for k in files if re.fullmatch(re.escape(stem) + r"\.z\d\d", k)),
                   key=lambda k: int(k[-2:]))
    if stem + ".zip" not in files:
        raise ValueError(f"Missing final volume for {stem}")
    return parts + [stem + ".zip"]


def parse_labels(data, site, split):
    """Validate one label table; return rows with derived single-vehicle fields."""
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError(f"Unexpected label columns in {site}/{split}.csv")
    rows, seen = [], set()
    for index, raw in enumerate(reader):
        try:
            values = {k: int(raw[k]) for k in ("dow", "hour", "minute", *COUNT_FIELDS)}
        except (TypeError, ValueError):
            raise ValueError(f"Non-integer label in {site}/{split}.csv row {index}") from None
        if not (0 <= values["dow"] <= 6 and 0 <= values["hour"] <= 23 and 0 <= values["minute"] <= 59):
            raise ValueError(f"Invalid time label in {site}/{split}.csv row {index}")
        if any(values[k] < 0 for k in COUNT_FIELDS):
            raise ValueError(f"Negative count in {site}/{split}.csv row {index}")
        if raw["split"] != split or not re.fullmatch(rf"{split}/\d{{5}}\.flac", raw["path"]):
            raise ValueError(f"Path/split mismatch in {site}/{split}.csv row {index}")
        if raw["path"] in seen:
            raise ValueError(f"Duplicate path in {site}/{split}.csv")
        seen.add(raw["path"])
        car, cv = values["car_left"] + values["car_right"], values["cv_left"] + values["cv_right"]
        single = car + cv == 1
        rows.append({
            "segment_id": f"DCASE2024T10|{site}|{raw['path']}", "site": site, "split": split,
            "member": f"{site}/{raw['path']}", "row_index": index, **values,
            "car_total": car, "cv_total": cv, "vehicle_total": car + cv,
            "single_vehicle": single,
            "single_class": ("car" if car else "cv") if single else None,
            "single_direction": (next(k.split("_")[1] for k in COUNT_FIELDS if values[k]) if single else None),
            "provenance_group_id": f"DCASE2024T10|{site}",
            "recording_session": "unknown",
        })
    return rows


def mark_isolation(rows):
    """Flag single-vehicle minutes whose listed neighbours are both empty.

    Neighbours are the same site/split rows at minute-1 and minute+1 on the same
    day-of-week clock. Absent neighbours are unknown, so they do not qualify. A
    weekday clock listed more than once (different, unidentified days) is
    ambiguous: neither it nor its neighbours can establish isolation.
    """
    index = defaultdict(list)
    for r in rows:
        index[(r["site"], r["split"], r["dow"], r["hour"] * 60 + r["minute"])].append(r)
    for (site, split, dow, clock), group in index.items():
        for r in group:
            r["weekday_clock_repeated"] = len(group) > 1
            neighbours = [index.get((site, split, dow, clock + step), []) for step in (-1, 1)]
            r["neighbours_listed"] = sum(len(n) == 1 for n in neighbours)
            r["isolated_single_vehicle"] = bool(
                r["single_vehicle"] and len(group) == 1 and
                all(len(n) == 1 and n[0]["vehicle_total"] == 0 for n in neighbours))
    return rows


def check_members(rows, members, site):
    """Every label row must name exactly one listed audio member, and vice versa."""
    audio = {m.name for m in members if m.is_audio and m.name.startswith(site + "/")}
    named = {r["member"] for r in rows}
    if named != audio:
        raise ValueError(f"{site}: label/audio inventory mismatch "
                         f"({len(named - audio)} unlisted, {len(audio - named)} unlabelled)")
    return len(audio)


def summarize(rows):
    """Per-site/split counts; support is reported, never pooled into a score."""
    keys = ("minutes", "repeated_weekday_clock_minutes", "vehicle_absent_minutes",
            "both_neighbours_listed", "car_passbys", "cv_passbys", "single_car", "single_cv",
            "isolated_single_car", "isolated_single_cv", "max_vehicles_per_minute")
    out = defaultdict(lambda: Counter(dict.fromkeys(keys, 0)))
    for r in rows:
        for key in [(r["site"], r["split"]), (r["site"], "all")]:
            c = out[key]
            c["minutes"] += 1
            c["both_neighbours_listed"] += r["neighbours_listed"] == 2
            c["repeated_weekday_clock_minutes"] += r["weekday_clock_repeated"]
            c["vehicle_absent_minutes"] += r["vehicle_total"] == 0
            c["car_passbys"] += r["car_total"]
            c["cv_passbys"] += r["cv_total"]
            if r["single_vehicle"]:
                c["single_" + r["single_class"]] += 1
                if r["isolated_single_vehicle"]:
                    c["isolated_single_" + r["single_class"]] += 1
            c["max_vehicles_per_minute"] = max(c["max_vehicles_per_minute"], r["vehicle_total"])
    return {f"{site}|{split}": dict(sorted(c.items())) for (site, split), c in sorted(out.items())}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def admit(output, *, fetcher=http_fetcher, records=None, limit=TRANSFER_LIMIT):
    """Fetch metadata members and archive directories into a new directory.

    ``records`` maps record IDs to Zenodo API JSON (fetched when omitted). The
    output directory must not exist; nothing is overwritten.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = records or {rid: record(rid) for rid in (DEVELOPMENT_RECORD, EVALUATION_RECORD)}
    started, hashes, receipts, all_rows, inventories = time.time(), {}, [], [], {}
    for rid, stems in [(DEVELOPMENT_RECORD, SITES + ("simulation",)), (EVALUATION_RECORD, SITES)]:
        files = {f["key"]: f for f in records[rid]["files"]}
        hashes[f"zenodo_{rid}.json"] = _write(output / f"zenodo_{rid}.json", _json(records[rid]))
        for stem in stems:
            parts = archive_parts(files, stem)
            volumes = Volumes([files[k]["size"] for k in parts],
                              fetcher([files[k]["links"]["self"] for k in parts]), max_bytes=limit)
            members, directory = central_directory(volumes)
            inventory = [{"name": m.name, "size": m.size, "compressed_size": m.compressed_size,
                          "crc32": f"{m.crc32:08x}", "audio": m.is_audio} for m in members]
            key = f"{rid}/{stem}"
            inventories[key] = inventory
            hashes[f"{key}/inventory.json"] = _write(output / key / "inventory.json", _json(inventory))
            if rid == DEVELOPMENT_RECORD and stem in SITES:
                site_rows = []
                for m in members:
                    if re.fullmatch(METADATA, m.name):
                        data = read_member(volumes, m, allow=METADATA)
                        hashes[f"{key}/{m.name}"] = _write(output / key / m.name, data)
                        if m.name.endswith(".csv"):
                            site_rows += parse_labels(data, stem, m.name.split("/")[1][:-4])
                        else:
                            json.loads(data)
                if {r["split"] for r in site_rows} != set(SPLITS):
                    raise ValueError(f"{stem}: expected train and val label tables")
                check_members(site_rows, members, stem)
                all_rows += site_rows
            elif rid == EVALUATION_RECORD and any(not m.is_audio and m.name.endswith(".csv") for m in members):
                raise ValueError("Evaluation archive unexpectedly contains label tables; review before use")
            receipts.append({"record": rid, "archive": stem, "volumes": parts,
                             "volume_sizes": [files[k]["size"] for k in parts],
                             "volume_md5": [files[k]["checksum"] for k in parts],
                             "directory": directory, "bytes_transferred": sum(volumes.transferred),
                             "requests": volumes.requests,
                             "audio_members_listed": sum(m.is_audio for m in members),
                             "audio_bytes_requested": 0})
    mark_isolation(all_rows)
    manifest = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in all_rows)
    hashes["segments.jsonl"] = _write(output / "segments.jsonl", manifest)
    summary = {"label_definition": LABEL_DEFINITION, "sites": summarize(all_rows),
               "segments": len(all_rows),
               "evaluation_record_label_tables": 0,
               "simulation_audio_members": sum(i["audio"] for i in inventories[f"{DEVELOPMENT_RECORD}/simulation"])}
    hashes["summary.json"] = _write(output / "summary.json", _json(summary))
    receipt = {"started_unix": started, "elapsed_s": time.time() - started,
               "bytes_transferred": sum(r["bytes_transferred"] for r in receipts),
               "audio_bytes_requested": 0, "archives": receipts, "transfer_limit_per_archive": limit}
    hashes["receipt.json"] = _write(output / "receipt.json", _json(receipt))
    _write(output / "hashes.json", _json(hashes))
    return summary, receipt


def main(argv=None):
    import argparse
    from ..paths import resolve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="new data: directory, e.g. data:DCASE2024_T10/metadata_v1")
    args = parser.parse_args(argv)
    summary, receipt = admit(resolve(args.output, must_exist=False))
    print(json.dumps({"segments": summary["segments"],
                      "bytes_transferred": receipt["bytes_transferred"],
                      "audio_bytes_requested": receipt["audio_bytes_requested"]}))


if __name__ == "__main__":
    main()
