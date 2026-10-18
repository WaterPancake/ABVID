"""Pinned, bounded access to the official Procedural Engine Sounds release."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

DATASET = "rdoerfler/procedural-engine-sounds"
REVISION = "e57c44ab64e8528a7b37fc693b553bf4b66440e7"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
SETS = {letter + ("_full_set" if letter in "ABCD" else "_large_set"):
        {"engine": engine, "cylinders_assumed": cylinders}
        for letter, engine, cylinders in zip("ABCDEFGH", ["E1", "E2", "E3", "E3", "E3", "E3", "E3", "E2"], [4, 8, 8, 8, 8, 8, 8, 8])}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_path(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or not part.parts:
        raise ValueError("Unsafe release path")
    path = (Path(root) / str(part)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Release path escapes storage")
    return path


def verify(path, row):
    if Path(path).stat().st_size != row["size"]:
        raise ValueError(f"Size mismatch: {row['path']}")
    if "lfs" in row:
        actual, expected = sha(path), row["lfs"]["oid"]
    else:
        contents = Path(path).read_bytes()
        actual = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
        expected = row["oid"]
    if actual != expected:
        raise ValueError(f"Content hash mismatch: {row['path']}")


def fetch(root, row, *, download):
    path = safe_path(root, row["path"])
    if path.exists():
        verify(path, row)  # Corruption never triggers a silent replacement.
        return path
    if not download:
        raise FileNotFoundError(f"Missing pinned input; use --download: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(URL + row["path"], timeout=60) as response, temporary.open("xb") as stream:
        count = 0
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > row["size"]:
                raise ValueError("Download exceeds pinned size")
            stream.write(block)
    verify(temporary, row)
    temporary.rename(path)
    return path


def inventory():
    """Read all remote file metadata, never audio. Pin every page to one revision."""
    prefix = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
    url, rows = prefix + "?recursive=true&limit=1000", []
    seen = set()
    while url:
        if not url.startswith(prefix + "?") or url in seen:
            raise ValueError("Unexpected inventory pagination")
        seen.add(url)
        with urllib.request.urlopen(url, timeout=60) as response:
            rows.extend(json.load(response))
            next_page = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
        url = next_page[1] if next_page else None
    files = sorted((r for r in rows if r["type"] == "file"), key=lambda r: r["path"])
    if len({r["path"] for r in files}) != len(files):
        raise ValueError("Duplicate inventory paths")
    return files


def make_lock(files):
    by_set = {name: {} for name in SETS}
    metadata = []
    for row in files:
        if row["path"].endswith(".wav"):
            p = PurePosixPath(row["path"])
            by_set[p.parent.name][p.name.split("_")[0]] = row
        else:
            metadata.append(row)
    common = sorted(set.intersection(*(set(rows) for rows in by_set.values())))
    if len(common) < 2:
        raise ValueError("Insufficient shared filename candidates")
    chosen = [common[0], common[len(common) // 2]]
    selected = []
    for name, rows in by_set.items():
        for identifier in chosen:
            row = dict(rows[identifier])
            row.update(SETS[name], capture_configuration=name, recording_session="unknown",
                       original_control_session="unknown", filename_trace_candidate=identifier,
                       role="synthetic_development_reconstruction_only",
                       conservative_group="entire_release_shared_control_and_timbre_ancestry")
            selected.append(row)
    if len(selected) != 16 or sum(r["size"] for r in selected) > 100_000_000:
        raise ValueError("Bounded download limit violated")
    return {"schema_version": 1, "dataset": DATASET, "revision": REVISION,
            "domain": "synthetic_recording_derived", "license": "CC-BY-NC-4.0",
            "inventory_sha256": hashlib.sha256(canonical(files)).hexdigest(),
            "inventory_audio_count": sum(len(v) for v in by_set.values()),
            "inventory_bytes": sum(r["size"] for r in files),
            "counts_by_set": {k: len(v) for k, v in by_set.items()},
            "selection": "lexicographic_first_and_midpoint_of_filename_prefix_intersection_all_eight_sets",
            "selected_prefixes": chosen, "selected": selected, "metadata": metadata,
            "physical_cylinder_count": "assumed_from_author_timbre_description_not_independently_verified",
            "splits": "one_conservative_development_role_no_validation_or_test"}


def read_metadata(root, lock):
    rows = {}
    for entry in lock["metadata"]:
        if entry["path"].endswith("_stats.csv"):
            name = PurePosixPath(entry["path"]).name.removesuffix("_stats.csv")
            for row in csv.DictReader(io.StringIO(safe_path(root, entry["path"]).read_text())):
                path = f"dataset/audio/{name}/{row['filename']}"
                if path in rows:
                    raise ValueError("Duplicate metadata row")
                rows[path] = row
    return rows


def decode(path, row):
    """Verify bytes BEFORE decode; annotations never enter the audio downmix."""
    verify(path, row)
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (48000, 4, "PCM_16"):
        raise ValueError("Unexpected annotated WAV format")
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    if not np.isfinite(data).all() or len(data) < 96000:
        raise ValueError("Nonfinite or short audio")
    rpm, torque = data[:, 2] * 10000, data[:, 3] * 1000
    if rpm.min() < 0 or rpm.max() > 10000 or torque.min() < -200 or torque.max() > 1000:
        raise ValueError("Controls outside declared prototype range")
    if np.sqrt(np.mean(data[:, :2] ** 2)) < 1e-7:
        raise ValueError("Silent audio")
    return data[:, :2], rpm, torque, sr


def audit(root, lock, files, *, download=False):
    started = time.perf_counter()
    if make_lock(files) != lock:
        raise ValueError("Inventory/selection lock mismatch")
    for row in lock["metadata"]:
        fetch(root, row, download=download)
    metadata = read_metadata(root, lock)
    audio_rows = {r["path"]: r for r in files if r["path"].endswith(".wav")}
    if set(audio_rows) != set(metadata):
        raise ValueError("Inventory and full metadata filenames differ")
    for key, meta in metadata.items():
        if int(meta["samplerate"]) != 48000 or float(meta["duration_sec"]) < 2:
            raise ValueError("Metadata sample-rate/duration violation")
        if abs(float(meta["size_MB"]) * 1024**2 - audio_rows[key]["size"]) > 2:
            raise ValueError("Metadata/inventory size mismatch")
    results, failures = [], []
    for row in lock["selected"]:
        try:
            path = fetch(root, row, download=download)
            audio, rpm, torque, sr = decode(path, row)
            control_digest = hashlib.sha256(np.column_stack((rpm, torque)).astype("<f8").tobytes()).hexdigest()
            numeric = {"duration_sec": len(audio)/sr}
            for name, x in (("rpm", rpm), ("torque", torque)):
                for suffix, func in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
                    numeric[f"{name}_{suffix}"] = float(func(x))
            error = max(abs(v - float(metadata[row["path"]][k])) for k, v in numeric.items())
            if error > 1e-6:
                raise ValueError(f"Embedded controls disagree with metadata: {error}")
            results.append({"path": row["path"], "sha256": sha(path), "control_sha256": control_digest,
                            "metadata_max_abs_error": error, "samples": len(audio), **numeric,
                            "audio_peak": float(np.max(np.abs(audio))),
                            "audio_clipped_samples": int(np.sum(np.abs(audio) >= 32767/32768)),
                            "recording_session": "unknown", "role": row["role"]})
        except (OSError, ValueError) as exc:
            failures.append({"path": row["path"], "error": str(exc), "type": type(exc).__name__})
    controls = {}
    for row in results:
        controls.setdefault(row["control_sha256"], []).append(row["path"])
    return {"status": "pass" if not failures else "fail", "scope": "full_inventory_and_metadata_bounded_audio_decode",
            "dataset": DATASET, "revision": REVISION, "full_metadata_rows": len(metadata),
            "decoded_files": len(results), "decoded_bytes": sum(r["size"] for r in lock["selected"] if any(v["path"] == r["path"] for v in results)),
            "files": results, "identical_control_groups": list(controls.values()),
            "failures": failures, "seconds": time.perf_counter()-started,
            "admission": "bounded_synthetic_development_only_not_real_physical_validation"}
