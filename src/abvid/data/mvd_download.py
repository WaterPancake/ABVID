"""Acquire the original public MVD release; never admit, split or score audio.

Uses the author's original Drive folder, a complete frozen HTML inventory,
provider CRC32C when supplied, local SHA-256, and WAV header checks only.
Original files stay outside the checkout. Existing bytes are never replaced.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import struct
import threading
import time
import wave

import requests

from abvid.paths import resolve, roots
from abvid.protocol import load
from abvid.provenance.hashing import sha256


REPOSITORY = "https://github.com/Ashhad785/MVD"
COMMIT = "bd797a0049b120e43d90ceb9da30ac3aee4e7979"
FOLDER_ID = "1kB-rEWlXDwuIPmEKp_qszpTqWRy_Gtmh"
FOLDER_URL = "https://drive.google.com/drive/folders/" + FOLDER_ID
PUBLISHED_URL = "https://drive.google.com/drive/folders/1VhPQAoAWs4TFDOEXauBQIZrCCM4_0qt9"
EXPECTED_COUNTS = {"car": 1005, "truck": 1077, "motorbike": 1122, "none": 1025}
LABELS = {"C": "car", "T": "truck", "M": "motorbike", "N": "none"}
PATTERN = re.compile(r"Recording_(\d+[A-Za-z]*)_([OAMH])_([CTMN])\.wav")
LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def crc_table():
    values = []
    for i in range(256):
        x = i
        for _ in range(8):
            x = (x >> 1) ^ (0x82F63B78 if x & 1 else 0)
        values.append(x)
    return tuple(values)


CRC_TABLE = crc_table()


def crc32c(data):
    value = 0xFFFFFFFF
    for byte in data:
        value = CRC_TABLE[(value ^ byte) & 255] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = dict(attrs).get("href", "")
            match = re.match(r"https://drive\.google\.com/file/d/([\w-]+)/view", url)
            if match:
                self.current = {"drive_file_id": match[1], "provider_url": url, "text": []}

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            row, self.current = self.current, None
            row["filename"] = "".join(row.pop("text")).strip()
            match = PATTERN.fullmatch(row["filename"])
            if not match:
                raise ValueError(f"Unexpected original release filename: {row['filename']}")
            row.update(provider_recording_token=match[1], microphone_token=match[2],
                       label_token=match[3], class_label_from_filename=LABELS[match[3]],
                       recording_session="unknown", physical_vehicle_id="unknown", site_id="unknown")
            self.rows.append(row)


def parse_inventory(data):
    parser = Inventory()
    parser.feed(data.decode("utf-8"))
    rows = sorted(parser.rows, key=lambda r: r["filename"])
    if (len(rows) != 4229 or len({r["filename"] for r in rows}) != len(rows) or
            len({r["drive_file_id"] for r in rows}) != len(rows)):
        raise ValueError("Expected the complete original 4,229-file inventory without duplicates")
    counts = Counter(r["class_label_from_filename"] for r in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Original-release class counts differ: {dict(counts)}")
    return rows


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "ABVID-MVD-research-download/1.0"
    return LOCAL.session


def inspect(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Response is not RIFF/WAVE")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("RIFF length does not match downloaded bytes")
    with wave.open(BytesIO(data), "rb") as audio:
        header = {"sample_rate_hz": audio.getframerate(), "channels": audio.getnchannels(),
                  "sample_width_bytes": audio.getsampwidth(), "frames": audio.getnframes(),
                  "compression": audio.getcomptype()}
    header["duration_seconds_from_header"] = header["frames"] / header["sample_rate_hz"]
    return header


def transfer(row, output, previous):
    fid, filename = row["drive_file_id"], row["filename"]
    path = output / "audio" / filename
    prior = previous.get(fid)
    if path.exists() and prior and sha256(path) == prior["sha256"]:
        data = path.read_bytes()
        if len(data) != prior["bytes"] or inspect(data) != prior["header"]:
            raise ValueError("Existing downloaded file differs from its receipt")
        return {**prior, "reused_verified_local_file": True}
    errors = []
    for attempt in range(4):
        try:
            response = session().get("https://drive.usercontent.google.com/download",
                                     params={"id": fid, "export": "download", "confirm": "t"}, timeout=(15, 45))
            response.raise_for_status()
            data = response.content
            if len(data) > 4_000_000:
                raise ValueError("Unexpectedly large MVD clip; no automatic expansion of the release")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != len(data):
                raise ValueError("HTTP content length differs")
            header = inspect(data)
            provider_hash = {}
            for item in response.headers.get("x-goog-hash", "").split(","):
                key, sep, value = item.strip().partition("=")
                if sep and key in ("crc32c", "md5"):
                    provider_hash[key] = value
            if "crc32c" in provider_hash:
                expected = int.from_bytes(base64.b64decode(provider_hash["crc32c"], validate=True), "big")
                if crc32c(data) != expected:
                    raise ValueError("Google CRC32C differs")
            if "md5" in provider_hash:
                if hashlib.md5(data).digest() != base64.b64decode(provider_hash["md5"], validate=True):
                    raise ValueError("Google MD5 differs")
            digest = hashlib.sha256(data).hexdigest()
            if path.exists():
                if sha256(path) != digest:
                    raise ValueError("Existing bytes differ; replacement is forbidden")
            else:
                temporary = path.with_suffix(".wav.part")
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, path)  # Atomic publication, without replacing an existing file.
                temporary.unlink()
            return {"drive_file_id": fid, "filename": filename, "status": "downloaded_header_verified",
                    "bytes": len(data), "sha256": digest, "header": header,
                    "provider_hash": provider_hash, "provider_crc32c_verified": "crc32c" in provider_hash,
                    "http_content_type": response.headers.get("Content-Type"),
                    "http_last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"),
                    "attempt": attempt + 1, "downloaded_utc": now(), "reused_verified_local_file": False}
        except (requests.RequestException, ValueError, OSError, wave.Error) as error:
            errors.append({"attempt": attempt + 1, "type": type(error).__name__, "message": str(error)[:500]})
            if attempt < 3:
                time.sleep(2 ** attempt)
    return {"drive_file_id": fid, "filename": filename, "status": "failed", "errors": errors}


def download(output_reference, inventory_html, workers, resume):
    if crc32c(b"123456789") != 0xE3069283:
        raise ValueError("CRC32C known-answer check failed")
    load(resolve("repo:configs/experiments/admission.json"), check_files=True)
    output = resolve(output_reference, must_exist=False)
    if (not output_reference.startswith("data:MVD/") or output.is_relative_to(roots()["repo"]) or
            output.is_relative_to(roots()["archive"])):
        raise ValueError("MVD downloads must use a new data:MVD/ directory outside the checkout/archive")
    snapshot = Path(inventory_html).read_bytes()
    rows = parse_inventory(snapshot)
    if shutil.disk_usage(roots()["data"]).free < 1_500_000_000:
        raise ValueError("Insufficient free space for the expected release and receipts")
    if resume:
        plan = json.loads((output / "download_plan.json").read_text())
        if plan["inventory_html_sha256"] != hashlib.sha256(snapshot).hexdigest():
            raise ValueError("Resume inventory differs from the frozen download plan")
        if plan["script_sha256"] != sha256(Path(__file__)):
            raise ValueError("Resume implementation differs from the frozen download plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "audio").mkdir()
        (output / "folder_inventory.html").write_bytes(snapshot)
        (output / "inventory.json").write_bytes(encoded(rows))
        plan = {"dataset": "MVD_original", "status": "download_only_pending_admission", "created_utc": now(),
                "repository": REPOSITORY, "repository_commit": COMMIT, "published_folder": PUBLISHED_URL,
                "actual_audio_folder": FOLDER_URL, "expected_files": 4229, "expected_class_counts": EXPECTED_COUNTS,
                "inventory_html_sha256": hashlib.sha256(snapshot).hexdigest(),
                "inventory_sha256": sha256(output / "inventory.json"), "script_sha256": sha256(Path(__file__)),
                "license_for_audio": "unknown", "recording_sessions": "unknown", "admitted": False,
                "model_access": False, "audio_samples_decoded": False, "header_inspection_only": True}
        (output / "download_plan.json").write_bytes(encoded(plan))
        (output / "download_implementation.py").write_bytes(Path(__file__).read_bytes())
    if sha256(output / "inventory.json") != plan["inventory_sha256"]:
        raise ValueError("Frozen file inventory changed")
    previous = {}
    log = output / "transfers.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "downloaded_header_verified":
                previous[r["drive_file_id"]] = r
    results = []
    started = time.perf_counter()
    with log.open("a") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(transfer, row, output, previous): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            if len(results) % 100 == 0 or result["status"] == "failed":
                print(json.dumps({"complete": len(results), "total": len(rows), "failed": sum(r["status"] == "failed" for r in results),
                                  "MiB": round(sum(r.get("bytes", 0) for r in results) / 2 ** 20, 1),
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    good = {r["filename"]: r for r in results if r["status"] == "downloaded_header_verified"}
    failures = [r for r in results if r["status"] == "failed"]
    for name, record in good.items():
        if sha256(output / "audio" / name) != record["sha256"]:
            raise ValueError("Final local SHA256 verification failed")
    if {p.name for p in (output / "audio").glob("*.wav")} != set(good):
        raise ValueError("Downloaded WAV inventory differs from successful transfers")
    summary = {"dataset": "MVD_original", "status": "complete_download_pending_admission" if not failures else "partial_download_pending_admission",
               "completed_utc": now(), "download_root": output_reference, "files": len(good), "expected_files": 4229,
               "failed": len(failures), "bytes": sum(r["bytes"] for r in good.values()),
               "class_counts": dict(Counter(r["class_label_from_filename"] for r in rows if r["filename"] in good)),
               "microphone_counts": dict(Counter(r["microphone_token"] for r in rows if r["filename"] in good)),
               "provider_crc32c_verified": sum(r["provider_crc32c_verified"] for r in good.values()),
               "all_local_sha256_rechecked": True, "unique_sha256": len({r["sha256"] for r in good.values()}),
               "header_profiles": dict(Counter(json.dumps(r["header"], sort_keys=True) for r in good.values())),
               "failures": failures, "elapsed_s": time.perf_counter() - started,
               "inventory_sha256": plan["inventory_sha256"], "script_sha256": plan["script_sha256"],
               "admitted": False, "license_for_audio": "unknown", "recording_session": "unknown",
               "model_fitting_or_feature_extraction": False}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = output / f"receipt_{stamp}.json"
    receipt.write_bytes(encoded(summary))
    manifest = output / f"files_{stamp}.jsonl"
    manifest.write_bytes(b"".join(json.dumps(good[name], sort_keys=True, allow_nan=False).encode() + b"\n" for name in sorted(good)))
    print(json.dumps({"status": summary["status"], "files": len(good), "failed": len(failures), "bytes": summary["bytes"],
                      "receipt": str(receipt), "manifest": str(manifest), "receipt_sha256": sha256(receipt), "manifest_sha256": sha256(manifest)}), flush=True)
    if failures:
        raise RuntimeError("Download incomplete; failed records retained for an explicit resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-html", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    download(args.output, args.inventory_html, args.workers, args.resume)


if __name__ == "__main__":
    main()
