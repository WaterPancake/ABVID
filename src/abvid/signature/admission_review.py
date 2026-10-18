"""Caption-hidden P1 review packets served offline without duplicating audio."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
from urllib.parse import unquote, urlsplit

import numpy as np

from abvid.paths import resolve
from abvid.provenance.hashing import canonical, sha256
from . import admission
from .config import external_path, configuration
from .trim import _spectrum_svg, _wave_svg
from . import review_ui

VERSION = "signature-admission-review-1"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def display_audio(path, maximum_seconds):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True, timeout=60, check=True)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if not len(samples) or len(samples) > (maximum_seconds + 1) * 8000 or not np.isfinite(samples).all():
        raise ValueError("Invalid or overlong display waveform")
    return samples.reshape(-1, 1)


def records(cfg):
    root = external_path(cfg["storage"]["data_root"])
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/record.json"))]
    result = [r for r in rows if r["status"] == "collected_pending_blind_review"]
    if len({r["asset_id"] for r in result}) != len(result):
        raise ValueError("Duplicate collected identities")
    return result


def validate_review_audio(row):
    audio = row["acquisition"]
    path = resolve(audio["audio_ref"])
    if sha256(path) != audio["audio_sha256"]:
        raise ValueError("Collected native audio hash changed")
    metadata, duration, stream = admission.probe(path)
    # The collector strips source tags. Verify that no title/artist/source names
    # were carried into either container or stream metadata before blind review.
    allowed = {
        "major_brand": {"M4A "}, "minor_version": {"512"},
        "compatible_brands": {"M4A isomiso2"}, "language": {"und"},
        "handler_name": {"SoundHandler"},
    }
    for section in [metadata["format"], *metadata["streams"]]:
        for key, value in section.get("tags", {}).items():
            if key == "encoder" and re.fullmatch(r"Lavf[0-9.]+", value):
                continue
            if key not in allowed or value not in allowed[key]:
                raise ValueError("Non-generic embedded metadata would unblind audio")
    if (int(stream["sample_rate"]) != audio["native_sample_rate_hz"] or
            stream["channels"] != audio["channels"] or
            abs(duration - audio["audio_duration_seconds"]) > 0.1):
        raise ValueError("Native review audio metadata changed")
    return path


def card(key, ordinal, duration, x):
    # Controls deliberately match the Chrome-confirmed P0 viewer. P1's different
    # protocol/version/batch binding is supplied separately.
    if not re.fullmatch(r"review-[0-9a-f]{24}", key):
        raise ValueError("Invalid opaque review identity")
    section = (review_ui.CARD.replace("__BLIND_ID__", key)
               .replace("__AUDIO_ROUTE__", f"/audio/{key}.m4a")
               .replace("__ORDINAL__", str(ordinal))
               .replace("__DURATION__", f"{duration:.3f}"))
    insertion = section.index('<div class="interval">')
    return section[:insertion] + _wave_svg(x) + _spectrum_svg(x, 8000) + section[insertion:]


def make_packets(config_ref=admission.CONFIG_REF, batch_size=8):
    cfg = admission.load_config(config_ref)
    if type(batch_size) is not int or not 1 <= batch_size <= 12:
        raise ValueError("Review batches must contain one to twelve recordings")
    root = external_path(cfg["storage"]["artifact_root"])
    blind = root / "blind"
    private = root / "private"
    if blind.exists() or private.exists():
        raise FileExistsError("Review packets are immutable; preserve existing handoff")
    rows = records(cfg)
    for row in rows:
        validate_review_audio(row)
    ordering = np.random.default_rng(17).permutation(len(rows))
    viewer_hash = review_ui.SOURCE_TEMPLATE_SHA256
    blind.mkdir()
    private.mkdir()
    manifest_hash = digest(rows)
    public_summary, private_summary = [], []
    audio_routes = {}
    for start in range(0, len(rows), batch_size):
        name = f"packet-{start // batch_size + 1:02d}"
        packet = blind / name
        packet.mkdir()
        public, mapping, cards = [], {}, []
        for offset, index in enumerate(ordering[start:start + batch_size]):
            row = rows[int(index)]
            audio = row["acquisition"]
            key = "review-" + secrets.token_hex(12)
            path = validate_review_audio(row)
            x = display_audio(path, cfg["collection"]["maximum_asset_seconds"])
            cards.append(card(key, offset + 1, audio["audio_duration_seconds"], x))
            public.append({
                "blind_id": key, "audio_file": "/audio/" + key + ".m4a",
                "duration_seconds": audio["audio_duration_seconds"],
                "audio_sha256": audio["audio_sha256"],
            })
            mapping[key] = {
                "asset_id": row["asset_id"], "audio_ref": audio["audio_ref"],
                "audio_sha256": audio["audio_sha256"], "source_sha256": audio["source_sha256"],
                "native_sample_rate_hz": audio["native_sample_rate_hz"],
                "channels": audio["channels"],
            }
            audio_routes[key] = mapping[key]
        private_data = {
            "protocol_id": cfg["id"], "config_sha256": digest(cfg),
            "input_records_sha256": manifest_hash, "mapping": mapping,
            "nonce": secrets.token_hex(16),
        }
        batch = {
            "protocol_id": cfg["id"], "tool_version": VERSION, "recordings": public,
            "private_commitment_sha256": digest(private_data),
        }
        batch_hash = digest(batch)
        meta = {"batch_sha256": batch_hash, "tool_version": VERSION}
        script = (review_ui.SCRIPT.replace("__META_JSON__", json.dumps(meta))
                  .replace("__TRIMMING_JSON__", json.dumps(configuration()["trimming"])))
        document = review_ui.HEAD + "".join(cards) + script
        disclaimer = ("The appearance of U.S. Department of War (DoW) visual information "
                      "does not imply or constitute DoW endorsement.")
        document = document.replace("<script>", f"<p>{disclaimer}</p><script>", 1)
        encoded = document.encode()
        if len(encoded) > admission.available_bytes(cfg):
            raise OSError("Storage budget cannot fit review packet")
        (packet / "index.html").write_bytes(encoded)
        admission.write_new(packet / "review_batch.json", batch)
        admission.write_new(private / (name + ".json"), {**private_data, "batch_sha256": batch_hash})
        public_summary.append({"packet": name, "recordings": len(public),
                               "href": name + "/index.html", "batch_sha256": batch_hash})
        private_summary.append({"packet": name, "mapping_sha256": digest(private_data)})
    links = "".join(f'<li><a href="{r["href"]}">Review packet {i+1}</a> '
                    f'({r["recordings"]} recordings)</li>' for i, r in enumerate(public_summary))
    index = ("<!doctype html><meta charset=utf-8><title>Recording review packets</title>"
             "<style>body{font:18px system-ui;max-width:850px;margin:3rem auto;padding:1rem}"
             "li{margin:1rem 0}</style><h1>Recording review packets</h1>"
             "<p>Review each packet without captions. Save its annotations separately.</p>"
             f"<ul>{links}</ul>").encode()
    if len(index) > admission.available_bytes(cfg):
        raise OSError("Storage budget cannot fit packet index")
    (blind / "index.html").write_bytes(index)
    admission.write_new(private / "audio_routes.json", {
        "protocol_id": cfg["id"], "config_sha256": digest(cfg), "routes": audio_routes,
    })
    result = {
        "recorded_at": admission.now(), "protocol_id": cfg["id"], "tool_version": VERSION,
        "recordings": len(rows), "packets": public_summary, "private_packets": private_summary,
        "input_records_sha256": manifest_hash, "viewer_template_sha256": viewer_hash,
        "viewer_component_source_sha256": sha256(Path(review_ui.__file__)),
        "audio_playback": "original-rate codec-copy audio via local opaque routes",
        "display_only_rate_hz": 8000, "audio_files_duplicated": False,
        "storage_used_bytes": admission.storage_used(cfg),
        "human_blind_review": "pending",
    }
    admission.write_new(root / "review_packet_receipt.json", result)
    return result


def serve(config_ref=admission.CONFIG_REF, port=8766):
    cfg = admission.load_config(config_ref)
    artifacts = external_path(cfg["storage"]["artifact_root"])
    blind = artifacts / "blind"
    routes_document = json.loads((artifacts / "private/audio_routes.json").read_text())
    if routes_document["protocol_id"] != cfg["id"] or routes_document["config_sha256"] != digest(cfg):
        raise ValueError("Audio routes belong to a different contract")
    routes = {}
    for key, row in routes_document["routes"].items():
        if not re.fullmatch(r"review-[0-9a-f]{24}", key):
            raise ValueError("Invalid opaque route")
        path = resolve(row["audio_ref"])
        if sha256(path) != row["audio_sha256"]:
            raise ValueError("Native audio hash changed before playback")
        routes["/audio/" + key + ".m4a"] = path

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self._send(head=True)

        def do_GET(self):
            self._send(head=False)

        def _send(self, head):
            url = unquote(urlsplit(self.path).path)
            if url in routes:
                path, content_type = routes[url], "audio/mp4"
            else:
                relative = Path(url.lstrip("/") or "index.html")
                path = (blind / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts or
                        not path.is_relative_to(blind) or not path.is_file() or
                        path.suffix not in {".html", ".json"}):
                    self.send_error(404)
                    return
                content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "application/json"
            total = path.stat().st_size
            lo, hi = 0, total - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if not match:
                    self.send_error(416)
                    return
                lo = int(match.group(1))
                hi = min(total - 1, int(match.group(2)) if match.group(2) else total - 1)
                if not 0 <= lo <= hi < total:
                    self.send_error(416)
                    return
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(hi - lo + 1))
            self.send_header("Accept-Ranges", "bytes")
            if range_header:
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{total}")
            self.end_headers()
            if not head:
                with path.open("rb") as stream:
                    stream.seek(lo)
                    remaining = hi - lo + 1
                    while remaining:
                        data = stream.read(min(262144, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(json.dumps({"url": f"http://127.0.0.1:{port}/", "recordings": len(routes),
                      "scope": "local offline blind packets and explicit audio routes only"}), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["packets", "serve"])
    parser.add_argument("--config", default=admission.CONFIG_REF)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.action == "packets":
        print(json.dumps(make_packets(args.config), indent=2))
    else:
        serve(args.config, args.port)


if __name__ == "__main__":
    main()
