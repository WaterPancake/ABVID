"""Bounded, separately authorized DVIDS media acquisition for P1 admission.

The immutable P0 contract is never relaxed. Collected recordings remain pending
human blind review and have no experimental role or verified session identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from abvid.paths import resolve
from abvid.provenance.hashing import sha256
from .config import external_path

CONFIG_REF = "repo:experiments/military_signature/p1_admission_5gb.json"
CONTRACT_HASH = "07c86a7a64dce473847e1a8928b1bde50b7375ac44194d4fcd8e5fc8e1cdea27"
TOOL_VERSION = "signature-admission-2"
MEDIA_HOST = "d34w7g4gy10iej.cloudfront.net"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(reference=CONFIG_REF):
    cfg = json.loads(resolve(reference).read_text())
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != CONTRACT_HASH:
        raise ValueError("Changed P1 admission contract; version a new authorization/configuration")
    for key in ("data_root", "artifact_root"):
        external_path(cfg["storage"][key])
    return cfg


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def storage_used(cfg):
    # Counting every path is conservative if a future packet uses hard links.
    total = 0
    for name in ("data_root", "artifact_root"):
        root = external_path(cfg["storage"][name])
        if root.exists():
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Symlinks are forbidden within the new collection")
                if path.is_file():
                    total += path.stat().st_size
    return total


def available_bytes(cfg):
    root = external_path(cfg["storage"]["data_root"])
    existing = next(p for p in [root, *root.parents] if p.exists())
    free = shutil.disk_usage(existing).free
    if free < cfg["storage"]["minimum_free_disk_bytes"]:
        raise OSError("Free-disk reserve reached")
    remaining = cfg["storage"]["maximum_total_bytes"] - storage_used(cfg)
    return min(remaining, free - cfg["storage"]["minimum_free_disk_bytes"])


def check_url(url, *, media=False):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ValueError("Only plain HTTPS metadata/media URLs are allowed")
    if media:
        if parsed.hostname != MEDIA_HOST or not parsed.path.startswith("/video/"):
            raise ValueError("Media URL is not the explicit DVIDS video CDN")
    elif parsed.hostname != "www.dvidshub.net":
        raise ValueError("Metadata URL left DVIDS")
    return url


class _Redirect(HTTPRedirectHandler):
    def __init__(self, media):
        self.media = media
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, media=self.media)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def response(url, cfg, *, media=False):
    check_url(url, media=media)
    opener = build_opener(_Redirect(media))
    return opener.open(Request(url, headers={"User-Agent": "ABVID-P1-Research-Admission/1.0"}),
                       timeout=cfg["collection"]["request_timeout_seconds"])


def fetch_text(url, cfg, destination, *, media=False):
    with response(url, cfg, media=media) as stream:
        content_type = stream.headers.get_content_type()
        binary_playlist = (content_type == "application/octet-stream" and
                           urlsplit(url).path.endswith(".m3u8"))
        if not (content_type.startswith("text/") or "mpegurl" in content_type or binary_playlist):
            raise ValueError("Unexpected metadata/playlist content type: " + content_type)
        limit = cfg["collection"]["maximum_metadata_bytes"]
        data = stream.read(limit + 1)
        if len(data) > limit or len(data) > available_bytes(cfg):
            raise ValueError("Metadata bound exceeded")
        if binary_playlist and not data.lstrip().startswith(b"#EXTM3U"):
            raise ValueError("Octet-stream playlist did not contain an HLS manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(data)
    return data.decode("utf-8", "replace")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.ignored += 1
        if tag in ("br", "p", "div", "tr"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignored = max(0, self.ignored - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def plain(fragment):
    parser = _Text()
    parser.feed(fragment)
    return " ".join("".join(parser.parts).split())


def _element(html, tag, css_class):
    match = re.search(rf'<{tag}\b[^>]*class="[^"]*\b{re.escape(css_class)}\b[^"]*"[^>]*>(.*?)</{tag}>',
                      html, flags=re.S | re.I)
    return match.group(1) if match else ""


def caption_classes(text):
    lower = text.lower()
    found = []
    if re.search(r"\bm1a[12]\b", lower) or (
            "abrams" in lower and re.search(r"\btanks?\b|\bmain battle\b", lower)):
        found.append("abrams")
    if re.search(r"\bleopard(?:[- ]?[12])?\b", lower):
        found.append("leopard")
    if "bradley" in lower and re.search(r"fighting|vehicle|convoy|maneuver|\bm2\b", lower):
        found.append("bradley")
    if re.search(r"\bstryker\b", lower):
        found.append("stryker")
    if re.search(r"\bm[- ]?109(?:a[67]|[- ]a[- ]?[67])?\b|\bpaladin\b", lower):
        found.append("m109")
    if re.search(r"\bhumvees?\b|\bhmmwvs?\b|high mobility multipurpose", lower):
        found.append("hmmwv")
    return found


def page_metadata(html, asset_id, url):
    title = plain(_element(html, "h1", "asset-title"))
    table = _element(html, "table", "video_info_table")
    fields = {}
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S | re.I):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S | re.I)
        if len(cells) == 2:
            fields[plain(cells[0]).rstrip(":")] = plain(cells[1]) or "unknown"
    caption_match = re.search(
        r'<div class="uk-width-1-1" style="clear: both;"></div>\s*<p>(.*?)</p>',
        html, re.S | re.I)
    caption = plain(caption_match.group(1)) if caption_match else ""
    copyright_info = plain(_element(html, "p", "copyright_info"))
    domains = [plain(v) for v in re.findall(
        r'<h4\b[^>]*class="[^"]*\bpublic-domain\b[^"]*"[^>]*>(.*?)</h4>', html, re.S | re.I)]
    duration_text = fields.get("Length", "")
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", duration_text)
    duration = sum(int(v) * factor for v, factor in zip(match.groups(), [3600, 60, 1])) if match else None
    media_urls = []
    for attrs in re.findall(r"<source\b([^>]+)>", html, re.S | re.I):
        src = re.search(r'\bsrc=["\']([^"\']+)["\']', attrs)
        if src and urlsplit(unescape(src.group(1))).path.endswith(".mp4"):
            media_urls.append(unescape(src.group(1)))
    h3s = [plain(s) for s in re.findall(r"<h3\b[^>]*>(.*?)</h3>", html, re.S | re.I)]
    producer = next((s.removeprefix("Video by ") for s in h3s if s.startswith("Video by ")), "unknown")
    unit = plain(_element(html, "h3", "the_unit")) or "unknown"
    if not title or not caption or fields.get("Video ID") != str(asset_id):
        raise ValueError("Incomplete/mismatched DVIDS title, caption or asset table")
    return {
        "asset_id": str(asset_id), "url": url, "title": title, "caption": caption,
        "caption_classes": caption_classes(title + " " + caption), "caption_confidence": "caption",
        "producer": producer, "unit": unit, "outlet": "DVIDS",
        "date_taken": fields.get("Date Taken", "unknown"),
        "date_posted": fields.get("Date Posted", "unknown"),
        "location": fields.get("Location", "unknown"), "VIRIN": fields.get("VIRIN", "unknown"),
        "source_filename": fields.get("Filename", "unknown"),
        "duration_seconds_from_page": duration,
        "public_domain_banner": any(s == "PUBLIC DOMAIN" for s in domains),
        "copyright_info": copyright_info, "mp4_urls": media_urls,
        "recording_session": "unknown", "session_evidence": "unknown",
        "physical_vehicle_id": "unknown", "independence_verified": False,
        "experimental_role": "unassigned", "engine_identity": "unknown",
        "audio_quality": "unknown", "foreground_seconds": "unknown",
        "background_support": "unknown", "blind_review_status": "pending",
    }


def rights_decision(row):
    text = " ".join([row["caption"], row["copyright_info"], row["unit"]]).lower()
    restricted = [
        r"editorial use only", r"no commercial", r"no archive", r"not for sale",
        r"not for redistribut", r"rights reserved", r"copyrighted",
        r"reuters", r"associated press", r"natochannel", r"nato channel",
        r"courtesy of", r"used with permission", r"\(c\)|©",
    ]
    reasons = [pattern for pattern in restricted if re.search(pattern, text)]
    if not row["public_domain_banner"] or not row["copyright_info"]:
        reasons.append("missing public-domain banner or asset usage statement")
    if row["unit"] == "unknown":
        reasons.append("originator unknown")
    return {
        "decision": "quarantine" if reasons else "collect_for_local_noncommercial_research",
        "basis": "Per-asset banner, originator and caption/usage restriction review plus DVIDS public-use notice",
        "policy_url": "https://www.dvidshub.net/about/copyright",
        "flags": reasons,
        "limitations": "Metadata-based admission, not a guarantee about unmarked third-party content; human audio review remains required",
        "media_redistribution_authorized": False,
    }


def historical_exclusions(cfg):
    ids, filenames, witnesses = set(), set(), []
    for ref in cfg["protected_catalogs"]:
        path = resolve(ref)
        text = path.read_text()
        ids.update(re.findall(r"dvidshub\.net/video/(\d+)", text))
        filenames.update(re.findall(r"/(DOD_[A-Za-z0-9_]+)(?:/|-|\.)", text))
        witnesses.append({"reference": ref, "sha256": sha256(path)})
    return ids, filenames, witnesses


def validate_playlist(text, base_url, maximum_seconds):
    if "#EXT-X-ENDLIST" not in text or "#EXTINF:" not in text:
        raise ValueError("Only finite media playlists are accepted")
    if "#EXT-X-KEY" in text or "#EXT-X-MAP" in text or "#EXT-X-BYTERANGE" in text:
        raise ValueError("Encrypted or unsupported media playlist")
    durations = [float(x) for x in re.findall(r"#EXTINF:([^,]+)", text)]
    if any(not math.isfinite(x) or x <= 0 for x in durations):
        raise ValueError("Invalid playlist segment durations")
    duration = sum(durations)
    if duration > maximum_seconds:
        raise ValueError("Playlist exceeds per-asset duration; long assets are not trimmed to fit")
    urls = [urljoin(base_url, line.strip()) for line in text.splitlines()
            if line.strip() and not line.startswith("#")]
    if len(urls) != len(durations):
        raise ValueError("Playlist segment/duration count differs")
    for url in urls:
        check_url(url, media=True)
    return duration


def select_hls(master):
    choices = []
    lines = master.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:"):
            attrs = dict(re.findall(r"([A-Z-]+)=([^,]+)", line))
            bandwidth = int(attrs.get("BANDWIDTH", "0"))
            resolution = attrs.get("RESOLUTION", "unknown")
            height = int(resolution.split("x")[-1]) if "x" in resolution else 0
            if i + 1 < len(lines) and height <= 720:
                url = lines[i + 1].strip()
                check_url(url, media=True)
                choices.append((bandwidth, url, resolution))
    if not choices:
        raise ValueError("No allowed finite HLS rendition")
    bandwidth, url, resolution = max(choices)
    return {"url": url, "bandwidth": bandwidth, "resolution": resolution}


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    metadata = json.loads(result.stdout)
    duration = float(metadata["format"]["duration"])
    audio = next((s for s in metadata["streams"] if s.get("codec_type") == "audio"), None)
    if not audio or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Media has no valid audio stream/duration")
    return metadata, duration, audio


def run_ffmpeg(args, destination, cfg, *, timeout=600):
    limit = min(cfg["collection"]["maximum_encoded_asset_bytes"],
                available_bytes(cfg) - cfg["storage"]["reserved_workspace_bytes"])
    if limit <= 0:
        raise OSError("Storage budget reached")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", *args, "-fs", str(limit), str(destination)]
    log = destination.with_suffix(destination.suffix + ".ffmpeg.log")
    with log.open("x") as handle:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=handle)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Bounded ffmpeg operation timed out")
                if available_bytes(cfg) < cfg["storage"]["reserved_workspace_bytes"]:
                    raise OSError("Storage reserve reached during ffmpeg")
                time.sleep(0.2)
            if process.returncode:
                raise ValueError(f"ffmpeg exit {process.returncode}; see retained log")
            if destination.stat().st_size >= limit:
                raise OSError("Output hit byte bound and may be truncated")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return command


def download_file(url, destination, cfg):
    check_url(url, media=True)
    limit = min(cfg["collection"]["maximum_encoded_asset_bytes"],
                available_bytes(cfg) - cfg["storage"]["reserved_workspace_bytes"])
    if limit <= 0:
        raise OSError("Storage budget reached")
    digest = hashlib.sha256()
    size = 0
    with response(url, cfg, media=True) as stream:
        expected = stream.headers.get("Content-Length")
        if expected is not None and int(expected) > limit:
            raise OSError("Advertised media exceeds available asset/storage budget")
        if stream.headers.get_content_type().startswith("text/"):
            raise ValueError("Media request returned text")
        with destination.open("xb") as output:
            while True:
                chunk = stream.read(262144)
                if not chunk:
                    break
                if size + len(chunk) > limit or len(chunk) > available_bytes(cfg):
                    raise OSError("Media byte/storage budget reached")
                output.write(chunk)
                output.flush()
                digest.update(chunk)
                size += len(chunk)
    if not size or (expected is not None and size != int(expected)):
        raise ValueError("Incomplete or empty media download")
    return {"url": url, "bytes": size, "sha256": digest.hexdigest(),
            "method": "unchanged explicit page MP4 download"}


def acquire(row, cfg, directory, excluded_filenames):
    maximum = cfg["collection"]["maximum_asset_seconds"]
    if row["duration_seconds_from_page"] is None or not 0 < row["duration_seconds_from_page"] <= maximum:
        raise ValueError("Missing/over-limit page duration")
    if row["source_filename"] in excluded_filenames:
        raise ValueError("Historical source filename exclusion")
    media = directory / "source.mp4"
    if row["mp4_urls"]:
        url = row["mp4_urls"][0]
        if any(filename in url for filename in excluded_filenames):
            raise ValueError("Historical media URL filename exclusion")
        acquisition = download_file(url, media, cfg)
        expected_duration = row["duration_seconds_from_page"]
    else:
        master_url = f"https://www.dvidshub.net/video/{row['asset_id']}.m3u8"
        master = fetch_text(master_url, cfg, directory / "master.m3u8")
        choice = select_hls(master)
        if any(filename in choice["url"] for filename in excluded_filenames):
            raise ValueError("Historical HLS filename exclusion")
        playlist = fetch_text(choice["url"], cfg, directory / "rendition.m3u8", media=True)
        expected_duration = validate_playlist(playlist, choice["url"], maximum)
        # A validated remote finite playlist is remuxed, with no codec encoding.
        command = run_ffmpeg(
            ["-protocol_whitelist", "https,tls,tcp", "-i", choice["url"],
             "-map", "0:v:0?", "-map", "0:a:0", "-c", "copy",
             "-t", str(maximum + 1), "-map_metadata", "-1"], media, cfg)
        acquisition = {**choice, "method": "validated finite HLS stream-copy remux",
                       "master_url": master_url, "command": command,
                       "sha256": sha256(media), "bytes": media.stat().st_size}
    metadata, duration, audio = probe(media)
    if duration > maximum or abs(duration - expected_duration) > 2.0:
        raise ValueError("Native duration exceeds limit or differs from full source")
    native = directory / "audio.m4a"
    command = run_ffmpeg(
        ["-i", str(media), "-map", "0:a:0", "-vn", "-c:a", "copy", "-map_metadata", "-1"],
        native, cfg, timeout=90)
    audio_metadata, audio_duration, native_stream = probe(native)
    if abs(audio_duration - duration) > 1.0:
        raise ValueError("Extracted audio duration differs from source")
    write_new(directory / "ffprobe_source.json", metadata)
    write_new(directory / "ffprobe_audio.json", audio_metadata)
    return {
        "source": acquisition, "source_ref": _reference(directory / "source.mp4", cfg),
        "source_sha256": sha256(media), "source_duration_seconds": duration,
        "audio_ref": _reference(native, cfg), "audio_sha256": sha256(native),
        "audio_duration_seconds": audio_duration,
        "audio_codec": native_stream["codec_name"],
        "native_sample_rate_hz": int(native_stream["sample_rate"]),
        "channels": native_stream["channels"],
        "source_audio_codec": audio["codec_name"],
        "source_audio_bit_rate": audio.get("bit_rate", "unknown"),
        "native_audio_command": command, "audio_reencoding_performed": False,
    }


def _reference(path, cfg):
    root = external_path(cfg["storage"]["data_root"])
    return cfg["storage"]["data_root"] + "/" + str(path.relative_to(root))


def collect(candidates_ref, config_ref=CONFIG_REF, *, retry_metadata_failures=False):
    cfg = load_config(config_ref)
    data = external_path(cfg["storage"]["data_root"])
    artifacts = external_path(cfg["storage"]["artifact_root"])
    data.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    excluded_ids, excluded_filenames, catalog_witnesses = historical_exclusions(cfg)
    identity = artifacts / "contract.json"
    if not identity.exists():
        write_new(identity, {"configuration": cfg, "config_sha256": sha256(resolve(config_ref)),
                             "tool_version": TOOL_VERSION, "catalog_witnesses": catalog_witnesses,
                             "started_at": now(), "source_code_sha256": sha256(Path(__file__))})
        for name, url in [("copyright", "https://www.dvidshub.net/about/copyright"),
                          ("faq", "https://www.dvidshub.net/about/faq")]:
            fetch_text(url, cfg, artifacts / (name + ".html"))
    else:
        existing = json.loads(identity.read_text())
        if existing["configuration"] != cfg or existing["catalog_witnesses"] != catalog_witnesses:
            raise ValueError("Resume contract/catalog witnesses changed")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        source_copy = artifacts / ("collector-source-" + TOOL_VERSION + "-" + stamp + ".py")
        with source_copy.open("xb") as handle:
            handle.write(Path(__file__).read_bytes())
        write_new(artifacts / ("resume-" + stamp + ".json"), {
            "recorded_at": now(), "tool_version": TOOL_VERSION,
            "source_code_sha256": sha256(Path(__file__)),
            "retained_source_copy": str(source_copy),
            "retry_metadata_failures": retry_metadata_failures,
            "prior_contract_source_code_sha256": existing["source_code_sha256"],
        })
    candidates = json.loads(resolve(candidates_ref).read_text())["candidates"]
    all_rows = [json.loads(p.read_text()) for p in sorted(data.glob("*/record.json"))]
    done_ids = {r["asset_id"] for r in all_rows}
    eligible = sum(r["status"] == "collected_pending_blind_review" for r in all_rows)
    for candidate in candidates[:cfg["collection"]["maximum_candidate_assets"]]:
        asset_id, url = str(candidate["asset_id"]), candidate["url"]
        if not re.fullmatch(r"\d+", asset_id):
            raise ValueError("Candidate ID must be numeric")
        prior = [r for r in all_rows if r["asset_id"] == asset_id]
        retryable = (retry_metadata_failures and prior and
                     not any(r["status"] == "collected_pending_blind_review" for r in prior) and
                     "Unexpected metadata/playlist content type: application/octet-stream" in
                     prior[-1].get("error", ""))
        if asset_id in done_ids and not retryable:
            continue
        if eligible >= cfg["collection"]["target_collected_assets"]:
            break
        if available_bytes(cfg) < cfg["storage"]["reserved_workspace_bytes"] + 20_000_000:
            break
        directory = data / (f"{asset_id}-attempt-{len(prior)+1}" if prior else asset_id)
        directory.mkdir(exist_ok=True)
        if list(directory.iterdir()):
            # An interruption may leave uncommitted bytes. Preserve them rather
            # than silently overwrite or treat partial downloads as complete.
            row = {"asset_id": asset_id, "url": url, "status": "interrupted_partial_quarantine",
                   "error": "Existing partial asset directory has no terminal record",
                   "recorded_at": now(), "media_access_performed": "unknown"}
        else:
            row = {"asset_id": asset_id, "url": url, "recorded_at": now(),
                   "discovery_classes": candidate.get("discovery_classes", []),
                   "media_access_performed": False, "attempt": len(prior) + 1,
                   "tool_version": TOOL_VERSION, "source_code_sha256": sha256(Path(__file__))}
            try:
                if asset_id in excluded_ids:
                    raise ValueError("Historical asset-ID exclusion; metadata/media not fetched")
                html = fetch_text(url, cfg, directory / "page.html")
                row.update(page_metadata(html, asset_id, url))
                row["page_sha256"] = sha256(directory / "page.html")
                if not row["caption_classes"]:
                    raise ValueError("No proposed caption class in title/caption")
                row["rights"] = rights_decision(row)
                if row["rights"]["decision"] == "quarantine":
                    raise ValueError("Rights quarantine: " + "; ".join(row["rights"]["flags"]))
                # The rights decision is persisted before any media request.
                write_new(directory / "rights_decision.json", row["rights"])
                row["media_access_attempted"] = True
                row["acquisition"] = acquire(row, cfg, directory, excluded_filenames)
                row["media_access_performed"] = True
                row["status"] = "collected_pending_blind_review"
                eligible += 1
            except Exception as error:
                row["status"] = "quarantine"
                row["error"] = f"{type(error).__name__}: {error}"
                row["media_access_performed"] = ((directory / "source.mp4").is_file() and
                    (directory / "source.mp4").stat().st_size > 0)
        row["completed_at"] = now()
        row["stored_bytes"] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        write_new(directory / "record.json", row)
        all_rows.append(row)
        done_ids.add(asset_id)
        print(json.dumps({"asset_id": asset_id, "status": row["status"],
                          "classes": row.get("caption_classes", []),
                          "error": row.get("error"),
                          "collected": eligible, "storage_bytes": storage_used(cfg)}), flush=True)
    summary = summarize(cfg, all_rows)
    checkpoint = artifacts / ("summary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_new(checkpoint, summary)
    return summary


def summarize(cfg, rows):
    collected = [r for r in rows if r["status"] == "collected_pending_blind_review"]
    return {
        "recorded_at": now(), "protocol_id": cfg["id"], "tool_version": TOOL_VERSION,
        "candidate_outcomes": len(rows), "collected_assets": len(collected),
        "unique_candidate_assets": len({r["asset_id"] for r in rows}),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "caption_class_asset_counts": dict(Counter(c for r in collected for c in r["caption_classes"])),
        "total_source_seconds": sum(r["acquisition"]["source_duration_seconds"] for r in collected),
        "storage_used_bytes": storage_used(cfg), "storage_limit_bytes": cfg["storage"]["maximum_total_bytes"],
        "maximum_asset_seconds": cfg["collection"]["maximum_asset_seconds"],
        "verified_independent_recording_groups": 0, "admitted_foreground_segments": 0,
        "human_blind_review": "pending", "model_or_CAST_fitting_performed": False,
        "experimental_roles": "unassigned",
        "limitations": ["Caption class counts are not session counts or interval labels",
                        "Collected playable media are pending human audio/contamination and provenance admission",
                        "No source/model recognition or generalization result is established"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "status"])
    parser.add_argument("--config", default=CONFIG_REF)
    parser.add_argument("--candidates")
    parser.add_argument("--retry-metadata-failures", action="store_true")
    args = parser.parse_args()
    if args.action == "collect":
        if not args.candidates:
            parser.error("collect requires --candidates")
        result = collect(args.candidates, args.config,
                         retry_metadata_failures=args.retry_metadata_failures)
    else:
        cfg = load_config(args.config)
        data = external_path(cfg["storage"]["data_root"])
        rows = [json.loads(p.read_text()) for p in sorted(data.glob("*/record.json"))]
        result = summarize(cfg, rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
