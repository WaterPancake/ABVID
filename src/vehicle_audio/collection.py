"""License-aware downloading of human-reviewed audio source records.

This module is deliberately separate from dataset generation.  The generator
never makes network requests; this collector is an explicit, operator-invoked
step that preserves source provenance beside every normalized WAV.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml

from vehicle_audio.metadata import ConditionSegment, validate_operating_condition


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "vehicle-audio-source-collector/0.1 (research dataset; contact project owner)"


@dataclass(frozen=True)
class SourceSpec:
    """A catalog entry describing one source file and its dataset placement."""

    id: str
    kind: str
    provider: str
    source_page: str
    output_path: str
    status: str
    expected_license: str | None
    admitted_to_corpus: bool | None = None
    file_title: str | None = None
    archive_identifier: str | None = None
    archive_file: str | None = None
    direct_url: str | None = None
    license_url: str | None = None
    vehicle_class: str | None = None
    vehicle_model: str | None = None
    recording_session: str | None = None
    operating_condition: str | None = None
    condition_segments: tuple[ConditionSegment, ...] = ()
    category: str | None = None
    attribution: str | None = None
    notes: str | None = None
    review_reason: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceSpec":
        required = ("id", "kind", "provider", "source_page", "output_path", "status")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"source record missing required fields: {missing}")
        output_path = str(value["output_path"])
        if Path(output_path).is_absolute() or ".." in Path(output_path).parts:
            raise ValueError(f"output_path must stay below the collection root: {output_path}")
        admitted_to_corpus = value.get("admitted_to_corpus")
        if admitted_to_corpus is not None and not isinstance(admitted_to_corpus, bool):
            raise ValueError("admitted_to_corpus must be boolean or null")
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            provider=str(value["provider"]),
            source_page=str(value["source_page"]),
            output_path=output_path,
            status=str(value["status"]),
            expected_license=(
                None if value.get("expected_license") is None else str(value["expected_license"])
            ),
            admitted_to_corpus=admitted_to_corpus,
            file_title=(None if value.get("file_title") is None else str(value["file_title"])),
            archive_identifier=(
                None
                if value.get("archive_identifier") is None
                else str(value["archive_identifier"])
            ),
            archive_file=(
                None if value.get("archive_file") is None else str(value["archive_file"])
            ),
            direct_url=(None if value.get("direct_url") is None else str(value["direct_url"])),
            license_url=(
                None if value.get("license_url") is None else str(value["license_url"])
            ),
            vehicle_class=(
                None if value.get("vehicle_class") is None else str(value["vehicle_class"])
            ),
            vehicle_model=(
                None if value.get("vehicle_model") is None else str(value["vehicle_model"])
            ),
            recording_session=(
                None
                if value.get("recording_session") is None
                else str(value["recording_session"])
            ),
            operating_condition=(
                None
                if value.get("operating_condition") is None
                else validate_operating_condition(str(value["operating_condition"]))
            ),
            condition_segments=tuple(
                ConditionSegment.from_mapping(segment)
                for segment in value.get("condition_segments", [])
            ),
            category=None if value.get("category") is None else str(value["category"]),
            attribution=(
                None if value.get("attribution") is None else str(value["attribution"])
            ),
            notes=None if value.get("notes") is None else str(value["notes"]),
            review_reason=(
                None if value.get("review_reason") is None else str(value["review_reason"])
            ),
        )


def load_catalog(path: str | Path) -> tuple[dict[str, Any], list[SourceSpec]]:
    """Load and validate a YAML catalog."""

    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("source catalog must contain a mapping")
    policy = document.get("collection_policy")
    raw_sources = document.get("sources")
    if not isinstance(policy, Mapping) or not isinstance(raw_sources, list):
        raise ValueError("catalog must contain collection_policy and a sources list")
    allowed = policy.get("allowed_licenses")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("collection_policy.allowed_licenses must be a list of strings")
    sources = [SourceSpec.from_mapping(item) for item in raw_sources]
    ids = [source.id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source ids must be unique")
    return dict(document), sources


def is_license_allowed(license_name: str | None, allowed: Iterable[str]) -> bool:
    """Return true only for an exact license name in the operator allowlist."""

    return license_name is not None and license_name.strip() in set(allowed)


def _request_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object from {url}")
    return value


def resolve_commons_file(file_title: str) -> dict[str, Any]:
    """Resolve current Commons file URL and machine-readable license metadata."""

    query = urlencode(
        {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime|sha1",
            "format": "json",
            "formatversion": "2",
        }
    )
    document = _request_json(f"{COMMONS_API}?{query}")
    pages = document.get("query", {}).get("pages", [])
    if not isinstance(pages, list) or len(pages) != 1:
        raise ValueError(f"Commons did not resolve exactly one page for {file_title!r}")
    page = pages[0]
    image_info = page.get("imageinfo", [])
    if not image_info:
        raise ValueError(f"Commons page is not a downloadable media file: {file_title}")
    info = image_info[0]
    extmetadata = info.get("extmetadata", {})

    def metadata_value(key: str) -> str | None:
        value = extmetadata.get(key, {}).get("value")
        return None if value is None else str(value)

    original_url = str(info.get("url", ""))
    if not original_url:
        raise ValueError(f"Commons file has no original URL: {file_title}")
    # Remove tracking query parameters while retaining the canonical file path.
    parts = urlsplit(original_url)
    canonical_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return {
        "provider": "wikimedia_commons",
        "file_title": str(page.get("title", file_title)),
        "source_url": f"https://commons.wikimedia.org/wiki/{quote(str(page.get('title', file_title)), safe=':')}",
        "download_url": canonical_url,
        "mime": info.get("mime"),
        "size": info.get("size"),
        "sha1": info.get("sha1"),
        "license": metadata_value("LicenseShortName"),
        "license_url": metadata_value("LicenseUrl"),
        "artist": metadata_value("Artist"),
        "credit": metadata_value("Credit"),
        "description": metadata_value("ImageDescription"),
        "date": metadata_value("Date"),
    }


def resolve_archive_item(
    identifier: str,
    requested_file: str | None = None,
) -> dict[str, Any]:
    """Resolve an Internet Archive item and one audio file from item metadata."""

    document = _request_json(f"https://archive.org/metadata/{quote(identifier, safe='')}")
    metadata = document.get("metadata")
    files = document.get("files")
    if not isinstance(metadata, Mapping) or not isinstance(files, list):
        raise ValueError(f"Internet Archive item is missing metadata/files: {identifier}")
    license_value = str(metadata.get("licenseurl") or metadata.get("license") or "")
    normalized_license: str | None
    if "publicdomain" in license_value.lower() or "public domain" in license_value.lower():
        normalized_license = "Public domain"
    elif "by-sa/4" in license_value.lower():
        normalized_license = "CC BY-SA 4.0"
    elif "by-sa/3" in license_value.lower():
        normalized_license = "CC BY-SA 3.0"
    elif "by/4" in license_value.lower():
        normalized_license = "CC BY 4.0"
    elif "by/3" in license_value.lower():
        normalized_license = "CC BY 3.0"
    elif "cc0" in license_value.lower():
        normalized_license = "CC0"
    else:
        normalized_license = None
    candidates = [
        item
        for item in files
        if isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and Path(str(item["name"])).suffix.lower() in {".wav", ".flac", ".ogg", ".oga", ".mp3"}
    ]
    if requested_file:
        if Path(requested_file).name != requested_file:
            raise ValueError(f"archive_file must be a plain item filename: {requested_file}")
        candidates = [item for item in candidates if item.get("name") == requested_file]
    if not candidates:
        raise ValueError(f"no requested audio file found in Internet Archive item: {identifier}")
    selected = candidates[0]
    filename = str(selected["name"])
    return {
        "provider": "internet_archive",
        "archive_identifier": identifier,
        "archive_file": filename,
        "source_url": f"https://archive.org/details/{identifier}",
        "download_url": f"https://archive.org/download/{quote(identifier, safe='')}/{quote(filename, safe='')}",
        "mime": selected.get("format"),
        "size": selected.get("size"),
        "sha1": selected.get("sha1"),
        "license": normalized_license,
        "license_url": license_value or None,
        "artist": metadata.get("creator"),
        "credit": metadata.get("contributor"),
        "description": metadata.get("description"),
        "date": metadata.get("date") or metadata.get("year"),
        "item_metadata": dict(metadata),
    }


def _direct_record(source: SourceSpec, allowed_licenses: list[str]) -> dict[str, Any]:
    """Resolve a cataloged direct URL whose rights were manually recorded."""

    if not source.direct_url:
        raise ValueError(f"direct source {source.id} has no direct_url")
    actual_license = source.expected_license
    if not is_license_allowed(actual_license, allowed_licenses):
        raise PermissionError(
            f"source {source.id} has unapproved direct-source license {actual_license!r}"
        )
    return {
        "provider": source.provider,
        "source_url": source.source_page,
        "download_url": source.direct_url,
        "mime": None,
        "size": None,
        "sha1": None,
        "license": actual_license,
        "license_url": source.license_url,
        "artist": source.attribution,
        "credit": None,
        "description": source.notes,
        "date": None,
    }


def _archive_record(source: SourceSpec, allowed_licenses: list[str]) -> dict[str, Any]:
    if not source.archive_identifier:
        raise ValueError(f"Internet Archive source {source.id} has no archive_identifier")
    resolved = resolve_archive_item(source.archive_identifier, source.archive_file)
    actual_license = resolved.get("license")
    if not is_license_allowed(actual_license, allowed_licenses):
        raise PermissionError(
            f"source {source.id} has unapproved Internet Archive license {actual_license!r}"
        )
    if source.expected_license and actual_license != source.expected_license:
        raise PermissionError(
            f"source {source.id} license changed: expected {source.expected_license!r}, "
            f"got {actual_license!r}"
        )
    return resolved


def _safe_output(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"output path escapes collection root: {relative_path}") from exc
    return candidate


def _retry_delay_seconds(retry_after: str | None, attempt: int) -> float:
    """Return a bounded retry delay while honoring provider cooldowns."""

    fallback = 10.0 * (attempt + 1)
    if retry_after is None:
        return fallback
    try:
        return max(5.0, min(float(retry_after), 900.0))
    except (TypeError, ValueError):
        return fallback


def _download(url: str, destination: Path, max_bytes: int) -> str:
    """Download one URL and return its SHA-256, enforcing a byte limit.

    Wikimedia may return 429 while several catalog files are being resolved.
    Retry only transient rate/server failures, and honor Retry-After when it is
    present; permission and policy failures remain immediate.
    """

    last_error: Exception | None = None
    for attempt in range(5):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "audio/ogg, audio/webm, video/ogg, */*",
            },
        )
        try:
            digest = hashlib.sha256()
            total = 0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(
                        f"refusing {url}: advertised size {content_length} exceeds {max_bytes} bytes"
                    )
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"refusing {url}: download exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    handle.write(chunk)
            return digest.hexdigest()
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "code", None)
            if status not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            retry_after = getattr(exc, "headers", {}).get("Retry-After")
            time.sleep(_retry_delay_seconds(retry_after, attempt))
    raise RuntimeError(f"download failed: {url}: {last_error}")


def _extract_audio(input_path: Path, output_path: Path) -> None:
    """Normalize audio/video to float WAV without changing its native sample rate."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to normalize collected media to WAV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "ffmpeg failed").strip()
        raise RuntimeError(f"could not extract audio from {input_path}: {detail}") from exc


def _commons_record(source: SourceSpec, allowed_licenses: list[str]) -> dict[str, Any]:
    if not source.file_title:
        raise ValueError(f"Commons source {source.id} has no file_title")
    resolved = resolve_commons_file(source.file_title)
    actual_license = resolved.get("license")
    if not is_license_allowed(actual_license, allowed_licenses):
        raise PermissionError(
            f"source {source.id} has unapproved Commons license {actual_license!r}"
        )
    if source.expected_license and actual_license != source.expected_license:
        raise PermissionError(
            f"source {source.id} license changed: expected {source.expected_license!r}, "
            f"got {actual_license!r}"
        )
    return resolved


def collect_sources(
    catalog_path: str | Path,
    output_root: str | Path,
    source_ids: Iterable[str],
    *,
    include_review_required: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    max_bytes: int = 500_000_000,
) -> list[dict[str, Any]]:
    """Resolve and optionally download selected catalog entries.

    Internet Archive records are resolved through their item metadata and are
    downloaded only when the item exposes an allowlisted rights marker. Files
    with missing or ambiguous rights remain review-gated in the catalog.
    """

    document, sources = load_catalog(catalog_path)
    allowed_licenses = [str(item) for item in document["collection_policy"]["allowed_licenses"]]
    requested = set(source_ids)
    by_id = {source.id: source for source in sources}
    unknown = requested - by_id.keys()
    if unknown:
        raise KeyError(f"unknown source ids: {sorted(unknown)}")
    selected = [source for source in sources if source.id in requested]
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in selected:
        base_record: dict[str, Any] = {
            "source_id": source.id,
            "kind": source.kind,
            "status": source.status,
            "source_page": source.source_page,
            "output_path": source.output_path,
            "expected_license": source.expected_license,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if source.status != "approved" and not include_review_required:
            base_record["result"] = "skipped_review_required"
            base_record["review_reason"] = source.review_reason
            records.append(base_record)
            continue
        if source.provider not in {"wikimedia_commons", "internet_archive", "direct"}:
            base_record["result"] = "unsupported_provider"
            base_record["reason"] = "provider is not implemented in this collector"
            records.append(base_record)
            continue
        try:
            if source.provider == "wikimedia_commons":
                resolved = _commons_record(source, allowed_licenses)
            elif source.provider == "internet_archive":
                resolved = _archive_record(source, allowed_licenses)
            else:
                resolved = _direct_record(source, allowed_licenses)
            license_url = resolved.get("license_url") or source.license_url
            base_record.update(
                {
                    "license": resolved.get("license"),
                    "license_url": license_url,
                    "download_url": resolved.get("download_url"),
                    "source_sha1": resolved.get("sha1"),
                    "source_size": resolved.get("size"),
                }
            )
            output_path = _safe_output(root, source.output_path)
            base_record["normalized_path"] = output_path.relative_to(root).as_posix()
            if dry_run:
                base_record["result"] = "ready"
                records.append(base_record)
                continue
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"output exists (use --overwrite): {output_path}")
            with tempfile.TemporaryDirectory(prefix=f"vehicle-audio-{source.id}-") as temp_dir:
                raw_path = Path(temp_dir) / "source.bin"
                raw_sha256 = _download(resolved["download_url"], raw_path, max_bytes)
                _extract_audio(raw_path, output_path)
            sidecar = {
                "source_id": source.id,
                "source_page": source.source_page,
                "download_url": resolved["download_url"],
                "provider_metadata": resolved,
                "license": resolved.get("license"),
                "license_url": license_url,
                "attribution": source.attribution,
                "raw_sha256": raw_sha256,
                "collected_at_utc": base_record["collected_at_utc"],
            }
            if source.kind == "target":
                sidecar.update(
                    {
                        "vehicle_class": source.vehicle_class,
                        "vehicle_model": source.vehicle_model,
                        "recording_session": source.recording_session,
                        "operating_condition": source.operating_condition or "unknown",
                        "condition_segments": [
                            segment.to_dict() for segment in source.condition_segments
                        ],
                    }
                )
                if source.admitted_to_corpus is not None:
                    sidecar["admitted_to_corpus"] = source.admitted_to_corpus
            elif source.kind == "background":
                sidecar["category"] = source.category
            sidecar_path = output_path.with_suffix(".json")
            with sidecar_path.open("w", encoding="utf-8") as handle:
                json.dump(sidecar, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            base_record.update(
                {
                    "result": "downloaded",
                    "raw_sha256": raw_sha256,
                    "sidecar_path": sidecar_path.relative_to(root).as_posix(),
                }
            )
        except Exception as exc:  # record failures so one bad source is auditable
            base_record["result"] = "error"
            base_record["error_type"] = type(exc).__name__
            base_record["error"] = str(exc)
        records.append(base_record)
    return records


def write_collection_manifest(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write an auditable JSONL result manifest, merging prior source results.

    Keeping successful rows when a later retry hits a rate limit makes the
    manifest useful across one-source-at-a-time collection runs.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prior = json.loads(line)
                    if isinstance(prior, Mapping) and prior.get("source_id"):
                        merged[str(prior["source_id"])] = dict(prior)
    for record in records:
        current = dict(record)
        source_id = str(current.get("source_id", ""))
        if not source_id:
            raise ValueError("collection manifest records require source_id")
        prior = merged.get(source_id)
        if prior and prior.get("result") == "downloaded" and current.get("result") != "downloaded":
            continue
        merged[source_id] = current
    with output_path.open("w", encoding="utf-8") as handle:
        for source_id in sorted(merged):
            handle.write(json.dumps(merged[source_id], sort_keys=True, allow_nan=False))
            handle.write("\n")


def sync_catalog_metadata(
    catalog_path: str | Path,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    """Apply reviewed catalog labels to sidecars for already collected audio."""

    document, sources = load_catalog(catalog_path)
    root = Path(output_root).resolve()
    records: list[dict[str, Any]] = []
    for source in sources:
        output_path = _safe_output(root, source.output_path)
        sidecar_path = output_path.with_suffix(".json")
        record: dict[str, Any] = {
            "source_id": source.id,
            "audio_path": source.output_path,
            "sidecar_path": sidecar_path.relative_to(root).as_posix(),
        }
        if not output_path.exists() or not sidecar_path.exists():
            record["result"] = "not_collected"
            records.append(record)
            continue
        with sidecar_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"source sidecar must contain an object: {sidecar_path}")
        metadata["catalog_version"] = document.get("version")
        if source.kind == "target":
            metadata.update(
                {
                    "vehicle_class": source.vehicle_class,
                    "vehicle_model": source.vehicle_model,
                    "recording_session": source.recording_session,
                    "operating_condition": source.operating_condition or "unknown",
                    "condition_segments": [
                        segment.to_dict() for segment in source.condition_segments
                    ],
                }
            )
            if source.admitted_to_corpus is not None:
                metadata["admitted_to_corpus"] = source.admitted_to_corpus
        elif source.kind == "background":
            metadata["category"] = source.category
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        record["result"] = "updated"
        records.append(record)
    return records
