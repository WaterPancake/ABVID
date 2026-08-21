#!/usr/bin/env python3
"""Explicit, provenance-preserving source collector.

Examples:
    uv run python scripts/collect_audio.py --list
    uv run python scripts/collect_audio.py --all-approved --output data
    uv run python scripts/collect_audio.py --source-id target-tracked-tr85m1-tank-range --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Permit running this file directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vehicle_audio.collection import collect_sources, load_catalog, write_collection_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/audio_sources.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="collection root; catalog output_path values are relative to this directory",
    )
    parser.add_argument("--source-id", action="append", default=[], help="select one source (repeatable)")
    parser.add_argument("--all-approved", action="store_true", help="select every approved source")
    parser.add_argument("--include-review-required", action="store_true", help="override the review gate")
    parser.add_argument("--dry-run", action="store_true", help="resolve licenses and URLs without downloading")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL result manifest (default: <output>/collection_manifest.jsonl)",
    )
    parser.add_argument("--list", action="store_true", help="list catalog records and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document, sources = load_catalog(args.catalog)
    if args.list:
        for source in sources:
            print(
                json.dumps(
                    {
                        "id": source.id,
                        "kind": source.kind,
                        "provider": source.provider,
                        "status": source.status,
                        "expected_license": source.expected_license,
                        "output_path": source.output_path,
                    },
                    sort_keys=True,
                )
            )
        return 0

    selected = set(args.source_id)
    if args.all_approved:
        selected.update(source.id for source in sources if source.status == "approved")
    if not selected:
        raise SystemExit("select --source-id at least once, or use --all-approved")
    manifest_path = args.manifest or args.output / "collection_manifest.jsonl"
    records: list[dict[str, object]] = []
    selected_in_catalog_order = [source.id for source in sources if source.id in selected]
    for source_id in selected_in_catalog_order:
        source_records = collect_sources(
            args.catalog,
            args.output,
            [source_id],
            include_review_required=args.include_review_required,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            max_bytes=args.max_bytes,
        )
        records.extend(source_records)
        # Checkpoint every source so an interruption or later provider failure
        # cannot discard provenance for files already collected successfully.
        write_collection_manifest(manifest_path, source_records)
    print(json.dumps({"manifest": str(manifest_path), "records": records}, indent=2, sort_keys=True))
    return 0 if all(record["result"] not in {"error", "unsupported_provider"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
