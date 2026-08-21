#!/usr/bin/env python3
"""Apply reviewed catalog labels to sidecars for already collected sources."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from vehicle_audio.collection import sync_catalog_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/audio_sources.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    records = sync_catalog_metadata(args.catalog, args.output)
    print(
        json.dumps(
            {
                "results": dict(Counter(record["result"] for record in records)),
                "updated": [
                    record["source_id"] for record in records if record["result"] == "updated"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
