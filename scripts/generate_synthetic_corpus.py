#!/usr/bin/env python3
"""Generate the separate Milestone 3 controlled vehicle-source corpus."""

from __future__ import annotations

import sys

from vehicle_audio.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["synthesize-sources", *sys.argv[1:]]))
