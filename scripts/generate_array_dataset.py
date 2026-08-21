#!/usr/bin/env python3
"""Compatibility wrapper for ``vehicle-audio generate-array``."""

from __future__ import annotations

import sys

from vehicle_audio.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate-array", *sys.argv[1:]]))
