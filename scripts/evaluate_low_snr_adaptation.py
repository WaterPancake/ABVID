#!/usr/bin/env python3
"""Adapt the frozen PANNs probe using low-SNR real-source corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_audio.low_snr_adaptation import run_low_snr_adaptation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-probe-bundle", type=Path, required=True)
    parser.add_argument("--panns-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    payload = run_low_snr_adaptation(
        arguments.config,
        arguments.manifest,
        arguments.base_probe_bundle,
        arguments.panns_checkpoint,
        arguments.output,
        device_name=arguments.device,
    )
    print(json.dumps(payload["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
