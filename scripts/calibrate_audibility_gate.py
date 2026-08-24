#!/usr/bin/env python3
"""Calibrate the event audibility gate on source-held-out development scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_audio.audibility_calibration import calibrate_audibility_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--adaptation-metrics", type=Path, required=True)
    parser.add_argument("--adaptation-models", type=Path, required=True)
    parser.add_argument("--panns-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    payload = calibrate_audibility_gate(
        arguments.config,
        arguments.native_manifest,
        arguments.adaptation_metrics,
        arguments.adaptation_models,
        arguments.panns_checkpoint,
        arguments.output,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    print(json.dumps(payload["gate_config"], indent=2, sort_keys=True))
    print(json.dumps(payload["selected"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
