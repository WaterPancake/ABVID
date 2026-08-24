"""Command-line interface for synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from vehicle_audio.array_dataset import ArrayDatasetGenerator
from vehicle_audio.config import load_config
from vehicle_audio.dataset import DatasetGenerator
from vehicle_audio.event_inference import (
    AudibilityGateConfig,
    EventWindowConfig,
    infer_audio_event,
)
from vehicle_audio.factorial_dataset import FactorialDatasetGenerator
from vehicle_audio.multichannel import load_array_config
from vehicle_audio.real_corpus import load_real_corpus_config, prepare_real_corpus
from vehicle_audio.source_simulation import generate_controlled_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-audio")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate",
        help="generate paired clean/corrupted audio and a JSONL manifest",
    )
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--targets", type=Path, required=True)
    generate_parser.add_argument("--backgrounds", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument(
        "--impulse-responses",
        type=Path,
        default=Path("data/impulse_responses"),
        help="IR directory; a missing/empty directory disables IR selection",
    )
    generate_parser.add_argument("--num-samples", type=int, required=True)
    generate_parser.add_argument("--seed", type=int, required=True)
    factorial_parser = subparsers.add_parser(
        "generate-factorial",
        help="generate a balanced paired nuisance/SNR corpus for Milestone 6",
    )
    factorial_parser.add_argument("--config", type=Path, required=True)
    factorial_parser.add_argument("--targets", type=Path, required=True)
    factorial_parser.add_argument("--backgrounds", type=Path, required=True)
    factorial_parser.add_argument("--output", type=Path, required=True)
    factorial_parser.add_argument(
        "--impulse-responses",
        type=Path,
        default=Path("data/impulse_responses"),
    )
    factorial_parser.add_argument("--corruption-views", type=int, default=1)
    factorial_parser.add_argument("--seed", type=int, required=True)
    source_parser = subparsers.add_parser(
        "synthesize-sources",
        help="generate a controlled procedural vehicle-source corpus",
    )
    source_parser.add_argument("--config", type=Path, required=True)
    source_parser.add_argument("--output", type=Path, required=True)
    source_parser.add_argument("--seed", type=int, required=True)
    source_parser.add_argument("--overwrite", action="store_true")
    array_parser = subparsers.add_parser(
        "generate-array",
        help="generate synchronized multichannel observations with known geometry",
    )
    array_parser.add_argument("--config", type=Path, required=True)
    array_parser.add_argument("--source-manifest", type=Path, required=True)
    array_parser.add_argument("--backgrounds", type=Path, required=True)
    array_parser.add_argument("--output", type=Path, required=True)
    array_parser.add_argument("--num-samples", type=int, required=True)
    array_parser.add_argument("--seed", type=int, required=True)
    array_parser.add_argument("--overwrite", action="store_true")
    real_parser = subparsers.add_parser(
        "prepare-real",
        help="window native real recordings without inventing SNR or corruption",
    )
    real_parser.add_argument("--config", type=Path, required=True)
    real_parser.add_argument("--targets", type=Path, required=True)
    real_parser.add_argument("--output", type=Path, required=True)
    real_parser.add_argument("--overwrite", action="store_true")
    inference_parser = subparsers.add_parser(
        "infer-event",
        help="classify one audio event using overlapping audibility-gated windows",
    )
    inference_parser.add_argument("--audio", type=Path, required=True)
    inference_parser.add_argument("--probe-bundle", type=Path, required=True)
    inference_parser.add_argument("--panns-checkpoint", type=Path, required=True)
    inference_parser.add_argument("--output", type=Path, required=True)
    inference_parser.add_argument(
        "--state-key", default="panns_audioset_paired_linear"
    )
    inference_parser.add_argument("--window-seconds", type=float, default=2.0)
    inference_parser.add_argument("--hop-seconds", type=float, default=1.0)
    inference_parser.add_argument("--channel", type=int, default=0)
    inference_parser.add_argument(
        "--relative-rms-threshold-db",
        type=float,
        required=True,
        help="required audibility gate relative to the loudest window; must be <= 0",
    )
    inference_parser.add_argument("--minimum-active-windows", type=int, default=3)
    inference_parser.add_argument("--batch-size", type=int, default=16)
    inference_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = load_config(args.config)
        generator = DatasetGenerator(
            config,
            args.targets,
            args.backgrounds,
            args.output,
            impulse_responses_root=args.impulse_responses,
        )
        summary = generator.generate(args.num_samples, args.seed)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-factorial":
        generator = FactorialDatasetGenerator(
            load_config(args.config),
            args.targets,
            args.backgrounds,
            args.output,
            impulse_responses_root=args.impulse_responses,
        )
        summary = generator.generate(
            seed=args.seed,
            corruption_views=args.corruption_views,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "synthesize-sources":
        summary = generate_controlled_corpus(
            args.config,
            args.output,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-array":
        generator = ArrayDatasetGenerator(
            load_array_config(args.config),
            args.source_manifest,
            args.backgrounds,
            args.output,
        )
        summary = generator.generate(
            args.num_samples,
            args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-real":
        summary = prepare_real_corpus(
            load_real_corpus_config(args.config),
            args.targets,
            args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "infer-event":
        result = infer_audio_event(
            args.audio,
            args.probe_bundle,
            args.panns_checkpoint,
            args.output,
            state_key=args.state_key,
            window_config=EventWindowConfig(
                window_seconds=args.window_seconds,
                hop_seconds=args.hop_seconds,
                channel=args.channel,
            ),
            gate_config=AudibilityGateConfig(
                relative_peak_threshold_db=args.relative_rms_threshold_db,
                minimum_active_windows=args.minimum_active_windows,
            ),
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        summary = {
            key: value
            for key, value in result["result"].items()
            if key != "windows"
        }
        summary["window_predictions_saved_to"] = str(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
