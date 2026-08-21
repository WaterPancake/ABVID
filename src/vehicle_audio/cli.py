"""Command-line interface for synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from vehicle_audio.array_dataset import ArrayDatasetGenerator
from vehicle_audio.config import load_config
from vehicle_audio.dataset import DatasetGenerator
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
