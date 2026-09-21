"""Extract a manifest-bound PANNs cache for native-real benchmark windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from vehicle_audio.baseline import CLASS_NAMES
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.pretrained_evaluation import (
    PANN_AUDIOSET_DIMENSION,
    PANN_EMBEDDING_DIMENSION,
    PANN_SAMPLE_RATE,
    PRETRAINED_EVALUATION_VERSION,
    _extract_panns_features,
    validate_panns_checkpoint,
)
from vehicle_audio.real_corpus import audit_real_manifest, validate_real_manifest_record


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_real_panns_features(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    channel: int = 0,
    batch_size: int = 16,
    forbidden_source_ids: Sequence[str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract frozen PANNs embeddings and AudioSet outputs for one real manifest."""

    if channel < 0 or batch_size <= 0:
        raise ValueError("channel must be nonnegative and batch_size must be positive")
    manifest = Path(manifest_path)
    records, loaded_sha256 = _load_jsonl(manifest)
    manifest_sha256 = _manifest_sha256(manifest)
    if manifest_sha256 != loaded_sha256:
        raise RuntimeError("manifest hash changed while loading")
    for record in records:
        validate_real_manifest_record(record)
    audit = audit_real_manifest(records, minimum_sessions_per_class=5)
    if not audit["full_protocol_ready"]:
        raise ValueError(
            f"native-real manifest failed audit: {audit['blocking_reasons']}"
        )
    forbidden = set(forbidden_source_ids)
    overlap = sorted(
        forbidden & {str(record.get("source_id", "")) for record in records}
    )
    if overlap:
        raise ValueError(f"native-real manifest contains protected sources: {overlap}")
    sample_rates = {int(record["sample_rate"]) for record in records}
    if len(sample_rates) != 1:
        raise ValueError(f"manifest has multiple sample rates: {sorted(sample_rates)}")
    sample_rate = next(iter(sample_rates))
    checkpoint = validate_panns_checkpoint(checkpoint_path)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"feature cache exists (use overwrite): {output}")

    try:
        from panns_inference import AudioTagging
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PANNs extraction requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=checkpoint["path"], device="cpu")
    embeddings, outputs = _extract_panns_features(
        records,
        manifest,
        "audio_path",
        tagging,
        channel=channel,
        sample_rate=sample_rate,
        batch_size=batch_size,
        label="native real benchmark",
    )
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = torch.tensor(
        [label_map[str(record["vehicle_class"])] for record in records],
        dtype=torch.long,
    )
    if embeddings.shape != (len(records), PANN_EMBEDDING_DIMENSION):
        raise RuntimeError("unexpected native-real PANNs embedding shape")
    if outputs.shape != (len(records), PANN_AUDIOSET_DIMENSION):
        raise RuntimeError("unexpected native-real PANNs AudioSet shape")
    payload: dict[str, Any] = {
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "real_manifest": str(manifest),
        "real_manifest_sha256": manifest_sha256,
        "panns_checkpoint_sha256": checkpoint["sha256"],
        "source_sample_rate": sample_rate,
        "panns_sample_rate": PANN_SAMPLE_RATE,
        "channel": channel,
        "forbidden_source_ids": sorted(forbidden),
        "real_embeddings": embeddings,
        "real_clipwise_outputs": outputs,
        "real_labels": labels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "feature_cache": str(output),
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint["sha256"],
        "observation_count": len(records),
        "embedding_shape": list(embeddings.shape),
        "clipwise_output_shape": list(outputs.shape),
        "protected_overlap": overlap,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--forbid-source-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = extract_real_panns_features(
        args.real_manifest,
        args.checkpoint,
        args.output,
        channel=args.channel,
        batch_size=args.batch_size,
        forbidden_source_ids=args.forbid_source_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
