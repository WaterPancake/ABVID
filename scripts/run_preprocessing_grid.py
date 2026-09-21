"""Uncapped reviewed native-real preprocessing search with nested session selection."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
import sklearn
import soundfile as sf
import torch
import yaml

from evaluate_embedding_context import evaluate
from vehicle_audio.audio import resample_audio
from vehicle_audio.baseline import ClassicalFeatureExtractor, FeatureConfig
from vehicle_audio.benchmark_followup import FrozenBenchmark, sha256, write_json
from vehicle_audio.collection import load_catalog
from vehicle_audio.preprocessing_grid import preprocessing_grid, preprocess
from vehicle_audio.pretrained_evaluation import validate_panns_checkpoint
from vehicle_audio.real_corpus import (
    RealCorpusConfig,
    audit_real_manifest,
    prepare_real_corpus,
)
from vehicle_audio.semantic_session_evaluation import SEMANTIC_AUDIOSET_FEATURES

CONFIG = Path("configs/benchmark_preprocessing_grid_v1.yaml")
PROTOCOL = Path("docs/preprocessing_grid_protocol.md")
CATALOG = Path("configs/audio_sources.yaml")
ARCHIVE = Path("configs/archived_startup_intervals.yaml")


def file_hashes(root):
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != ".DS_Store" and p.name != "artifact_sha256.json"
    }


def verify_hashes(root, mapping):
    for relative, expected in mapping.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"artifact hash mismatch: {root / relative}")


def prepare(output, config, frozen):
    _, sources = load_catalog(CATALOG)
    admitted = {
        s.id: s for s in sources if s.kind == "target" and s.admitted_to_corpus is True
    }
    if set(admitted) != {r["source_id"] for r in frozen.records}:
        raise ValueError("unexpected development source changes")
    for source in admitted.values():
        sidecar = json.loads(
            (Path("data") / source.output_path).with_suffix(".json").read_text()
        )
        if sidecar["condition_segments"] != [
            s.to_dict() for s in source.condition_segments
        ]:
            raise ValueError("catalog/sidecar segment mismatch")
        if not source.condition_segments or any(
            s.operating_condition == "startup" for s in source.condition_segments
        ):
            raise ValueError("startup or unsegmented active source")
    prepare_real_corpus(RealCorpusConfig(**config["dataset"]), "data/targets", output)
    manifest = output / "real_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    policy = yaml.safe_load(Path("configs/benchmark_v0_1.yaml").read_text())
    forbidden = set(
        policy["consumed_locked_source_ids"] + policy["future_confirmation_source_ids"]
    )
    blocked_sessions = {s.recording_session for s in sources if s.id in forbidden}
    archive = yaml.safe_load(ARCHIVE.read_text())["intervals"]
    for r in records:
        if (
            r["source_id"] not in admitted
            or r["source_id"] in forbidden
            or r["recording_session"] in blocked_sessions
        ):
            raise ValueError("protected/unadmitted source before inference")
        if r["operating_condition"] in config["excluded_operating_conditions"]:
            raise ValueError("excluded condition")
        if not any(
            s.start_seconds <= r["window_start_seconds"]
            and r["window_end_seconds"] <= s.end_seconds
            and s.operating_condition == r["operating_condition"]
            for s in admitted[r["source_id"]].condition_segments
        ):
            raise ValueError("unreviewed window")
        if any(
            a["source_id"] == r["source_id"]
            and r["window_start_seconds"] < a["end_seconds"]
            and a["start_seconds"] < r["window_end_seconds"]
            for a in archive
        ):
            raise ValueError("startup overlap")
    audit = audit_real_manifest(records, minimum_sessions_per_class=5)
    if not audit["full_protocol_ready"]:
        raise ValueError(audit)
    counts = {
        c: len({r["recording_session"] for r in records if r["vehicle_class"] == c})
        for c in ("tracked", "wheeled")
    }
    if counts != {"tracked": 7, "wheeled": 5}:
        raise ValueError(counts)
    write_json(
        output / "audit.json",
        {
            "manifest_sha256": sha256(manifest),
            "audit": audit,
            "sessions": counts,
            "windows": len(records),
            "windows_by_class": dict(Counter(r["vehicle_class"] for r in records)),
            "windows_by_session": dict(
                Counter(r["recording_session"] for r in records)
            ),
            "windows_by_condition": dict(
                Counter(r["operating_condition"] for r in records)
            ),
            "protected_overlap": [],
            "startup_overlap": [],
            "archived_startup_seconds": 41.5,
        },
    )
    return records


def extract(waveforms, records, variant, tagging, classical, output, config):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"semantic35": [], "classical53": [], "embedding2048": []}
    diagnostics = []
    feature_indices = [i for i, _ in SEMANTIC_AUDIOSET_FEATURES]
    for offset in range(0, len(waveforms), config["batch_size"]):
        raw = waveforms[offset : offset + config["batch_size"]]
        processed, gain = preprocess(raw, variant, config["dataset"]["sample_rate"])
        audio = torch.from_numpy(processed)
        with torch.inference_mode():
            arrays["classical53"].append(classical(audio))
            scores, embeddings = tagging.inference(
                resample_audio(audio, 16000, 32000).numpy()
            )
        arrays["semantic35"].append(torch.from_numpy(scores[:, feature_indices]))
        arrays["embedding2048"].append(torch.from_numpy(embeddings))
        for j, y in enumerate(processed):
            diagnostics.append(
                {
                    "sample_id": records[offset + j]["sample_id"],
                    "waveform_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                    "gain": float(gain[j]),
                    "rms": float(np.sqrt(np.mean(y.astype(float) ** 2))),
                    "peak": float(np.max(np.abs(y))),
                }
            )
    arrays = {k: torch.cat(v).float() for k, v in arrays.items()}
    if any(not torch.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite features")
    torch.save(arrays, output / "features.pt")
    write_json(output / "waveform_diagnostics.json", diagnostics)
    write_json(
        output / "extraction.json",
        {
            "preprocessing": variant.to_dict(),
            "seconds": time.perf_counter() - start,
            "feature_shapes": {k: list(v.shape) for k, v in arrays.items()},
            "artifact_sha256": file_hashes(output),
        },
    )
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/preprocessing_grid_v1")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    torch.set_num_threads(config["torch_threads"])
    b = FrozenBenchmark()
    b.verify_native(b.predict(b.semantic, b.classical))
    root = args.output
    code_hashes = {
        str(p): sha256(p)
        for p in [
            CONFIG,
            PROTOCOL,
            CATALOG,
            ARCHIVE,
            Path(__file__),
            Path("scripts/evaluate_embedding_context.py"),
            *sorted(Path("src/vehicle_audio").glob("*.py")),
        ]
    }
    if root.exists():
        if not args.resume:
            raise FileExistsError(root)
        metadata = json.loads((root / "experiment.json").read_text())
        if metadata["study_input_sha256"] != code_hashes:
            raise ValueError("study inputs/code changed; use a new version")
        verify_hashes(root, metadata["dataset_artifact_sha256"])
        records = [
            json.loads(line)
            for line in (root / "dataset/real_manifest.jsonl").read_text().splitlines()
        ]
    else:
        root.mkdir(parents=True)
        records = prepare(root / "dataset", config, b)
        metadata = b.metadata(config)
        metadata.update(
            study_input_sha256=code_hashes,
            dataset_version=sha256(root / "dataset/real_manifest.jsonl"),
            dataset_artifact_sha256=file_hashes(root),
            model_checkpoint=validate_panns_checkpoint(config["checkpoint"]),
            train_test_split="per-model splits.json; selected grid splits saved after search",
            versions={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
                "torch": str(torch.__version__),
            },
        )
        write_json(root / "experiment.json", metadata)
        shutil.copyfile(CATALOG, root / "audio_sources.yaml")
        shutil.copyfile(ARCHIVE, root / "archived_startup_intervals.yaml")
        shutil.copyfile(CONFIG, root / "protocol.yaml")
        shutil.copyfile(PROTOCOL, root / "protocol.md")
    waveforms = np.stack(
        [
            sf.read(root / "dataset" / r["audio_path"], dtype="float32")[0]
            for r in records
        ]
    )
    labels = torch.tensor([int(r["vehicle_class"] == "wheeled") for r in records])
    from panns_inference import AudioTagging

    tagging = AudioTagging(checkpoint_path=config["checkpoint"], device="cpu")
    classical = ClassicalFeatureExtractor(FeatureConfig()).eval()
    variants = preprocessing_grid(
        config["grid"]["remove_dc"],
        config["grid"]["highpass_hz"],
        config["grid"]["rms_dbfs"],
    )
    print(
        f"DATASET: {len(records)} uncapped reviewed windows; 24 preprocessing variants; 7 tracked / 5 wheeled sessions",
        flush=True,
    )
    for number, variant in enumerate(variants, 1):
        directory = root / "variants" / variant.name
        extraction = directory / "extraction.json"
        if extraction.exists():
            verify_hashes(
                directory, json.loads(extraction.read_text())["artifact_sha256"]
            )
            features = torch.load(directory / "features.pt", weights_only=True)
        else:
            features = extract(
                waveforms, records, variant, tagging, classical, directory, config
            )
        for name in ("semantic35", "classical53"):
            output = directory / name
            if (output / "experiment.json").exists():
                verify_hashes(
                    output,
                    json.loads((output / "experiment.json").read_text())[
                        "artifact_sha256"
                    ],
                )
            else:
                evaluate(features[name], labels, records, output, config)
                experiment = json.loads((output / "experiment.json").read_text())
                experiment["parent_experiment"] = "../../../experiment.json"
                experiment["preprocessing"] = variant.to_dict()
                write_json(output / "experiment.json", experiment)
        print(f"COMPLETED {number}/{len(variants)} {variant.name}", flush=True)
    write_json(root / "artifact_sha256.json", file_hashes(root))
    write_json(
        root / "complete.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "variants": len(variants),
            "representations": ["semantic35", "classical53"],
            "windows": len(records),
        },
    )


if __name__ == "__main__":
    main()
