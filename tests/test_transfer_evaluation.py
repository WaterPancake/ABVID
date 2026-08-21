from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vehicle_audio.audio import save_audio
from vehicle_audio.transfer_evaluation import (
    evaluate_transfer,
    stratified_fraction_indices,
    validate_transfer_protocol,
)


def _write_transfer_manifests(root: Path) -> tuple[Path, Path]:
    sample_rate = 8_000
    num_samples = 4_000
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    synthetic_rows: list[dict[str, object]] = []
    real_rows: list[dict[str, object]] = []
    for domain in ("synthetic", "real"):
        for class_index, vehicle_class in enumerate(("tracked", "wheeled")):
            for session_index in range(3):
                session = f"{domain}_{vehicle_class}_session_{session_index}"
                for clip_index in range(2):
                    base = 180.0 if vehicle_class == "tracked" else 380.0
                    domain_shift = 20.0 if domain == "real" else 0.0
                    frequency = base + domain_shift + session_index + clip_index
                    waveform = 0.2 * torch.sin(2.0 * torch.pi * frequency * time)
                    relative = Path(domain) / f"{session}_{clip_index}.wav"
                    save_audio(root / relative, waveform.unsqueeze(0), sample_rate)
                    common: dict[str, object] = {
                        "sample_id": f"{domain}_{vehicle_class}_{session_index}_{clip_index}",
                        "vehicle_class": vehicle_class,
                        "recording_session": session,
                        "sample_rate": sample_rate,
                        "num_channels": 1,
                        "num_samples": num_samples,
                        "corrupted_path": relative.name,
                    }
                    if domain == "synthetic":
                        common.update(
                            {
                                "source_domain": "procedural_synthetic",
                                "snr_db": 10.0,
                            }
                        )
                        synthetic_rows.append(common)
                    else:
                        common.update(
                            {
                                "source_domain": "real_recording",
                                "observation_domain": "native_real_recording",
                                "audio_path": relative.name,
                                "metadata_path": f"metadata/{common['sample_id']}.json",
                                "source_id": f"source-{session}",
                                "snr_db": None,
                                "window_start_sample": clip_index * num_samples,
                                "provenance_complete": True,
                                "content_review_status": "reviewed_segment",
                            }
                        )
                        real_rows.append(common)
    synthetic_manifest = root / "synthetic" / "manifest.jsonl"
    real_manifest = root / "real" / "real_manifest.jsonl"
    synthetic_manifest.parent.mkdir(parents=True, exist_ok=True)
    real_manifest.parent.mkdir(parents=True, exist_ok=True)
    synthetic_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in synthetic_rows),
        encoding="utf-8",
    )
    real_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in real_rows),
        encoding="utf-8",
    )
    return synthetic_manifest, real_manifest


def test_stratified_fraction_is_deterministic_and_keeps_both_classes() -> None:
    labels = torch.tensor([0] * 10 + [1] * 6)
    candidates = tuple(range(16))
    first = stratified_fraction_indices(labels, candidates, 0.1, seed=42)
    second = stratified_fraction_indices(labels, candidates, 0.1, seed=42)

    assert first == second
    assert {int(labels[index]) for index in first} == {0, 1}


def test_transfer_protocol_rejects_insufficient_real_sessions() -> None:
    synthetic = [
        {
            "source_domain": "procedural_synthetic",
            "vehicle_class": vehicle_class,
            "sample_rate": 8_000,
        }
        for vehicle_class in ("tracked", "wheeled")
    ]
    real = [
        {
            "sample_id": f"real_{vehicle_class}",
            "audio_path": "audio.wav",
            "metadata_path": "metadata.json",
            "vehicle_class": vehicle_class,
            "recording_session": f"{vehicle_class}_only_session",
            "source_domain": "real_recording",
            "source_id": f"source-{vehicle_class}",
            "sample_rate": 8_000,
            "num_channels": 1,
            "num_samples": 2_000,
            "window_start_sample": 0,
            "snr_db": None,
            "provenance_complete": True,
            "content_review_status": "reviewed_segment",
        }
        for vehicle_class in ("tracked", "wheeled")
    ]
    with pytest.raises(ValueError, match="at least three independent"):
        validate_transfer_protocol(synthetic, real)


def test_transfer_protocol_rejects_duration_domain_cue() -> None:
    synthetic = [
        {
            "source_domain": "procedural_synthetic",
            "vehicle_class": vehicle_class,
            "sample_rate": 8_000,
            "num_samples": 32_000,
        }
        for vehicle_class in ("tracked", "wheeled")
    ]
    real = [
        {
            "sample_id": f"real_{vehicle_class}_{session}",
            "audio_path": "audio.wav",
            "metadata_path": "metadata.json",
            "vehicle_class": vehicle_class,
            "recording_session": f"{vehicle_class}_session_{session}",
            "source_domain": "real_recording",
            "source_id": f"source-{vehicle_class}-{session}",
            "sample_rate": 8_000,
            "num_channels": 1,
            "num_samples": 16_000,
            "window_start_sample": 0,
            "snr_db": None,
            "provenance_complete": True,
            "content_review_status": "reviewed_segment",
        }
        for vehicle_class in ("tracked", "wheeled")
        for session in range(3)
    ]

    with pytest.raises(ValueError, match="observation shapes differ"):
        validate_transfer_protocol(synthetic, real)


def test_toy_transfer_protocol_writes_all_required_artifacts(tmp_path: Path) -> None:
    synthetic_manifest, real_manifest = _write_transfer_manifests(tmp_path)
    output = tmp_path / "run"
    results = evaluate_transfer(
        synthetic_manifest,
        real_manifest,
        output,
        real_fractions=(0.5,),
        seed=7,
        pretrain_epochs=1,
        finetune_epochs=1,
        real_only_epochs=1,
        batch_size=4,
        split_seed=11,
        device_name="cpu",
    )

    assert results["protocol_status"] == "complete"
    assert results["training_config"]["split_seed"] == 11
    assert results["test_domain"] == "held-out native real recording sessions"
    assert results["split_counts"]["real"] == {"train": 4, "validation": 4, "test": 4}
    assert results["experiments"]["A_real_only"]["test"]["support"] == 4
    assert results["experiments"]["B_synthetic_only"]["test"]["support"] == 4
    assert (
        results["experiments"]["C_synthetic_pretraining_limited_real"]
        ["50_percent_real"]["test"]["support"]
        == 4
    )
    for name in (
        "metrics.json",
        "experiment.json",
        "splits.json",
        "models.pt",
        "domain_features.pt",
        "learning_curve.csv",
        "learning_curve.png",
        "embedding_projection.csv",
        "embedding_projection.png",
    ):
        assert (output / name).is_file()
    split_payload = json.loads((output / "splits.json").read_text())
    assert split_payload["seed"] == 11
    for domain in ("synthetic", "real"):
        groups = [
            set(split_payload[domain][name]["groups"])
            for name in ("train", "validation", "test")
        ]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
