from __future__ import annotations

import json
from pathlib import Path

import torch

from vehicle_audio.audio import save_audio
from vehicle_audio.invariance_evaluation import (
    METHODS,
    build_factorial_split_and_partitions,
    evaluate_invariance,
    hard_negative_partner_indices,
    hard_positive_partner_indices,
)


def _write_invariance_manifests(root: Path) -> tuple[Path, Path]:
    sample_rate = 8_000
    num_samples = 4_000
    time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    synthetic_root = root / "synthetic"
    real_root = root / "real"
    synthetic_rows: list[dict[str, object]] = []
    real_rows: list[dict[str, object]] = []
    nuisance_conditions = (
        ("rain", "near_broadside", False, "idle"),
        ("animals/insects", "near_broadside", False, "accelerating"),
        ("road traffic", "near_broadside", False, "steady_speed"),
        ("rain", "near_broadside", True, "decelerating"),
        ("rain", "far_offset", False, "steady_speed"),
    )

    for class_index, vehicle_class in enumerate(("tracked", "wheeled")):
        base_frequency = 180.0 if vehicle_class == "tracked" else 380.0
        for session_index in range(3):
            session = f"synthetic_{vehicle_class}_session_{session_index}"
            for clip_index, (
                background,
                geometry,
                microphone_applied,
                operating_condition,
            ) in enumerate(nuisance_conditions):
                frequency = base_frequency + 2 * session_index + clip_index
                clean = 0.2 * torch.sin(2.0 * torch.pi * frequency * time)
                noise_generator = torch.Generator().manual_seed(
                    1000 * class_index + 100 * session_index + clip_index
                )
                corrupted = clean + 0.015 * torch.randn(
                    num_samples, generator=noise_generator
                )
                sample_id = f"synthetic_{vehicle_class}_{session_index}_{clip_index}"
                sample_dir = synthetic_root / sample_id
                save_audio(sample_dir / "clean.wav", clean.unsqueeze(0), sample_rate)
                save_audio(
                    sample_dir / "corrupted.wav", corrupted.unsqueeze(0), sample_rate
                )
                synthetic_rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_class": vehicle_class,
                        "vehicle_id": f"toy_{vehicle_class}",
                        "recording_session": session,
                        "source_domain": "procedural_synthetic",
                        "operating_condition": operating_condition,
                        "sample_rate": sample_rate,
                        "num_channels": 1,
                        "num_samples": num_samples,
                        "clean_path": f"{sample_id}/clean.wav",
                        "corrupted_path": f"{sample_id}/corrupted.wav",
                        "snr_db": float(20 - 5 * clip_index),
                        "background_category": background,
                        "source_listener_geometry": {"geometry_id": geometry},
                        "augmentations": {
                            "microphone_response": {
                                "applied": microphone_applied,
                            }
                        },
                    }
                )

            real_session = f"real_{vehicle_class}_session_{session_index}"
            for clip_index in range(2):
                frequency = base_frequency + 20 + session_index + clip_index
                waveform = 0.2 * torch.sin(2.0 * torch.pi * frequency * time)
                sample_id = f"real_{vehicle_class}_{session_index}_{clip_index}"
                relative_path = f"{sample_id}.wav"
                save_audio(real_root / relative_path, waveform.unsqueeze(0), sample_rate)
                real_rows.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": relative_path,
                        "metadata_path": f"metadata/{sample_id}.json",
                        "vehicle_class": vehicle_class,
                        "recording_session": real_session,
                        "source_domain": "real_recording",
                        "observation_domain": "native_real_recording",
                        "source_id": f"source-{real_session}",
                        "sample_rate": sample_rate,
                        "num_channels": 1,
                        "num_samples": num_samples,
                        "window_start_sample": clip_index * num_samples,
                        "snr_db": None,
                        "provenance_complete": True,
                        "content_review_status": "reviewed_segment",
                    }
                )

    synthetic_manifest = synthetic_root / "manifest.jsonl"
    real_manifest = real_root / "real_manifest.jsonl"
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


def test_partner_mining_is_deterministic_and_respects_pair_semantics() -> None:
    records = [
        {
            "sample_id": "tracked_idle",
            "vehicle_class": "tracked",
            "vehicle_id": "tracked_a",
            "recording_session": "tracked_run_1",
            "operating_condition": "idle",
            "background_category": "rain",
            "snr_db": 10.0,
        },
        {
            "sample_id": "tracked_accelerating",
            "vehicle_class": "tracked",
            "vehicle_id": "tracked_a",
            "recording_session": "tracked_run_2",
            "operating_condition": "accelerating",
            "background_category": "rain",
            "snr_db": 10.0,
        },
        {
            "sample_id": "wheeled_idle",
            "vehicle_class": "wheeled",
            "vehicle_id": "wheeled_a",
            "recording_session": "wheeled_run_1",
            "operating_condition": "idle",
            "background_category": "rain",
            "snr_db": 10.0,
        },
    ]
    indices = tuple(range(len(records)))

    positives = hard_positive_partner_indices(records, indices)
    negatives = hard_negative_partner_indices(records, indices)

    assert positives == hard_positive_partner_indices(records, indices)
    assert negatives == hard_negative_partner_indices(records, indices)
    assert positives[0] == 1
    assert positives[1] == 0
    assert positives[2] is None
    assert negatives[0] == 2
    assert negatives[1] == 2
    assert records[negatives[0]]["vehicle_class"] != records[0]["vehicle_class"]


def test_factorial_split_produces_balanced_controlled_partitions() -> None:
    records: list[dict[str, object]] = []
    for vehicle_class in ("tracked", "wheeled"):
        for geometry, group_count in (("near_broadside", 3), ("far_offset", 1)):
            for group_index in range(group_count):
                session = f"{vehicle_class}_{geometry}_{group_index}"
                for background in ("rain", "road traffic"):
                    for snr_db in (10.0, 0.0):
                        for microphone in (False, True):
                            records.append(
                                {
                                    "sample_id": f"sample_{len(records)}",
                                    "vehicle_class": vehicle_class,
                                    "recording_session": session,
                                    "background_category": background,
                                    "snr_db": snr_db,
                                    "corruption_view": 0,
                                    "factorial_dataset_version": 1,
                                    "source_listener_geometry": {
                                        "geometry_id": geometry
                                    },
                                    "augmentations": {
                                        "microphone_response": {
                                            "applied": microphone
                                        }
                                    },
                                }
                            )

    split, partitions = build_factorial_split_and_partitions(
        records,
        42,
        heldout_noise="road traffic",
        heldout_geometry="far_offset",
    )

    assert len(split.train) == len(split.validation) == 16
    assert len(split.test) == 32
    assert len(partitions["train_in_distribution"]) == 4
    assert len(partitions["validation_in_distribution"]) == 4
    assert len(partitions["seen_corruption"]) == 4
    assert len(partitions["unseen_noise"]) == 4
    assert len(partitions["unseen_microphone"]) == 4
    assert len(partitions["unseen_environment"]) == 4
    for indices in partitions.values():
        counts = {
            vehicle_class: sum(
                records[index]["vehicle_class"] == vehicle_class for index in indices
            )
            for vehicle_class in ("tracked", "wheeled")
        }
        assert counts["tracked"] == counts["wheeled"]


def test_toy_invariance_protocol_writes_required_artifacts(tmp_path: Path) -> None:
    synthetic_manifest, real_manifest = _write_invariance_manifests(tmp_path)
    output = tmp_path / "run"

    results = evaluate_invariance(
        synthetic_manifest,
        real_manifest,
        output,
        epochs=1,
        batch_size=4,
        seed=7,
        split_seed=11,
        device_name="cpu",
    )

    assert results["protocol_status"] == "complete"
    assert results["training_config"]["split_seed"] == 11
    assert set(results["methods"]) == set(METHODS)
    assert results["training_config"]["hard_positive_pair_count"] == 4
    assert results["training_config"]["hard_negative_pair_count"] == 4
    for method in METHODS:
        evaluations = results["methods"][method]["evaluations"]
        for condition in (
            "seen_corruption",
            "unseen_noise",
            "unseen_microphone",
            "unseen_environment",
            "all_corruptions",
            "native_real",
            "native_real_all_sessions",
        ):
            assert evaluations[condition]["support"] > 0
        all_real = evaluations["native_real_all_sessions"]
        assert all_real["support"] == 12
        assert all_real["session_count"] == 6
        assert set(all_real["class_session_mean_recall"]) == {"tracked", "wheeled"}
        assert len(all_real["per_session"]) == 6

    for name in (
        "metrics.json",
        "experiment.json",
        "splits.json",
        "models.pt",
        "paired_features.pt",
        "condition_summary.csv",
        "snr_robustness.csv",
        "condition_comparison.png",
        "accuracy_vs_snr.png",
    ):
        assert (output / name).is_file()

    splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert splits["seed"] == 11
    for domain in ("synthetic", "real"):
        groups = [
            set(splits[domain][name]["groups"])
            for name in ("train", "validation", "test")
        ]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
