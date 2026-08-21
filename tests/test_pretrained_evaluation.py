from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from vehicle_audio.baseline import CLASS_NAMES, grouped_stratified_split
from vehicle_audio.invariance_evaluation import build_factorial_split_and_partitions
from vehicle_audio.pretrained_evaluation import (
    METHODS,
    PANN_AUDIOSET_DIMENSION,
    PANN_EMBEDDING_DIMENSION,
    evaluate_precomputed_panns,
    evaluate_real_adaptation,
    validate_panns_checkpoint,
)


def _records_and_embeddings() -> tuple[object, ...]:
    synthetic: list[dict[str, object]] = []
    real: list[dict[str, object]] = []
    synthetic_labels: list[int] = []
    real_labels: list[int] = []
    for class_index, vehicle_class in enumerate(CLASS_NAMES):
        for geometry, group_count in (("near_broadside", 3), ("far_offset", 1)):
            for group_index in range(group_count):
                session = f"synthetic_{vehicle_class}_{geometry}_{group_index}"
                for background in ("rain", "road traffic"):
                    for snr_db in (10.0, 0.0):
                        for microphone in (False, True):
                            synthetic.append(
                                {
                                    "sample_id": f"synthetic_{len(synthetic)}",
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
                            synthetic_labels.append(class_index)
        for session_index in range(3):
            for sample_index in range(2):
                real.append(
                    {
                        "sample_id": f"real_{vehicle_class}_{session_index}_{sample_index}",
                        "vehicle_class": vehicle_class,
                        "vehicle_id": f"real_{vehicle_class}_{session_index}",
                        "source_id": f"source_{vehicle_class}_{session_index}",
                        "recording_session": f"real_{vehicle_class}_{session_index}",
                    }
                )
                real_labels.append(class_index)

    generator = torch.Generator().manual_seed(123)
    clean = 0.03 * torch.randn(
        len(synthetic), PANN_EMBEDDING_DIMENSION, generator=generator
    )
    corrupted = clean + 0.01 * torch.randn(
        len(synthetic), PANN_EMBEDDING_DIMENSION, generator=generator
    )
    native_real = 0.03 * torch.randn(
        len(real), PANN_EMBEDDING_DIMENSION, generator=generator
    )
    clean_clipwise = 0.03 * torch.randn(
        len(synthetic), PANN_AUDIOSET_DIMENSION, generator=generator
    )
    corrupted_clipwise = clean_clipwise + 0.01 * torch.randn(
        len(synthetic), PANN_AUDIOSET_DIMENSION, generator=generator
    )
    native_real_clipwise = 0.03 * torch.randn(
        len(real), PANN_AUDIOSET_DIMENSION, generator=generator
    )
    synthetic_label_tensor = torch.tensor(synthetic_labels)
    real_label_tensor = torch.tensor(real_labels)
    clean[:, 0] += torch.where(synthetic_label_tensor == 0, 1.0, -1.0)
    corrupted[:, 0] += torch.where(synthetic_label_tensor == 0, 1.0, -1.0)
    native_real[:, 0] += torch.where(real_label_tensor == 0, 1.0, -1.0)
    clean_clipwise[:, 0] += torch.where(
        synthetic_label_tensor == 0, 1.0, -1.0
    )
    corrupted_clipwise[:, 0] += torch.where(
        synthetic_label_tensor == 0, 1.0, -1.0
    )
    native_real_clipwise[:, 0] += torch.where(
        real_label_tensor == 0, 1.0, -1.0
    )
    snrs = torch.tensor([float(record["snr_db"]) for record in synthetic])
    return (
        synthetic,
        real,
        clean,
        corrupted,
        clean_clipwise,
        corrupted_clipwise,
        synthetic_label_tensor,
        snrs,
        native_real,
        native_real_clipwise,
        real_label_tensor,
    )


def test_frozen_panns_probes_use_controlled_partitions_deterministically() -> None:
    (
        synthetic,
        real,
        clean,
        corrupted,
        clean_clipwise,
        corrupted_clipwise,
        synthetic_labels,
        snrs,
        native_real,
        native_real_clipwise,
        real_labels,
    ) = _records_and_embeddings()
    _, nuisance = build_factorial_split_and_partitions(
        synthetic, 42, heldout_noise="road traffic", heldout_geometry="far_offset"
    )
    real_split = grouped_stratified_split(real, 42)
    arguments = (
        clean,
        corrupted,
        clean_clipwise,
        corrupted_clipwise,
        synthetic_labels,
        snrs,
        native_real,
        native_real_clipwise,
        real_labels,
        synthetic,
        real,
        nuisance,
        real_split,
    )
    keyword_arguments = {
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "seed": 7,
        "device": torch.device("cpu"),
    }

    first, first_states = evaluate_precomputed_panns(
        *arguments, **keyword_arguments
    )
    second, second_states = evaluate_precomputed_panns(
        *arguments, **keyword_arguments
    )

    assert set(first) == set(METHODS)
    assert first == second
    for method in METHODS:
        assert set(first_states[method]) == set(second_states[method])
        for name in first_states[method]:
            assert torch.equal(first_states[method][name], second_states[method][name])
        evaluations = first[method]["evaluations"]
        assert evaluations["all_corruptions"]["support"] > 0
        assert evaluations["native_real"]["support"] == 4
        assert evaluations["native_real_all_sessions"]["session_count"] == 6
        assert evaluations["native_real_all_sessions"]["support"] == 12


def test_real_adaptation_is_deterministic_and_keeps_test_held_out() -> None:
    (
        synthetic,
        real,
        clean,
        corrupted,
        clean_clipwise,
        corrupted_clipwise,
        synthetic_labels,
        snrs,
        native_real,
        native_real_clipwise,
        real_labels,
    ) = _records_and_embeddings()
    _, nuisance = build_factorial_split_and_partitions(
        synthetic, 42, heldout_noise="road traffic", heldout_geometry="far_offset"
    )
    real_split = grouped_stratified_split(real, 42)
    _, states = evaluate_precomputed_panns(
        clean,
        corrupted,
        clean_clipwise,
        corrupted_clipwise,
        synthetic_labels,
        snrs,
        native_real,
        native_real_clipwise,
        real_labels,
        synthetic,
        real,
        nuisance,
        real_split,
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0001,
        seed=7,
        device=torch.device("cpu"),
    )
    arguments = (
        states["panns_audioset_paired_linear"],
        native_real_clipwise,
        real_labels,
        real,
        real_split,
        (0.1, 1.0),
    )
    keyword_arguments = {
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "seed": 7,
        "subset_seed": 42,
        "device": torch.device("cpu"),
    }

    first, first_states = evaluate_real_adaptation(
        *arguments, **keyword_arguments
    )
    second, second_states = evaluate_real_adaptation(
        *arguments, **keyword_arguments
    )

    assert first == second
    assert {row["key"] for row in first} == {"10_percent_real", "100_percent_real"}
    assert set(first[0]["real_training_sample_ids"]).issubset(
        first[1]["real_training_sample_ids"]
    )
    test_sample_ids = {str(real[index]["sample_id"]) for index in real_split.test}
    for row in first:
        assert row["native_real"]["support"] == len(real_split.test)
        assert set(row["real_training_class_support"]) == set(CLASS_NAMES)
        assert not test_sample_ids.intersection(row["real_training_sample_ids"])
        assert all(value > 0 for value in row["real_training_class_support"].values())
    for key in first_states:
        for name in first_states[key]:
            assert torch.equal(first_states[key][name], second_states[key][name])


def test_checkpoint_validation_records_hash_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "vehicle_audio.pretrained_evaluation.PANN_CHECKPOINT_SHA256", digest
    )

    metadata = validate_panns_checkpoint(checkpoint)

    assert metadata["sha256"] == digest
    assert metadata["size_bytes"] == len(b"test checkpoint")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_panns_checkpoint(checkpoint)
