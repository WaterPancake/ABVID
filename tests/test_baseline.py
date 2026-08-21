from __future__ import annotations

import pytest
import torch

from vehicle_audio.baseline import (
    ClassicalFeatureExtractor,
    FeatureConfig,
    LogisticRegressionBaseline,
    LogMelExtractor,
    SmallAudioCNN,
    classification_metrics,
    grouped_stratified_split,
    holdout_split,
)


def _grouped_records() -> list[dict[str, object]]:
    return [
        {
            "vehicle_class": vehicle_class,
            "snr_db": snr_db,
            "recording_session": f"{vehicle_class}_session_{session}",
        }
        for vehicle_class in ("tracked", "wheeled")
        for session in range(3)
        for snr_db in (-5.0, 10.0)
        for _ in range(2)
    ]


def test_grouped_split_is_disjoint_and_never_splits_recording_sessions() -> None:
    records = _grouped_records()
    split = grouped_stratified_split(records, seed=42)

    partitions = (set(split.train), set(split.validation), set(split.test))
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(range(len(records)))
    session_sets = [
        {str(records[index]["recording_session"]) for index in indices}
        for indices in partitions
    ]
    assert not session_sets[0] & session_sets[1]
    assert not session_sets[0] & session_sets[2]
    assert not session_sets[1] & session_sets[2]
    for indices in partitions:
        assert {records[index]["vehicle_class"] for index in indices} == {
            "tracked",
            "wheeled",
        }


def test_grouped_split_rejects_insufficient_sessions() -> None:
    records = [
        {
            "vehicle_class": vehicle_class,
            "snr_db": 5.0,
            "recording_session": f"{vehicle_class}_{session}",
        }
        for vehicle_class in ("tracked", "wheeled")
        for session in range(2)
    ]
    with pytest.raises(ValueError, match="at least three"):
        grouped_stratified_split(records, seed=1)


def test_holdout_split_keeps_complete_operating_conditions() -> None:
    records = [
        {"vehicle_class": vehicle_class, "operating_condition": condition}
        for vehicle_class in ("tracked", "wheeled")
        for condition in ("idle", "accelerating", "steady_speed", "decelerating")
        for _ in range(3)
    ]
    split = holdout_split(
        records,
        field="operating_condition",
        validation_values=("idle",),
        test_values=("decelerating",),
    )

    assert {records[index]["operating_condition"] for index in split.validation} == {"idle"}
    assert {records[index]["operating_condition"] for index in split.test} == {"decelerating"}
    assert {records[index]["operating_condition"] for index in split.train} == {
        "accelerating",
        "steady_speed",
    }


def test_classical_log_mel_and_model_shapes_are_finite() -> None:
    config = FeatureConfig(
        sample_rate=8_000,
        n_fft=256,
        hop_length=64,
        n_mels=24,
        n_mfcc=12,
        f_max=4_000,
    )
    waveforms = torch.randn((3, 2_000), generator=torch.Generator().manual_seed(9))
    log_mels = LogMelExtractor(config)(waveforms)
    classical = ClassicalFeatureExtractor(config)(waveforms)
    cnn_logits = SmallAudioCNN()(log_mels)
    classical_logits = LogisticRegressionBaseline(classical.shape[1])(classical)

    assert log_mels.shape[:3] == (3, 1, 24)
    assert classical.shape == (3, 2 * 12 + 13)
    assert cnn_logits.shape == (3, 2)
    assert classical_logits.shape == (3, 2)
    assert torch.isfinite(log_mels).all()
    assert torch.isfinite(classical).all()
    assert torch.isfinite(cnn_logits).all()
    assert torch.isfinite(classical_logits).all()


def test_classification_metrics_report_full_metrics_calibration_and_snr() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.3, 0.7],
        ]
    )
    predictions = probabilities.argmax(dim=1)
    targets = torch.tensor([0, 1, 1, 1, 0, 1])
    snrs = torch.tensor([10.0, 10.0, 10.0, -5.0, -5.0, -5.0])

    metrics = classification_metrics(predictions, targets, snrs, probabilities)

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx((1 / 2 + 3 / 4) / 2)
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["per_class"]["tracked"]["precision"] == pytest.approx(1 / 2)
    assert metrics["per_snr"]["10"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_snr"]["-5"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["calibration"]["expected_calibration_error"] >= 0.0
    assert metrics["calibration"]["negative_log_likelihood"] >= 0.0
