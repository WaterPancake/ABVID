from __future__ import annotations

import torch

from vehicle_audio.locked_fusion_evaluation import summarize_locked_predictions


def test_locked_summary_uses_frozen_threshold_for_windows_and_sessions() -> None:
    records = [
        {
            "sample_id": "tracked_0",
            "vehicle_class": "tracked",
            "recording_session": "new_tracked",
        },
        {
            "sample_id": "tracked_1",
            "vehicle_class": "tracked",
            "recording_session": "new_tracked",
        },
        {
            "sample_id": "wheeled_0",
            "vehicle_class": "wheeled",
            "recording_session": "new_wheeled",
        },
        {
            "sample_id": "wheeled_1",
            "vehicle_class": "wheeled",
            "recording_session": "new_wheeled",
        },
    ]
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.65, 0.35], [0.55, 0.45], [0.1, 0.9]]
    )
    labels = torch.tensor([0, 0, 1, 1])

    metrics, sessions, predictions = summarize_locked_predictions(
        probabilities,
        labels,
        records,
        wheeled_threshold=0.4,
    )

    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["calibration_available"] is False
    assert sessions["new_tracked"]["correct"] is True
    assert sessions["new_wheeled"]["correct"] is True
    assert torch.equal(predictions, labels)
