from __future__ import annotations

import pytest
import torch

from vehicle_audio.event_inference import (
    AudibilityGateConfig,
    EventWindowConfig,
    aggregate_event_predictions,
)


def test_temporal_aggregation_excludes_faint_event_edges() -> None:
    probabilities = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )
    rms = torch.tensor([0.01, 0.08, 0.10, 0.07, 0.01])

    result = aggregate_event_predictions(
        probabilities,
        rms,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        window_seconds=2.0,
        gate=AudibilityGateConfig(
            relative_peak_threshold_db=-6.0,
            minimum_active_windows=3,
        ),
    )

    assert result["status"] == "classified"
    assert result["predicted_class"] == "tracked"
    assert result["active_window_count"] == 3
    assert result["windows"][0]["audibility_gate_passed"] is False
    assert result["windows"][4]["audibility_gate_passed"] is False
    assert result["gate"]["is_vehicle_presence_detector"] is False


def test_temporal_aggregation_abstains_when_too_few_windows_pass() -> None:
    result = aggregate_event_predictions(
        torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]),
        torch.tensor([0.1, 0.01, 0.01]),
        [0.0, 1.0, 2.0],
        window_seconds=2.0,
        gate=AudibilityGateConfig(
            relative_peak_threshold_db=-3.0,
            minimum_active_windows=2,
        ),
    )

    assert result["status"] == "insufficient_audible_audio"
    assert result["predicted_class"] is None
    assert result["class_probabilities"] is None


def test_event_window_configuration_rejects_gapped_windows() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        EventWindowConfig(window_seconds=1.0, hop_seconds=2.0)


def test_audibility_gate_requires_explicit_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="at most zero"):
        AudibilityGateConfig(relative_peak_threshold_db=1.0)
