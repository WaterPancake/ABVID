from __future__ import annotations

import torch

from vehicle_audio.audibility_calibration import (
    select_gate_threshold,
    threshold_summary,
)


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for vehicle_class, session in (
        ("tracked", "tracked_session"),
        ("wheeled", "wheeled_session"),
    ):
        for index, rms in enumerate((0.01, 0.1, 0.09, 0.08, 0.01)):
            rows.append(
                {
                    "sample_id": f"{session}_{index}",
                    "vehicle_class": vehicle_class,
                    "recording_session": session,
                    "rms": rms,
                }
            )
    return rows


def test_threshold_summary_counts_abstention_as_incorrect() -> None:
    records = _records()
    probabilities = torch.tensor([[0.8, 0.2]] * 5 + [[0.2, 0.8]] * 5)

    value = threshold_summary(
        probabilities,
        records,
        relative_peak_threshold_db=-3.0,
        minimum_active_windows=4,
    )

    assert value["sessions_abstained"] == 2
    assert value["sessions_correct"] == 0
    assert value["session_balanced_accuracy"] == 0.0


def test_threshold_selection_uses_session_then_window_accuracy() -> None:
    summaries = [
        {
            "relative_peak_threshold_db": -12.0,
            "session_balanced_accuracy": 1.0,
            "coverage": 0.5,
            "active_window_metrics": {"balanced_accuracy": 0.9},
        },
        {
            "relative_peak_threshold_db": -24.0,
            "session_balanced_accuracy": 1.0,
            "coverage": 0.9,
            "active_window_metrics": {"balanced_accuracy": 0.8},
        },
        {
            "relative_peak_threshold_db": -6.0,
            "session_balanced_accuracy": 0.8,
            "coverage": 0.4,
            "active_window_metrics": {"balanced_accuracy": 1.0},
        },
    ]

    assert select_gate_threshold(summaries)["relative_peak_threshold_db"] == -12.0
