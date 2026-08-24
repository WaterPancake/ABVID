from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vehicle_audio.historical_checkpoint_audit import (
    load_audit_config,
    summarize_probabilities,
)


def test_summary_uses_argmax_and_session_mean_probabilities() -> None:
    records = [
        {
            "sample_id": "tracked_0",
            "vehicle_class": "tracked",
            "recording_session": "tracked_session",
        },
        {
            "sample_id": "tracked_1",
            "vehicle_class": "tracked",
            "recording_session": "tracked_session",
        },
        {
            "sample_id": "wheeled_0",
            "vehicle_class": "wheeled",
            "recording_session": "wheeled_session",
        },
    ]
    probabilities = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    labels = torch.tensor([0, 0, 1])

    metrics, sessions, predictions = summarize_probabilities(
        probabilities, labels, records
    )

    assert predictions.tolist() == [0, 1, 1]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["snr_available"] is False
    assert sessions["tracked_session"]["predicted_class"] == "tracked"
    assert sessions["tracked_session"]["correct"] is True
    assert sessions["wheeled_session"]["correct"] is True


def test_audit_config_rejects_duplicate_model_ids(tmp_path: Path) -> None:
    config = {
        "audit_version": 1,
        "selection_policy": "evaluate all predeclared models",
        "models": [
            {
                "id": "duplicate",
                "milestone": 2,
                "architecture": "cnn",
                "kind": "standalone",
                "checkpoint": "first.pt",
                "training_domain": "synthetic",
            },
            {
                "id": "duplicate",
                "milestone": 3,
                "architecture": "cnn",
                "kind": "standalone",
                "checkpoint": "second.pt",
                "training_domain": "synthetic",
            },
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate model id"):
        load_audit_config(path)


def test_repository_audit_roster_is_predeclared_and_complete() -> None:
    config = load_audit_config("configs/historical_checkpoint_audit.yaml")
    identifiers = {model["id"] for model in config["models"]}

    assert len(identifiers) == 29
    assert {model["milestone"] for model in config["models"]} == {2, 3, 4, 5, 6}
    assert "m4_single_mic_1" in identifiers
    assert "m6_frozen_semantic_classical_fusion" in identifiers
    assert sum(model["kind"] == "preserved_result" for model in config["models"]) == 1
