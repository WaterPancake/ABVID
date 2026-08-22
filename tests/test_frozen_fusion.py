from __future__ import annotations

import numpy as np
import pytest
import torch

from vehicle_audio.frozen_fusion import (
    predict_frozen_fusion,
    select_and_fit_frozen_fusion,
    validate_locked_records,
)


pytest.importorskip("sklearn")


def _toy_data() -> tuple[
    list[dict[str, str]], torch.Tensor, torch.Tensor, torch.Tensor
]:
    records: list[dict[str, str]] = []
    semantic_rows: list[list[float]] = []
    classical_rows: list[list[float]] = []
    labels: list[int] = []
    generator = np.random.default_rng(123)
    for class_index, vehicle_class in enumerate(("tracked", "wheeled")):
        sign = -1.0 if class_index == 0 else 1.0
        for session_index in range(3):
            session = f"{vehicle_class}_{session_index}"
            for sample_index in range(3):
                records.append(
                    {
                        "sample_id": f"{session}_{sample_index}",
                        "vehicle_class": vehicle_class,
                        "recording_session": session,
                        "source_id": f"source_{session}",
                    }
                )
                semantic_rows.append(
                    [sign + 0.02 * generator.standard_normal(), float(session_index)]
                )
                classical_rows.append(
                    [sign + 0.02 * generator.standard_normal(), float(sample_index)]
                )
                labels.append(class_index)
    return (
        records,
        torch.tensor(semantic_rows, dtype=torch.float32),
        torch.tensor(classical_rows, dtype=torch.float32),
        torch.tensor(labels),
    )


def test_freeze_selects_global_threshold_and_serializable_predictor() -> None:
    records, semantic, classical, labels = _toy_data()

    selection, state, splits = select_and_fit_frozen_fusion(
        semantic,
        classical,
        labels,
        records,
        c_values=(0.1, 1.0),
        threshold_values=(0.25, 0.5, 0.75),
        seed=42,
    )
    probabilities, predictions = predict_frozen_fusion(semantic, classical, state)

    assert selection["selection_protocol_status"] == "development_selection_only"
    assert selection["selected_wheeled_threshold"] in {0.25, 0.5, 0.75}
    assert selection["selected_threshold_aggregate"]["outer_fold_count"] == 9
    assert selection["selected_threshold_aggregate"]["balanced_accuracy"]["mean"] >= 0.9
    assert len(splits["folds"]) == 9
    assert probabilities.shape == (18, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(18), atol=1e-6)
    assert torch.equal(predictions, labels)


def test_locked_validation_rejects_development_provenance_overlap() -> None:
    locked = [
        {
            "sample_id": "locked_t",
            "vehicle_class": "tracked",
            "recording_session": "new_tracked",
            "source_id": "new_source_t",
            "original_media_source": "https://example.test/tracked",
            "normalized_source_sha256": "tracked-sha",
            "provenance_complete": True,
        },
        {
            "sample_id": "locked_w",
            "vehicle_class": "wheeled",
            "recording_session": "new_wheeled",
            "source_id": "new_source_w",
            "original_media_source": "https://example.test/wheeled",
            "normalized_source_sha256": "wheeled-sha",
            "provenance_complete": True,
        },
    ]
    development = {
        "recording_session": ["old_tracked", "old_wheeled"],
        "source_id": ["old_source"],
    }

    observed = validate_locked_records(locked, development)
    assert observed["recording_session"] == ["new_tracked", "new_wheeled"]
    locked[1]["source_id"] = "old_source"
    with pytest.raises(ValueError, match="overlaps development source_id"):
        validate_locked_records(locked, development)
