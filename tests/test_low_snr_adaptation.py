from __future__ import annotations

import torch

from vehicle_audio.low_snr_adaptation import (
    adapt_probe_fixed_epochs,
    leave_session_pair_out_folds,
)
from vehicle_audio.pretrained_evaluation import StandardizedLinearProbe


def _records() -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"{vehicle_class}_{session}_{sample}",
            "vehicle_class": vehicle_class,
            "recording_session": f"{vehicle_class}_{session}",
        }
        for vehicle_class in ("tracked", "wheeled")
        for session in range(3)
        for sample in range(2)
    ]


def test_leave_session_pair_out_covers_every_pair_without_group_leakage() -> None:
    records = _records()
    folds = leave_session_pair_out_folds(records)

    assert len(folds) == 9
    for fold in folds:
        train_sessions = {
            records[index]["recording_session"] for index in fold["train_indices"]
        }
        test_sessions = {
            records[index]["recording_session"] for index in fold["test_indices"]
        }
        assert train_sessions.isdisjoint(test_sessions)
        assert len(test_sessions) == 2


def test_fixed_epoch_adaptation_is_deterministic() -> None:
    records = _records()
    labels = torch.tensor(
        [0 if record["vehicle_class"] == "tracked" else 1 for record in records]
    )
    features = torch.randn(len(records), 5, generator=torch.Generator().manual_seed(1))
    base = StandardizedLinearProbe(torch.zeros(5), torch.ones(5), 2)
    initial_state = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    arguments = (
        initial_state,
        features,
        labels,
        records,
        tuple(range(len(records))),
    )
    keywords = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "seed": 42,
        "device": torch.device("cpu"),
    }

    first, first_history = adapt_probe_fixed_epochs(*arguments, **keywords)
    second, second_history = adapt_probe_fixed_epochs(*arguments, **keywords)

    assert first_history == second_history
    assert all(torch.equal(first[name], second[name]) for name in first)
