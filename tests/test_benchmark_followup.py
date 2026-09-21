import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from vehicle_audio.benchmark_followup import ensemble_probability, validate_split

spec = importlib.util.spec_from_file_location(
    "corruption_runner",
    Path(__file__).parents[1] / "scripts/evaluate_controlled_corruption.py",
)
corruption = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corruption)


def test_saved_probability_matches_sklearn():
    x = np.random.default_rng(42).normal(size=(30, 5))
    y = np.arange(30) % 2
    scaler = StandardScaler().fit(x)
    model = LogisticRegression().fit(scaler.transform(x), y)
    state = {
        "feature_mean": torch.tensor(scaler.mean_),
        "feature_scale": torch.tensor(scaler.scale_),
        "coefficient": torch.tensor(model.coef_),
        "intercept": torch.tensor(model.intercept_),
    }
    np.testing.assert_allclose(
        ensemble_probability(x, {"c": state}),
        model.predict_proba(scaler.transform(x))[:, 1],
    )


def test_session_leakage_rejected():
    records = [
        {"sample_id": "a", "recording_session": "same"},
        {"sample_id": "b", "recording_session": "same"},
    ]
    with pytest.raises(ValueError, match="session leakage"):
        validate_split(
            records,
            {
                "train_sample_ids": ["a"],
                "test_sample_ids": ["b"],
                "test_sessions": {"tracked": ["same"]},
            },
        )


def test_incomplete_split_rejected():
    records = [
        {"sample_id": "a", "recording_session": "one"},
        {"sample_id": "b", "recording_session": "two"},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        validate_split(records, {"train_sample_ids": [], "test_sample_ids": ["b"]})


@pytest.mark.parametrize("snr", [30, 20, 10, 5, 0, -5, -10])
def test_corruption_deterministic_and_snr(snr):
    g = torch.Generator().manual_seed(42)
    x, n = torch.randn((1, 32000), generator=g), torch.randn((1, 32000), generator=g)
    a, ma = corruption.corrupt_window(x, n, snr)
    b, mb = corruption.corrupt_window(x, n, snr)
    assert torch.equal(a, b) and ma == mb
    assert abs(ma["measured_snr_db"] - snr) < 1e-5
    assert a.abs().max() <= 0.990001 and a.shape == x.shape
    assert ma["common_gain"] < 1


def test_offset_stable_and_no_short_background():
    assert corruption.stable_offset(
        42, "sample", "background", 99999, 32000
    ) == corruption.stable_offset(42, "sample", "background", 99999, 32000)
    assert corruption.stable_offset(42, "sample", "background", 32000, 32000) == 0
    with pytest.raises(ValueError, match="complete window"):
        corruption.stable_offset(42, "sample", "background", 100, 32000)


def test_flat_microphone_response_is_identity_below_peak_limit():
    x = torch.sin(torch.arange(32000) * 0.01).unsqueeze(0) * 0.2
    y, meta = corruption.corrupt_window(
        x, response={"frequencies_hz": [0, 8000], "gains_db": [0, 0]}
    )
    torch.testing.assert_close(x, y)
    assert meta["common_gain"] == 1
