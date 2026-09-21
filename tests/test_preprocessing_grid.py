import numpy as np
import pytest
from pathlib import Path
import yaml

from vehicle_audio.preprocessing_grid import Preprocessing, preprocess, selection_key


def test_control_exact_and_deterministic():
    x = np.random.default_rng(1).normal(0, 0.1, (3, 32000)).astype(np.float32)
    y, gain = preprocess(x, Preprocessing())
    assert np.array_equal(x, y)
    assert np.array_equal(gain, np.ones(3))
    a = preprocess(x, Preprocessing(True, 50, -20))[0]
    assert np.array_equal(a, preprocess(x, Preprocessing(True, 50, -20))[0])
    assert a.shape == x.shape


def test_dc_silence_and_gain_cap():
    y, _ = preprocess(np.ones((1, 32000), dtype=np.float32), Preprocessing(True))
    assert np.max(np.abs(y)) == 0
    y, gain = preprocess(
        np.zeros((1, 32000), dtype=np.float32), Preprocessing(True, 80, -20)
    )
    assert np.max(np.abs(y)) == 0 and gain[0] == 1
    x = np.full((1, 32000), 1e-4, dtype=np.float32)
    _, gain = preprocess(x, Preprocessing(rms_dbfs=-20))
    assert gain[0] == pytest.approx(10 ** (12 / 20))


def test_highpass_rejects_low_frequency_not_high():
    t = np.arange(32000) / 16000
    x = np.stack([np.sin(2 * np.pi * f * t) for f in (10, 1000)]).astype(np.float32)
    y, _ = preprocess(x, Preprocessing(highpass_hz=80))
    assert np.std(y[0, 1000:-1000]) < 0.01
    assert np.std(y[1, 1000:-1000]) == pytest.approx(np.std(x[1]), rel=0.01)


def test_rms_target_and_batch_independence():
    t = np.arange(32000) / 16000
    x = np.stack(
        [0.05 * np.sin(2 * np.pi * 300 * t), 0.2 * np.sin(2 * np.pi * 400 * t)]
    ).astype(np.float32)
    config = Preprocessing(True, 30, -20)
    y, gain = preprocess(x, config)
    assert np.sqrt(np.mean(y**2, axis=1)) == pytest.approx([0.1, 0.1], rel=1e-6)
    # Another recording in the batch cannot change this recording's frontend.
    single, single_gain = preprocess(x[:1], config)
    assert np.array_equal(single[0], y[0])
    assert single_gain[0] == gain[0]


def test_gain_peak_limit_no_clipping_and_invalid_input():
    x = np.zeros((1, 32000), np.float32)
    x[0, 123] = 0.9
    y, gain = preprocess(x, Preprocessing(rms_dbfs=-20))
    assert np.max(y) <= 0.990001
    assert gain[0] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        preprocess(np.full((1, 100), np.nan), Preprocessing())
    with pytest.raises(ValueError):
        preprocess(x, Preprocessing(highpass_hz=9000))


def test_selection_ignores_outer_results():
    fold = {
        "selected_regularization_c": 0.01,
        "candidate_selection": [
            {
                "regularization_c": 0.01,
                "mean_balanced_accuracy": 0.6,
                "median_balanced_accuracy": 0.5,
            }
        ],
        "metrics": {"balanced_accuracy": 0.0},
    }
    first = selection_key(Preprocessing(), fold, 0)
    fold["metrics"]["balanced_accuracy"] = 1.0
    assert selection_key(Preprocessing(), fold, 0) == first
    assert first > selection_key(Preprocessing(True), fold, 1)


def test_active_catalog_excludes_startup_preserves_archive():
    catalog = yaml.safe_load(Path("configs/audio_sources.yaml").read_text())
    admitted = [s for s in catalog["sources"] if s.get("admitted_to_corpus") is True]
    assert all(s["operating_condition"] != "startup" for s in admitted)
    assert all(
        segment["operating_condition"] != "startup"
        for s in admitted
        for segment in s.get("condition_segments", [])
    )
    history = yaml.safe_load(
        Path("configs/archived_startup_intervals.yaml").read_text()
    )
    assert len(history["intervals"]) == 3
    assert (
        sum(s["end_seconds"] - s["start_seconds"] for s in history["intervals"]) == 41.5
    )
