import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "logmel_diagnostics",
    Path(__file__).parents[1] / "scripts/plot_native_logmel_diagnostics.py",
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_relative_db_preserves_shared_gain_difference():
    power = np.array([[1.0, 0.1], [0.01, 0.0]])
    expected = np.array([[0.0, -10.0], [-20.0, -100.0]])
    np.testing.assert_allclose(diagnostics.relative_db(power, 1.0), expected)
    np.testing.assert_allclose(
        diagnostics.relative_db(power * 4, 4.0)[power > 0],
        diagnostics.relative_db(power, 1.0)[power > 0],
    )
    assert np.isfinite(diagnostics.relative_db(np.zeros((2, 2)), 0)).all()


def test_correctness_uses_each_held_out_threshold_not_mean_score():
    benchmark = SimpleNamespace(labels=np.array([1]))
    folds = [
        {
            "fold_id": "a",
            "indices": np.array([0]),
            "threshold": 0.3,
            "semantic": np.array([0.4]),
            "classical": np.array([0.4]),
            "fusion": np.array([0.4]),
        },
        {
            "fold_id": "b",
            "indices": np.array([0]),
            "threshold": 0.6,
            "semantic": np.array([0.4]),
            "classical": np.array([0.4]),
            "fusion": np.array([0.4]),
        },
    ]
    scores = diagnostics.window_scores(benchmark, folds)[0]
    assert scores["fusion_p_wheeled"] == 0.4
    assert scores["fusion_correct"] == 0.5
    assert scores["semantic_correct"] == 0
    assert scores["classical_correct"] == 0
