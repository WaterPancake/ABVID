import importlib.util
from pathlib import Path

import torch

from vehicle_audio.semantic_session_evaluation import nested_leave_session_pair_out

spec = importlib.util.spec_from_file_location(
    "embedding_context", Path("scripts/evaluate_embedding_context.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cached_nested_fits_exactly_match_uncached():
    records = [
        {
            "sample_id": f"{c}_{s}_{w}",
            "recording_session": f"{c}_{s}",
            "vehicle_class": c,
        }
        for c in ("tracked", "wheeled")
        for s in range(3)
        for w in range(2)
    ]
    features = torch.randn(12, 4, generator=torch.Generator().manual_seed(4))
    labels = torch.tensor([0] * 6 + [1] * 6)
    a = nested_leave_session_pair_out(features, labels, records, c_values=(0.01, 1.0))
    b = nested_leave_session_pair_out(
        features, labels, records, c_values=(0.01, 1.0), cache_fits=True
    )
    assert a[0] == b[0]
    assert a[2] == b[2]
    for fold in a[1]:
        for field in a[1][fold]["nested_selected"]:
            assert torch.equal(
                a[1][fold]["nested_selected"][field],
                b[1][fold]["nested_selected"][field],
            )


def test_context_stays_in_one_reviewed_interval():
    catalog = {
        "source": {
            "condition_segments": [
                {"start_seconds": 0, "end_seconds": 7, "operating_condition": "idle"},
                {
                    "start_seconds": 8,
                    "end_seconds": 20,
                    "operating_condition": "moving",
                },
            ]
        }
    }
    records = [
        {
            "source_id": "source",
            "window_start_seconds": a,
            "window_end_seconds": a + 2,
            "operating_condition": c,
        }
        for a, c in [
            (0, "idle"),
            (2, "idle"),
            (5, "idle"),
            (8, "moving"),
            (11, "moving"),
        ]
    ]
    assert module.matched_indices(records, catalog, 4) == [1, 4]
    assert module.matched_indices(records, catalog, 8) == [4]
    assert module.matched_indices(records, catalog, 2) == list(range(5))


def test_paired_bootstrap_identical_is_zero():
    sessions = {
        f"{c}_{i}": {"class": c, "mean_recall": i / 3}
        for c in ("tracked", "wheeled")
        for i in range(3)
    }
    report = {h: {"sessions": sessions} for h in ("ensemble", "selected")}
    actual = module.paired_summary(report, report)
    assert actual["ensemble"]["paired_session_bootstrap_95_percent"] == [0, 0]
