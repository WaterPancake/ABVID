import copy

import pytest

from vehicle_audio.state_identity_audit import (
    audit_inventory,
    render_report,
    state_group,
    worksheet_segments,
)


def example():
    source = {
        "recording_session": "one-session",
        "vehicle_class": "tracked",
        "vehicle_model": "A and B",
        "condition_segments": [
            {"start_seconds": 0, "end_seconds": 3, "operating_condition": "idle"},
            {"start_seconds": 3, "end_seconds": 6, "operating_condition": "steady_speed"},
        ],
    }
    reviews = {"source": [
        {"start_seconds": 0, "end_seconds": 3, "tag": "idle", "vehicle": "A"},
        {"start_seconds": 3, "end_seconds": 6, "tag": "steady_speed", "vehicle": "B"},
    ]}
    records = [
        {"source_id": "source", "sample_id": str(start), "recording_session": "one-session",
         "window_start_seconds": start, "window_end_seconds": start + 2,
         "operating_condition": "idle" if start < 3 else "steady_speed",
         "vehicle_model": "A and B"}
        for start in (0, 1, 3, 4)
    ]
    return records, {"source": source}, reviews


def test_never_pairs_different_models_in_same_session():
    args = example()
    original = copy.deepcopy(args)
    result = audit_inventory(*args)
    assert args == original
    assert result["summary"]["same_model_pair_candidate_sessions"] == 0
    assert result["summary"]["windows_in_multi_model_sources"] == 4
    assert result["summary"]["approved_seconds"] == 6  # Not 4 windows times 2.
    assert result["summary"]["recording_sessions"] == 1


def test_same_model_is_not_same_individual_and_mixed_is_unresolved():
    records, sources, reviews = example()
    for row in reviews["source"]:
        row["vehicle"] = "A convoy"
    result = audit_inventory(records, sources, reviews)
    assert result["summary"]["same_model_pair_candidate_sessions"] == 1
    assert result["summary"]["verified_same_individual_pairs"] == 0
    assert result["inventory"][0]["convoy_caveat"]
    assert "convoy" in render_report(result)
    assert state_group("mixed") == state_group("unknown") == "unresolved"
    with pytest.raises(ValueError):
        state_group("startup")


@pytest.mark.parametrize("mutation", ["condition", "bounds", "identity", "extra", "window", "overlap"])
def test_inconsistent_annotations_fail_closed(mutation):
    records, sources, reviews = example()
    if mutation == "condition":
        reviews["source"][0]["tag"] = "mixed"
    elif mutation == "bounds":
        reviews["source"][0]["end_seconds"] = 2.9
    elif mutation == "identity":
        reviews["source"][0]["vehicle"] = None
    elif mutation == "extra":
        reviews["source"].append(reviews["source"][0])
    elif mutation == "window":
        records[0]["window_end_seconds"] = 4
    else:
        sources["source"]["condition_segments"][1]["start_seconds"] = 2
    with pytest.raises(ValueError):
        audit_inventory(records, sources, reviews)


def test_parser_filters_non_development_and_excluded_segments():
    text = '''### A
- Source ID: `source`
```yaml
segments:
  - {start: "00:00", end: "00:01.5", decision: keep, tag: idle, vehicle: A}
  - {start: "00:02", end: "00:03", decision: exclude, tag: startup, vehicle: A}
```
### Reserved
- Source ID: `reserved`
```yaml
not even a valid review block
```
'''
    rows = worksheet_segments(text, {"source"})
    assert len(rows["source"]) == 1
    assert rows["source"][0]["end_seconds"] == 1.5
    with pytest.raises(ValueError):
        worksheet_segments(text, {"missing"})
    with pytest.raises(ValueError):
        worksheet_segments(text + text, {"source"})
