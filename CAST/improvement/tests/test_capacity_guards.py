"""Prior evidence must be complete before the next raw-data stage."""
from copy import deepcopy
import pytest
from cast_improvement import capacity_data as data
from cast.config import ROOT, save, sha


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("generated_waveform_descriptor_donor_calibration_replays", 89999),
    ("scores_recomputed", 1799), ("outer_held_audio_access", True),
    ("class_centers_and_scales_recomputed_from_training_only", False),
    ("all_fold_ancestry_disjoint", False), ("summary_sha256", "changed"),
])
def test_capacity_rejects_missing_or_changed_audit(tmp_path, monkeypatch, field, value):
    summary = {"candidates": {"example": {"passes": False}}}
    save(tmp_path/"summary.json", summary)
    receipt = {"passed": True, "summary_sha256": sha(tmp_path/"summary.json"),
               "generated_waveform_descriptor_donor_calibration_replays": 90000,
               "scores_recomputed": 1800, "outer_held_audio_access": False,
               "all_fold_ancestry_disjoint": True,
               "class_centers_and_scales_recomputed_from_training_only": True}
    receipt[field] = value
    save(tmp_path/"audit_verification.json", receipt)
    monkeypatch.setattr(data, "prior_frozen", lambda _: (None, deepcopy(summary)))
    with pytest.raises(ValueError, match="audited prior-v2"):
        data.require_prior_audit(tmp_path)
