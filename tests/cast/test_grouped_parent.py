"""Synthetic-only production-budget tests for the grouped same-parent adapter."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
import subprocess
from types import MappingProxyType

import numpy as np
import pytest
import soundfile as sf
import torch

from abvid.cast import grouped_parent as parent
from abvid.cast.grouped_design import LINKS, known
from abvid.cast.audio import normalize, observation, shape
from abvid.cast.config import canonical
from abvid.cast.coverage import describe
from abvid.cast.population.engine import SmoothRenderer
from abvid.cast.renderer import tensors
from abvid.cast.roles import FoldRole
from abvid.cast.synthetic import fixtures
from abvid.paths import archive_root, resolve


def _inputs():
    numerical = json.loads(resolve(
        "repo:experiments/cast_coverage/grouped_coverage/numerical_contract.json"
    ).read_text())
    source = {"file_id": "synthetic_grouped_parent_fixture", "dataset_id": "IDMT",
              "canonical_class": "car", "class_id": 0, "provenance_group_id": "training_group",
              "site_id": "training_site", "recording_session": "unknown",
              "relative_path": "explicitly_synthetic_fixture_not_a_recording.wav"}
    held = {**source, "file_id": "synthetic_held_fixture", "provenance_group_id": "held_group"}
    role = FoldRole("group_0", "group", (source["file_id"],), (held["file_id"],),
                    ("held_group",), ("training_group",),
                    MappingProxyType({r["file_id"]: canonical(r) for r in (source, held)}),
                    MappingProxyType({k: frozenset([held[k]]) if known(held.get(k)) else frozenset()
                                      for k in LINKS}),
                    MappingProxyType({"car": (source["file_id"],), "truck": ()}),
                    "unit_test_metadata_only")
    return numerical, source, held, role


@pytest.fixture(scope="module")
def completed_parent():
    numerical, source, _, role = _inputs()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        cfg = numerical["renderer_configs"]["smooth8_v1"]
        with torch.no_grad():
            truth = shape(SmoothRenderer(cfg, "grouped_parent_truth").render(
                tensors(fixtures()["mixture"]), "check")[0, 0]).numpy()
        raw = np.stack((truth, truth), axis=1).astype(np.float64)
        prepared = parent.prepare_observation(raw, 16000, source, numerical, role=role)
        result = parent.calibrate_parent(
            prepared["target"], source, numerical, role=role,
            original=prepared["original"], normalization=prepared["normalization"])
        yield numerical, source, role, prepared, result
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def test_full_budget_chain_preserves_ancestry_alternatives_and_artifact_bytes(completed_parent):
    numerical, source, _, prepared, result = completed_parent
    assert result["status"] == "complete"
    assert tuple(result["stages"]) == parent.STAGES
    assert result["final_row"] == result["stages"]["v11"]
    assert len(result["fit"]["history"]) == 301
    assert [s["seed"] for s in result["fit"]["starts"]] == [42, 123, 456, 789]
    assert result["fit"]["ancestry"]["parent"] == source
    assert result["fit"]["original_level"] == prepared["normalization"]
    assert result["fit"]["physical_metadata"]["RPM"] == "unknown"
    assert set(result["stage_seconds"]) == set(parent.STAGES)
    for stage, row in result["stages"].items():
        assert row["parent"] == source
        assert row["parent"]["recording_session"] == "unknown"
        assert row["file_id"] == source["file_id"]
        assert row["fit_sha256"] == result["artifact_sha256"]["fit.json"]
        assert json.loads(result["artifacts"][stage + ".json"]) == row
    for name, data in result["artifacts"].items():
        assert hashlib.sha256(data).hexdigest() == result["artifact_sha256"][name]
    wave, sr = sf.read(BytesIO(result["artifacts"]["original_shape.wav"]), dtype="float32")
    assert sr == 16000 and np.array_equal(wave, prepared["target"].numpy())
    assert result["observation"]["descriptors"] == {k: v.tolist() for k, v in
        describe(wave, numerical["renderer_configs"]["smooth8_v1"], numerical["metric_config"]).items()}
    with np.load(BytesIO(result["artifacts"]["components.npz"]), allow_pickle=False) as data:
        assert all(np.array_equal(data[k], v) for k, v in result["components"].items())
    assert result["stages"]["v7"]["mixture_calibration"]["input_hashes"] == {
        name: result["artifact_sha256"][name]
        for name in ("fit.json", "components.npz", "original_shape.wav")}
    for stage, key in (("v9", "spectral_calibration"), ("v11", "resolution_calibration")):
        calibration = result["stages"][stage][key]
        assert len(calibration["history"]) == 151
        assert calibration["settings"]["steps"] == 150
        assert calibration["input_id"] == source["file_id"]
    assert len(result["final_row"]["parameters"]["noise_weights"]) == 16
    assert len(result["final_row"]["parameters"]["envelope_knots"]) == 41


def test_preprocessing_uses_original_contract_and_does_not_modify_inputs():
    numerical, source, _, role = _inputs()
    raw = np.random.default_rng(19).normal(size=(100000, 2))
    before = raw.copy()
    prepared = parent.prepare_observation(raw, 48000, source, numerical, role=role)
    cfg = numerical["renderer_configs"]["smooth8_v1"]
    original = observation(raw, 48000, cfg)
    target, stats = normalize(original, cfg)
    assert np.array_equal(raw, before)
    assert np.array_equal(prepared["original"], original)
    assert torch.equal(prepared["target"], target)
    assert prepared["normalization"] == stats


def test_held_parent_rejected_before_preprocessing_or_fit(monkeypatch):
    numerical, source, held, role = _inputs()
    def forbidden(*args, **kwargs):
        pytest.fail("A held parent reached waveform processing")
    monkeypatch.setattr(parent, "audio_observation", forbidden)
    monkeypatch.setattr(parent, "fit_smooth", forbidden)
    with pytest.raises(ValueError, match="Held ancestor|training role"):
        parent.prepare_observation(object(), 16000, held, numerical, role=role)
    with pytest.raises(ValueError, match="Held ancestor|training role"):
        parent.calibrate_parent(object(), held, numerical, role=role)
    changed = {**source, "recording_session": "invented_session"}
    with pytest.raises(ValueError, match="frozen source"):
        parent.prepare_observation(object(), 16000, changed, numerical, role=role)


@pytest.mark.parametrize("change", ["fit_budget", "spectral_budget", "descriptor_floor"])
def test_changed_numerical_recipe_rejected_before_waveform_access(change, monkeypatch):
    numerical, source, _, role = _inputs()
    if change == "fit_budget":
        numerical["renderer_configs"]["smooth8_v1"]["optimizer"]["steps"] = 2
    elif change == "spectral_budget":
        numerical["expected_spectrum_calibration"]["steps"] = 2
    else:
        numerical["metric_config"]["spectral_probability_floor"] = 1e-6
    monkeypatch.setattr(parent, "audio_observation", lambda *a: pytest.fail("Changed recipe ran"))
    with pytest.raises(ValueError, match="Changed"):
        parent.prepare_observation(object(), 16000, source, numerical, role=role)


def test_technical_failure_keeps_completed_fit_calibrations_and_artifacts(completed_parent, monkeypatch):
    numerical, source, role, prepared, complete = completed_parent
    def fail(*args, **kwargs):
        raise FloatingPointError("declared synthetic failure injection")
    monkeypatch.setattr(parent.resolution_bank, "fit_record", fail)
    with pytest.raises(parent.ParentCalibrationError) as caught:
        parent.calibrate_parent(prepared["target"], source, numerical, role=role,
                                original=prepared["original"], fit_result=complete["fit"])
    partial = caught.value.partial
    assert caught.value.stage == "v11"
    assert partial["status"] == "technical_failure"
    assert tuple(partial["stages"]) == parent.STAGES[:-1]
    assert partial["final_row"] is None
    assert len(partial["fit"]["history"]) == 301
    assert "fit.json" in partial["artifacts"] and "v10.json" in partial["artifacts"]
    assert "v11.json" not in partial["artifacts"]
    assert "v9" in partial["flags"]


def test_wrong_parent_saved_fit_and_wrong_observation_fail_closed(completed_parent):
    numerical, source, role, prepared, complete = completed_parent
    wrong = deepcopy(complete["fit"])
    wrong["input_id"] = "other_parent"
    with pytest.raises(parent.ParentCalibrationError, match="fit input"):
        parent.calibrate_parent(prepared["target"], source, numerical, role=role, fit_result=wrong)
    with pytest.raises(ValueError, match="normalization ancestry"):
        parent.calibrate_parent(prepared["target"], source, numerical, role=role,
                                original=prepared["original"] + np.float32(.01),
                                normalization=prepared["normalization"])


PARITY = r'''
import json,sys
import numpy as np
import torch
from cast_improvement.parent_mixture_bank import derived_row
from cast_improvement.envelope_prior import derive_bank as envelope
from cast_improvement.spectrum_bank import fit_record as spectrum
from cast_improvement.temporal_prior import derive_bank as temporal
from cast_improvement.resolution_bank import fit_record as resolution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
r=json.load(sys.stdin)
cfgs,metric=r['configs'],r['metric']
metric['held_group']='held_group'
base,obs=r['base'],r['observation']
cfg=cfgs['smooth8_v1']
rows={}
rows['v7']=derived_row(base,np.asarray(r['harmonic'],dtype=np.float32),
    np.asarray(r['noise'],dtype=np.float32),obs['descriptors']['bands'],r['hashes'],cfg)
rows['v8']=envelope([rows['v7']],[obs],cfg,metric['held_group'])[0]
rows['v9']=spectrum(rows['v8'],obs,cfg,metric,base['file_id'])
rows['v10']=temporal([rows['v9']],[obs],cfgs['smooth8_temporal41_v10'],metric['held_group'])[0]
rows['v11']=resolution(rows['v10'],obs,cfgs['spectrum16_temporal41_v11'],metric,base['file_id'])
print(json.dumps(rows,allow_nan=False,sort_keys=True))
'''


@pytest.mark.archive
def test_full_calibration_chain_matches_isolated_historical_functions(completed_parent):
    numerical, _, _, _, result = completed_parent
    archive = archive_root()
    interpreter = archive / ".venv/bin/python"
    if not interpreter.exists():
        pytest.skip("Preserved historical interpreter is unavailable")
    for reference in numerical["references"]:
        if reference["path"].startswith("archive:CAST/") and reference["path"].endswith(".py"):
            assert hashlib.sha256(resolve(reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    request = {"configs": numerical["renderer_configs"], "metric": numerical["metric_config"],
               "base": result["stages"]["smooth8"], "observation": result["observation"],
               "harmonic": result["components"]["harmonic"].tolist(),
               "noise": result["components"]["noise"].tolist(),
               "hashes": {name: result["artifact_sha256"][name] for name in
                          ("fit.json", "components.npz", "original_shape.wav")}}
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(
        str(archive / subdir) for subdir in
        ("CAST/improvement/src", "CAST/generalization/src", "CAST/src")))
    completed = subprocess.run([str(interpreter), "-B", "-c", PARITY],
                               input=json.dumps(request, allow_nan=False), text=True,
                               capture_output=True, check=True, env=env, cwd=archive, timeout=120)
    assert json.loads(completed.stdout) == {stage: result["stages"][stage] for stage in parent.STAGES[1:]}
