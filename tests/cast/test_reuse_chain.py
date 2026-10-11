"""Failure-focused audit regression; fixtures never contain research audio."""
from copy import deepcopy
import json

import pytest

from abvid.cast import reuse_chain as audit


@pytest.fixture
def fit_fixture():
    parameters = {"spacing_hz": [100., 100., 100.], "harmonic_weights": [.125]*8,
                  "noise_weights": [.125]*8, "harmonic_fraction": .5,
                  "envelope_knots": [1.]*5}
    cfg = {"seeds": {"run": 42, "starts": [42, 123, 456, 789]},
           "initialization": {"fixture": True}, "optimizer": {"steps": 300}}
    history = [{"step": i, "fit_losses": [float(302-i+j) for j in range(4)]}
               for i in range(301)]
    starts = [{"seed": seed, "initial_parameters": deepcopy(parameters),
               "parameters": deepcopy(parameters), "best_step": 300,
               "initial_fit_loss": history[0]["fit_losses"][j],
               "fit_loss": history[-1]["fit_losses"][j],
               "initial_check_loss": 5.+j, "check_loss": 3.+j}
              for j, seed in enumerate(cfg["seeds"]["starts"])]
    fit = {"input_id": "fixture", "known_harmonic_weights": None,
           "streams": audit.streams("fixture"), "initialization": {"rule": cfg["initialization"]},
           "starts": starts, "history": history, "winner": 0, "parameters": parameters,
           "fit_loss": 2., "check_loss": 3., "baseline_fit_loss": 302.,
           "baseline_check_loss": 5., "baseline_check_winner": 0,
           "selection": "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream"}
    return fit, cfg


def test_all_saved_fit_states_and_starts_are_accepted(fit_fixture):
    fit, cfg = fit_fixture
    audit.validate_fit(fit, "fixture", cfg)


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f["starts"].pop(), "four starts"),
    (lambda f: f["history"].pop(100), "301-state"),
    (lambda f: f["history"][10]["fit_losses"].__setitem__(0, float("nan")), "loss history"),
    (lambda f: f["starts"][0].__setitem__("best_step", 299), "best-state"),
    (lambda f: f.__setitem__("winner", 1), "winner"),
    (lambda f: f["streams"]["check"].__setitem__(0, 99), "streams"),
    (lambda f: f["parameters"]["spacing_hz"].__setitem__(0, 500), "winner"),
])
def test_changed_fitting_history_is_rejected(fit_fixture, mutation, match):
    fit, cfg = fit_fixture
    mutation(fit)
    with pytest.raises(ValueError, match=match):
        audit.validate_fit(fit, "fixture", cfg)


def test_later_equal_minimum_is_not_first_minimum(fit_fixture):
    fit, cfg = fit_fixture
    fit["history"][299]["fit_losses"][0] = 2.
    with pytest.raises(ValueError, match="best-state"):
        audit.validate_fit(fit, "fixture", cfg)


def test_source_session_unknown_remains_unknown():
    row = {"dataset_id": "IDMT", "release_id": "IDMT_V1", "canonical_class": "car",
           "provider_channel_pair": "CH34", "original_channel_ids": [3, 4],
           "recording_session": None}
    audit._same_parent(row, deepcopy(row))
    changed = {**row, "recording_session": "invented_session"}
    with pytest.raises(ValueError, match="recording_session"):
        audit._same_parent(changed, row)


@pytest.mark.parametrize("path", ["dataset/MELAUDIS/audio/a.wav",
                                  "dataset/IDMT_Traffic/audio/../../other.wav",
                                  "dataset/IDMT_Traffic/audio/nested/a.wav",
                                  "dataset/IDMT_Traffic/audio/a.npz"])
def test_raw_audio_path_is_confined(path):
    with pytest.raises(ValueError):
        audit.raw_reference({"source_path": path}, "archive:dataset/IDMT_Traffic/audio")


def test_named_candidate_path_only():
    assert audit.raw_reference({"source_path": "dataset/IDMT_Traffic/audio/a.wav"},
                               "archive:dataset/IDMT_Traffic/audio").endswith("/a.wav")


def mixture_fixture():
    p = {"spacing_hz": [100.]*3, "harmonic_weights": [.125]*8,
         "noise_weights": [.125]*8, "harmonic_fraction": .5, "envelope_knots": [1.]*5}
    before = {"file_id": "fixture", "class": "car", "group": "g", "parent": {},
              "fit_sha256": "fit-hash", "flags": {"scientific_flags": ["ambiguity"]}, "parameters": p}
    cal = {"file_id": "fixture", "class": "car", "group": "g",
           "original_harmonic_fraction": .5, "harmonic_fraction": .2,
           "observed_bands": [.125]*8, "flags": ["noise_dominant"]}
    after = deepcopy(before)
    after["parameters"]["harmonic_fraction"] = .2
    after.update(mixture_calibration=cal, mixture_calibration_sha256=audit.digest(cal))
    return before, after, {"descriptors": {"bands": [.125]*8}}


def test_ambiguity_is_preserved_not_a_rejection():
    before, after, obs = mixture_fixture()
    audit.validate_transition("v7", before, after, obs, "fit-hash", {}, {})


@pytest.mark.parametrize("change", ["parent", "fixed_control", "component_target", "hash", "flags"])
def test_changed_stage_ancestry_or_fixed_control_rejected(change):
    before, after, obs = mixture_fixture()
    if change == "parent":
        after["parent"] = {"file_id": "someone_else"}
    elif change == "fixed_control":
        after["parameters"]["spacing_hz"][0] = 90.
    elif change == "component_target":
        obs["descriptors"]["bands"][0] = .99
    elif change == "hash":
        after["mixture_calibration_sha256"] = "incorrect"
    else:
        after["flags"] = {}
    with pytest.raises(ValueError):
        audit.validate_transition("v7", before, after, obs, "fit-hash", {}, {})


def test_hash_closure_does_not_accept_changed_files(tmp_path, monkeypatch):
    path = tmp_path/"artifact.json"
    path.write_text('{}\n')
    monkeypatch.setattr(audit, "resolve", lambda ref, **kw: path)
    checks = audit.Checks()
    expected = audit.sha256(path)
    path.write_text('{"changed": true}\n')
    checks.check("closure", lambda: checks.hashed("artifact:file.json", expected))
    assert "SHA256 mismatch" in checks.failures[0]
    assert checks.hashes["artifact:file.json"] != expected


def test_missing_shared_chain_retains_every_candidate_ineligible(tmp_path, monkeypatch):
    source = tmp_path/"source.jsonl"
    source.write_text('{"file_id":"a"}\n{"file_id":"b"}\n')
    numerical = tmp_path/"numerical.json"
    numerical.write_text('{}\n')
    def resolve(ref, **kwargs):
        return {"artifact:sources": source, "repo:numerical": numerical}.get(ref, tmp_path/"missing")
    monkeypatch.setattr(audit, "resolve", resolve)
    c = {"candidate_ids": ["a", "b"], "stages": {s: "archive:"+s for s in audit.STAGES},
         "per_parent_closure_files": audit.SAVED_FILES,
         "references": [], "source_manifest": {"path": "artifact:sources", "sha256": audit.sha256(source)},
         "numerical_contract": {"path": "repo:numerical", "sha256": audit.sha256(numerical)},
         "raw_source_root": "archive:dataset/IDMT_Traffic/audio"}
    records, summary = audit.audit_saved_parents(c)
    assert [r["file_id"] for r in records] == ["a", "b"]
    assert all(not r["eligible"] and r["failures"] for r in records)
    assert summary["eligible_count"] == 0 and summary["shared_failures"]
    assert records == audit.audit_saved_parents(c)[0]


def test_saved_environment_version_difference_rejected():
    env = {"machine": "arm64", "python": "3.11", "versions": {"torch": "2.13.0"}}
    altered = deepcopy(env)
    altered["versions"]["torch"] = "different"
    with pytest.raises(ValueError, match="environment versions"):
        audit._environment({"environment": altered}, env)


@pytest.fixture
def saved_parent(tmp_path, monkeypatch):
    root = tmp_path/"archive"
    parent = root/"parents"/"fixture"
    parent.mkdir(parents=True)
    hashes = {}
    for name in audit.SAVED_FILES["v9"]:
        path = parent/name
        path.write_bytes(b"non-audio audit fixture")
        hashes[name] = audit.sha256(path)
    (parent/"hashes.json").write_text(json.dumps(hashes))
    monkeypatch.setattr(audit, "roots", lambda: {"archive": root})
    monkeypatch.setattr(audit, "resolve", lambda ref, **kw: (root/ref.split(":", 1)[1]).resolve())
    return parent, hashes, "archive:parents/fixture"


def test_exact_parent_payload_closure(saved_parent):
    _, hashes, ref = saved_parent
    checks = audit.Checks()
    assert audit._saved_closure(checks, ref, "v9") == hashes
    assert not checks.failures and len(checks.hashes) == 5


@pytest.mark.parametrize("extra", ["../held_descriptors.jsonl", "nested/held.npy", "/absolute.wav", "extra.wav"])
def test_arbitrary_closure_entry_rejected_before_payload_read(saved_parent, monkeypatch, extra):
    parent, hashes, ref = saved_parent
    hashes[extra] = "f"*64
    (parent/"hashes.json").write_text(json.dumps(hashes))
    read = []
    original = audit.sha256
    monkeypatch.setattr(audit, "sha256", lambda path: (read.append(path), original(path))[1])
    with pytest.raises(ValueError, match="exact saved closure inventory"):
        audit._saved_closure(audit.Checks(), ref, "v9")
    assert read == [parent/"hashes.json"]


@pytest.mark.parametrize("linked", ["hashes.json", "derived_row.json", "calibration_reconstruction_0.wav"])
def test_payload_symlink_rejected_before_following_or_other_payload_read(saved_parent, monkeypatch, linked):
    parent, _, ref = saved_parent
    target = parent.parent/"forbidden_asset"
    target.write_bytes(b"must never read")
    (parent/linked).unlink()
    (parent/linked).symlink_to(target)
    read = []
    original = audit.sha256
    monkeypatch.setattr(audit, "sha256", lambda path: (read.append(path), original(path))[1])
    with pytest.raises(ValueError, match="symlink"):
        audit._saved_closure(audit.Checks(), ref, "v9")
    assert read == ([] if linked == "hashes.json" else [parent/"hashes.json"])


def test_parent_directory_symlink_rejected_before_any_read(saved_parent, monkeypatch):
    parent, _, ref = saved_parent
    displaced = parent.parent/"other_parent"
    parent.rename(displaced)
    parent.symlink_to(displaced, target_is_directory=True)
    def forbidden_hash(path):
        pytest.fail(f"Unexpected file read through parent symlink: {path}")
    monkeypatch.setattr(audit, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="symlink"):
        audit._saved_closure(audit.Checks(), ref, "v9")
