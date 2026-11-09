"""Synthetic serialization/replay and additive execution-pin regression checks.

Only temporary fixtures and repository source/configuration bytes are opened.
No archive, source recording, saved parent or scientific output is accessed.
"""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from abvid.cast import cast4a_serialization_config as config
from abvid.cast import cast4a_serialization_runner as runner
from abvid.cast import grouped_design, grouped_execution_config, grouped_runner
from abvid.cast.config import digest
from abvid.provenance.hashing import sha256


class SyntheticPopulation:
    def __init__(self):
        self.bank = [{"file_id": "synthetic_parent"}]
        self.observed = [{"file_id": "synthetic_parent", "descriptors": {"bands": [0.25]}}]
        self.statistics = {"bank_sha256": digest(self.bank),
                           "observations_sha256": digest(self.observed)}
        self.scales = {"car": {"bands": [1.]}}
        self.fingerprint = digest(self.statistics)

    def draw(self, planned):
        rng = np.random.default_rng(planned["noise_seed"])
        wave = (rng.standard_normal(32000) * 0.1).astype(np.float32)
        record = {"planned_sample_id": planned["id"], "parent": "synthetic_parent",
                  "recording_session": None, "population_fingerprint": self.fingerprint}
        return record, wave, {"bands": np.array([0.25, 0.5])}


def sample():
    planned = next(grouped_design.planned_schedule([{"id": "group_0"}]))
    return SyntheticPopulation(), planned


def next_second():
    # A full elapsed second guarantees distinct libsndfile time(NULL) values;
    # no mocked clock or chance of two writes landing in the same second.
    before = time.time()
    time.sleep(1.1)
    assert int(time.time()) > int(before)


def test_npy_generation_gate_replays_after_second_boundary(tmp_path, monkeypatch):
    population, planned = sample()
    role = SimpleNamespace(partition_id="group_0", design_identity_sha256="synthetic_design")
    context = {"output": tmp_path / "revision", "config_sha256": "synthetic_revision"}
    folder = context["output"] / "folds" / role.partition_id
    runner._checkpoint(folder / "population", {
        "bank.json": runner.json_bytes(population.bank),
        "observations.json": runner.json_bytes(population.observed),
        "statistics.json": runner.json_bytes(population.statistics),
        "scales.json": runner.json_bytes(population.scales)},
        runner._metadata(context, role, "training_population", fingerprint=population.fingerprint),
        replay_only=False)
    # Exercise the real immutable gate on one synthetic sample, without the
    # scientific schedule, parent fitting or any held-source access.
    monkeypatch.setattr(runner, "_schedule", lambda *args: [planned])
    first = runner.generation_gate(context, role, population)
    path = runner._sample_path(folder, planned)
    metadata, payload = runner.read_bundle(path)
    assert set(payload) == {"waveform.npy", "planned.json", "record.json", "descriptors.json"}
    restored = np.load(BytesIO(payload["waveform.npy"]), allow_pickle=False)
    wave = population.draw(planned)[1]
    assert restored.dtype == np.float32 and restored.shape == (32000,)
    assert restored.tobytes() == wave.tobytes()
    assert metadata["waveform_float32_sha256"] == runner.sha(wave.tobytes())
    assert metadata["waveform_format"] == "npy" and metadata["waveform_sample_rate_hz"] == 16000
    legacy, _ = grouped_runner.sample_payload(population, planned)
    assert all(payload[name] == legacy[name] for name in payload if name.endswith(".json"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}

    next_second()
    second = runner.generation_gate(context, role, population, replay_only=True)

    assert first == second
    assert runner.read_bundle(path) == (metadata, payload)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}


def peak_timestamp(audio):
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    offset = 12
    while offset + 8 <= len(audio):
        size = struct.unpack_from("<I", audio, offset + 4)[0]
        if audio[offset:offset + 4] == b"PEAK":
            return offset + 12
        offset += 8 + size + size % 2
    pytest.fail("Plain FLOAT WAV no longer contains the legacy PEAK chunk")


def test_plain_float_wav_replay_changes_only_peak_timestamp(tmp_path):
    population, planned = sample()
    # The preserved v1 path calls plain soundfile.write(..., subtype="FLOAT").
    first, details = grouped_runner.sample_payload(population, planned)
    folder = tmp_path / "legacy_sample"
    grouped_runner._checkpoint(folder, first, details, replay_only=False)

    next_second()
    second, replay_details = grouped_runner.sample_payload(population, planned)

    assert details == replay_details
    assert all(first[name] == second[name] for name in first if name != "waveform.wav")
    a, b = first["waveform.wav"], second["waveform.wav"]
    offset = peak_timestamp(a)
    assert peak_timestamp(b) == offset
    assert struct.unpack_from("<I", b, offset)[0] > struct.unpack_from("<I", a, offset)[0]
    assert a[:offset] + a[offset + 4:] == b[:offset] + b[offset + 4:]
    decoded_a, sr_a = sf.read(BytesIO(a), dtype="float32")
    decoded_b, sr_b = sf.read(BytesIO(b), dtype="float32")
    assert sr_a == sr_b == 16000
    assert decoded_a.tobytes() == decoded_b.tobytes() == population.draw(planned)[1].tobytes()
    with pytest.raises(ValueError, match="Replay differs"):
        grouped_runner._checkpoint(folder, second, replay_details, replay_only=True)
    assert grouped_runner.read_bundle(folder) == (details, first)


@pytest.fixture
def revision_contract(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads((repo / config.CONFIG.split(":", 1)[1]).read_text())
    candidate = tmp_path / "config.json"
    candidate.write_bytes(runner.json_bytes(cfg))
    original_resolve, original_checked = config.resolve, config.checked
    metadata = tmp_path / "synthetic_metadata.json"
    metadata.write_text("{}\n")
    opened = []

    def resolve(reference, **kwargs):
        if reference == config.CONFIG:
            return candidate
        assert reference.startswith("repo:"), "Test tried to resolve external data"
        return original_resolve(reference, **kwargs)

    def checked(reference):
        if reference["path"].startswith("artifact:"):
            opened.append(reference["path"])
            return metadata
        assert reference["path"].startswith("repo:"), "Test tried to open archive data"
        return original_checked(reference)

    monkeypatch.setattr(config, "resolve", resolve)
    monkeypatch.setattr(config, "checked", checked)
    monkeypatch.setattr(config, "artifact_path", lambda ref: tmp_path / ref.split(":", 1)[1])
    monkeypatch.setattr(config.readiness, "environment", lambda: deepcopy(cfg["expected_environment"]))
    return SimpleNamespace(config=cfg, candidate=candidate, opened=opened, repo=repo)


def test_revision_contract_pins_new_runner_and_preserves_v1_inventory(revision_contract):
    fixture = revision_contract
    cfg = config.load_contract()
    original_path = fixture.repo / grouped_execution_config.CONFIG.split(":", 1)[1]
    original = json.loads(original_path.read_text())
    assert cfg["previous_execution"] == {"path": grouped_execution_config.CONFIG,
                                         "sha256": sha256(original_path)}
    assert cfg["output"] == config.OUTPUT and cfg["output"] != original["output"]
    assert cfg["entrypoint"]["module"] == "abvid.cast.cast4a_serialization_runner"
    for key in ("implementation_references", "test_references"):
        assert all(reference in cfg[key] for reference in original[key])
        assert all(sha256(fixture.repo / ref["path"].split(":", 1)[1]) == ref["sha256"] for ref in cfg[key])
    modules, tests = grouped_execution_config.reference_inventory()
    assert modules == [ref["path"] for ref in original["implementation_references"]]
    assert tests == [ref["path"] for ref in original["test_references"]]
    assert fixture.opened == [cfg[key]["path"] for key in ("design_lock", "readiness_lock")]


@pytest.mark.parametrize("change", ["old_output", "protocol", "previous_pin", "serialization", "runner_pin", "test_pin", "environment"])
def test_revision_contract_rejects_protocol_and_pin_changes(revision_contract, monkeypatch, change):
    cfg = deepcopy(revision_contract.config)
    if change == "old_output":
        cfg["output"] = grouped_execution_config.OUTPUT
    elif change == "protocol":
        cfg["numerical_contract"]["sha256"] = "0" * 64
    elif change == "previous_pin":
        cfg["previous_execution"]["sha256"] = "0" * 64
    elif change == "serialization":
        cfg["waveform_serialization"]["format"] = "wav"
    elif change in {"runner_pin", "test_pin"}:
        key = "implementation_references" if change == "runner_pin" else "test_references"
        ref = next(ref for ref in cfg[key] if "cast4a_serialization" in ref["path"])
        ref["sha256"] = "0" * 64
    else:
        monkeypatch.setattr(config.readiness, "environment", lambda: {})
    revision_contract.candidate.write_bytes(runner.json_bytes(cfg))
    with pytest.raises(ValueError):
        config.load_contract()


@pytest.mark.parametrize("command", ["plan", "run", "verify"])
def test_revision_cli_stays_metadata_only_without_execute(monkeypatch, capsys, command):
    monkeypatch.setattr(config, "load_context", lambda ref: {"plan": {"output": config.OUTPUT}})
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("Metadata command executed science"))
    assert runner.main([command]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"output": config.OUTPUT, "requested_command": command, "executed": False}
