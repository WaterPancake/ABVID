import copy
import hashlib
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from abvid.cli import main
from abvid.paths import resolve
from abvid.protocol import command, load


def test_all_registry_contracts_are_structurally_valid():
    paths = list((Path(__file__).parents[1] / "configs/experiments").glob("*.json"))
    required = {"admission", "r0_reproduction", "h1_baselines", "h2_simulation", "cast_coverage"}
    names = {p.stem for p in paths}
    assert required <= names <= required | {"cast_grouped_coverage"}
    assert all(load(p)["id"] == p.stem for p in paths)
    assert {load(p)["phase"] for p in paths} == {
        "admission", "reproduction", "baseline", "simulation", "source_coverage"}


@pytest.fixture
def example():
    return json.loads((Path(__file__).parents[1] / "configs/experiments/h1_baselines.json").read_text())


@pytest.mark.parametrize("change", ["unknown_field", "new_mode", "shell_arguments", "absolute_path", "traversal", "bad_hash"])
def test_config_fails_closed(tmp_path, example, change):
    cfg = copy.deepcopy(example)
    if change == "unknown_field": cfg["test_tuning"] = True
    if change == "new_mode": cfg["execution"]["mode"] = "train"
    if change == "shell_arguments": cfg["execution"]["arguments"] = "--config $(something)"
    if change == "absolute_path": cfg["contract"]["path"] = "archive:/tmp/contract.json"
    if change == "traversal": cfg["contract"]["path"] = "archive:../contract.json"
    if change == "bad_hash": cfg["contract"]["sha256"] = "unknown"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): load(path)


def test_changed_contract_is_rejected(tmp_path, example, monkeypatch):
    monkeypatch.setenv("ABVID_ARCHIVE_ROOT", str(tmp_path))
    contract = tmp_path / "contract.json"; contract.write_text("{}")
    script = tmp_path / "script.py"; script.write_text("pass\n")
    example["contract"]["path"] = "archive:contract.json"
    example["execution"]["script"] = "archive:script.py"
    example["reports"] = []
    path = tmp_path / "config.json"; path.write_text(json.dumps(example))
    with pytest.raises(ValueError, match="Changed frozen input"): load(path, check_files=True)


def test_unconfigured_symlink_escape_rejected(tmp_path, monkeypatch):
    root = tmp_path / "data"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("ABVID_DATA_ROOT", str(root))
    with pytest.raises(ValueError, match="escapes"): resolve("data:escape", must_exist=False)


@pytest.fixture
def metadata_config(tmp_path, monkeypatch, example):
    """A hashed metadata wrapper with all inputs inside a temporary checkout."""
    monkeypatch.setenv("ABVID_REPO_ROOT", str(tmp_path))
    script = tmp_path / "src/abvid/cast/grouped_design.py"
    script.parent.mkdir(parents=True)
    script.write_text("# metadata fixture; never executed\n")
    contract = tmp_path / "protocol.md"
    contract.write_text("Metadata-only fixture\n")
    cfg = copy.deepcopy(example)
    cfg.update(id="cast_grouped_coverage", phase="source_coverage", reports=[])
    cfg["contract"] = {"path": "repo:protocol.md", "sha256": hashlib.sha256(contract.read_bytes()).hexdigest()}
    cfg["execution"] = {
        "mode": "metadata_only", "environment": "cast",
        "script": "repo:src/abvid/cast/grouped_design.py",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "module": "abvid.cast.grouped_design",
        "arguments": ["freeze", "--config", "repo:configs/experiments/cast_grouped_coverage.json"],
    }
    path = tmp_path / "configs/experiments/cast_grouped_coverage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_metadata_command_uses_current_checkout(metadata_config, tmp_path):
    path, cfg = metadata_config
    assert load(path, check_files=True) == cfg
    assert command(cfg) == ([str(tmp_path / ".venv/bin/python"), "-m", "abvid.cast.grouped_design",
                            "freeze", "--config", "repo:configs/experiments/cast_grouped_coverage.json"], tmp_path)


@pytest.mark.parametrize("field,value", [
    ("mode", "scientific"),
    ("environment", "reproduction"),
    ("script", "repo:src/abvid/cast/other.py"),
    ("script", "repo:src/abvid/cast/../cast/grouped_design.py"),
    ("script", "archive:src/abvid/cast/grouped_design.py"),
    ("module", None),
    ("module", "abvid.cast.other"),
    ("arguments", "freeze --config $(anything)"),
    ("arguments", ["run", "--config", "repo:configs/experiments/cast_grouped_coverage.json"]),
    ("arguments", ["freeze", "--config", "repo:configs/experiments/other.json"]),
    ("arguments", ["freeze", "--config", "repo:configs/experiments/cast_grouped_coverage.json", "--execute"]),
    ("script_sha256", "unknown"),
    ("extra_option", True),
])
def test_metadata_execution_is_exact(metadata_config, field, value):
    path, cfg = metadata_config
    cfg["execution"][field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load(path)
    with pytest.raises(ValueError):
        command(cfg)


@pytest.mark.parametrize("reference", ["contract", "script"])
def test_metadata_hashes_are_checked(metadata_config, reference):
    path, cfg = metadata_config
    source = cfg["contract"]["path"] if reference == "contract" else cfg["execution"]["script"]
    resolve(source).write_text("changed input\n")
    with pytest.raises(ValueError, match="Changed frozen input"):
        load(path, check_files=True)


def test_prepare_prints_without_executing(metadata_config, monkeypatch, capsys):
    path, cfg = metadata_config
    def unexpected_run(*args, **kwargs):
        pytest.fail("prepare must not execute without --execute")
    monkeypatch.setattr("abvid.cli.subprocess.run", unexpected_run)
    main(["prepare", str(path)])
    cmd, cwd = command(cfg)
    assert capsys.readouterr().out == f"cd {shlex.quote(str(cwd))}\n{shlex.join(cmd)}\n"


def test_prepare_execute_clears_archived_pythonpath(metadata_config, monkeypatch):
    path, cfg = metadata_config
    monkeypatch.setenv("PYTHONPATH", "/old/archive/CAST/src")
    calls = []
    def record_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr("abvid.cli.subprocess.run", record_run)
    with pytest.raises(SystemExit) as exc:
        main(["prepare", str(path), "--execute"])
    assert exc.value.code == 7
    assert len(calls) == 1
    cmd, cwd = command(cfg)
    assert calls[0][0] == cmd
    assert calls[0][1]["cwd"] == cwd
    assert calls[0][1]["check"] is False
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_replay_rejects_metadata_preparation(metadata_config, monkeypatch, capsys):
    path, _ = metadata_config
    def unexpected_run(*args, **kwargs):
        pytest.fail("replay must reject metadata-only execution")
    monkeypatch.setattr("abvid.cli.subprocess.run", unexpected_run)
    with pytest.raises(SystemExit) as exc:
        main(["replay", str(path), "--execute"])
    assert exc.value.code == 2
    assert "replay requires execution.mode=frozen_replay" in capsys.readouterr().err


@pytest.mark.parametrize("environment,module", [("reproduction", None), ("cast", "cast_improvement.width_evaluation")])
def test_historical_commands_keep_archive_environment(tmp_path, example, monkeypatch, capsys, environment, module):
    monkeypatch.setenv("ABVID_ARCHIVE_ROOT", str(tmp_path))
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    relative_script = "CAST/improvement/src/cast_improvement/width_evaluation.py" if module else "script.py"
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# historical fixture; never executed\n")
    cfg = copy.deepcopy(example)
    cfg["contract"] = {"path": "archive:contract.json", "sha256": hashlib.sha256(contract.read_bytes()).hexdigest()}
    cfg["execution"].update(environment=environment, module=module, script=f"archive:{relative_script}",
                            script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), arguments=["--help"])
    cfg["reports"] = []
    path = tmp_path / "wrapper.json"
    path.write_text(json.dumps(cfg))
    python = tmp_path / (".venv/bin/python" if environment == "cast" else "experiments/reproduction/.venv/bin/python")
    expected = [str(python), "-m", module, "--help"] if module else [str(python), str(script), "--help"]
    assert command(load(path, check_files=True)) == (expected, tmp_path)
    calls = []
    def record_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("abvid.cli.subprocess.run", record_run)
    monkeypatch.setenv("PYTHONPATH", "/unrelated/package/path")
    with pytest.raises(SystemExit) as exc:
        main(["replay", str(path), "--execute"])
    assert exc.value.code == 0
    assert calls[0][0] == expected
    assert calls[0][1]["cwd"] == tmp_path
    cast_path = os.pathsep.join(str(tmp_path / p) for p in (
        "CAST/src", "CAST/generalization/src", "CAST/improvement/src"))
    prefix = f"PYTHONPATH={shlex.quote(cast_path)} " if environment == "cast" else ""
    assert capsys.readouterr().out == f"cd {shlex.quote(str(tmp_path))}\n{prefix}{shlex.join(expected)}\n"
    if environment == "cast":
        assert calls[0][1]["env"]["PYTHONPATH"] == cast_path
    else:
        assert "PYTHONPATH" not in calls[0][1]["env"]
    calls.clear()
    with pytest.raises(SystemExit) as exc:
        main(["prepare", str(path), "--execute"])
    assert exc.value.code == 2
    assert "prepare requires execution.mode=metadata_only" in capsys.readouterr().err
    assert not calls
