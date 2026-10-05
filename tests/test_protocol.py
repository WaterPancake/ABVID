import copy
import json
from pathlib import Path

import pytest

from abvid.paths import resolve
from abvid.protocol import load


def test_all_registry_contracts_are_structurally_valid():
    paths = list((Path(__file__).parents[1] / "configs/experiments").glob("*.json"))
    assert len(paths) == 5
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
