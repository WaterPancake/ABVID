"""Synthetic-only regression and exact original-layout smooth8 fit parity."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from abvid.cast.audio import shape
from abvid.cast.population.engine import SmoothRenderer, model_config
from abvid.cast.renderer import tensors
from abvid.cast.smooth_fit import fit
from abvid.cast.synthetic import fixtures
from abvid.paths import archive_root, repository_root, resolve


ARCHIVED_ENGINE_SHA256 = (
    "78af0f34dd24e784601ac7ada639c9e88c69ec2457642cdcbc16e4adafa0a070"
)

# Both implementations run in fresh processes with their own import roots. The
# original archive never enters the packaged module's import graph, and bytecode
# writes are disabled. The only observation constructed is a declared fixture.
PARITY_PROGRAM = r'''
import hashlib
import importlib.metadata
import json
import platform
import sys
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
if sys.argv[1] == "archive":
    from cast.audio import shape
    from cast.renderer import tensors
    from cast_improvement.engine import SmoothRenderer, fit
else:
    from abvid.cast.audio import shape
    from abvid.cast.renderer import tensors
    from abvid.cast.population.engine import SmoothRenderer
    from abvid.cast.smooth_fit import fit

request = json.load(sys.stdin)
cfg, input_id = request["config"], request["input_id"]
with torch.no_grad():
    target = shape(SmoothRenderer(cfg, input_id + "_truth").render(
        tensors(request["truth"]), "check")[0, 0])
result = fit(target, cfg, input_id, request["known_harmonic_weights"])
fit_seconds = result.pop("fit_seconds")
with torch.no_grad():
    waveforms, components = SmoothRenderer(cfg, input_id).render(
        tensors(result["parameters"]), "check", components=True)
def sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
print(json.dumps({
    "result": result,
    "fit_seconds": fit_seconds,
    "target_sha256": sha(target),
    "waveform_sha256": sha(waveforms),
    "component_sha256": {key: sha(value) for key, value in components.items()},
    "environment": {
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "torch")},
    },
}, sort_keys=True, allow_nan=False))
'''


@pytest.fixture
def deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)


def test_short_fit_preserves_inputs_alternatives_and_best_fit_winner(deterministic_cpu):
    cfg = model_config()
    cfg["optimizer"]["steps"] = 2
    frozen_cfg = deepcopy(cfg)
    with torch.no_grad():
        target = shape(SmoothRenderer(cfg, "smooth_fit_regression_truth").render(
            tensors(fixtures()["mixture"]), "check")[0, 0])
    frozen_target = target.clone()
    first = fit(target, cfg, "smooth_fit_regression")
    second = fit(target, cfg, "smooth_fit_regression")
    first_time, second_time = first.pop("fit_seconds"), second.pop("fit_seconds")
    assert first_time >= 0 and second_time >= 0
    assert first == second
    assert cfg == frozen_cfg
    assert torch.equal(target, frozen_target)
    assert [row["seed"] for row in first["starts"]] == [42, 123, 456, 789]
    assert [row["step"] for row in first["history"]] == [0, 1, 2]
    losses = np.array([row["fit_losses"] for row in first["history"]])
    assert losses.shape == (3, 4)
    assert np.array_equal(losses.min(axis=0), [row["fit_loss"] for row in first["starts"]])
    assert first["winner"] == int(np.argmin(losses.min(axis=0)))
    assert first["parameters"] == first["starts"][first["winner"]]["parameters"]
    assert first["selection"] == (
        "fitted_winner_by_fit_stream_only_baseline_minimum_on_common_check_stream"
    )
    assert len(first["streams"]["fit"]) == len(first["streams"]["check"]) == 2
    assert set(first["streams"]["fit"]).isdisjoint(first["streams"]["check"])


def _archived_inputs():
    root = archive_root()
    engine = root / "CAST/improvement/src/cast_improvement/engine.py"
    snapshot = root / "CAST/improvement/runs/cast_smooth8_v1_20261004_r1/snapshot.json"
    if not engine.is_file() or not snapshot.is_file():
        pytest.skip("Preserved smooth8 implementation/environment metadata unavailable")
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == ARCHIVED_ENGINE_SHA256
    # The frozen recipe pins every base dependency used by the original fitter.
    contract = json.loads(resolve(
        "repo:experiments/cast_coverage/grouped_coverage/numerical_contract.json"
    ).read_text())
    names = {
        "archive:CAST/src/cast/" + name + ".py"
        for name in ("audio", "config", "fit", "renderer")
    } | {"archive:CAST/improvement/src/cast_improvement/engine.py"}
    references = [row for row in contract["references"] if row["path"] in names]
    assert len(references) == len(names)
    for row in references:
        assert hashlib.sha256(resolve(row["path"]).read_bytes()).hexdigest() == row["sha256"]
    recorded = json.loads(snapshot.read_text())["environment"]
    expected_env = {
        "python": recorded["python"].split()[0],
        "versions": {name: recorded["versions"][name] for name in ("numpy", "scipy", "torch")},
    }
    return root, expected_env


def _run_implementation(mode, request, archive, expected_env):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if mode == "archive":
        env["PYTHONPATH"] = os.pathsep.join(str(archive / subdir) for subdir in (
            "CAST/improvement/src", "CAST/generalization/src", "CAST/src"
        ))
        cwd = archive
        interpreter = archive / ".venv/bin/python"
        if not interpreter.is_file():
            pytest.skip("Preserved historical CAST interpreter unavailable")
    else:
        env["PYTHONPATH"] = str(repository_root() / "src")
        cwd = repository_root()
        interpreter = sys.executable
    completed = subprocess.run(
        [str(interpreter), "-B", "-c", PARITY_PROGRAM, mode],
        input=json.dumps(request, allow_nan=False), text=True,
        capture_output=True, check=True, cwd=cwd, env=env, timeout=180,
    )
    result = json.loads(completed.stdout)
    assert result["environment"] == expected_env, "Exact historical numerical environment required"
    return result


def _write_parity_evidence(request, old, new, archive):
    """Optionally retain synthetic-only proof outside the checkout, without replacing it."""
    output = os.environ.get("ABVID_SMOOTH_FIT_PARITY_OUTPUT")
    if output is None:
        return
    if not output.startswith("artifact:"):
        raise ValueError("Parity evidence must use an artifact: storage path")
    directory = resolve(output, must_exist=False)
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {
        "scope": "synthetic_fixture_packaging_parity_only",
        "real_audio_opened": False,
        "exact_parity_excluding_elapsed_seconds": True,
        "request": request,
        "archive": old,
        "package": new,
        "archive_interpreter": str(archive / ".venv/bin/python"),
        "package_interpreter": sys.executable,
        "archived_engine_sha256": ARCHIVED_ENGINE_SHA256,
        "packaged_fit_sha256": hashlib.sha256(resolve(
            "repo:src/abvid/cast/smooth_fit.py"
        ).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(resolve(
            "repo:tests/cast/test_smooth_fit.py"
        ).read_bytes()).hexdigest(),
        "deterministic_algorithms": True,
        "torch_threads": 1,
        "torch_interop_threads": 1,
    }
    with (directory / (request["input_id"] + ".json")).open("x") as handle:
        json.dump(evidence, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


@pytest.mark.archive
@pytest.mark.parametrize("fixture_name,steps,known_weights", [
    ("mixture", 2, False),
    ("single_tone", 2, True),
    ("mixture", 300, False),
], ids=["short_mixture", "short_fixed_harmonics", "full_300_step_mixture"])
def test_exact_archived_fit_parity(fixture_name, steps, known_weights):
    archive, expected_env = _archived_inputs()
    cfg = model_config()
    assert cfg["optimizer"]["steps"] == 300
    cfg["optimizer"]["steps"] = steps
    truth = fixtures()[fixture_name]
    request = {
        "config": cfg, "truth": truth,
        "input_id": f"smooth_fit_parity_{fixture_name}_{steps}",
        "known_harmonic_weights": truth["harmonic_weights"] if known_weights else None,
    }
    old = _run_implementation("archive", request, archive, expected_env)
    new = _run_implementation("package", request, archive, expected_env)
    old_time, new_time = old.pop("fit_seconds"), new.pop("fit_seconds")
    assert new == old  # Every loss, start, control, seed, wave and component hash.
    result = new["result"]
    assert len(result["history"]) == steps + 1
    assert [row["seed"] for row in result["starts"]] == [42, 123, 456, 789]
    if known_weights:
        for row in result["starts"]:
            assert row["parameters"]["harmonic_weights"] == row["initial_parameters"]["harmonic_weights"]
    _write_parity_evidence(
        request, dict(old, fit_seconds=old_time), dict(new, fit_seconds=new_time), archive,
    )
    print(json.dumps({
        "fixture": fixture_name, "steps": steps, "exact_parity": True,
        "history_states": len(result["history"]), "starts": len(result["starts"]),
        "archive_fit_seconds": old_time, "packaged_fit_seconds": new_time,
        "result_sha256": hashlib.sha256(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "waveform_sha256": new["waveform_sha256"],
    }, sort_keys=True))
