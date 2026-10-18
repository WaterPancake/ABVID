import json

import numpy as np
import pytest

pytest.importorskip("numba")

from abvid.paths import resolve  # noqa: E402
from abvid.simulation.propagation import build_plan, exact_trajectory, render_sources  # noqa: E402
from abvid.simulation.propagation_v2 import render, shift  # noqa: E402


def trajectory(samples=4000, moving=False):
    t = np.arange(samples) / 8000
    x = (12.0 * (t - t.mean())) if moving else np.zeros(samples)
    return np.column_stack([np.full(samples, 20.0), x, np.full(samples, 0.5)])


def source(samples=4000, seed=0):
    return np.random.default_rng(seed).standard_normal((samples, 2))


@pytest.mark.parametrize("moving", [False, True])
def test_causal_mode_reproduces_v1_exactly(moving):
    plan, x = build_plan(trajectory(moving=moving)), source()
    v1 = render_sources(x, plan)
    np.testing.assert_array_equal(render(x, plan, air="causal")["total"], v1["P1"])
    np.testing.assert_array_equal(render(x, plan, air="off", ground=False)["total"], v1["P0"])
    record = render(x, plan, air="causal")["latency"]
    assert record["implicit_direct"] == 5 and record["implicit_reflected"] == 10
    assert record["relative_reflected_minus_direct"] == 5


def test_centered_equals_causal_after_declared_shift_for_static_geometry():
    plan, x = build_plan(trajectory()), source(seed=1)
    causal = render(x, plan, air="causal")["total"]
    centered = render(x, plan, air="centered", direct_latency=5, reflected_latency=10)
    np.testing.assert_allclose(centered["total"], causal, rtol=0, atol=1e-12)
    assert centered["latency"]["relative_reflected_minus_direct"] == 5


def test_default_relative_arrival_follows_geometry_only():
    plan = build_plan(trajectory())
    impulse = np.zeros(4000)
    impulse[1000] = 1.0
    geometric = plan["delays"][0, 1] + plan["delays"][0, 2] - plan["delays"][0, 0]
    ground_offset = int(np.argmax(np.abs(plan["ground"][0])))
    for air, extra in (("centered", 0), ("causal", 5)):
        out = render(impulse, plan, air=air)
        direct_peak = np.argmax(np.abs(out["direct"][:, 0]))
        reflected_peak = np.argmax(np.abs(out["reflected"][:, 0]))
        assert abs((reflected_peak - direct_peak) - (geometric + ground_offset + extra)) <= 1
        assert out["latency"]["relative_reflected_minus_direct"] == extra


def test_centered_air_filters_are_zero_phase_and_positive():
    plan = build_plan(trajectory())
    for coefficients in (plan["air"][0][0], plan["air"][1][0], plan["air"][2][0]):
        np.testing.assert_allclose(coefficients, coefficients[::-1], rtol=0, atol=1e-15)
        center = len(coefficients) // 2
        omega = np.linspace(0, np.pi, 257)
        response = np.exp(-1j * np.outer(omega, np.arange(len(coefficients)) - center)) @ coefficients
        assert np.abs(response.imag).max() < 1e-12 and response.real.min() > 0


def test_declared_latency_is_validated():
    x = np.arange(1.0, 6.0)[:, None]
    np.testing.assert_array_equal(shift(x, 2)[:, 0], [0, 0, 1, 2, 3])
    for bad in (-1, 1.5):
        with pytest.raises(ValueError):
            shift(x, bad)
    plan = build_plan(trajectory(), air=False)
    with pytest.raises(ValueError, match="air"):
        render(source(), plan, air="centered")
    with pytest.raises(ValueError, match="air must be"):
        render(source(), build_plan(trajectory()), air="half")


@pytest.mark.archive
def test_centered_mode_reproduces_archived_attenuation_only_render():
    corpus = resolve("archive:experiments/h2/corpus/source_path_20261004_r1")
    geometries = {r["geometry_id"]: r for r in map(json.loads, resolve(
        "archive:experiments/h2/frozen/source_path_20261004_v1/geometries.jsonl").open())}
    sources = {r["dry_source_id"]: r for r in map(json.loads, (corpus / "source_manifest.jsonl").open())}
    observations = resolve("archive:experiments/h2_air_latency/corpus/attenuation_latency_20261004_v1/observations.jsonl")
    target = next(r for r in map(json.loads, observations.open()) if r["cell"] == "A1L0")
    geometry = geometries[target["geometry_id"]]
    wave = np.load(corpus / sources[target["dry_source_id"]]["path"], allow_pickle=False)
    plan = build_plan(exact_trajectory(geometry), np.array(geometry["mic_xyz_m"], dtype=float))
    full = render(wave, plan, air="centered")["total"][:, 0]
    import hashlib
    digest = hashlib.sha256(np.asarray(full, dtype="<f8", order="C").tobytes()).hexdigest()
    assert digest == target["rendered_waveform_sha256"]
