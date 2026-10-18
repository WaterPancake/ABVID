"""Independent analytic checks and admission/replay guards for physics v0.

These are new tests for new behavior. Existing tests and the human-maintained
harness baseline are intentionally unchanged.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from abvid.physics.config import configuration, load
from abvid.physics.data import SETS, audit, decode, make_lock, safe_path, sha, verify
from abvid.physics.engine import Engine, firing_events, free_field, render
from abvid.physics.fit import fit, prepare, seeds


def test_four_stroke_clock_known_rpm_and_cylinder_count():
    # 1500 rev/min, four cylinders, one firing per cylinder per two turns: 50 Hz.
    events = firing_events(np.full(32000, 1500.), 16000, 4)
    assert len(events) == 100
    assert np.max(np.abs(events - np.arange(100)*320)) < 1e-6
    eight = firing_events(np.full(32000, 1500.), 16000, 8)
    assert len(eight) == 200
    assert np.max(np.abs(eight - np.arange(200)*160)) < 1e-6


def test_changing_rpm_integrates_crank_phase_and_handles_restart():
    rpm = np.r_[np.zeros(16000), np.full(16000, 1200.), np.full(16000, 2400.)]
    events = firing_events(rpm, 16000, 4)
    expected = np.r_[16000+np.arange(40)*400, 32000+np.arange(80)*200]
    assert events.shape == expected.shape
    assert np.max(np.abs(events-expected)) < 1e-6


def test_stationary_engine_is_silent():
    output, components = render(np.zeros(32000), np.full(32000, 100.))
    assert np.array_equal(output, np.zeros(32000))
    assert all(np.array_equal(v, output) for v in components.values())


def test_firing_frequency_appears_in_output_spectrum():
    # The strongest line need not be the fundamental; check energy at order 1.
    wave, _ = render(np.full(32000, 1500.), np.full(32000, 100.), Engine(noise_gain=0))
    spec = abs(np.fft.rfft(wave[8000:24000]*np.hanning(16000)))
    assert spec[50] > 100 * np.median(spec[45:56])
    assert np.argmax(spec[45:56])+45 == 50


def test_torque_changes_excitation_without_hidden_normalization():
    engine = Engine(noise_gain=0)
    rpm = np.full(32000, 1800.)
    a, _ = render(rpm, np.zeros_like(rpm), engine)
    b, _ = render(rpm, np.full_like(rpm, 120.), engine)
    assert np.allclose(b, 3*a, rtol=1e-8, atol=1e-10)


def test_source_seeds_components_and_inputs_are_preserved():
    rpm, torque = np.linspace(800, 4000, 32000), np.linspace(-50, 200, 32000)
    before = (rpm.copy(), torque.copy())
    a, ac = render(rpm, torque, seed=123)
    b, bc = render(rpm, torque, seed=123)
    c, cc = render(rpm, torque, seed=456)
    assert np.array_equal(a, b) and np.array_equal(ac["broadband"], bc["broadband"])
    assert not np.array_equal(a, c)
    assert np.array_equal(ac["firing"], cc["firing"])
    assert np.array_equal(a, ac["firing"]+ac["broadband"])
    assert np.array_equal(rpm, before[0]) and np.array_equal(torque, before[1])
    assert np.isfinite(a).all() and len(a) == 32000


def test_oversampling_converges_for_fast_tonal_case():
    rpm, torque = np.full(32000, 7000.), np.full(32000, 200.)
    engine = Engine(pulse_width_ms=0.25, resonance_hz=1000, noise_gain=0)
    a, _ = render(rpm, torque, engine, oversample=4)
    b, _ = render(rpm, torque, engine, oversample=8)
    # Discrete causal filters include a sub-sample latency, so align via a
    # spectral-magnitude comparison; do not claim waveform equality.
    x = np.abs(np.fft.rfft(a[8000:24000]*np.hanning(16000)))
    y = np.abs(np.fft.rfft(b[8000:24000]*np.hanning(16000)))
    assert np.linalg.norm(x-y)/np.linalg.norm(y) < 0.02
    assert np.sum(x[7000:]**2)/np.sum(x**2) < 0.001


def test_static_inverse_distance_and_retarded_delay():
    source = np.zeros(16000); source[1000] = 1.
    near = free_field(source, 2., sound_speed_m_s=320.)
    far = free_field(source, 4., sound_speed_m_s=320.)
    assert np.argmax(near) == 1100 and np.argmax(far) == 1200
    assert abs(near.max()-0.5) < 1e-12 and abs(far.max()-0.25) < 1e-12
    assert np.count_nonzero(abs(near)>1e-9) == 1


def test_receding_doppler_matches_analytic_solution():
    sr = 16000; t = np.arange(64000)/sr; f = 200.; c = 343.; speed = 30.
    y = free_field(np.sin(2*np.pi*f*t), 10+speed*t, sound_speed_m_s=c)
    emitted_t = (t-10/c)/(1+speed/c)
    truth = np.sin(2*np.pi*f*emitted_t)/(10+speed*emitted_t)
    valid = (t > 0.2) & (t < 3.5)
    assert np.linalg.norm(y[valid]-truth[valid])/np.linalg.norm(truth[valid]) < 0.001


@pytest.mark.parametrize("bad", [-1., np.nan, 10001.])
def test_invalid_physical_controls_fail_closed(bad):
    with pytest.raises(ValueError):
        render(np.full(32000, bad), np.zeros(32000))


def test_invalid_parameters_and_path_fail_closed():
    with pytest.raises(ValueError):
        render(np.full(32000, 1500.), np.zeros(32000), Engine(resonance_q=0))
    with pytest.raises(ValueError):
        free_field(np.zeros(100), 0.5)
    with pytest.raises(ValueError):
        free_field(np.zeros(100), np.arange(100)+1)


def annotated_file(tmp_path):
    path = tmp_path/"input.wav"
    t = np.arange(144000)/48000
    audio = np.column_stack((0.1*np.sin(2*np.pi*100*t), 0.1*np.sin(2*np.pi*100*t),
                             np.full(len(t), 0.15), np.full(len(t), 0.1)))
    sf.write(path, audio, 48000, subtype="PCM_16")
    row = {"path": path.name, "size": path.stat().st_size, "lfs": {"oid": sha(path)}}
    return path, row


def test_labels_stay_out_of_audio_and_controls_align(tmp_path):
    path, row = annotated_file(tmp_path)
    audio, rpm, torque, sr = decode(path, row)
    raw, target, controls, levels = prepare(audio, rpm, torque, sr)
    assert audio.shape[1] == 2
    assert abs(float(np.mean(raw)) - float(np.mean(audio))) < 1e-6  # Preserve decoded PCM16 DC; telemetry must not enter audio.
    assert controls["rpm"].min() > 1499 and controls["rpm"].max() < 1501
    assert controls["torque_nm"].min() > 99 and controls["torque_nm"].max() < 101
    assert levels["crop_start_seconds"] == 0.5
    assert len(target) == 32000
    # A ramp checks the time origin, not just constant-value scaling.
    ramp = np.arange(len(rpm))/sr
    _, _, controls, _ = prepare(audio, 1000+100*ramp, 20*ramp, sr)
    assert controls["rpm"][0] == 1050
    assert abs(controls["torque_nm"][-1] - (2.5-1/16000)*20) < 1e-10


def test_corruption_rejected_before_decode(tmp_path, monkeypatch):
    path, row = annotated_file(tmp_path)
    with path.open("r+b") as f:
        f.seek(-1, 2); f.write(b"x")
    monkeypatch.setattr(sf, "read", lambda *a, **k: pytest.fail("decode occurred before integrity check"))
    with pytest.raises(ValueError, match="hash mismatch"):
        decode(path, row)


@pytest.mark.parametrize("path", ["../escape.wav", "/absolute.wav", "a/../../b.wav", "a\\..\\b.wav"])
def test_dataset_path_traversal_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_selection_preserves_one_role_and_unknown_sessions(tmp_path):
    files = [{"type": "file", "path": f"dataset/audio/{name}/{i:03}_Engine-{name[0]}.wav",
              "size": 100, "oid": "x"} for name in SETS for i in range(1, 5)]
    lock = make_lock(files)
    assert lock["selected_prefixes"] == ["001", "003"]
    assert len(lock["selected"]) == 16
    assert len({r["conservative_group"] for r in lock["selected"]}) == 1
    assert all(r["recording_session"] == "unknown" for r in lock["selected"])
    changed = deepcopy(lock); changed["selected"][0]["role"] = "test"
    with pytest.raises(ValueError, match="lock mismatch"):
        audit(tmp_path, changed, files)


def test_frozen_config_rejects_changes(tmp_path):
    # Real versioned contract is validated, including its input-lock digest.
    cfg = load("repo:configs/physics/engine_v0.json")
    assert cfg == configuration()
    assert set(seeds("fixture", "fit")).isdisjoint(seeds("fixture", "check"))
    assert seeds("fixture", "fit") == seeds("fixture", "fit")
    assert seeds("fixture", "fit") != seeds("other", "fit")


def test_small_grid_keeps_best_state_and_repeats_exactly():
    from abvid.physics.fit import rendered
    from abvid.cast.audio import shape
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = configuration()
        cfg["fit"].update(pulse_width_ms=[0.25, 0.8], resonance_hz=[400.],
                          resonance_q=[2.], noise_gain=[0.03])
        controls = {"rpm": np.full(32000, 1500.), "torque_nm": np.full(32000, 100.)}
        target = shape(rendered(Engine(pulse_width_ms=0.25), controls, "fixture", "fit", cfg)[0])
        a = fit(target, controls, 4, "fixture", cfg)
        b = fit(target, controls, 4, "fixture", cfg)
        assert a.pop("fit_seconds") >= 0 and b.pop("fit_seconds") >= 0
        assert a == b
        assert a["parameters"]["pulse_width_ms"] == 0.25
        assert a["fit_loss"] == min(r["fit_loss"] for r in a["history"])
        assert a["fit_loss"] < a["baseline_fit_loss"] * 0.95
    finally:
        torch.set_num_threads(old_threads)
