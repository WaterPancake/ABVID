"""Complete, versioned numerical contract for the small physics comparison."""
import json
from copy import deepcopy

from abvid.paths import resolve
from .data import REVISION, sha

SPEC = {
    "schema_version": 1,
    "id": "physics_engine_v0",
    "phase": "synthetic_development_reconstruction",
    "data_root": f"data:physics_engine/procedural_engine_sounds/{REVISION}",
    "input_lock": "repo:experiments/physics_engine/input_lock.json",
    "input_lock_sha256": "b7150b140184f15438a96d2138a70c9bac651c82456ed37f65176232c8cc2671",
    "audio": {"sample_rate_hz": 16000, "samples": 32000,
              "route": "channels_1_2_only_mean_native_to8k_to16k_center2s_kaiser5",
              "controls": "linear_interpolation_at_cropped_audio_sample_times_no_audio_normalization",
              "objective": "unchanged_CAST_v0_unit_RMS_MRSTFT",
              "physics_bandlimit": "16k_to8k_to16k_kaiser5",
              "playback": "restore_target_AC_RMS_one_common_gain_peak_at_most_0.95"},
    "source": {"version": "four_stroke_pulse_resonator_v0", "oversample": 4,
               "preroll_seconds": 0.25, "torque_reference_nm": 400.0, "idle_gain": 0.15,
               "pulse_width_ms": 0.8, "resonance_hz": 400.0, "resonance_q": 2.0,
               "noise_gain": 0.03, "pressure_units": "relative_uncalibrated",
               "firing": "uniform_cylinders_four_stroke_no_firing_order_inference",
               "comparison_path": "identity_no_extra_propagation"},
    "fit": {"method": "exhaustive_grid_fit_stream_minimum_first_tie",
            "pulse_width_ms": [0.25, 0.8, 2.0], "resonance_hz": [150.0, 400.0, 1000.0],
            "resonance_q": [0.7, 2.0], "noise_gain": [0.0, 0.03, 0.12],
            "fit_realizations": 2, "check_realizations": 2,
            "seed": 42, "seed_derivation": "sha256_canonical_[42,input_id,physics,purpose,index]_first8_big_mod_2**63",
            "equivalent_relative_loss": 0.05, "minimum_check_improvement_fraction": 0.05},
    "cast_references": ["v0_original_full_300_steps_four_starts", "smooth8_full_300_steps_four_starts"],
    "demo": {"seconds": 8, "rpm_start": 1000, "rpm_end": 4000,
             "torque_start_nm": 20, "torque_end_nm": 200,
             "cylinders": 4, "path_x_start_m": -60, "path_x_end_m": 60,
             "path_y_m": 5, "sound_speed_m_s": 343.0, "seed": 123},
    "hardware": {"device": "cpu", "torch_threads": 1, "interop_threads": 1,
                 "deterministic_algorithms": True},
    "scope": {"role": "one_conservative_synthetic_development_group",
              "independent_generalization": False, "classifier": None,
              "protected_or_target_audio_access": False},
}


def configuration():
    return deepcopy(SPEC)


def load(reference):
    cfg = json.loads(resolve(reference).read_text())
    if cfg != SPEC:
        raise ValueError("Unresolved or changed physics v0 configuration; version a new contract")
    if sha(resolve(cfg["input_lock"])) != cfg["input_lock_sha256"]:
        raise ValueError("Input selection lock hash mismatch")
    resolve(cfg["data_root"], must_exist=False)
    return cfg
