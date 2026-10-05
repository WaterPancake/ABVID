"""Complete v0 contract. No experiment settings are inferred from results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "CAST"
SPEC = {
    "schema_version": 1,
    "experiment_id": "cast_observation_pilot_v0",
    "phase": "CAST-2",
    "parent_protocol": "h1_common_budget_v1.3",
    "data": {
        "manifest": "experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/source_manifest.jsonl",
        "manifest_sha256": "12f60e038a9d45e2e0767b81586cfa0cb2fd62692ce9f5ee39314a567a317dd0",
        "source_lock": "experiments/h1_diagnostics/frozen/source_diagnostics_20261003_v1/lock.json",
        "source_lock_sha256": "85485b6ee8f2139a629293d75e5fb36d3655871a15c1fb266c31480c2aecb7a8",
        "parent_lock": "experiments/h1/frozen/h1_common_budget_v1.3/lock.json",
        "parent_lock_sha256": "3e427a2f4ee084ecdd4102dd3a9863a55cc06496abdb9688258ec26cbaaf901e",
        "classes": {"car": 0, "truck": 1}, "parent_fold": 0, "selection_seed": 42,
        "excluded_group": "connected_4001f06f57cfeee7",
        "pilot_max_per_group_class": 5,
        "selection_rule": "lexicographic_file_id_within_group_class_from_existing_H1_train_ids",
        "selected_ids_file": "selected_manifest.jsonl",
        "allowed_audio_root": "dataset/IDMT_Traffic/audio",
        "dataset_id": "IDMT", "release_id": "IDMT_V1", "target_access": "none",
        "role": "source_training_in_sample_reconstruction_only",
    },
    "audio": {
        "profile": "core8_2s", "sample_rate_hz": 16000, "samples": 32000,
        "intermediate_rate_hz": 8000, "channels": "arithmetic_mean_two_stored_CH34",
        "decode_dtype": "float64", "output_dtype": "float32",
        "resampler": "scipy.signal.resample_poly", "kaiser_beta": 5.0,
        "padtype": "constant_zero", "crop": "center_floor_start_after_resampling",
        "short_audio": "reject_no_padding",
    },
    "normalization": {
        "fit": "dc_removed_unit_rms_shape_only", "silence_rms": 1e-7,
        "numerical_rms_floor": 1e-12,
        "playback": "restore_original_AC_RMS_then_one_common_gain",
        "playback_peak_limit": 0.95, "wav_subtype": "FLOAT",
    },
    "renderer": {
        "version": "cast_harmonic_noise_v0", "internal_sample_rate_hz": 8000,
        "samples": 16000, "spacing_hz_bounds": [10.0, 400.0], "spacing_knots": 3,
        "harmonics": 8, "harmonic_weights": "softmax_amplitude_sum_one",
        "noise_edges_hz": [0, 125, 250, 500, 1000, 1500, 2000, 3000, 4000],
        "noise_filter": "orthogonal_rectangular_rfft_masks_DC_and_Nyquist_excluded",
        "noise_weights": "softmax_energy_sum_one_sqrt_amplitudes",
        "mix": "clamped_linear_harmonic_energy_fraction_0_1",
        "component_normalization": "DC_removed_unit_RMS_before_mixing",
        "envelope_knots": 5, "envelope_bounds": [0.1, 3.0],
        "envelope_normalization": "interpolated_curve_unit_RMS",
        "interpolation": "linear_endpoint_inclusive",
        "phase": "seeded_per_harmonic_uniform_0_2pi_shared_across_starts",
        "alias_rule": "instantaneous_harmonic_frequency_strictly_below_Nyquist",
        "upsampler": "differentiable_FIR_equivalent_scipy_resample_poly_2_1",
        "upsampler_taps": 41, "dtype": "float32_with_float64_phase_accumulation",
    },
    "loss": {
        "fft_sizes": [256, 512, 1024, 2048], "hop_divisor": 4,
        "window": "periodic_Hann", "center": False, "fft_normalized": False,
        "onesided": True, "epsilon": 1e-5, "magnitude_floor": "add_epsilon_inside_log",
        "relative": "mean_abs_difference_over_mean_target_plus_epsilon",
        "log": "mean_abs_natural_log_difference", "resolution_weights": [0.25]*4,
        "term_weights": [1.0, 1.0], "regularization": "none",
        "boundary": "valid_frames_no_padding_discard_incomplete_tail",
    },
    "optimizer": {"name": "adam", "lr": 0.03, "spacing_lr": 0.001, "betas": [0.9, 0.999],
                  "epsilon": 1e-8, "steps": 300, "best_state": "minimum_fit_loss_including_initial",
                  "gradient_clip_norm": 100.0, "spacing_coordinate_scale_hz": 100.0},
    "initialization": {
        "search_fft": 8192, "segment_samples": 8000, "centers_samples": [4000, 16000, 28000],
        "grid_step_hz": 2.0, "candidate_separation_hz": 8.0,
        "salience": "sum_interpolated_magnitudes_at_harmonics_divided_by_harmonic_order",
        "trajectory": "strongest_salience_within_40Hz_log_parabolic_refinement",
        "trajectory_radius_hz": 40.0, "weights": "observed_center_harmonic_magnitude_floor_1e-6_noise_uniform",
        "harmonic_weight_floor": 1e-6,
        "trajectory_endpoints": "linear_extrapolate_segment_center_estimates_to_0_1_2_seconds_then_clip",
        "mix": [0.5, 0.5, 0.5, 0.5], "envelope": "flat",
        "weight_logit_jitter_std": 0.1,
    },
    "seeds": {"run": 42, "starts": [42, 123, 456, 789], "fit_realizations": 2,
              "check_realizations": 2, "check_noise_stream": "disjoint",
              "derivation": "SHA256_canonical_JSON_[run,input_id,component,purpose,index]_first8_big_mod_2**63"},
    "diagnostics": {
        "envelope_frame_samples": 800, "modulation_max_hz": 10.0,
        "spectral_fft": 2048, "salience_resolvable_ratio": 5.0,
        "equally_good_relative_loss": 0.05, "spacing_disagreement_hz": 7.8125,
        "mix_disagreement": 0.2, "noise_dominant_mix_threshold": 0.1,
        "boundary_fraction": 0.01, "weight_boundary_threshold": 0.001,
        "scientific_failure_check_improvement_fraction": 0.05,
        "poor_spectral_relative_threshold": 0.75, "poor_envelope_rmse_threshold": 0.25,
    },
    "acceptance": {
        "single_tone_identifiability": "fix_known_fundamental_only_weights_for_unambiguous_recovery_also_test_unrestricted_tone_ambiguity",
        "single_tone_spacing_tolerance_hz": 7.8125,
        "nontrivial_fixtures": ["harmonic", "noise", "changing", "mixture"],
        "minimum_fit_loss_improvement_fraction": 0.05,
        "harmonic_spacing_tolerance_hz": 7.8125, "changing_spacing_tolerance_hz": 15.625,
        "noise_band_L1_tolerance": 0.35,
        "resampler_max_abs_tolerance": 2e-6, "alias_energy_fraction_tolerance": 1e-5,
        "replay_max_abs_tolerance": 0.0, "finite_difference_relative_tolerance": 0.03,
    },
    "hardware": {"device": "cpu", "torch_threads": 1, "interop_threads": 1,
                 "deterministic_algorithms": True},
    "classifier": None, "distribution": None, "output_root": "CAST/runs",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate(cfg):
    # A version names a complete numerical contract. Changes require a new code/config
    # version, not undocumented defaults or mutation of a resolved run.
    if cfg != SPEC:
        raise ValueError("Unresolved, unsupported, or changed v0 config; all settings must match the versioned contract")
    canonical(cfg)
    return cfg


def load(path):
    return validate(json.loads(Path(path).read_text()))


def configuration():
    return deepcopy(SPEC)
