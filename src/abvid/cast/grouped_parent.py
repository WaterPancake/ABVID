"""Same-parent CAST-4A calibration, with no file or population access.

The caller validates the frozen execution contract and guards/hash-checks raw
file access before decoding. This module independently enforces the supplied
training role and returns immutable-ready artifact bytes, never writes files,
and never estimates a statistic from another recording.
"""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import time

import numpy as np
import soundfile as sf
import torch

from abvid.cast.artifacts import diagnose
from abvid.cast.audio import normalize, observation as audio_observation, shape
from abvid.cast.config import digest
from abvid.cast.coverage import describe
from abvid.cast.fit import Objective
from abvid.cast.population import envelope_prior, parent_mixture_bank
from abvid.cast.population import resolution_bank, resolution_calibration, resolution_model
from abvid.cast.population import spectrum_bank, spectrum_calibration, temporal_prior
from abvid.cast.population.engine import SmoothRenderer, model_config
from abvid.cast.renderer import tensors
from abvid.cast.reuse_chain import validate_fit, validate_transition
from abvid.cast.roles import FoldRole
from abvid.cast.smooth_fit import fit as fit_smooth


STAGES = ("smooth8", "v7", "v8", "v9", "v10", "v11")
CALIBRATIONS = {"v7": "mixture_calibration", "v8": "envelope_calibration",
                "v9": "spectral_calibration", "v10": "temporal_calibration",
                "v11": "resolution_calibration"}


class ParentCalibrationError(RuntimeError):
    """Technical failure retaining every completed stage and artifact payload."""

    def __init__(self, stage, partial, cause):
        super().__init__(f"Parent calibration failed at {stage}: {cause}")
        self.stage = stage
        self.partial = partial
        self.cause = cause


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _wav_bytes(value):
    stream = BytesIO()
    sf.write(stream, np.asarray(value), 16000, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def _npz_bytes(values):
    stream = BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


def _put(result, name, value):
    if name in result["artifacts"]:
        raise ValueError(f"Artifact would be replaced: {name}")
    result["artifacts"][name] = value
    result["artifact_sha256"][name] = hashlib.sha256(value).hexdigest()


def _guard(source_metadata, role):
    if not isinstance(role, FoldRole):
        raise TypeError("A verified FoldRole is required")
    if not isinstance(source_metadata, dict):
        raise ValueError("Full admitted source metadata is required")
    record = role.validate_parent(source_metadata)
    if record["source_metadata"] != source_metadata:
        raise ValueError("Pass the complete frozen source metadata without omissions")
    return record


def _numerical(numerical):
    """Reject recipe drift while leaving historical role metadata nonexecutable."""
    configs = numerical["renderer_configs"]
    factories = {"smooth8_v1": model_config,
                 "smooth8_temporal41_v10": temporal_prior.model_config,
                 "spectrum16_temporal41_v11": resolution_model.model_config}
    metadata = {"data", "experiment_id", "phase", "parent_protocol", "output_root"}
    for name, factory in factories.items():
        actual = {k: v for k, v in configs[name].items() if k not in metadata}
        expected = {k: v for k, v in factory().items() if k not in metadata}
        if actual != expected:
            raise ValueError(f"Changed inherited numerical configuration: {name}")
    settings = {k: numerical["expected_spectrum_calibration"][k]
                for k in ("steps", "lr", "betas", "epsilon", "gradient_clip")}
    if settings != spectrum_calibration.SETTINGS or settings != resolution_calibration.SETTINGS:
        raise ValueError("Changed inherited spectral optimization settings")
    metric = numerical["metric_config"]
    if metric["log_spectrum_bins"] != 64 or metric["spectral_probability_floor"] != 1e-8:
        raise ValueError("Changed same-parent descriptor recipe")
    return configs, settings, metric


def prepare_observation(raw_stereo, sample_rate, source_metadata, numerical, *, role):
    """Preprocess an already guarded decode; no file access is performed here."""
    _guard(source_metadata, role)
    configs, _, _ = _numerical(numerical)
    cfg = configs["smooth8_v1"]
    original = audio_observation(raw_stereo, sample_rate, cfg)
    target, level = normalize(original, cfg)
    return {"target": target, "original": original, "normalization": level}


def calibrate_parent(target, source_metadata, numerical, *, role, fit_result=None,
                     original=None, normalization=None):
    """Run the fixed smooth8 -> v7 -> v8 -> v9 -> v10 -> v11 chain.

    ``target`` is the exact normalized float32 32000-sample tensor returned by
    :func:`prepare_observation`. An optional fit must have been admitted by the
    caller's separate same-parent cache audit; this function also validates its
    complete history and replays all four checking vectors on this target.
    Historical checking realizations become calibration inputs after smooth8,
    and are never represented as independent validation.
    """
    role_record = _guard(source_metadata, role)
    configs, settings, inherited_metric = _numerical(numerical)
    cfg = configs["smooth8_v1"]
    file_id = source_metadata["file_id"]
    if (not isinstance(target, torch.Tensor) or target.dtype != torch.float32 or
            target.device.type != "cpu" or target.shape != (32000,) or
            not torch.isfinite(target).all() or target.requires_grad):
        raise ValueError("Expected a finite detached CPU float32 32000-sample target")
    if abs(float(target.mean())) > 1e-6 or abs(float(target.square().mean()) - 1) > 1e-6:
        raise ValueError("Target must be DC removed and unit RMS")
    if original is not None:
        original = np.asarray(original)
        if original.dtype != np.float32 or original.shape != (32000,):
            raise ValueError("Original must be the preprocessed float32 observation")
        expected, level = normalize(original, cfg)
        if not torch.equal(target, expected) or (normalization is not None and normalization != level):
            raise ValueError("Observation normalization ancestry differs")
        normalization = level
    elif normalization is not None:
        raise ValueError("Original observation is required to verify level metadata")
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Fixed CPU one-thread deterministic execution is required")
    metric = deepcopy(inherited_metric)
    # Every known held group has already been rejected by FoldRole. The old
    # one-group numerical helpers retain their own additional single-group check.
    metric["held_group"] = role.held_groups[0]
    result = {"file_id": file_id, "source_metadata": deepcopy(source_metadata),
              "role_validation": role_record, "stages": {}, "flags": {}, "fit": None,
              "observation": None, "components": {}, "artifacts": {}, "artifact_sha256": {},
              "stage_seconds": {}, "status": "incomplete", "final_row": None}
    stage = "smooth8"
    started = time.perf_counter()
    try:
        _put(result, "original_shape.wav", _wav_bytes(target.numpy()))
        if original is not None:
            _put(result, "original.wav", _wav_bytes(original))
        fitted = fit_smooth(target, cfg, file_id) if fit_result is None else deepcopy(fit_result)
        result["fit"] = fitted
        validate_fit(fitted, file_id, cfg)
        renderer = SmoothRenderer(cfg, file_id)
        objective = Objective(target, cfg)
        with torch.no_grad():
            for alternative in fitted["starts"]:
                for key, loss_key in (("parameters", "check_loss"),
                                      ("initial_parameters", "initial_check_loss")):
                    actual = float(objective(renderer.render(tensors(alternative[key]), "check")).mean())
                    if abs(actual - alternative[loss_key]) > 1e-5:
                        raise ValueError("Recomputed same-parent checking loss differs")
            raw, components = renderer.render(tensors(fitted["parameters"]), "check", components=True)
            reconstruction = shape(raw)[0, 0].numpy()
            _, terms = objective(raw, detail=True)
        components = {k: v.detach().numpy() for k, v in components.items()}
        result["components"] = {**components, "check_waveforms": raw.numpy()}
        _put(result, "components.npz", _npz_bytes(result["components"]))
        _put(result, "reconstructed_shape.wav", _wav_bytes(reconstruction))
        _put(result, "reconstructed_raw.wav", _wav_bytes(raw[0, 0].numpy()))
        diagnostics = diagnose(target.numpy(), reconstruction, fitted, cfg)
        flags = {"numerical_failure": False, "scientific_flags": diagnostics["flags"],
                 "boundary_hits": diagnostics["boundary_hits"]}
        saved_fit = {**fitted, "config_sha256": digest(cfg),
                     "ancestry": {"parent": deepcopy(source_metadata),
                                  "grouped_design_identity_sha256": role.design_identity_sha256,
                                  "initial_training_partition": role.partition_id},
                     "original_level": deepcopy(normalization) if normalization is not None else "unknown",
                     "diagnostics": diagnostics,
                     "per_resolution_check": {n: {k: v[0].tolist() for k, v in fields.items()}
                                              for n, fields in terms.items()},
                     "physical_metadata": {k: "unknown" for k in
                                           ("RPM", "load", "throttle", "physical_vehicle_id", "weather", "distance")}}
        if fit_result is not None:
            saved_fit["supplied_fit_record_sha256"] = digest(fit_result)
        result["fit"] = saved_fit
        _put(result, "fit.json", _json_bytes(saved_fit))
        _put(result, "failure.json", _json_bytes(flags))
        common = {"file_id": file_id, "class": source_metadata["canonical_class"],
                  "group": source_metadata["provenance_group_id"], "parent": deepcopy(source_metadata)}
        # FLOAT WAV stores the tensor's exact float32 values. Historically the
        # observation descriptors were computed from that saved float32 read.
        stored_target, sr = sf.read(BytesIO(result["artifacts"]["original_shape.wav"]), dtype="float32")
        if sr != 16000 or not np.array_equal(stored_target, target.numpy()):
            raise ValueError("FLOAT WAV observation round trip changed samples")
        observed = {**common, "descriptors": {k: v.tolist() for k, v in
                    describe(stored_target, cfg, metric).items()}}
        result["observation"] = observed
        _put(result, "observation.json", _json_bytes(observed))
        before = {**common, "parameters": deepcopy(fitted["parameters"]), "flags": flags,
                  "fit_sha256": result["artifact_sha256"]["fit.json"]}
        result["stages"][stage] = before
        result["flags"][stage] = flags
        _put(result, stage + ".json", _json_bytes(before))
        result["stage_seconds"][stage] = time.perf_counter() - started
        for stage in STAGES[1:]:
            started = time.perf_counter()
            stage_cfg = configs["spectrum16_temporal41_v11" if stage == "v11" else
                                "smooth8_temporal41_v10" if stage == "v10" else "smooth8_v1"]
            if stage == "v7":
                hashes = {name: result["artifact_sha256"][name] for name in
                          ("fit.json", "components.npz", "original_shape.wav")}
                after = parent_mixture_bank.derived_row(
                    before, components["harmonic"], components["noise"],
                    observed["descriptors"]["bands"], hashes, cfg)
            elif stage == "v8":
                after = envelope_prior.derive_bank([before], [observed], cfg, metric["held_group"])[0]
            elif stage == "v9":
                after = spectrum_bank.fit_record(before, observed, cfg, metric, file_id)
            elif stage == "v10":
                after = temporal_prior.derive_bank([before], [observed], stage_cfg, metric["held_group"])[0]
            else:
                after = resolution_bank.fit_record(before, observed, stage_cfg, metric, file_id)
            validate_transition(stage, before, after, observed,
                                result["artifact_sha256"]["fit.json"], stage_cfg, settings)
            result["stages"][stage] = after
            result["flags"][stage] = after[CALIBRATIONS[stage]]["flags"]
            _put(result, stage + ".json", _json_bytes(after))
            if stage in ("v9", "v11"):
                cls = SmoothRenderer if stage == "v9" else resolution_model.ResolutionRenderer
                with torch.no_grad():
                    waves = cls(stage_cfg, file_id).render(tensors(after["parameters"]), "check")[0].numpy()
                for i, wave in enumerate(waves):
                    _put(result, f"{stage}_reconstruction_{i}.wav", _wav_bytes(wave))
            result["stage_seconds"][stage] = time.perf_counter() - started
            before = after
        result["final_row"] = before
        result["status"] = "complete"
        return result
    except Exception as error:
        result["status"] = "technical_failure"
        result["failed_stage"] = stage
        result["stage_seconds"][stage] = time.perf_counter() - started
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise ParentCalibrationError(stage, result, error) from error
