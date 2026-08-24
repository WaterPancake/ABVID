"""Audibility-gated temporal aggregation for one-channel vehicle events.

The audibility gate is intentionally separate from vehicle detection.  Relative
energy can suppress faint event edges, but it cannot determine whether an
arbitrary loud sound is a vehicle.  A future presence detector can supply the
same boolean window mask without changing the temporal aggregation interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from vehicle_audio.baseline import CLASS_NAMES
from vehicle_audio.invariance_evaluation import _git_commit
from vehicle_audio.pretrained_evaluation import (
    PANN_AUDIOSET_DIMENSION,
    PANN_SAMPLE_RATE,
    StandardizedLinearProbe,
    validate_panns_checkpoint,
)


EVENT_INFERENCE_VERSION = 1


@dataclass(frozen=True)
class EventWindowConfig:
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    channel: int = 0

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("window and hop duration must be positive")
        if self.hop_seconds > self.window_seconds:
            raise ValueError("hop duration must not exceed window duration")
        if self.channel < 0:
            raise ValueError("channel must be nonnegative")


@dataclass(frozen=True)
class AudibilityGateConfig:
    relative_peak_threshold_db: float
    minimum_active_windows: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_peak_threshold_db):
            raise ValueError("relative peak threshold must be finite")
        if self.relative_peak_threshold_db > 0:
            raise ValueError("relative peak threshold must be at most zero dB")
        if self.minimum_active_windows <= 0:
            raise ValueError("minimum_active_windows must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_event_predictions(
    probabilities: torch.Tensor,
    rms: torch.Tensor,
    window_start_seconds: Sequence[float],
    *,
    window_seconds: float,
    gate: AudibilityGateConfig,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Gate faint windows and mean the retained class probabilities over time."""

    probabilities = probabilities.detach().to(torch.float32).cpu()
    rms = rms.detach().to(torch.float32).cpu()
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("probabilities must have shape [windows, classes]")
    if rms.shape != (len(probabilities),):
        raise ValueError("rms must have one value per window")
    if len(window_start_seconds) != len(probabilities) or not len(probabilities):
        raise ValueError("window starts must have one nonempty value per window")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(rms).all():
        raise ValueError("event inputs contain NaN or Inf")
    if (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(len(probabilities)),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("rows must contain normalized class probabilities")
    if (rms < 0).any():
        raise ValueError("rms values must be nonnegative")

    peak_rms = float(rms.max())
    if peak_rms <= 0:
        relative_db = torch.full_like(rms, float("-inf"))
        active = torch.zeros_like(rms, dtype=torch.bool)
    else:
        relative_db = 20.0 * torch.log10((rms / peak_rms).clamp_min(1e-12))
        active = relative_db >= gate.relative_peak_threshold_db
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    status = (
        "classified"
        if len(active_indices) >= gate.minimum_active_windows
        else "insufficient_audible_audio"
    )
    aggregate_probabilities: dict[str, float] | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    if status == "classified":
        aggregate = probabilities[active_indices].mean(dim=0)
        predicted_index = int(aggregate.argmax())
        predicted_class = str(class_names[predicted_index])
        confidence = float(aggregate[predicted_index])
        aggregate_probabilities = {
            str(name): float(aggregate[index])
            for index, name in enumerate(class_names)
        }

    windows: list[dict[str, Any]] = []
    for index, start_seconds in enumerate(window_start_seconds):
        prediction_index = int(probabilities[index].argmax())
        windows.append(
            {
                "window_index": index,
                "start_seconds": float(start_seconds),
                "end_seconds": float(start_seconds) + window_seconds,
                "rms": float(rms[index]),
                "relative_peak_db": float(relative_db[index]),
                "audibility_gate_passed": bool(active[index]),
                "predicted_class": str(class_names[prediction_index]),
                "class_probabilities": {
                    str(name): float(probabilities[index, class_index])
                    for class_index, name in enumerate(class_names)
                },
            }
        )
    return {
        "status": status,
        "predicted_class": predicted_class,
        "classification_confidence": confidence,
        "class_probabilities": aggregate_probabilities,
        "window_count": len(probabilities),
        "active_window_count": len(active_indices),
        "active_window_fraction": len(active_indices) / len(probabilities),
        "peak_rms": peak_rms,
        "gate": {
            **asdict(gate),
            "method": "window_rms_relative_to_event_peak",
            "is_vehicle_presence_detector": False,
        },
        "aggregation": "unweighted_mean_probability_over_active_windows",
        "windows": windows,
    }


def _window_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    config: EventWindowConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    window_samples = round(config.window_seconds * sample_rate)
    hop_samples = round(config.hop_seconds * sample_rate)
    if len(waveform) < window_samples:
        raise ValueError("audio is shorter than one inference window")
    starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    windows = torch.stack(
        [waveform[start : start + window_samples] for start in starts]
    )
    rms = windows.square().mean(dim=1).sqrt()
    return windows, rms, [start / sample_rate for start in starts]


def _load_probe(
    bundle_path: Path, state_key: str
) -> tuple[StandardizedLinearProbe, dict[str, Any]]:
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if tuple(payload.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("probe class names do not match tracked/wheeled order")
    models = payload.get("models")
    if not isinstance(models, Mapping) or state_key not in models:
        raise ValueError(f"probe bundle has no model state {state_key!r}")
    state = models[state_key]
    if state["feature_mean"].shape != (PANN_AUDIOSET_DIMENSION,):
        raise ValueError("event inference requires a 527-output AudioSet probe")
    model = StandardizedLinearProbe(
        state["feature_mean"], state["feature_standard_deviation"], len(CLASS_NAMES)
    )
    model.load_state_dict(state)
    model.eval()
    provenance = {
        field: payload[field]
        for field in (
            "training_config",
            "synthetic_manifest_sha256",
            "real_manifest_sha256",
            "pretrained_encoder",
            "class_names",
        )
        if field in payload
    }
    return model, provenance


def infer_audio_event(
    audio_path: str | Path,
    probe_bundle: str | Path,
    panns_checkpoint: str | Path,
    output_path: str | Path,
    *,
    state_key: str = "panns_audioset_paired_linear",
    window_config: EventWindowConfig = EventWindowConfig(),
    gate_config: AudibilityGateConfig,
    batch_size: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run frozen PANNs/probe inference and audibility-aware event aggregation."""

    source = Path(audio_path)
    bundle = Path(probe_bundle)
    panns_path = Path(panns_checkpoint)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite event result: {output}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if not 0 <= window_config.channel < samples.shape[1]:
        raise ValueError(
            f"requested channel {window_config.channel} but audio has {samples.shape[1]}"
        )
    waveform = torch.from_numpy(samples[:, window_config.channel].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError("input audio contains NaN or Inf")
    windows, rms, starts = _window_audio(waveform, sample_rate, window_config)
    if sample_rate != PANN_SAMPLE_RATE:
        windows = audio_functional.resample(windows, sample_rate, PANN_SAMPLE_RATE)

    checkpoint_metadata = validate_panns_checkpoint(panns_path)
    try:
        from panns_inference import AudioTagging
    except ImportError as error:
        raise RuntimeError(
            "event inference requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=str(panns_path), device="cpu")
    clipwise_batches: list[torch.Tensor] = []
    for start in range(0, len(windows), batch_size):
        clipwise_output, _ = tagging.inference(
            windows[start : start + batch_size].numpy()
        )
        clipwise_batches.append(
            torch.from_numpy(np.asarray(clipwise_output)).to(torch.float32)
        )
    clipwise_outputs = torch.cat(clipwise_batches)
    if clipwise_outputs.shape != (len(windows), PANN_AUDIOSET_DIMENSION):
        raise RuntimeError("PANNs returned an unexpected AudioSet output shape")
    probe, probe_provenance = _load_probe(bundle, state_key)
    with torch.inference_mode():
        probabilities = torch.softmax(probe(clipwise_outputs), dim=1)
    aggregate = aggregate_event_predictions(
        probabilities,
        rms,
        starts,
        window_seconds=window_config.window_seconds,
        gate=gate_config,
    )
    payload = {
        "event_inference_version": EVENT_INFERENCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_audio": str(source),
        "input_audio_sha256": _sha256(source),
        "input_sample_rate": sample_rate,
        "input_channels": samples.shape[1],
        "window_config": asdict(window_config),
        "probe_bundle": str(bundle),
        "probe_bundle_sha256": _sha256(bundle),
        "probe_state_key": state_key,
        "probe_provenance": probe_provenance,
        "pretrained_encoder": checkpoint_metadata,
        "result": aggregate,
        "protocol_status": "engineering_audibility_gate_not_vehicle_detection",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
