"""Calibrate the relative-energy event gate using source-held-out predictions."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from vehicle_audio.baseline import CLASS_NAMES, classification_metrics
from vehicle_audio.event_inference import AudibilityGateConfig
from vehicle_audio.invariance_evaluation import _git_commit, _load_jsonl
from vehicle_audio.low_snr_adaptation import _probabilities
from vehicle_audio.pretrained_evaluation import (
    PRETRAINED_EVALUATION_VERSION,
    _extract_panns_features,
    validate_panns_checkpoint,
)


AUDIBILITY_CALIBRATION_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_gate_calibration_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("gate calibration config must contain a mapping")
    result = dict(value)
    if result.get("calibration_version") != AUDIBILITY_CALIBRATION_VERSION:
        raise ValueError("unsupported audibility calibration version")
    if int(result.get("minimum_active_windows", 0)) <= 0:
        raise ValueError("minimum_active_windows must be positive")
    candidates = tuple(
        float(threshold)
        for threshold in result.get("relative_peak_threshold_candidates_db", ())
    )
    if not candidates or any(threshold > 0 for threshold in candidates):
        raise ValueError("gate thresholds must be a nonempty nonpositive list")
    if len(set(candidates)) != len(candidates):
        raise ValueError("gate thresholds must be unique")
    if not isinstance(result.get("selection_rule"), str):
        raise ValueError("gate calibration must document its selection rule")
    result["relative_peak_threshold_candidates_db"] = candidates
    return result


def validate_native_calibration_records(
    records: Sequence[Mapping[str, Any]], manifest: Path
) -> dict[str, Any]:
    sessions: dict[str, set[str]] = {name: set() for name in CLASS_NAMES}
    root = manifest.parent
    for index, record in enumerate(records):
        for field in (
            "sample_id",
            "audio_path",
            "vehicle_class",
            "recording_session",
            "sample_rate",
            "rms",
        ):
            if field not in record:
                raise ValueError(f"native record {index} has no {field}")
        vehicle_class = str(record["vehicle_class"])
        if vehicle_class not in CLASS_NAMES:
            raise ValueError("native calibration has an unsupported class")
        if record.get("source_domain") != "real_recording":
            raise ValueError("gate calibration requires native real recordings")
        if record.get("content_review_status") != "reviewed_segment":
            raise ValueError("gate calibration requires reviewed segments")
        sessions[vehicle_class].add(str(record["recording_session"]))
        if not (root / str(record["audio_path"])).is_file():
            raise FileNotFoundError(root / str(record["audio_path"]))
    if any(len(values) < 3 for values in sessions.values()):
        raise ValueError("gate calibration requires at least three sessions per class")
    return {
        "window_count": len(records),
        "sessions_by_class": {
            vehicle_class: sorted(values)
            for vehicle_class, values in sessions.items()
        },
    }


def threshold_summary(
    probabilities: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    *,
    relative_peak_threshold_db: float,
    minimum_active_windows: int,
) -> dict[str, Any]:
    """Summarize one threshold over source-held-out native probabilities."""

    if probabilities.shape != (len(records), len(CLASS_NAMES)):
        raise ValueError("probabilities have the wrong shape")
    session_results: dict[str, Any] = {}
    selected_indices: list[int] = []
    class_session_correct: dict[str, list[float]] = {
        name: [] for name in CLASS_NAMES
    }
    for session in sorted({str(record["recording_session"]) for record in records}):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["recording_session"]) == session
        ]
        classes = {str(records[index]["vehicle_class"]) for index in indices}
        if len(classes) != 1:
            raise ValueError(f"session spans classes: {session}")
        vehicle_class = next(iter(classes))
        rms = torch.tensor([float(records[index]["rms"]) for index in indices])
        peak = float(rms.max())
        relative_db = (
            torch.full_like(rms, float("-inf"))
            if peak <= 0
            else 20.0 * torch.log10((rms / peak).clamp_min(1e-12))
        )
        active_positions = torch.nonzero(
            relative_db >= relative_peak_threshold_db, as_tuple=False
        ).flatten()
        active_indices = [indices[int(position)] for position in active_positions]
        sufficient = len(active_indices) >= minimum_active_windows
        predicted_class = None
        mean_probability = None
        correct = False
        if sufficient:
            selected_indices.extend(active_indices)
            mean = probabilities[active_indices].mean(dim=0)
            predicted_class = CLASS_NAMES[int(mean.argmax())]
            mean_probability = {
                name: float(mean[index]) for index, name in enumerate(CLASS_NAMES)
            }
            correct = predicted_class == vehicle_class
        class_session_correct[vehicle_class].append(float(correct))
        session_results[session] = {
            "vehicle_class": vehicle_class,
            "predicted_class": predicted_class,
            "correct": correct,
            "abstained": not sufficient,
            "active_window_count": len(active_indices),
            "window_count": len(indices),
            "active_window_fraction": len(active_indices) / len(indices),
            "mean_probability": mean_probability,
        }
    session_balanced_accuracy = sum(
        sum(values) / len(values) for values in class_session_correct.values()
    ) / len(CLASS_NAMES)
    if selected_indices:
        selected = torch.tensor(selected_indices, dtype=torch.long)
        labels = torch.tensor(
            [CLASS_NAMES.index(str(records[index]["vehicle_class"])) for index in selected]
        )
        window_metrics = classification_metrics(
            probabilities[selected].argmax(dim=1),
            labels,
            torch.zeros(len(selected)),
            probabilities[selected],
        )
        window_metrics.pop("per_snr")
    else:
        window_metrics = None
    return {
        "relative_peak_threshold_db": relative_peak_threshold_db,
        "minimum_active_windows": minimum_active_windows,
        "coverage": len(selected_indices) / len(records),
        "active_window_count": len(selected_indices),
        "session_balanced_accuracy": session_balanced_accuracy,
        "sessions_correct": sum(
            bool(value["correct"]) for value in session_results.values()
        ),
        "session_count": len(session_results),
        "sessions_abstained": sum(
            bool(value["abstained"]) for value in session_results.values()
        ),
        "active_window_metrics": window_metrics,
        "session_results": session_results,
    }


def select_gate_threshold(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen lexicographic threshold selection rule."""

    if not summaries:
        raise ValueError("threshold summaries must not be empty")

    def key(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
        window_metrics = value.get("active_window_metrics")
        window_balanced_accuracy = (
            -1.0
            if window_metrics is None
            else float(window_metrics["balanced_accuracy"])
        )
        threshold = float(value["relative_peak_threshold_db"])
        return (
            float(value["session_balanced_accuracy"]),
            window_balanced_accuracy,
            float(value["coverage"]),
            -abs(threshold),
        )

    selected = max(summaries, key=key)
    return dict(selected)


def _native_feature_cache(
    records: Sequence[Mapping[str, Any]],
    manifest: Path,
    output: Path,
    manifest_sha256: str,
    panns_checkpoint: Path,
    *,
    channel: int,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    encoder = validate_panns_checkpoint(panns_checkpoint)
    path = output / "native_panns_features.pt"
    expected = {
        "audibility_calibration_version": AUDIBILITY_CALIBRATION_VERSION,
        "pretrained_evaluation_version": PRETRAINED_EVALUATION_VERSION,
        "manifest_sha256": manifest_sha256,
        "panns_checkpoint_sha256": encoder["sha256"],
        "channel": channel,
    }
    if path.is_file():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if all(cache.get(key) == value for key, value in expected.items()):
            return cache["clipwise_outputs"], encoder
    try:
        from panns_inference import AudioTagging
    except ImportError as error:
        raise RuntimeError(
            "audibility calibration requires `uv sync --extra pretrained`"
        ) from error
    tagging = AudioTagging(checkpoint_path=str(panns_checkpoint), device="cpu")
    _, outputs = _extract_panns_features(
        records,
        manifest,
        "audio_path",
        tagging,
        channel=channel,
        sample_rate=int(records[0]["sample_rate"]),
        batch_size=batch_size,
        label="native gate calibration",
    )
    torch.save({**expected, "clipwise_outputs": outputs}, path)
    return outputs, encoder


def calibrate_audibility_gate(
    config_path: str | Path,
    native_manifest_path: str | Path,
    adaptation_metrics_path: str | Path,
    adaptation_models_path: str | Path,
    panns_checkpoint: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 64,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Calibrate on source-held-out native predictions from adaptation outer folds."""

    config_path = Path(config_path)
    manifest_path = Path(native_manifest_path)
    metrics_path = Path(adaptation_metrics_path)
    models_path = Path(adaptation_models_path)
    panns_path = Path(panns_checkpoint)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gate calibration: {output}")
    output.mkdir(parents=True)
    config = load_gate_calibration_config(config_path)
    records, manifest_sha256 = _load_jsonl(manifest_path)
    corpus_summary = validate_native_calibration_records(records, manifest_path)
    adaptation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    adaptation_models = torch.load(models_path, map_location="cpu", weights_only=True)
    if adaptation_metrics.get("manifest_sha256") != adaptation_models.get(
        "manifest_sha256"
    ):
        raise ValueError("adaptation metrics and models refer to different data")
    outputs, encoder = _native_feature_cache(
        records,
        manifest_path,
        output,
        manifest_sha256,
        panns_path,
        channel=int(adaptation_metrics["configuration"]["channel"]),
        batch_size=batch_size,
    )
    fold_by_id = {
        str(fold["fold_id"]): fold for fold in adaptation_metrics["fold_results"]
    }
    probability_sum = torch.zeros((len(records), len(CLASS_NAMES)))
    prediction_count = torch.zeros(len(records), dtype=torch.long)
    device = torch.device(device_name)
    for fold_id, state in adaptation_models["outer_fold_models"].items():
        fold = fold_by_id[str(fold_id)]
        test_sessions = {
            str(fold["tracked_test_session"]),
            str(fold["wheeled_test_session"]),
        }
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["recording_session"]) in test_sessions
        ]
        probabilities = _probabilities(
            state, outputs, batch_size=batch_size, device=device
        )
        probability_sum[indices] += probabilities[indices]
        prediction_count[indices] += 1
    if (prediction_count == 0).any():
        raise RuntimeError("fold models did not cover every native record")
    probabilities = probability_sum / prediction_count.unsqueeze(1)
    summaries = [
        threshold_summary(
            probabilities,
            records,
            relative_peak_threshold_db=float(threshold),
            minimum_active_windows=int(config["minimum_active_windows"]),
        )
        for threshold in config["relative_peak_threshold_candidates_db"]
    ]
    selected = select_gate_threshold(summaries)
    gate = AudibilityGateConfig(
        relative_peak_threshold_db=float(selected["relative_peak_threshold_db"]),
        minimum_active_windows=int(config["minimum_active_windows"]),
    )
    payload = {
        "audibility_calibration_version": AUDIBILITY_CALIBRATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "configuration": config,
        "configuration_path": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "native_manifest": str(manifest_path),
        "native_manifest_sha256": manifest_sha256,
        "corpus_summary": corpus_summary,
        "adaptation_metrics": str(metrics_path),
        "adaptation_metrics_sha256": _sha256(metrics_path),
        "adaptation_models": str(models_path),
        "adaptation_models_sha256": _sha256(models_path),
        "pretrained_encoder": encoder,
        "prediction_repetitions": {
            name: sorted(
                {
                    int(prediction_count[index])
                    for index, record in enumerate(records)
                    if record["vehicle_class"] == name
                }
            )
            for name in CLASS_NAMES
        },
        "threshold_summaries": summaries,
        "selected": selected,
        "gate_config": {
            **gate.__dict__,
            "method": "window_rms_relative_to_event_peak",
            "calibration_domain": "source-held-out native development sessions",
            "is_vehicle_presence_detector": False,
        },
        "protocol_status": config["status"],
        "limitations": [
            "Relative energy does not distinguish vehicles from other loud sounds.",
            "Calibration uses seven development sessions and needs a fresh external test.",
            "A real vehicle-presence detector still requires vehicle-absent negative recordings.",
        ],
    }
    (output / "calibration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gate_config.json").write_text(
        json.dumps(payload["gate_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "threshold_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = [
            {
                "relative_peak_threshold_db": value["relative_peak_threshold_db"],
                "coverage": value["coverage"],
                "active_window_balanced_accuracy": (
                    None
                    if value["active_window_metrics"] is None
                    else value["active_window_metrics"]["balanced_accuracy"]
                ),
                "session_balanced_accuracy": value["session_balanced_accuracy"],
                "sessions_correct": value["sessions_correct"],
                "sessions_abstained": value["sessions_abstained"],
            }
            for value in summaries
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload
