"""Import human P1 blind reviews and export bounded native-frame audio clips."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import soundfile as sf

from abvid.paths import resolve
from abvid.provenance.hashing import sha256
from . import admission, admission_review
from .config import configuration, external_path
from .manifest import CONTAMINANTS, MOTION, REASONS


def validate_intervals(intervals, recordings):
    fields = {"blind_id", "start_seconds", "end_seconds", "kind",
              "audible_vehicle_count", "motion_state", "contaminants", "rejection_reasons"}
    if not isinstance(intervals, list):
        raise ValueError("Intervals must be an array")
    by_id = {r["blind_id"]: r for r in recordings}
    reviewed, accepted = set(), {}
    trimming = configuration()["trimming"]
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != fields:
            raise ValueError("Unexpected interval fields")
        key = interval["blind_id"]
        if key not in by_id:
            raise ValueError("Unknown blind recording")
        lo, hi = interval["start_seconds"], interval["end_seconds"]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (lo, hi)):
            raise ValueError("Interval times must be finite JSON numbers")
        if not 0 <= lo < hi <= by_id[key]["duration_seconds"]:
            raise ValueError("Interval exceeds loaded audio bounds")
        kind, count = interval["kind"], interval["audible_vehicle_count"]
        if kind not in {"foreground", "background", "rejected"}:
            raise ValueError("Unknown interval decision")
        if not (count == "unknown" or (type(count) is int and count in {0, 1, 2})):
            raise ValueError("Invalid audible vehicle count")
        if interval["motion_state"] not in MOTION:
            raise ValueError("Invalid motion state")
        for name, allowed in [("contaminants", CONTAMINANTS), ("rejection_reasons", REASONS)]:
            values = interval[name]
            if (not isinstance(values, list) or any(v not in allowed for v in values) or
                    len(set(values)) != len(values)):
                raise ValueError("Invalid " + name)
        if kind == "rejected":
            if not interval["rejection_reasons"]:
                raise ValueError("Rejected intervals require a reason")
        else:
            if interval["rejection_reasons"]:
                raise ValueError("Accepted intervals cannot have rejection reasons")
            if count != (1 if kind == "foreground" else 0):
                raise ValueError("Foreground requires one audible vehicle; background requires zero")
            if not trimming["minimum_" + kind + "_seconds"] <= hi - lo <= trimming["maximum_segment_seconds"]:
                raise ValueError("Accepted duration outside frozen trimming rule")
            if kind == "foreground" and set(interval["contaminants"]) & set(trimming["forbidden_foreground_contaminants"]):
                raise ValueError("Contaminated foreground must be rejected")
            for a, b in accepted.get(key, []):
                if a < hi and lo < b:
                    raise ValueError("Accepted intervals overlap")
            accepted.setdefault(key, []).append((lo, hi))
        reviewed.add(key)
    if reviewed != set(by_id):
        raise ValueError("Every recording requires an interval or logged rejection")
    return intervals


def import_packet(annotation_path, packet_name, output_ref, config_ref=admission.CONFIG_REF):
    cfg = admission.load_config(config_ref)
    if not re.fullmatch(r"packet-\d{2}", packet_name):
        raise ValueError("Invalid packet name")
    artifacts = external_path(cfg["storage"]["artifact_root"])
    batch = json.loads((artifacts / "blind" / packet_name / "review_batch.json").read_text())
    private = json.loads((artifacts / "private" / (packet_name + ".json")).read_text())
    annotations = json.loads(Path(annotation_path).read_text())
    if set(annotations) != {"batch_sha256", "tool_version", "trimmer_id", "intervals"}:
        raise ValueError("Unexpected annotation fields")
    if (annotations["tool_version"] != admission_review.VERSION or
            batch["tool_version"] != admission_review.VERSION or batch["protocol_id"] != cfg["id"]):
        raise ValueError("Wrong annotation protocol/tool version")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", annotations["trimmer_id"]):
        raise ValueError("Invalid reviewer ID")
    commitment = {k: v for k, v in private.items() if k != "batch_sha256"}
    batch_hash = admission_review.digest(batch)
    if (batch["private_commitment_sha256"] != admission_review.digest(commitment) or
            private["config_sha256"] != admission_review.digest(cfg) or
            private["batch_sha256"] != batch_hash or annotations["batch_sha256"] != batch_hash):
        raise ValueError("Annotation/private mapping/batch commitment mismatch")
    rows = admission_review.records(cfg)
    if private["input_records_sha256"] != admission_review.digest(rows):
        raise ValueError("Collection changed after review packet was frozen")
    public = {r["blind_id"]: r for r in batch["recordings"]}
    if set(public) != set(private["mapping"]):
        raise ValueError("Public/private recording identities differ")
    by_id = {r["asset_id"]: r for r in rows}
    for key, source in private["mapping"].items():
        row = by_id[source["asset_id"]]
        if (public[key]["audio_sha256"] != source["audio_sha256"] or
                row["acquisition"]["audio_sha256"] != source["audio_sha256"]):
            raise ValueError("Source audio binding changed")
        admission_review.validate_review_audio(row)
    validate_intervals(annotations["intervals"], batch["recordings"])
    segments = []
    for interval in annotations["intervals"]:
        source = private["mapping"][interval["blind_id"]]
        segment_id = "segment-" + admission_review.digest(
            [packet_name, annotations["trimmer_id"], interval])[:24]
        segment = {
            **interval, "segment_id": segment_id, "asset_id": source["asset_id"],
            "audio_ref": source["audio_ref"], "audio_sha256": source["audio_sha256"],
            "source_sha256": source["source_sha256"],
            "native_sample_rate_hz": source["native_sample_rate_hz"],
            "channels": source["channels"], "trimmer_id": annotations["trimmer_id"],
            "experimental_role": "unassigned", "recording_session": "unknown",
            "physical_vehicle_id": "unknown", "independence_verified": False,
            "model_label": "unknown", "model_label_confidence": "unknown",
            "distance_m": "unknown", "mic_type": "unknown",
            "caption_class_candidates": by_id[source["asset_id"]]["caption_classes"],
            "blind_review": True,
        }
        segments.append(segment)
    result = {
        "protocol_id": cfg["id"], "tool_version": admission_review.VERSION,
        "recorded_at": admission.now(), "packet": packet_name, "batch_sha256": batch_hash,
        "annotation_sha256": sha256(Path(annotation_path)),
        "config_sha256": admission_review.digest(cfg),
        "input_records_sha256": private["input_records_sha256"], "segments": segments,
        "note": "Human interval review; source captions are separate candidates, not verified model/engine identities",
    }
    output = external_path(output_ref)
    if not output.is_relative_to(artifacts):
        raise ValueError("Annotation outputs must stay inside the allocated P1 artifact root")
    preserved_input = output.with_name(output.name + ".input.json")
    result["annotation_input_ref"] = output_ref + ".input.json"
    encoded = json.dumps(result, indent=2).encode()
    original_input = Path(annotation_path).read_bytes()
    if len(encoded) + len(original_input) > admission.available_bytes(cfg):
        raise OSError("Storage budget cannot fit imported annotations")
    preserved_input.parent.mkdir(parents=True, exist_ok=True)
    with preserved_input.open("xb") as handle:
        handle.write(original_input)
    admission.write_new(output, result)
    return result


def export_native(imported_ref, output_ref, config_ref=admission.CONFIG_REF):
    cfg = admission.load_config(config_ref)
    imported = json.loads(resolve(imported_ref).read_text())
    if imported["protocol_id"] != cfg["id"] or imported["config_sha256"] != admission_review.digest(cfg):
        raise ValueError("Wrong imported review contract")
    source_rows = admission_review.records(cfg)
    if imported["input_records_sha256"] != admission_review.digest(source_rows):
        raise ValueError("Source collection changed")
    by_id = {r["asset_id"]: r for r in source_rows}
    original_input = resolve(imported["annotation_input_ref"])
    if sha256(original_input) != imported["annotation_sha256"]:
        raise ValueError("Original human annotation input changed")
    original_annotations = json.loads(original_input.read_text())
    if (original_annotations["batch_sha256"] != imported["batch_sha256"] or
            original_annotations["tool_version"] != admission_review.VERSION or
            len(original_annotations["intervals"]) != len(imported["segments"])):
        raise ValueError("Original annotation binding/count changed")
    packet = imported["packet"]
    if not re.fullmatch(r"packet-\d{2}", packet):
        raise ValueError("Invalid imported packet identity")
    artifacts = external_path(cfg["storage"]["artifact_root"])
    batch = json.loads((artifacts / "blind" / packet / "review_batch.json").read_text())
    private = json.loads((artifacts / "private" / (packet + ".json")).read_text())
    commitment = {k: v for k, v in private.items() if k != "batch_sha256"}
    if (admission_review.digest(batch) != imported["batch_sha256"] or
            private["batch_sha256"] != imported["batch_sha256"] or
            batch["private_commitment_sha256"] != admission_review.digest(commitment)):
        raise ValueError("Imported review packet/private commitment changed")
    interval_fields = {"blind_id", "start_seconds", "end_seconds", "kind",
                       "audible_vehicle_count", "motion_state", "contaminants", "rejection_reasons"}
    for segment, original in zip(imported["segments"], original_annotations["intervals"]):
        if {k: segment[k] for k in interval_fields} != original:
            raise ValueError("Imported interval differs from the retained human annotation")
        row = by_id.get(segment["asset_id"])
        mapped = private["mapping"].get(segment["blind_id"])
        if not mapped or any(segment[k] != mapped[k] for k in
                             ("asset_id", "audio_ref", "audio_sha256", "native_sample_rate_hz", "channels")):
            raise ValueError("Imported interval changed its original blind source identity")
        if not row or any(segment[k] != row["acquisition"][k] for k in
                          ("audio_ref", "audio_sha256", "native_sample_rate_hz", "channels")):
            raise ValueError("Clip source differs from the admitted collection binding")
        if segment["experimental_role"] != "unassigned":
            raise ValueError("P1 admission cannot allocate experimental roles")
    output = external_path(output_ref)
    if not output.is_relative_to(artifacts) or output.exists():
        raise ValueError("Choose a new output directory inside the allocated P1 artifact root")
    output.mkdir(parents=True)
    clip_records, rejected, budget_deferred = [], [], []
    for segment in imported["segments"]:
        if segment["kind"] == "rejected":
            rejected.append(segment)
            continue
        source = resolve(segment["audio_ref"])
        if sha256(source) != segment["audio_sha256"]:
            raise ValueError("Native source audio changed")
        rate, channels = segment["native_sample_rate_hz"], segment["channels"]
        lo = math.floor(segment["start_seconds"] * rate)
        hi = math.ceil(segment["end_seconds"] * rate)
        predicted_bytes = (hi - lo) * channels * 4 + 1024
        if predicted_bytes + 16384 > admission.available_bytes(cfg):
            budget_deferred.append(segment["segment_id"])
            continue
        target = output / (segment["segment_id"] + ".wav")
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source), "-map", "0:a:0",
            "-vn", "-af", f"atrim=start_sample={lo}:end_sample={hi},asetpts=PTS-STARTPTS",
            "-c:a", "pcm_f32le", "-map_metadata", "-1", "-fs", str(predicted_bytes), str(target),
        ]
        subprocess.run(command, capture_output=True, timeout=120, check=True)
        info = sf.info(target)
        if info.samplerate != rate or info.channels != channels or info.frames != hi - lo:
            raise ValueError("Native-frame export metadata mismatch")
        decoded = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:a:0",
             "-f", "f32le", "-"], capture_output=True, timeout=120, check=True)
        original = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, channels)
        actual, _ = sf.read(target, dtype="float32", always_2d=True)
        if hi > len(original) or not np.array_equal(actual, original[lo:hi]):
            raise ValueError("Export did not preserve decoded native samples")
        if admission.storage_used(cfg) > cfg["storage"]["maximum_total_bytes"]:
            raise OSError("Storage budget exceeded during export")
        clip_records.append({
            "segment_id": segment["segment_id"], "asset_id": segment["asset_id"],
            "source_audio_sha256": segment["audio_sha256"],
            "start_frame": lo, "end_frame": hi, "sample_rate_hz": rate, "channels": channels,
            "native_samples_equal": True, "clip_sha256": sha256(target),
            "clip_ref": output_ref + "/" + target.name,
        })
    result = {
        "protocol_id": cfg["id"], "recorded_at": admission.now(),
        "imported_review_sha256": sha256(resolve(imported_ref)),
        "clips": clip_records, "rejections": rejected, "budget_deferred_segments": budget_deferred,
        "experimental_roles": "unassigned", "model_or_CAST_fitting_performed": False,
    }
    admission.write_new(output / "export.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["import", "export"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--packet")
    args = parser.parse_args()
    if args.action == "import":
        if not args.packet:
            parser.error("import requires --packet")
        result = import_packet(args.input, args.packet, args.output)
    else:
        result = export_native(args.input, args.output)
    print(json.dumps({"segments": len(result.get("segments", [])),
                      "clips": len(result.get("clips", [])),
                      "budget_deferred": len(result.get("budget_deferred_segments", []))}, indent=2))


if __name__ == "__main__":
    main()
