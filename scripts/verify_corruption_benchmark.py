"""Verify corruption replay, input windows, hashes and held-out prediction membership."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from vehicle_audio.benchmark_followup import (
    FrozenBenchmark,
    MANIFEST,
    sha256,
    write_json,
)
from vehicle_audio.real_corpus import RealCorpusConfig, _read_window

spec = importlib.util.spec_from_file_location(
    "corruption_runner", Path(__file__).with_name("evaluate_controlled_corruption.py")
)
corruption = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corruption)


def verify(output):
    b = FrozenBenchmark()
    lock = json.loads((output / "run_lock.json").read_text())
    config = lock["config"]
    results = json.loads((output / "results.json").read_text())
    for path, expected in lock["code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"code changed: {path}")
    backgrounds, snapshot = corruption.prepare_backgrounds(config, b)
    if snapshot != lock["backgrounds"]:
        raise ValueError("background snapshot changed")
    original = {}
    corpus_config = RealCorpusConfig()
    for r in b.records:
        candidate = SimpleNamespace(
            source=SimpleNamespace(path=Path("data/targets") / r["target_source"]),
            start_sample=r["window_start_sample"],
        )
        rebuilt, _, _ = _read_window(candidate, corpus_config)
        samples, sr = sf.read(
            MANIFEST.parent / r["audio_path"], dtype="float32", always_2d=True
        )
        actual = torch.from_numpy(samples.T.copy())
        if sr != 16000 or not torch.equal(rebuilt, actual):
            raise ValueError(
                f"native window does not match reviewed source: {r['sample_id']}"
            )
        if (
            sha256(MANIFEST.parent / r["audio_path"])
            != lock["native_window_sha256"][r["sample_id"]]
        ):
            raise ValueError("native window changed")
        original[r["sample_id"]] = actual
    checked, max_snr_error = 0, 0.0
    offsets = {}
    for name, result in results.items():
        if name == "native":
            continue
        full = json.loads((output / "conditions" / f"{name}.json").read_text())
        condition = full["condition"]
        if sha256(full["prediction_path"]) != full["prediction_sha256"]:
            raise ValueError("prediction tensor changed")
        if len(full["transforms"]) != len(b.records) or {
            x["sample_id"] for x in full["transforms"]
        } != set(original):
            raise ValueError("transformation coverage mismatch")
        for transform in full["transforms"]:
            sample_id = transform["sample_id"]
            if "background_id" in condition:
                background = backgrounds[condition["background_id"]]
                offset = corruption.stable_offset(
                    config["seed"],
                    sample_id,
                    condition["background_id"],
                    background.shape[-1],
                    32000,
                )
                if offset != transform["background_offset_samples_16khz"]:
                    raise ValueError("background crop offset mismatch")
                key = (sample_id, condition["background_id"])
                if key in offsets and offsets[key] != offset:
                    raise ValueError("background crop changed across SNR levels")
                offsets[key] = offset
                _, meta = corruption.corrupt_window(
                    original[sample_id],
                    background[:, offset : offset + 32000],
                    condition["snr_db"],
                )
                max_snr_error = max(
                    max_snr_error, abs(meta["measured_snr_db"] - condition["snr_db"])
                )
            else:
                _, meta = corruption.corrupt_window(
                    original[sample_id], response=condition["response"]
                )
            if any(transform[k] != v for k, v in meta.items()):
                raise ValueError("corrupted waveform/metadata not deterministic")
            checked += 1
        payload = torch.load(
            full["prediction_path"], weights_only=True, map_location="cpu"
        )
        replay = b.predict(
            payload["semantic_features"].numpy(), payload["classical_features"].numpy()
        )
        for actual, saved, split in zip(
            replay, payload["fold_predictions"], b.splits, strict=True
        ):
            selected = {b.records[i]["sample_id"] for i in saved["indices"].tolist()}
            if selected != set(split["test_sample_ids"]):
                raise ValueError("predictions contain non-held-out windows")
            for key in ("semantic", "classical", "fusion"):
                np.testing.assert_array_equal(actual[key], saved[key].numpy())
    if checked != 30 * 593 or max_snr_error > 1e-5:
        raise ValueError("wrong protocol size or SNR tolerance exceeded")
    write_json(
        output / "verification.json",
        {
            "native_windows_rebuilt_from_hash_verified_sources": len(original),
            "corrupted_waveforms_reproduced_exactly": checked,
            "conditions": len(results) - 1,
            "folds_per_condition": len(b.splits),
            "held_out_prediction_membership_verified": True,
            "saved_probe_predictions_reproduced_exactly": True,
            "background_crops_constant_across_snr": True,
            "maximum_absolute_snr_error_db": max_snr_error,
            "protected_source_overlap": [],
            "results_sha256": sha256(output / "results.json"),
            "verifier_sha256": sha256(__file__),
        },
    )
    print(
        f"Verified {len(original)} native windows, {checked} corrupted waveforms, and {len(b.splits) * (len(results) - 1)} fold-condition predictions; maximum SNR error {max_snr_error:.2g} dB."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/v0.1/corruption_7t5w_v1")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify(args.output)
