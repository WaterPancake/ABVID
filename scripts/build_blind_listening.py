"""Build a local anonymous listening test; never serve the private answer key."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from vehicle_audio.benchmark_followup import sha256, write_json
from vehicle_audio.development_corpus import MANIFEST, MANIFEST_SHA256, load_development
from vehicle_audio.listening import select_anchors
from vehicle_audio.real_corpus import RealCorpusConfig, _read_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/blind_listening_v1")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, sources, waves, audit = load_development()
    anchors = select_anchors(records, sources)
    public = args.output / "participant"
    private = args.output / "private"
    public.mkdir(parents=True)
    private.mkdir()
    positions = {r["sample_id"]: i for i, r in enumerate(records)}
    rng = random.Random(43)
    trials = []
    for seconds in (2, 6):
        order = list(anchors)
        rng.shuffle(order)
        for r in order:
            ident = f"clip_{rng.getrandbits(64):016x}"
            center = (r["window_start_seconds"] + r["window_end_seconds"]) / 2
            if seconds == 2:
                wave = waves[positions[r["sample_id"]]]
            else:
                source = Path("data") / sources[r["source_id"]]["output_path"]
                candidate = SimpleNamespace(
                    source=SimpleNamespace(path=source),
                    start_sample=round((center - 3) * 16000),
                )
                wave = _read_window(candidate, RealCorpusConfig(window_seconds=6))[
                    0
                ].numpy()[0]
            if wave.shape != (seconds * 16000,) or not np.isfinite(wave).all():
                raise ValueError("invalid listening excerpt")
            path = public / f"{ident}.wav"
            sf.write(path, wave, 16000, subtype="FLOAT")
            restored, sr = sf.read(path, dtype="float32")
            if sr != 16000 or not np.array_equal(restored, wave):
                raise ValueError("lossless audio export failed")
            trials.append(
                {
                    "id": ident,
                    "seconds": seconds,
                    "file": path.name,
                    "audio_sha256": sha256(path),
                    "anchor_sample_id": r["sample_id"],
                    "source_id": r["source_id"],
                    "recording_session": r["recording_session"],
                    "vehicle_class": r["vehicle_class"],
                    "source_page": r["source_page"],
                    "license": r["license"],
                    "attribution": r["attribution"],
                    "start_seconds": center - seconds / 2,
                    "end_seconds": center + seconds / 2,
                }
            )
    test_id = (
        "abvid-listening-v1-"
        + hashlib.sha256(json.dumps(trials, sort_keys=True).encode()).hexdigest()[:12]
    )
    participant = {
        "test_id": test_id,
        "trials": [{k: t[k] for k in ("id", "seconds", "file")} for t in trials],
    }
    template = Path("src/vehicle_audio/listening_ui.html").read_text()
    (public / "index.html").write_text(
        template.replace("__TRIAL_DATA__", json.dumps(participant))
    )
    write_json(private / "answer_key.json", {"test_id": test_id, "trials": trials})
    shutil.copy2(MANIFEST, private / "development_manifest.jsonl")
    write_json(
        private / "experiment.json",
        {
            "domain": "human listening to native development real audio",
            "seed": 42,
            "dataset_sha256": MANIFEST_SHA256,
            "audit": audit,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
            "model_checkpoint": None,
            "metrics": "pending real participant responses",
            "train_test_split": "not a model split; development-only human diagnostic",
            "hashes": {
                str(p): sha256(p)
                for p in [
                    Path(__file__),
                    Path("src/vehicle_audio/listening.py"),
                    Path("src/vehicle_audio/development_corpus.py"),
                    Path("src/vehicle_audio/listening_ui.html"),
                    Path("docs/blind_listening_protocol.md"),
                    Path("configs/audio_sources.yaml"),
                ]
            },
            "participant_artifacts": {p.name: sha256(p) for p in public.iterdir()},
        },
    )
    print(
        f"Ready: {public / 'index.html'} ({len(trials)} trials); serve participant ONLY"
    )


if __name__ == "__main__":
    main()
