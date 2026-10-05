"""Audit current development labels without editing corpus/model artifacts."""

import argparse
import subprocess
from pathlib import Path

from vehicle_audio.benchmark_followup import sha256, write_json
from vehicle_audio.development_corpus import MANIFEST, load_development
from vehicle_audio.state_identity_audit import (
    audit_inventory,
    render_report,
    worksheet_segments,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/state_identity_audit_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, sources, _, corpus_audit = load_development()
    worksheet = Path("docs/audio_segment_review.md")
    reviews = worksheet_segments(worksheet.read_text(), {r["source_id"] for r in records})
    result = audit_inventory(records, sources, reviews)
    inputs = [MANIFEST, worksheet, Path("configs/audio_sources.yaml"),
              Path("configs/benchmark_v0_1.yaml"), Path("configs/archived_startup_intervals.yaml"),
              Path(__file__), Path("src/vehicle_audio/state_identity_audit.py"),
              Path("src/vehicle_audio/development_corpus.py")]
    result["provenance"] = {
        "domain": "native-real development metadata audit; no training or evaluation",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "seed": None,
        "seed_reason": "deterministic inventory; no sampling",
        "model_checkpoint": None,
        "split": "existing development sessions only; no split changes",
        "corpus_audit": corpus_audit,
        "validation": "development source/window hashes, sidecars, bounds, startup and protected exclusions verified",
        "media_review_performed": False,
        "protected_media_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "audit.json", result)
    (args.output / "inventory.md").write_text(render_report(result))
    print(result["summary"])
    print(args.output / "inventory.md")


if __name__ == "__main__":
    main()
