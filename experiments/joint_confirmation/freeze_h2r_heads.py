"""Freeze the H2R secondary head list for the joint confirmation pass.

Decided 2026-10-11: H2R uses only the existing latency-free heads (H2 F00/F10 and
the latency control's A1L0 at S0/S1), with no new training. This writes a
create-only manifest of archive paths and recomputed SHA256 values.
Run: uv run --extra cast python experiments/joint_confirmation/freeze_h2r_heads.py
"""
import json
from pathlib import Path

from abvid.paths import resolve
from abvid.provenance.hashing import sha256

OUT = Path(__file__).with_name("h2r_heads.json")
H2 = "archive:experiments/h2/models/source_path_20261004_r1"
LATENCY = "archive:experiments/h2_air_latency/models/attenuation_latency_20261004_v1"
CELLS = {("h2", None, "F00"): "S0_P0", ("h2", None, "F10"): "S1_P0",
         ("latency", "S0", "A1L0"): "S0_P1c", ("latency", "S1", "A1L0"): "S1_P1c"}


def main():
    heads = []
    for study, root in (("h2", H2), ("latency", LATENCY)):
        folder = resolve(root)
        for fit in json.loads((folder / "fits.json").read_text()):
            cell = CELLS.get((study, fit.get("source_level"), fit["cell"]))
            if cell is None:
                continue
            path = folder / fit["model_path"] if study == "h2" else resolve("archive:" + fit["model_path"])
            reference = (f"{root}/{fit['model_path']}" if study == "h2" else "archive:" + fit["model_path"])
            digest = sha256(path)
            if digest != fit["model_sha256"]:
                raise ValueError(f"Head hash mismatch: {fit['fit_id']}")
            heads.append({"cell": cell, "representation": fit["representation"], "bank": fit["replicate_seed"],
                          "fit_id": fit["fit_id"], "model": reference, "sha256": digest})
    heads.sort(key=lambda h: (h["representation"], h["cell"], h["bank"]))
    if len(heads) != 40 or len({(h["representation"], h["cell"], h["bank"]) for h in heads}) != 40:
        raise ValueError("Expected 40 unique heads: 2 representations x 4 cells x 5 banks")
    manifest = {"schema_version": 1, "role": "H2R pre-declared secondary (no confirmatory claim)",
                "decided": "2026-10-11", "training_input_route": "H2 8 kHz render, receiver RMS, 8->16 kHz, 2 s",
                "cells": {"S0_P0": "H2 F00", "S1_P0": "H2 F10", "S0_P1c": "latency-control A1L0, S0",
                          "S1_P1c": "latency-control A1L0, S1"},
                "heads": heads}
    with OUT.open("x") as stream:
        stream.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"heads": len(heads), "manifest_sha256": sha256(OUT)}))


if __name__ == "__main__":
    main()
