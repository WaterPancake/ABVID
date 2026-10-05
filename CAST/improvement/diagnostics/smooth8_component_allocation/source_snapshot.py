"""Source-only component allocation diagnostic; never changes fitted parameters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
import numpy as np
import soundfile as sf

from cast.artifacts import descriptors
from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read
from .data import HERE, setup_cpu
from .evaluate import load_banks


def diagnose(run, dest):
    setup_cpu(); started = time.perf_counter()
    configs, banks, observations, _ = load_banks(run, "full")
    cfg, bank = configs["smooth8"], banks["smooth8"]
    audited = read(run/"fit_full_verification.json")
    if not audited["passed"] or audited["fit_summary_sha256"] != sha(run/"fit_full_summary.json"):
        raise ValueError("Require intact full fit audit")
    observed = {r["file_id"]: np.array(r["descriptors"]["bands"]) for r in observations}
    records = []
    for row in bank:
        folder = run/"fits"/row["file_id"]
        fraction = row["parameters"]["harmonic_fraction"]
        original, sr = sf.read(folder/"reconstructed_shape.wav", dtype="float32")
        baseline = np.abs(descriptors(original, cfg)["band_proportions"]-observed[row["file_id"]]).sum()
        entry = {"file_id": row["file_id"], "class": row["class"], "group": row["group"],
                 "harmonic_fraction": fraction, "baseline_band_L1": float(baseline),
                 "fit_sha256": row["fit_sha256"], "offset_diagnostics": {},
                 "note": "same-source component diagnostic, not held evaluation or a new fit"}
        with np.load(folder/"components.npz", allow_pickle=False) as parts:
            harmonic = parts["harmonic"][0, 0].astype(np.float64)
            noise = parts["noise"][0, 0].astype(np.float64)
        for delta in (-2., -1.):
            if fraction <= 0 or fraction >= 1:
                entry["offset_diagnostics"][str(delta)] = {"skipped": "boundary component not recoverable by saved-component scaling"}
                continue
            changed = 1/(1+np.exp(-(np.log(fraction/(1-fraction))+delta)))
            wave = harmonic*np.sqrt(changed/fraction)+noise*np.sqrt((1-changed)/(1-fraction))
            error = float(np.abs(descriptors(wave, cfg)["band_proportions"]-observed[row["file_id"]]).sum())
            entry["offset_diagnostics"][str(delta)] = {"harmonic_fraction": float(changed), "band_L1": error,
                                                       "absolute_band_L1_reduction": float(baseline-error)}
        records.append(entry)
    summary = {}
    for label in ("car", "truck"):
        rows = [r for r in records if r["class"] == label]
        summary[label] = {"count": len(rows), "median_fitted_harmonic_fraction": float(np.median([r["harmonic_fraction"] for r in rows])), "offsets": {}}
        for delta in ("-2.0", "-1.0"):
            kept = [r for r in rows if "skipped" not in r["offset_diagnostics"][delta]]
            summary[label]["offsets"][delta] = {
                "used": len(kept), "skipped": len(rows)-len(kept),
                "improved": sum(r["offset_diagnostics"][delta]["absolute_band_L1_reduction"] > 0 for r in kept),
                "median_absolute_band_L1_reduction": float(np.median([r["offset_diagnostics"][delta]["absolute_band_L1_reduction"] for r in kept])),
                "group_median_absolute_band_L1_reduction": {
                    g: float(np.median([r["offset_diagnostics"][delta]["absolute_band_L1_reduction"] for r in kept if r["group"] == g]))
                    for g in sorted({r["group"] for r in kept})}}
    dest.mkdir(parents=True, exist_ok=False)
    save(dest/"per_parent.json", records)
    save(dest/"summary.json", {"classes": summary, "outer_access": False, "real_audio_opened": False,
         "fit_run": str(run.relative_to(ROOT)), "fit_audit_sha256": sha(run/"fit_full_verification.json"),
         "code_sha256": sha(Path(__file__)), "records_sha256": sha(dest/"per_parent.json"),
         "seconds": time.perf_counter()-started,
         "limitation": "Approximate saved-component scaling; bound cases retained as skipped, no generated or held coverage claim"})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-id", required=True); a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", a.output_id): p.error("Use a simple ID")
    diagnose(HERE/"runs/cast_smooth8_v1_20261004_r1", HERE/"diagnostics"/a.output_id)
