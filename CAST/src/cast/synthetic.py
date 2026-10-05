"""Preregistered fixtures and acceptance criteria, run before real decoding."""
import json
import time
import numpy as np
import torch

from .audio import shape
from .config import digest, save
from .fit import fit
from .renderer import Renderer, tensors


def fixtures():
    base = {"spacing_hz": [170.0]*3, "harmonic_weights": [0.6, .2, .1, .05, .03, .01, .005, .005],
            "noise_weights": [.04, .06, .1, .3, .25, .15, .06, .04], "harmonic_fraction": 1.0,
            "envelope_knots": [1.0]*5}
    return {
        "single_tone": {**base, "spacing_hz": [233.25]*3, "harmonic_weights": [1., 0, 0, 0, 0, 0, 0, 0]},
        "unrestricted_tone": {**base, "spacing_hz": [233.25]*3, "harmonic_weights": [1., 0, 0, 0, 0, 0, 0, 0]},
        "harmonic": base,
        "noise": {**base, "harmonic_fraction": 0.0},
        "changing": {**base, "spacing_hz": [130., 145., 170.], "envelope_knots": [.4, .9, 1.7, .8, .3]},
        "mixture": {**base, "spacing_hz": [105., 110., 100.], "harmonic_fraction": .55,
                    "envelope_knots": [.5, .9, 1.4, 1., .6]},
        "missing_fundamental": {**base, "spacing_hz": [95.]*3, "harmonic_weights": [0, .65, 0, .35, 0, 0, 0, 0]},
    }


def assess(name, truth, result, cfg):
    a = cfg["acceptance"]
    p = result["parameters"]
    improvement = 1-result["fit_loss"]/result["baseline_fit_loss"]
    checks = {"finite": np.isfinite(result["fit_loss"])}
    measurements = {"fit_improvement_fraction": improvement,
                    "spacing_max_error_hz": float(np.max(np.abs(np.array(p["spacing_hz"])-truth["spacing_hz"])))}
    if name in a["nontrivial_fixtures"]:
        checks["meaningful_improvement"] = improvement >= a["minimum_fit_loss_improvement_fraction"]
    if name in ("single_tone", "harmonic", "changing"):
        key = {"single_tone": "single_tone_spacing_tolerance_hz", "harmonic": "harmonic_spacing_tolerance_hz", "changing": "changing_spacing_tolerance_hz"}[name]
        checks["frequency_recovery"] = measurements["spacing_max_error_hz"] <= a[key]
    if name == "noise":
        err = float(np.abs(np.array(p["noise_weights"])-truth["noise_weights"]).sum())
        measurements["noise_band_L1_error"] = err
        checks["noise_spectrum_recovery"] = err <= a["noise_band_L1_tolerance"]
    if name == "missing_fundamental":
        # Both 95 Hz with even harmonics and 190 Hz with harmonics 1/2 are
        # admissible explanations. This is an explicit structural ambiguity.
        checks["ambiguity_exposed"] = True
        measurements["ambiguity"] = {"flag": "harmonic_order_not_identifiable",
                                     "alternative_spacing_hz": [95.0, 190.0],
                                     "physical_RPM": "unknown"}
    if name == "unrestricted_tone":
        order = int(np.argmax(p["harmonic_weights"]))+1
        error = float(np.max(np.abs(np.array(p["spacing_hz"])*order-truth["spacing_hz"])))
        checks["observed_tone_frequency_recovery"] = error <= a["single_tone_spacing_tolerance_hz"]
        measurements["dominant_tone_max_error_hz"] = error
        measurements["ambiguity"] = {"flag": "harmonic_order_not_identifiable", "dominant_harmonic_order": order,
                                      "physical_RPM": "unknown"}
    return {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "measurements": measurements}


def run(out, cfg):
    from .artifacts import save_fit
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save(out/"acceptance_preregistered.json", {"config_sha256": digest(cfg), "criteria": cfg["acceptance"], "fixtures": fixtures()})
    records = []
    for name, truth in fixtures().items():
        input_id = "synthetic_"+name
        # Fixture realization is separate from both optimization/checking seeds.
        target = shape(Renderer(cfg, input_id+"_truth").render(tensors(truth), "check")[0, 0]).detach()
        known = truth["harmonic_weights"] if name == "single_tone" else None
        result = fit(target, cfg, input_id, known_harmonic_weights=known)
        assessment = assess(name, truth, result, cfg)
        record = {"fixture": name, **assessment, "fit_seconds": result["fit_seconds"]}
        save_fit(out/name, target.numpy(), result, cfg, {"origin": "CAST_renderer_synthetic_fixture", "truth": truth, "input_id": input_id})
        save(out/name/"acceptance.json", record)
        records.append(record)
        print(json.dumps(record), flush=True)
    # Demonstrate global-gain/envelope-scale ambiguity numerically without
    # pretending those unidentifiable controls have unique truth.
    p = fixtures()["mixture"]
    renderer = Renderer(cfg, "ambiguity_gain")
    a = shape(renderer.render(tensors(p)))
    q = {**p, "envelope_knots": [v*1.2 for v in p["envelope_knots"]]}
    b = shape(renderer.render(tensors(q))*7)
    error = float((a-b).abs().max())
    ambiguity = {"flag": "gain_envelope_scale_unidentifiable", "max_shape_difference": error,
                 "passed": error < 3e-6, "alternatives": [p, q], "global_gains": [1, 7]}
    save(out/"gain_ambiguity.json", ambiguity)
    summary = {"passed": all(r["passed"] for r in records) and ambiguity["passed"], "records": records,
               "gain_ambiguity": ambiguity, "elapsed_seconds": time.perf_counter()-started,
               "config_sha256": digest(cfg), "domain": "synthetic_to_synthetic_reconstruction"}
    save(out/"summary.json", summary)
    return summary
