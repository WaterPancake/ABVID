"""Numeric components extracted from CAST/improvement/src/cast_improvement/envelope_prior.py."""
from copy import deepcopy


import numpy as np


from scipy.optimize import minimize


from abvid.cast.config import digest


from abvid.cast.renderer import validate_controls


from abvid.cast.population.parent_mixture_bank import draw as mixture_draw


MODES = ("mixture_only", "mixture_envelope")


def frame_basis():
    t = np.linspace(0, 1, 16000)
    knots = np.linspace(0, 1, 5)
    basis = np.stack([np.interp(t, knots, np.eye(5)[i]) for i in range(5)], axis=1)
    return basis.reshape(40,400,5).mean(axis=1)


def calibrate(original, target):
    original, target = np.asarray(original, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (original.shape != (5,) or target.shape != (40,) or not np.isfinite(original).all()
        or not np.isfinite(target).all() or (original < .1).any() or (original > 3).any()
        or (target < 0).any() or target.mean() <= 1e-12):
        raise ValueError("Invalid envelope controls or observed RMS trajectory")
    basis = frame_basis()
    gauge = basis.mean(axis=0)
    amplitude = float(gauge@original)
    scale = amplitude/float(target.mean())
    y = target*scale

    def fun(x): return float(np.square(basis@x-y).mean())
    def jac(x): return 2*basis.T@(basis@x-y)/40

    solved = minimize(fun, original, jac=jac, method="SLSQP", bounds=[(.1,3.)]*5,
                      constraints=[{"type":"eq", "fun":lambda x: float(gauge@x-amplitude), "jac":lambda x:gauge}],
                      options={"ftol":1e-12,"maxiter":300})
    x = solved.x
    flags = []
    if (not solved.success or not np.isfinite(x).all() or (x < .1-1e-12).any() or (x > 3+1e-12).any()
        or abs(gauge@x-amplitude)>1e-8 or fun(x)>fun(original)+1e-12):
        x = original.copy(); flags.append("envelope_calibration_solver_failure_original_retained")
    x = np.clip(x, .1, 3.)
    if ((x <= .10000001) | (x >= 2.99999999)).any(): flags.append("calibrated_envelope_boundary")
    return {"original_envelope_knots":original.tolist(), "envelope_knots":x.tolist(),
            "observed_envelope":target.tolist(), "target_scale":scale, "preserved_mean_amplitude":amplitude,
            "result_mean_amplitude":float(gauge@x), "original_frame_objective":fun(original),
            "calibrated_frame_objective":fun(x), "flags":flags,
            "solver":{"success":bool(solved.success), "status":int(solved.status), "message":str(solved.message),
                      "iterations":int(solved.nit), "function_evaluations":int(solved.nfev)}}


def derive_bank(bank, observations, cfg, held_group):
    obs = {r["file_id"]:r for r in observations}
    ids = [r["file_id"] for r in bank]
    if len(ids)!=len(set(ids)) or len(obs)!=len(observations) or set(ids)!=set(obs) or any(r["group"]==held_group for r in bank):
        raise ValueError("Envelope calibration requires exactly the non-outer training parents")
    result = []
    for row in bank:
        target = obs[row["file_id"]]
        if any(row[k] != target[k] for k in ("file_id","class","group","parent")):
            raise ValueError("Envelope calibration ancestry differs")
        if cfg["renderer"]["envelope_bounds"] != [.1,3.]:
            raise ValueError("Envelope bounds changed")
        record = {"file_id":row["file_id"],"class":row["class"],"group":row["group"],
                  "observed_descriptor_sha256":digest(target), "original_fit_sha256":row["fit_sha256"],
                  **calibrate(row["parameters"]["envelope_knots"],target["descriptors"]["envelope"])}
        copied = deepcopy(row)
        copied["parameters"]["envelope_knots"] = record["envelope_knots"]
        copied["envelope_calibration"] = record
        copied["envelope_calibration_sha256"] = digest(record)
        validate_controls(copied["parameters"],cfg)
        result.append(copied)
    return result


def draw(bank, cfg, metric, label, seed, index, arm, variant, fold):
    if variant not in MODES: raise ValueError("Undeclared envelope variant")
    row = mixture_draw(bank,cfg,metric,label,seed,index,arm,"parent_calibrated",fold)
    donors = {r["file_id"]:r for r in bank}
    ancestors = {}
    if variant == "mixture_envelope":
        for parent in row["parent_ids"]:
            donor = donors[parent]; rec = donor["envelope_calibration"]
            if (any(rec[k]!=donor[k] for k in ("file_id","class","group"))
                or digest(rec)!=donor["envelope_calibration_sha256"]
                or rec["envelope_knots"]!=donor["parameters"]["envelope_knots"]):
                raise ValueError("Changed envelope calibration ancestry or parameters")
            ancestors[parent] = donor["envelope_calibration_sha256"]
    return {**row,"envelope_calibration_ancestors":ancestors}
