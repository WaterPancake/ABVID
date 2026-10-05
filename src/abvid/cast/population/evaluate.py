"""Numeric components extracted from CAST/improvement/src/cast_improvement/evaluate.py."""
from copy import deepcopy


import numpy as np


import torch


from abvid.cast.audio import shape


from abvid.cast.renderer import seed_for, tensors, validate_controls


from abvid.cast.coverage import render as render_v0


from abvid.cast.population.engine import SmoothRenderer


def transformed(p):
    """Positive/log-simplex/logit coordinates; remove envelope scale gauge."""
    out = []
    out.extend(np.log(p["spacing_hz"]))
    for key in ("harmonic_weights", "noise_weights"):
        a = np.log(np.maximum(p[key], 1e-8));out.extend(a-a.mean())
    mix = np.clip(p["harmonic_fraction"], 1e-4, 1-1e-4)
    out.append(np.log(mix/(1-mix)))
    env = np.log(p["envelope_knots"]);out.extend(env-env.mean())
    return np.array(out)


def inverse(z, cfg):
    def softmax(x):
        x = np.exp(x-x.max());return (x/x.sum()).tolist()
    p = {"spacing_hz": np.exp(z[:3]).clip(*cfg["renderer"]["spacing_hz_bounds"]).tolist(),
         "harmonic_weights": softmax(z[3:11]), "noise_weights": softmax(z[11:19]),
         "harmonic_fraction": float(1/(1+np.exp(-np.clip(z[19], -15, 15)))),
         "envelope_knots": np.exp(z[20:25]).clip(*cfg["renderer"]["envelope_bounds"]).tolist()}
    validate_controls(p, cfg)
    return p


def temper(bank, temperature, cfg):
    if temperature not in (1., 1.1, 1.25, 1.5):
        raise ValueError("Temperature outside declared grid")
    if temperature == 1:
        return deepcopy(bank)
    result = []
    for c in ("car", "truck"):
        rows = [r for r in bank if r["class"] == c]
        groups = sorted({r["group"] for r in rows})
        center = np.mean([np.mean([transformed(r["parameters"]) for r in rows if r["group"] == g], axis=0) for g in groups], axis=0)
        for row in rows:
            z = center+temperature*(transformed(row["parameters"])-center)
            result.append({**row, "parameters": inverse(z, cfg)})
    return sorted(result, key=lambda r: r["file_id"])


def rendered(record, cfg, variant):
    if variant == "v0":
        return render_v0(record, cfg)
    r = SmoothRenderer(cfg, record["input_id"], sampling=True)
    with torch.no_grad():
        wave = shape(r.render(tensors(record["parameters"]), "sampling"))[0, 0].numpy()
    return wave, {"phase": seed_for(cfg, record["input_id"], "phase", "sampling"),
                  "noise": [seed_for(cfg, record["input_id"], "noise", "sampling", 0)], "purpose": "sampling"}


def aggregate(records, metric_cfg):
    result = {}
    keys = sorted({(r["variant"], r["temperature"]) for r in records})
    for variant, temperature in keys:
        classes = {}
        for c in metric_cfg["class_order"]:
            arms = {}
            for arm in metric_cfg["arms"]:
                selected = [r for r in records if (r["variant"], r["temperature"], r["class"], r["arm"]) == (variant, temperature, c, arm)]
                arms[arm] = {"W1": float(np.mean([r["score"]["W1"] for r in selected])),
                             "coverage": float(np.mean([r["score"]["coverage"] for r in selected])),
                             "family_spread": {f: float(np.mean([r["score"]["families"][f]["spread_ratio"] for r in selected])) for f in metric_cfg["primary_families"]}}
            gains = {a: 1-arms["joint"]["W1"]/arms[a]["W1"] for a in ("prototype", "marginals")}
            classes[c] = {"arms": arms, "gains": gains,
                          "passes": {"coverage": arms["joint"]["coverage"] >= .8, "gain": min(gains.values()) >= .025,
                                     "spread": all(.5 <= x <= 2 for x in arms["joint"]["family_spread"].values())}}
        spread_pass = all(r["passes"]["spread"] for r in classes.values())
        shortfall = max([0.]+[.8-r["arms"]["joint"]["coverage"] for r in classes.values()]+[.025-g for r in classes.values() for g in r["gains"].values()])
        w1 = float(np.mean([r["arms"]["joint"]["W1"] for r in classes.values()]))
        result[f"{variant}_T{temperature:g}"] = {"variant": variant, "temperature": temperature, "classes": classes,
            "selection_key": [not spread_pass, shortfall, w1, temperature], "passes": all(all(r["passes"].values()) for r in classes.values())}
    return result
