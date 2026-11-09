"""Per-clip verdicts of the saved real-only (R) heads for the listening page.

No fitting: applies the diagnosis's hash-checked R heads to its saved features.
Every per-stage metric recomputed from these per-clip values must equal the
published results before anything is written. Run in the baseline environment
after listen.py:

    .venv-baseline/bin/python -B experiments/cast_utility/stage_diagnostic/verdicts.py
    .venv/bin/python -B experiments/cast_utility/stage_diagnostic/listen.py --page

The full-band column has no verdict: no head ever saw that route.
"""
import json

import numpy as np

from abvid.cast.grouped_storage import read_bundle
from abvid.cast.stage_diagnostic import (BASE_ENV, REPS, cached_role, design, destination, environment,
                                         load_arrays, stage_plan)
from abvid.paths import resolve

RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
OUT = "artifact:cast/listening/stage_diagnosis_20261025_v1"


def main():
    import joblib
    from abvid.learning.utility import head_metrics
    environment(BASE_ENV)
    d, parents, identity = design(RUN)
    for p in parents:
        if (p["source"]["class_id"] == 1) != (p["source"]["canonical_class"] == "truck"):
            raise ValueError("Class id does not mean truck")
    original = cached_role(d, "real")
    models = {rep: joblib.load(resolve(d["head_refs"][rep + "__R"])) for rep in REPS}
    _, payload = read_bundle(destination(RUN) / "scores")
    published = {(r["representation"], r["scope"], r["input"]): r["metrics"]
                 for r in json.loads(payload["results.json"])["rows"] if r["head"] == "R"}
    inputs = {("original", 0): (d["train_ids"], original, "original")}
    for name, stage, purpose, realization, _ in stage_plan(d):
        arrays = load_arrays(RUN, name, identity)
        inputs[("v11_fresh" if purpose == "fresh" else stage, realization)] = (arrays["ids"].astype(str).tolist(), arrays, name)
    labels = dict(zip(d["train_ids"], original["y"]))
    verdicts, checked = {rep: {} for rep in REPS}, 0
    for (column, realization), (ids, arrays, name) in inputs.items():
        for rep in REPS:
            p = models[rep].predict_proba(arrays[rep].astype(np.float64))[:, 1]
            verdicts[rep].update({f"{fid[:12]}|{column}|{realization}": round(float(v), 4) for fid, v in zip(ids, p)})
            index = dict(zip(ids, p))
            for scope, wanted in (("all380", d["train_ids"]), ("balanced50", d["secondary_ids"])):
                key = (rep, scope, name)
                if key not in published or not set(wanted) <= set(index):
                    continue
                y = np.array([labels[i] for i in wanted])
                if head_metrics(y, np.array([index[i] for i in wanted])) != published[key]:
                    raise ValueError(f"Per-clip verdicts do not reproduce published metrics: {key}")
                checked += 1
    out = resolve(OUT) / "verdicts.js"
    with out.open("x") as handle:
        handle.write("window.VERDICTS = " + json.dumps({"head": "R (real-only, role utility_s42_0)", "p_truck": verdicts}) + ";\n")
    print(f"{sum(map(len, verdicts.values()))} verdicts; {checked} published metric rows reproduced exactly -> {out}")


if __name__ == "__main__":
    main()
