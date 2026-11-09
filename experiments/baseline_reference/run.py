"""B1 reference recipe (b1_reference_v1): parity check, scores and all-IDMT heads.

Contract: experiments/baseline_reference/PROTOCOL.md. Baseline environment.
Reads only B1's cached IDMT features/results and, for the feature-parity gate,
16 IDMT source clips. The recipe lives in src/abvid/learning/reference.py.

    .venv-baseline/bin/python -B experiments/baseline_reference/run.py check   # gates only, no scores, no writes
    .venv-baseline/bin/python -B experiments/baseline_reference/run.py run     # frozen protocol required
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from io import BytesIO
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np

from abvid.learning import reference as recipe
from abvid.learning.grouped_search import fit, fold_bootstrap, metrics
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

PROTOCOL = "repo:experiments/baseline_reference/PROTOCOL.md"
PROTOCOL_SHA = "1950f4e15928d58a8b3761ce7df1f36a633988ed019a77f50e95ba1276d32be1"
CODE = ("repo:experiments/baseline_reference/run.py", "repo:src/abvid/learning/reference.py",
        "repo:src/abvid/learning/grouped_search.py")
B1 = "artifact:baseline_strength/b1_20261012_v1"
B1_SHA = {"features.npz": "aab4b3aac1072bbd6ede4213e224802d512b987efc41dd193354d8facbc38d93",
          "results.json": "46399d342855a6ea672be430f9a8bcfafa4baaa962c112487bd4313fbe43a62e"}
H1_LOCK = "archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json"
H1_CONFIG = "archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json"
SOURCES = "artifact:cast/grouped_coverage/cast4a_design_20261007_v1/sources.jsonl"
AUDIO = "archive:dataset/IDMT_Traffic/audio"
BEATS = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
OUT = "artifact:baseline_strength/b1_reference_20261103_v1"
ENV = {"python": "3.11.13", "numpy": "1.26.4", "scikit-learn": "1.4.2", "torch": "2.11.0", "librosa": "0.11.0"}
FEATURE_N, FEATURE_TOL, SEED, DRAWS, WORKERS = 16, 1e-5, 20261103, 10000, 6
KEYS = ("auroc", "balanced_accuracy", "macro_f1", "average_precision")


def environment():
    actual = {"python": platform.python_version(), **{k: version(k) for k in ENV if k != "python"}}
    if actual != ENV:
        raise ValueError(f"Wrong numerical environment: {actual}")


def check_protocol():
    if PROTOCOL_SHA is None or sha256(resolve(PROTOCOL)) != PROTOCOL_SHA:
        raise ValueError("Protocol is not frozen or changed after freeze")


def inputs():
    """Gate 1: pinned B1 inputs, plus the H1 folds they were computed on."""
    root = resolve(B1)
    for name, expected in B1_SHA.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Gate 1 failed, B1 input changed: {name}")
    results = json.loads((root / "results.json").read_text())
    folds = json.loads(resolve(H1_LOCK).read_text())["folds"]
    if results["folds"] != [f["held_out_group"] for f in folds]:
        raise ValueError("B1 fold order differs from the H1 lock")
    return results, folds


_DATA = {}


def _init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    with np.load(resolve(B1) / "features.npz", allow_pickle=False) as f:
        _DATA.update(y=f["y"], group=f["group"], mfcc26=recipe.representation("mfcc26", mfcc=f["full__mfcc26"]),
                     beats_avg=recipe.representation("beats_avg", beats_layers=f["full__beats_layers"]))


def _confusion_recalls(y, p):
    pred = p > 0.5
    return {"truck_recall": float(pred[y == 1].mean()), "car_recall": float((~pred[y == 0]).mean())}


def _fold(task):
    """Inner and outer fits of one representation on one outer fold, for every C."""
    rep, held = task
    x, y, groups = _DATA[rep].astype(np.float64), _DATA["y"], _DATA["group"]
    train, test = np.flatnonzero(groups != held), np.flatnonzero(groups == held)
    inner = recipe.inner_scores(x[train], y[train], groups[train])
    outer = {}
    for c in recipe.CS:
        model = fit(x[train], y[train], c, True)
        p = model.predict_proba(x[test])[:, 1]
        outer[c] = {**metrics(y[test], p), **_confusion_recalls(y[test], p),
                    "train_auroc": metrics(y[train], model.predict_proba(x[train])[:, 1])["auroc"]}
    return {"rep": rep, "held": held, "inner": inner, "outer": outer}


def _all_idmt(rep):
    x, y, groups = _DATA[rep].astype(np.float64), _DATA["y"], _DATA["group"]
    model, info = recipe.fit_reference(x, y, groups)
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return rep, buffer.getvalue(), {"c": info["c"], "inner_auroc_by_c": info["inner_auroc_by_c"],
                                     "train_auroc": metrics(y, model.predict_proba(x)[:, 1])["auroc"]}


def fit_parity(results, folds, pool):
    """Gate 2: packaged fits reproduce B1's saved inner and outer metrics exactly."""
    raw = {tuple(r["config"]): r for r in results["raw"]}
    done = list(pool.map(_fold, [(rep, f["held_out_group"]) for rep in recipe.REPRESENTATIONS for f in folds]))
    order = [f["held_out_group"] for f in folds]
    worst, checked = 0.0, 0
    for row in done:
        i = order.index(row["held"])
        for k, c in enumerate(recipe.CS):
            saved = raw[(row["rep"], "full", "ball", c)]
            pairs = [(row["outer"][c], saved["outer"][i][0])] + list(zip(row["inner"][k], saved["inner"][i]))
            for new, old in pairs:
                worst = max(worst, max(abs(new[key] - old[key]) for key in KEYS))
                checked += 1
    if worst != 0.0:
        raise ValueError(f"Gate 2 failed, fits differ from B1: max {worst}")
    return done, {"fits_compared": checked, "max_abs_difference": worst}


def feature_parity():
    """Gate 3: the packaged route re-extracts B1's cached features for the first clips."""
    import soundfile as sf
    import torch
    from abvid.data.audio import observation
    from abvid.representations.beats import extract_layer_embeddings, load_encoder
    from abvid.representations.mfcc import mfcc_features
    lock = json.loads(resolve(H1_LOCK).read_text())
    cfg_path = resolve(H1_CONFIG)
    if sha256(cfg_path) != lock["artifacts_sha256"]["config.json"]:
        raise ValueError("Frozen H1 config changed")
    cfg = json.loads(cfg_path.read_text())
    rows = {r["file_id"]: r for r in map(json.loads, resolve(SOURCES).open())}
    with np.load(resolve(B1) / "features.npz", allow_pickle=False) as f:
        ids = f["ids"][:FEATURE_N].astype(str).tolist()
        cached = {"beats_layers": f["full__beats_layers"][:FEATURE_N], "mfcc26": f["full__mfcc26"][:FEATURE_N]}
    if ids != sorted(rows)[:FEATURE_N]:
        raise ValueError("Feature cache order differs from the source rows")
    torch.set_num_threads(cfg["hardware"]["torch_threads"])
    torch.set_num_interop_threads(1)
    beats, _ = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve(BEATS))))
    pre = recipe.preprocessing(cfg["preprocessing"])
    waves = []
    for i in ids:
        data = (resolve(AUDIO) / Path(rows[i]["source_path"]).name).read_bytes()
        if hashlib.sha256(data).hexdigest() != rows[i]["source_file_sha256"]:
            raise ValueError(f"Source changed: {i}")
        waves.append(observation(*sf.read(BytesIO(data), dtype="float64", always_2d=True), pre))
    waves = np.stack(waves)
    fresh = {"beats_layers": extract_layer_embeddings(beats, torch.from_numpy(waves)).numpy(),
             "mfcc26": np.stack([mfcc_features(w.astype(np.float64), cfg["mfcc"]) for w in waves])}
    worst = {k: float(np.abs(fresh[k] - cached[k]).max()) for k in cached}
    if any(v > FEATURE_TOL for v in worst.values()):
        raise ValueError(f"Gate 3 failed, features differ: {worst}")
    return worst


def summarize(done, folds, results):
    order = [f["held_out_group"] for f in folds]
    fold_site = results["fold_site"]
    r0 = np.array(results["reference"]["auroc_folds"])
    out = {}
    for rep in recipe.REPRESENTATIONS:
        rows = sorted((r for r in done if r["rep"] == rep), key=lambda r: order.index(r["held"]))
        per_fold = []
        for r in rows:
            c, means = recipe.select_c(r["inner"])
            per_fold.append({"held_out_group": r["held"], "c": c, "inner_auroc_by_c": dict(zip(map(str, recipe.CS), means)),
                             **r["outer"][c]})
        values = {k: np.array([p[k] for p in per_fold]) for k in (*KEYS, "truck_recall", "car_recall", "train_auroc")}
        out[rep] = {"per_fold": per_fold,
                    "mean": {k: fold_bootstrap(v, fold_site, DRAWS, SEED) for k, v in values.items()},
                    "worst_group": {k: {"value": float(values[k].min()), "held_out_group": order[int(values[k].argmin())]}
                                    for k in KEYS},
                    "gain_over_h1_reference_auroc": fold_bootstrap(values["auroc"] - r0, fold_site, DRAWS, SEED)}
    return out


def provenance():
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree": git("status", "--short").splitlines(), "platform": platform.platform(), "environment": ENV,
            "code_sha256": {ref: sha256(resolve(ref)) for ref in CODE}}


def main(stage):
    environment()
    if stage == "run":
        check_protocol()
    results, folds = inputs()
    with ProcessPoolExecutor(WORKERS, initializer=_init) as pool:
        done, parity = fit_parity(results, folds, pool)
        print(json.dumps({"gate1_inputs": "ok", "gate2_fit_parity": parity}), flush=True)
        features = feature_parity()
        print(json.dumps({"gate3_feature_parity_max": features}), flush=True)
        if stage == "check":
            return
        heads = list(pool.map(_all_idmt, recipe.REPRESENTATIONS))
    summary = summarize(done, folds, results)
    out = resolve(OUT, must_exist=False)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for rep, blob, info in heads:
        path = out / "heads" / f"{rep}.joblib"
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(blob)
        manifest[rep] = {"file": f"heads/{rep}.joblib", "sha256": sha256(path), **info}
    result = {"id": recipe.RECIPE_ID, "protocol_sha256": PROTOCOL_SHA, "b1_inputs_sha256": B1_SHA,
              "recipe": {"representations": recipe.REPRESENTATIONS, "cs": recipe.CS,
                         "intermediate_rate_hz": recipe.INTERMEDIATE_RATE_HZ, "class_weight": "balanced"},
              "gates": {"fit_parity": parity, "feature_parity_max": features},
              "folds": [f["held_out_group"] for f in folds], "fold_site": results["fold_site"],
              "h1_reference_auroc_folds": results["reference"]["auroc_folds"],
              "bootstrap": {"draws": DRAWS, "seed": SEED}, "representations": summary,
              "all_idmt_heads": manifest, "provenance": provenance()}
    with (out / "results.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, allow_nan=False)
    for rep in recipe.REPRESENTATIONS:
        m, g = summary[rep]["mean"], summary[rep]["gain_over_h1_reference_auroc"]
        print(f"{rep:9s} AUROC {m['auroc']['estimate']:.4f} {m['auroc']['ci95']} | BA {m['balanced_accuracy']['estimate']:.4f}"
              f" | AP {m['average_precision']['estimate']:.4f} | F1 {m['macro_f1']['estimate']:.4f}"
              f" | gain over H1 {g['estimate']:+.4f} {g['ci95']} site {g['site_ci95']}"
              f" | C {[p['c'] for p in summary[rep]['per_fold']]} | all-IDMT C {manifest[rep]['c']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "run"))
    main(parser.parse_args().stage)
