"""B1 baseline strength within IDMT (baseline environment). Protocol: PROTOCOL.md.

  features  decode once; core8/full routes; BEATs per-layer, CNN14, MFCC26
  fit       96 configurations: outer fits, inner leave-one-group-out, nested selection
Run: .venv-baseline/bin/python experiments/baseline_strength/run.py <stage>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from abvid.data.audio import observation
from abvid.learning.grouped_search import REFERENCE, fit, fold_bootstrap, grid, metrics, nested_select
from abvid.paths import resolve
from abvid.provenance.hashing import sha256
from abvid.representations import cnn14
from abvid.representations.beats import extract_layer_embeddings, load_encoder
from abvid.representations.mfcc import mfcc_features

OUT = "artifact:baseline_strength/b1_20261012_v1"
SOURCES = "artifact:cast/grouped_coverage/cast4a_design_20261007_v1/sources.jsonl"
H1_LOCK = "archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json"
H1_CONFIG = "archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json"
BEATS = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
CNN14 = ("artifact:models/panns/Cnn14_mAP=0.431.pth", "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31")
AUDIO = "archive:dataset/IDMT_Traffic/audio"
SEEDS = (42, 123, 456, 789, 1024)
PROTOCOL = Path(__file__).with_name("PROTOCOL.md")


def provenance():
    repo = resolve("repo:.")
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty_paths": len(git("status", "--porcelain").splitlines()), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(),
            "script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(PROTOCOL)}


def inputs():
    lock = json.loads(resolve(H1_LOCK).read_text())
    cfg_path = resolve(H1_CONFIG)
    if sha256(cfg_path) != lock["artifacts_sha256"]["config.json"]:
        raise ValueError("Frozen H1 config changed")
    rows = {r["file_id"]: r for r in map(json.loads, resolve(SOURCES).open())}
    return lock, json.loads(cfg_path.read_text()), rows


def features():
    started = time.perf_counter()
    lock, cfg, rows = inputs()
    out = resolve(OUT, must_exist=False)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg["hardware"]["torch_threads"])
    torch.set_num_interop_threads(1)
    beats, _ = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve(BEATS))))
    panns = cnn14.load(resolve(CNN14[0]), CNN14[1])
    ids = sorted(rows)
    routes = {"core8": cfg["preprocessing"], "full": dict(cfg["preprocessing"], intermediate_rate_hz=16000)}
    arrays = {f"{r}__{name}": [] for r in routes for name in ("beats_layers", "cnn14", "mfcc26")}
    audio = resolve(AUDIO)
    batch = cfg["hardware"]["batch_size"]
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        raw = []
        for i in chunk:
            data = (audio / Path(rows[i]["source_path"]).name).read_bytes()
            if __import__("hashlib").sha256(data).hexdigest() != rows[i]["source_file_sha256"]:
                raise ValueError(f"Source changed: {i}")
            raw.append(sf.read(BytesIO(data), dtype="float64", always_2d=True))
        for route, pre in routes.items():
            waves = np.stack([observation(x, sr, pre) for x, sr in raw])
            arrays[f"{route}__beats_layers"].append(extract_layer_embeddings(beats, torch.from_numpy(waves)).numpy())
            up = np.stack([resample_poly(w.astype(np.float64), 2, 1, window=("kaiser", 5.0)) for w in waves]).astype(np.float32)
            arrays[f"{route}__cnn14"].append(cnn14.embed(panns, torch.from_numpy(up))[0].numpy())
            arrays[f"{route}__mfcc26"].append(np.stack([mfcc_features(w.astype(np.float64), cfg["mfcc"]) for w in waves]))
        if (start // batch) % 50 == 0:
            print(f"features {start + len(chunk)}/{len(ids)} {time.perf_counter() - started:.0f}s", flush=True)
    saved = {k: np.concatenate(v).astype(np.float32) for k, v in arrays.items()}
    if not all(np.isfinite(v).all() for v in saved.values()):
        raise ValueError("Nonfinite features")
    with (out / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, ids=np.array(ids), y=np.array([rows[i]["class_id"] for i in ids]),
                            group=np.array([rows[i]["provenance_group_id"] for i in ids]), **saved)
    summary = {"stage": "features", "elapsed_s": time.perf_counter() - started,
               "features_sha256": sha256(out / "features.npz"), "provenance": provenance()}
    (out / "features_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("elapsed_s", "features_sha256")}))


_DATA = {}


def _matrix(encoder, route):
    f = _DATA["f"]
    layers = f[f"{route}__beats_layers"]
    return {"beats_l12": lambda: layers[:, 11], "beats_l6": lambda: layers[:, 5],
            "beats_avg": lambda: layers.mean(axis=1), "cnn14": lambda: f[f"{route}__cnn14"],
            "mfcc26": lambda: f[f"{route}__mfcc26"],
            "beats_l12+mfcc26": lambda: np.hstack([layers[:, 11], f[f"{route}__mfcc26"]])}[encoder]()


def _init(path):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    data = np.load(path, allow_pickle=False)
    _DATA["f"] = {k: data[k] for k in data.files}
    _DATA["index"] = {fid: k for k, fid in enumerate(_DATA["f"]["ids"])}


def _rows(ids):
    return np.array([_DATA["index"][i] for i in ids])


def _score(x, y, train, test, c, balanced):
    try:
        model = fit(x[train], y[train], c, balanced)
    except Exception as error:  # convergence or numerical failure is recorded, never replaced
        return {"failure": f"{type(error).__name__}: {error}"}
    return metrics(y[test], model.predict_proba(x[test])[:, 1])


def run_config(task):
    (encoder, route, budget, c), folds = task
    f = _DATA["f"]
    x, y, groups = _matrix(encoder, route).astype(np.float64), f["y"], f["group"]
    balanced = budget == "ball"
    outer, inner = [], []
    for fold in folds:
        held = fold["held_out_group"]
        test = np.flatnonzero(groups == held)
        selections = ([_rows(fold["train_ids"][str(s)]) for s in SEEDS] if budget == "b190"
                      else [np.flatnonzero(groups != held)])
        outer.append([_score(x, y, train, test, c, balanced) for train in selections])
        base = _rows(fold["train_ids"]["42"]) if budget == "b190" else np.flatnonzero(groups != held)
        scores = []
        for inner_group in sorted(set(groups[groups != held])):
            train = base[groups[base] != inner_group]
            scores.append(_score(x, y, train, np.flatnonzero(groups == inner_group), c, balanced))
        inner.append(scores)
    return {"config": [encoder, route, budget, c], "outer": outer, "inner": inner}


def fit_all():
    started = time.perf_counter()
    lock, _, rows = inputs()
    out = resolve(OUT)
    path = out / "features.npz"
    summary = json.loads((out / "features_summary.json").read_text())
    if sha256(path) != summary["features_sha256"]:
        raise ValueError("Feature cache changed")
    folds = lock["folds"]
    configs = grid()
    with ProcessPoolExecutor(max_workers=6, initializer=_init, initargs=(str(path),)) as pool:
        results = list(pool.map(run_config, [(cfg, folds) for cfg in configs]))
    group_site = {r["provenance_group_id"]: r["site_id"] for r in rows.values()}
    fold_site = [group_site[f["held_out_group"]] for f in folds]

    def fold_scores(result, key="auroc"):
        vals = []
        for fits in result["outer"]:
            ok = [m[key] for m in fits if "failure" not in m]
            vals.append(np.mean(ok) if len(ok) == len(fits) else np.nan)
        return np.array(vals)

    table = {tuple(r["config"]): r for r in results}
    failures = sum("failure" in m for r in results for fits in r["outer"] + r["inner"] for m in fits)
    inner_auroc = {k: [np.mean([m.get("auroc", np.nan) for m in fits]) for fits in r["inner"]] for k, r in table.items()}
    selected, procedure = [], []
    for i in range(len(folds)):
        scores = [inner_auroc[k][i] if np.isfinite(inner_auroc[k][i]) else -np.inf for k in configs]
        choice = nested_select(scores, configs)
        selected.append(list(choice))
        procedure.append(fold_scores(table[tuple(choice)])[i])
    reference = fold_scores(table[REFERENCE])
    rows_out = []
    for k in configs:
        entry = {"config": list(k)}
        for key in ("auroc", "balanced_accuracy", "macro_f1", "average_precision"):
            s = fold_scores(table[k], key)
            entry[key] = None if np.isnan(s).any() else float(s.mean())
            entry[key + "_folds"] = [None if np.isnan(v) else float(v) for v in s]
        entry["inner_auroc"] = [float(v) for v in inner_auroc[k]]
        rows_out.append(entry)
    procedure = np.array(procedure)
    result = {"protocol": "baseline_strength_v1", "configs": rows_out, "selected_per_fold": selected,
              "folds": [f["held_out_group"] for f in folds], "fold_site": fold_site,
              "reference": {"config": list(REFERENCE), "auroc_folds": reference.tolist(),
                            "auroc": float(reference.mean())},
              "procedure": {"auroc_folds": procedure.tolist(), "auroc": float(procedure.mean())},
              "primary_delta_auroc": fold_bootstrap(procedure - reference, fold_site),
              "fit_failures": int(failures), "elapsed_s": time.perf_counter() - started,
              "features_sha256": summary["features_sha256"], "provenance": provenance(),
              "raw": results}
    with (out / "results.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=True)
    print(json.dumps({k: result[k] for k in ("reference", "procedure", "primary_delta_auroc", "selected_per_fold",
                                             "fit_failures", "elapsed_s")}, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features", "fit"))
    {"features": features, "fit": fit_all}[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
