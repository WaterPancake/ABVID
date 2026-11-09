"""CAST reconstruction on held recordings: fit, features, score.

Contract: experiments/cast_utility/held_reconstruction/PROTOCOL.md (frozen
2026-11-01; its SHA256 is checked at every stage). Reads the 4,413 admitted IDMT
car/truck clips under that protocol only. Conditional reconstructions are
scored, never trained on, banked or used for coverage. Create-only chunk bundles
under artifact:cast/diagnostics/; an interrupted stage resumes at the next chunk.

    .venv/bin/python -B experiments/cast_utility/held_reconstruction/run.py fit
    .venv-baseline/bin/python -B experiments/cast_utility/held_reconstruction/run.py features
    .venv-baseline/bin/python -B experiments/cast_utility/held_reconstruction/run.py score
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from io import BytesIO
import json

import numpy as np

from abvid.cast.grouped_storage import read_bundle, verify_bundle, write_bundle
from abvid.cast.stage_diagnostic import (BASE_ENV, CAST_ENV, HEAD_MANIFEST_SHA, REPS, design, destination, encoded,
                                         environment, feature_path, load_arrays, npy, npz, provenance, wave_hash)
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

PROTOCOL = "repo:experiments/cast_utility/held_reconstruction/PROTOCOL.md"
PROTOCOL_SHA = "8d50023893789db488737250fff923711a7f215e08cedd0e12e76cabf051d0a0"
CODE = ("repo:experiments/cast_utility/held_reconstruction/run.py", "repo:src/abvid/cast/time_varying.py",
        "repo:src/abvid/cast/smooth_fit.py")
STAGE_RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
INPUT = "artifact:cast/utility/cast5a_run_20261011_v1"
RUN = "artifact:cast/diagnostics/cast_held_reconstruction_20261101_v1"
GATE_ROLE, TV_FOLD, CHUNK, WORKERS = "utility_s42_0", 0, 100, 8
REPLAY, PARITY, SEED, DRAWS = 1e-6, 1e-5, 20261101, 10000
KEEPS, LOSES = -0.03, -0.10
METRICS = ("macro_f1", "balanced_accuracy", "auroc", "average_precision", "positive_recall", "negative_recall")


def check_protocol():
    if sha256(resolve(PROTOCOL)) != PROTOCOL_SHA:
        raise ValueError("Protocol changed after freeze")


def code_hashes():
    return {ref: sha256(resolve(ref)) for ref in CODE}


def cache():
    with np.load(feature_path("real"), allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def chunks(ids):
    return [ids[i:i+CHUNK] for i in range(0, len(ids), CHUNK)]


def own_design():
    _, payload = read_bundle(destination(RUN) / "design")
    return json.loads(payload["design.json"])


def _fit_one(job):
    import torch
    from abvid.cast import time_varying as tv
    from abvid.cast.population.engine import SmoothRenderer
    from abvid.cast.renderer import tensors
    from abvid.cast.smooth_fit import fit as smooth_fit
    from abvid.data.augmentation import shape
    fid, target, arm, cfg = job
    fit, renderer = (smooth_fit, SmoothRenderer) if arm == "smooth8" else (tv.fit, tv.TimeVaryingRenderer)
    result = fit(torch.from_numpy(target), cfg, fid)
    with torch.no_grad():
        raw = renderer(cfg, fid).render(tensors(result["parameters"]), "check")[0].numpy()
    waves = np.stack([shape(raw[r]).astype(np.float32) for r in (0, 1)])
    record = {key: result[key] for key in ("input_id", "parameters", "winner", "fit_loss", "check_loss", "fit_seconds")}
    record["starts"] = [{k: s[k] for k in ("parameters", "best_step", "fit_loss", "check_loss")} for s in result["starts"]]
    if arm == "tv3":
        record["parity"] = tv.parity(result["initial_static_parameters"], cfg, fid)
    return record, waves


def fit_stage():
    from abvid.cast.audio import normalize, observation
    from abvid.cast.grouped_parents import _configure_cpu, read_source
    from abvid.cast.utility_design import _cast4a_context, load_context
    environment(CAST_ENV)
    check_protocol()
    _configure_cpu()
    d, _, identity = design(STAGE_RUN)
    roles = load_context()["roles"]
    real = cache()
    ids, expected = real["ids"].astype(str).tolist(), dict(zip(real["ids"].astype(str), real["wave_sha256"].astype(str)))
    if len(ids) != 4413 or int(real["y"].sum()) != 511:
        raise ValueError("Unexpected admitted IDMT inventory")
    gate_ids = list(roles[GATE_ROLE].train_ids)
    if gate_ids != d["train_ids"]:
        raise ValueError("Gate role differs from the stage diagnosis parents")
    smooth8_cfg = d["renderer_configs"]["smooth8_v1"]
    tv_cfg = copy.deepcopy(smooth8_cfg)
    tv_cfg["renderer"]["weight_knots"] = 3
    out = destination(RUN)
    if not (out / "design").exists():
        write_bundle(out / "design", {"design.json": encoded({
            "id": "cast_held_reconstruction_v1", "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
            "stage_run": STAGE_RUN, "stage_design_identity": identity, "input_run": INPUT,
            "smooth8_ids": gate_ids + [i for i in ids if i not in set(gate_ids)],
            "tv3_ids": list(roles[GATE_ROLE].held_ids), "configs": {"smooth8": smooth8_cfg, "tv3": tv_cfg},
            "fit_code_sha256": code_hashes(), "cast_environment": CAST_ENV, "provenance": provenance()})},
            {"kind": "cast_held_reconstruction_design"})
    own = own_design()
    if own["protocol_sha256"] != PROTOCOL_SHA or own["configs"] != {"smooth8": smooth8_cfg, "tv3": tv_cfg}:
        raise ValueError("Design differs from the frozen run")
    reference = {}
    for r in (0, 1):
        _, payload = read_bundle(destination(STAGE_RUN) / "render" / f"smooth8_check_{r}")
        reference[r] = dict(zip(d["train_ids"], np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)))
    c4 = _cast4a_context()
    base = smooth8_cfg

    def original(fid):
        raw, sr = read_source(c4, fid)
        target = normalize(observation(raw, sr, base), base)[0].numpy()
        if wave_hash(target) != expected[fid]:
            raise ValueError(f"Gate 1 failed, original hash drift: {fid}")
        return target

    replay, parity = 0.0, 0.0
    with ProcessPoolExecutor(WORKERS, initializer=_configure_cpu) as pool:
        for arm, arm_ids, cfg in (("smooth8", own["smooth8_ids"], smooth8_cfg), ("tv3", own["tv3_ids"], tv_cfg)):
            for k, chunk in enumerate(chunks(arm_ids)):
                folder = out / "fits" / arm / f"chunk_{k:03d}"
                if folder.exists():
                    verify_bundle(folder)
                    continue
                results = list(pool.map(_fit_one, [(fid, original(fid), arm, cfg) for fid in chunk]))
                waves = np.stack([w for _, w in results])
                for fid, (record, w) in zip(chunk, results):
                    if arm == "smooth8" and fid in reference[0]:
                        replay = max(replay, *(float(np.abs(w[r] - reference[r][fid]).max()) for r in (0, 1)))
                        if replay > REPLAY:
                            raise ValueError(f"Gate 3 failed, smooth8 replay differs: {fid} {replay}")
                    if arm == "tv3":
                        parity = max(parity, record["parity"])
                        if parity > PARITY:
                            raise ValueError(f"Gate 4 failed, tv3 parity: {fid} {parity}")
                write_bundle(folder, {"waves.npy": npy(waves), "ids.json": encoded(chunk),
                                      "fits.jsonl": "".join(json.dumps(r, allow_nan=False) + "\n" for r, _ in results).encode(),
                                      "records.json": encoded([{"file_id": f, "wave_sha256": [wave_hash(x) for x in w]}
                                                               for f, (_, w) in zip(chunk, results)])},
                             {"kind": "cast_held_reconstruction_fits", "arm": arm, "chunk": k,
                              "gates": {"smooth8_replay_max": replay, "tv3_parity_max": parity},
                              "fit_code_sha256": code_hashes()})
                print(json.dumps({"stage": "fits", "arm": arm, "chunk": k, "of": len(chunks(arm_ids)),
                                  "replay_max": replay, "parity_max": parity}), flush=True)
    print(json.dumps({"stage": "fit_done"}), flush=True)


def features_stage():
    import torch
    from abvid.representations.beats import extract_embeddings, load_encoder
    from abvid.representations.mfcc import mfcc_features
    environment(BASE_ENV)
    check_protocol()
    d, _, identity = design(STAGE_RUN)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, _ = load_encoder(dict(d["h1_config"]["encoder"], checkpoint=str(resolve(d["checkpoint"]))))

    def extract(waves):
        beats = np.concatenate([extract_embeddings(encoder, torch.from_numpy(waves[i:i+8])).numpy() for i in range(0, len(waves), 8)])
        return {"BEATs768": beats.astype(np.float32),
                "MFCC26": np.stack([mfcc_features(w.astype(np.float64), d["h1_config"]["mfcc"]) for w in waves])}

    _, payload = read_bundle(destination(STAGE_RUN) / "render" / "smooth8_check_0")
    control = extract(np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)[:16])
    reference = load_arrays(STAGE_RUN, "smooth8_check_0", identity)
    for rep in REPS:
        if not np.allclose(control[rep], reference[rep][:16], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Feature extraction control failed: {rep}")
    out = destination(RUN)
    for arm in ("smooth8", "tv3"):
        for folder in sorted((out / "fits" / arm).iterdir()):
            target = out / "features" / arm / folder.name
            if target.exists():
                verify_bundle(target)
                continue
            _, payload = read_bundle(folder)
            ids, waves = json.loads(payload["ids.json"]), np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
            per = [extract(np.ascontiguousarray(waves[:, r])) for r in (0, 1)]
            arrays = {rep: np.stack([p[rep] for p in per], 1) for rep in REPS}
            write_bundle(target, {"features.npz": npz(ids=np.array(ids), **arrays)},
                         {"kind": "cast_held_reconstruction_features", "arm": arm, "environment": BASE_ENV})
            print(json.dumps({"stage": "features", "arm": arm, "chunk": folder.name}), flush=True)


def heads(roles):
    import joblib
    path = resolve(INPUT) / "learning" / "heads" / "manifest.json"
    if sha256(path) != HEAD_MANIFEST_SHA:
        raise ValueError("Heads manifest changed")
    entries = {e["head"]: e for e in json.loads(path.read_text())}
    models = {}
    for rid in roles:
        for rep in REPS:
            for arm in ("R", "C"):
                name = f"{rep}__{arm}__{rid}"
                file = resolve(INPUT) / "learning" / "heads" / "models" / f"{name}.joblib"
                if sha256(file) != entries[name]["sha256"]:
                    raise ValueError(f"Head file changed: {name}")
                models[name] = joblib.load(file)
    return models


def recon_features():
    out, features = destination(RUN), {}
    for arm in ("smooth8", "tv3"):
        features[arm] = {}
        for folder in sorted((out / "features" / arm).iterdir()):
            _, payload = read_bundle(folder)
            with np.load(BytesIO(payload["features.npz"]), allow_pickle=False) as value:
                for i, fid in enumerate(value["ids"].astype(str)):
                    features[arm][fid] = {rep: value[rep][i] for rep in REPS}
    return features


def score_stage():
    from abvid.cast.utility_design import load_context
    from abvid.evaluation.ranking import selection_draws
    from abvid.learning.utility import head_metrics
    from abvid.learning.utility_stats import SEEDS, grids, summary, weighted
    environment(BASE_ENV)
    check_protocol()
    own = own_design()
    roles = load_context()["roles"]
    real = cache()
    index = {fid: k for k, fid in enumerate(real["ids"].astype(str))}
    recon = recon_features()
    if len(recon["smooth8"]) != 4413 or set(recon["tv3"]) != set(own["tv3_ids"]):
        raise ValueError("Reconstruction features are incomplete")
    models = heads(roles)
    scores = json.loads((resolve(INPUT) / "learning" / "scores" / "results.json").read_text())
    published = {h["head"]: h for h in scores["per_head"]}
    fold_site = {int(k): v for k, v in scores["statistics"]["design"]["fold_site"].items()}
    parts, per_head, flips, tv_rows = [], [], [], []
    for rid, role in sorted(roles.items()):
        fold = int(rid.rsplit("_", 1)[1])
        parts.append({"id": rid, "analysis": role.analysis, "fold": fold})
        for rep in REPS:
            for head in ("R", "C"):
                name = f"{rep}__{head}__{rid}"
                model = models[name]
                for side, ids in (("held", list(role.held_ids)), ("train", list(role.train_ids))):
                    y = real["y"][[index[i] for i in ids]]
                    p0 = model.predict_proba(real[rep][[index[i] for i in ids]].astype(np.float64))[:, 1]
                    m0 = head_metrics(y, p0)
                    if side == "held" and any(m0[k] != published[name][k] for k in m0):
                        raise ValueError(f"Gate 2 failed, held original metrics differ: {name}")
                    arms = {"original": [m0]}
                    for arm in ("smooth8", "tv3"):
                        if not all(i in recon[arm] for i in ids):
                            continue
                        x = np.stack([recon[arm][i][rep] for i in ids]).astype(np.float64)
                        ps = [model.predict_proba(x[:, r])[:, 1] for r in (0, 1)]
                        arms[arm] = [head_metrics(y, p) for p in ps]
                        if arm == "smooth8" and side == "held" and head == "R":
                            right0 = (p0 > .5) == (y == 1)
                            flips.append({"head": name, **{cls: int((right0 & ((ps[0] > .5) != (y == 1)) & (y == c)).sum())
                                                           for c, cls in ((0, "car"), (1, "truck"))}})
                    for arm, ms in arms.items():
                        per_head.append({"partition": rid, "representation": rep, "arm": f"{head}_{side}_{arm}",
                                         **{k: float(np.mean([m[k] for m in ms])) for k in METRICS}})
                    if fold == TV_FOLD and side == "held" and "tv3" in arms:
                        tv_rows.append({"head": name, **{arm: {k: float(np.mean([m[k] for m in ms])) for k in METRICS}
                                                         for arm, ms in arms.items()}})
    grid, folds = grids(per_head, parts)
    sites = sorted(set(fold_site.values()))
    rng = np.random.Generator(np.random.PCG64(SEED))
    sw = np.vstack([np.ones(len(SEEDS), int), selection_draws(rng, DRAWS, len(SEEDS))])
    fw = np.vstack([np.ones(len(folds), int), selection_draws(rng, DRAWS, len(folds))])
    site = np.vstack([np.ones(len(sites), int), selection_draws(rng, DRAWS, len(sites))])
    site_fw = site[:, [sites.index(fold_site[f]) for f in folds]]

    def arm_mean(rep, arm, key, weights=fw):
        return weighted(grid[(rep, arm, key)], sw, weights)

    contrasts, sensitivity, means = {}, {}, {}
    for rep in REPS:
        for head in ("R", "C"):
            for key in METRICS:
                for side in ("held", "train"):
                    for arm in ("original", "smooth8"):
                        means.setdefault(rep, {})[f"{head}_{side}_{arm}:{key}"] = summary(arm_mean(rep, f"{head}_{side}_{arm}", key))
                held = arm_mean(rep, f"{head}_held_smooth8", key) - arm_mean(rep, f"{head}_held_original", key)
                train = arm_mean(rep, f"{head}_train_smooth8", key) - arm_mean(rep, f"{head}_train_original", key)
                contrasts.setdefault(rep, {})[f"{head}:{key}"] = {"held_drop": summary(held), "train_drop": summary(train),
                                                                  "memorization_gap": summary(train - held)}
                site_held = (arm_mean(rep, f"{head}_held_smooth8", key, site_fw) -
                             arm_mean(rep, f"{head}_held_original", key, site_fw))
                sensitivity.setdefault(rep, {})[f"{head}:{key}"] = summary(site_held)
    primary = contrasts["BEATs768"]["R:auroc"]["held_drop"]
    lo, hi = primary["ci95"]
    decision = "keeps_generalizable_cue" if lo > KEEPS else "loses_generalizable_cue" if hi < LOSES else "partial"

    def retained(rep, side):
        o, s = (means[rep][f"R_{side}_{a}:auroc"]["estimate"] for a in ("original", "smooth8"))
        return (s - .5)/(o - .5)

    per_fold = {}
    for fold in folds:
        rows = [h for h in per_head if int(h["partition"].rsplit("_", 1)[1]) == fold]
        held = list(roles[next(r for r in roles if r.endswith(f"_{fold}"))].held_ids)
        per_fold[fold] = {"held_n": len(held), "held_trucks": int(real["y"][[index[i] for i in held]].sum()),
                          **{f"{rep}:{arm}": float(np.mean([h["auroc"] for h in rows
                                                            if h["representation"] == rep and h["arm"] == f"R_held_{arm}"]))
                             for rep in REPS for arm in ("original", "smooth8")}}
    result = {"id": "cast_held_reconstruction_v1", "protocol_sha256": PROTOCOL_SHA, "target_access": False,
              "new_model_fits": 0, "decision": decision, "primary_beats_r_held_auroc_drop": primary,
              "primary_site_sensitivity": sensitivity["BEATs768"]["R:auroc"],
              "retained_above_chance": {rep: {side: retained(rep, side) for side in ("held", "train")} for rep in REPS},
              "contrasts": contrasts, "site_sensitivity": sensitivity, "arm_means": means, "per_fold": per_fold,
              "held_flips_smooth8_r0": flips, "tv3_fold0_held": tv_rows, "per_head": per_head,
              "gates": {"held_original_metrics_reproduced": True, "smooth8_replay_and_tv3_parity": "checked during fit"},
              "bootstrap": {"draws": DRAWS, "seed": SEED, "folds": folds, "fold_site": {str(k): v for k, v in fold_site.items()}},
              "score_code_sha256": code_hashes(), "provenance": provenance()}
    write_bundle(destination(RUN) / "scores", {"results.json": encoded(result)}, {"kind": "cast_held_reconstruction_scores"})
    print(json.dumps({"decision": decision, "primary": primary, "site": result["primary_site_sensitivity"],
                      "retained_above_chance": result["retained_above_chance"]}, indent=1))
    for rep in REPS:
        for key in ("auroc", "balanced_accuracy", "positive_recall", "negative_recall"):
            c = contrasts[rep][f"R:{key}"]
            print(f"{rep:9s} R {key:18s} held {means[rep][f'R_held_original:{key}']['estimate']:.4f} -> "
                  f"{means[rep][f'R_held_smooth8:{key}']['estimate']:.4f} (drop {c['held_drop']['estimate']:+.4f} "
                  f"[{c['held_drop']['ci95'][0]:+.4f}, {c['held_drop']['ci95'][1]:+.4f}]) | train drop "
                  f"{c['train_drop']['estimate']:+.4f} | gap {c['memorization_gap']['estimate']:+.4f}")
    print(json.dumps({"per_fold": per_fold}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "features", "score"))
    {"fit": fit_stage, "features": features_stage, "score": score_stage}[parser.parse_args().stage]()
