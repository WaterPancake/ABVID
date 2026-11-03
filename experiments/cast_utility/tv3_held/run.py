"""Time-varying renderer (tv3) on held recordings: check, fit, features, score.

Contract: experiments/cast_utility/tv3_held/PROTOCOL.md. Its SHA256 is checked
by every writing stage. Reuses the smooth8 fits/features and the fold-0 tv3 fits
of cast_held_reconstruction_v1, fits tv3 for the other 3,843 IDMT clips and
scores them with the saved CAST-5A heads. Reconstructions are scored, never
trained on, banked or used for coverage. Create-only chunk bundles under
artifact:cast/diagnostics/; an interrupted stage resumes at the next chunk.

    .venv/bin/python -B experiments/cast_utility/tv3_held/run.py check-fit      # gate 3, no writes
    .venv-baseline/bin/python -B experiments/cast_utility/tv3_held/run.py check-score  # gates 2 and 5, no writes
    .venv/bin/python -B experiments/cast_utility/tv3_held/run.py fit
    .venv-baseline/bin/python -B experiments/cast_utility/tv3_held/run.py features
    .venv-baseline/bin/python -B experiments/cast_utility/tv3_held/run.py score
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json

import numpy as np

from abvid.cast.grouped_storage import read_bundle, verify_bundle, write_bundle
from abvid.cast.stage_diagnostic import (BASE_ENV, CAST_ENV, HEAD_MANIFEST_SHA, REPS, design, destination, encoded,
                                         environment, feature_path, npy, npz, provenance, wave_hash)
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

PROTOCOL = "repo:experiments/cast_utility/tv3_held/PROTOCOL.md"
PROTOCOL_SHA = "9f0e30656096448726e7c7731851766caed0b14b675b066ac6f71d687c229946"
CODE = ("repo:experiments/cast_utility/tv3_held/run.py", "repo:src/abvid/cast/time_varying.py",
        "repo:src/abvid/cast/smooth_fit.py")
FIT_CODE = ("repo:src/abvid/cast/time_varying.py", "repo:src/abvid/cast/smooth_fit.py")
STAGE_RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
CAST5A = "artifact:cast/utility/cast5a_run_20261011_v1"
SOURCE = "artifact:cast/diagnostics/cast_held_reconstruction_20261101_v1"
SOURCE_IDENTITY = {"design": "1e0131673009c21b88027c915f58e3e8eb4ac12ffc2f682fffec21528b983325",
                   "scores": "b7949da3dc322d5bc8c3db91358ae0019565bfd555aef3ca92db2b1fa3fa49e0"}
SOURCE_PROTOCOL_SHA = "8d50023893789db488737250fff923711a7f215e08cedd0e12e76cabf051d0a0"
RUN = "artifact:cast/diagnostics/cast_tv3_held_20261102_v1"
SEEN_FOLD, DECISION_FOLDS = 0, (1, 2, 3, 4, 5)
CHUNK, WORKERS, REPLAY_N, FEATURE_N = 100, 8, 8, 16
REPLAY, PARITY, FEATURE_TOL = 1e-6, 1e-5, 1e-5
SEED, DRAWS, HELPS = 20261102, 10000, 0.03
METRICS = ("macro_f1", "balanced_accuracy", "auroc", "average_precision", "positive_recall", "negative_recall")
ARMS = ("original", "smooth8", "tv3")


def check_protocol():
    if PROTOCOL_SHA is None or sha256(resolve(PROTOCOL)) != PROTOCOL_SHA:
        raise ValueError("Protocol is not frozen or changed after freeze")


def code_hashes(refs=CODE):
    return {ref: sha256(resolve(ref)) for ref in refs}


def chunks(ids):
    return [ids[i:i+CHUNK] for i in range(0, len(ids), CHUNK)]


def source_design():
    """The reused run's design, after checking its pinned identity and fit code."""
    root = resolve(SOURCE)
    for name, identity in SOURCE_IDENTITY.items():
        if verify_bundle(root / name)["identity_sha256"] != identity:
            raise ValueError(f"Reused bundle changed: {name}")
    _, payload = read_bundle(root / "design")
    d = json.loads(payload["design.json"])
    if d["id"] != "cast_held_reconstruction_v1" or d["protocol_sha256"] != SOURCE_PROTOCOL_SHA:
        raise ValueError("Unexpected reused design")
    recorded = {ref: d["fit_code_sha256"][ref] for ref in FIT_CODE}
    if code_hashes(FIT_CODE) != recorded:
        raise ValueError("tv3 fitting code differs from the reused run")
    return d


def held_groups(roles, ids):
    """Fold -> held ids; the six held groups must partition the inventory."""
    groups = {}
    for rid, role in roles.items():
        fold = int(rid.rsplit("_", 1)[1])
        if groups.setdefault(fold, list(role.held_ids)) != list(role.held_ids):
            raise ValueError(f"Roles of fold {fold} disagree on the held group")
    flat = [i for fold in sorted(groups) for i in groups[fold]]
    if sorted(flat) != sorted(ids) or len(flat) != len(set(flat)):
        raise ValueError("Held groups do not partition the clips")
    return groups


def cache():
    with np.load(feature_path("real"), allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def reused_tv3(arrays=True):
    """fid -> (two check waves) from the reused fold-0 tv3 fit bundles."""
    out = {}
    for folder in sorted((resolve(SOURCE) / "fits" / "tv3").iterdir()):
        _, payload = read_bundle(folder)
        ids = json.loads(payload["ids.json"])
        waves = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False) if arrays else [None] * len(ids)
        out.update(zip(ids, waves))
    return out


def _fit_one(job):
    import torch
    from abvid.cast import time_varying as tv
    from abvid.cast.renderer import tensors
    from abvid.data.augmentation import shape
    fid, target, cfg = job
    result = tv.fit(torch.from_numpy(target), cfg, fid)
    with torch.no_grad():
        raw = tv.TimeVaryingRenderer(cfg, fid).render(tensors(result["parameters"]), "check")[0].numpy()
    waves = np.stack([shape(raw[r]).astype(np.float32) for r in (0, 1)])
    record = {key: result[key] for key in ("input_id", "parameters", "winner", "fit_loss", "check_loss", "fit_seconds")}
    record["starts"] = [{k: s[k] for k in ("parameters", "best_step", "fit_loss", "check_loss")} for s in result["starts"]]
    record["parity"] = tv.parity(result["initial_static_parameters"], cfg, fid)
    return record, waves


def fit_stage(preflight=False):
    from abvid.cast.audio import normalize, observation
    from abvid.cast.grouped_parents import _configure_cpu, read_source
    from abvid.cast.utility_design import _cast4a_context, load_context
    environment(CAST_ENV)
    if not preflight:
        check_protocol()
    _configure_cpu()
    src = source_design()
    cfg = src["configs"]["tv3"]
    real = cache()
    ids = real["ids"].astype(str).tolist()
    expected = dict(zip(ids, real["wave_sha256"].astype(str)))
    if len(ids) != 4413 or int(real["y"].sum()) != 511:
        raise ValueError("Unexpected admitted IDMT inventory")
    groups = held_groups(load_context()["roles"], ids)
    if groups[SEEN_FOLD] != src["tv3_ids"]:
        raise ValueError("Reused tv3 fits are not the fold-0 held group")
    seen = set(src["tv3_ids"])
    new_ids = [i for i in src["smooth8_ids"] if i not in seen]
    if len(new_ids) != 3843:
        raise ValueError("Unexpected number of clips to fit")
    c4 = _cast4a_context()

    def original(fid):
        raw, sr = read_source(c4, fid)
        target = normalize(observation(raw, sr, src["configs"]["smooth8"]), src["configs"]["smooth8"])[0].numpy()
        if wave_hash(target) != expected[fid]:
            raise ValueError(f"Gate 1 failed, original hash drift: {fid}")
        return target

    reused = reused_tv3()
    replay_ids = src["tv3_ids"][:REPLAY_N]
    with ProcessPoolExecutor(WORKERS, initializer=_configure_cpu) as pool:
        results = list(pool.map(_fit_one, [(fid, original(fid), cfg) for fid in replay_ids]))
        replay = max(float(np.abs(w - reused[fid]).max()) for fid, (_, w) in zip(replay_ids, results))
        if replay > REPLAY:
            raise ValueError(f"Gate 3 failed, tv3 replay differs: {replay}")
        seconds = float(np.median([r["fit_seconds"] for r, _ in results]))
        print(json.dumps({"gate3_tv3_replay_max": replay, "median_fit_seconds": seconds,
                          "fits_to_run": len(new_ids)}), flush=True)
        if preflight:
            return
        out = destination(RUN)
        if not (out / "design").exists():
            write_bundle(out / "design", {"design.json": encoded({
                "id": "cast_tv3_held_v1", "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
                "source_run": SOURCE, "source_identity": SOURCE_IDENTITY, "stage_run": STAGE_RUN,
                "new_tv3_ids": new_ids, "config": cfg, "fit_code_sha256": code_hashes(),
                "cast_environment": CAST_ENV, "provenance": provenance()})},
                {"kind": "cast_tv3_held_design"})
        own = own_design()
        if own["protocol_sha256"] != PROTOCOL_SHA or own["config"] != cfg or own["new_tv3_ids"] != new_ids:
            raise ValueError("Design differs from the frozen run")
        parity = 0.0
        for k, chunk in enumerate(chunks(new_ids)):
            folder = out / "fits" / "tv3" / f"chunk_{k:03d}"
            if folder.exists():
                verify_bundle(folder)
                continue
            results = list(pool.map(_fit_one, [(fid, original(fid), cfg) for fid in chunk]))
            for fid, (record, _) in zip(chunk, results):
                parity = max(parity, record["parity"])
                if parity > PARITY:
                    raise ValueError(f"Gate 4 failed, tv3 parity: {fid} {parity}")
            write_bundle(folder, {"waves.npy": npy(np.stack([w for _, w in results])), "ids.json": encoded(chunk),
                                  "fits.jsonl": "".join(json.dumps(r, allow_nan=False) + "\n" for r, _ in results).encode(),
                                  "records.json": encoded([{"file_id": f, "wave_sha256": [wave_hash(x) for x in w]}
                                                           for f, (_, w) in zip(chunk, results)])},
                         {"kind": "cast_tv3_held_fits", "chunk": k, "fit_code_sha256": code_hashes(),
                          "gates": {"tv3_replay_max": replay, "tv3_parity_max": parity}})
            print(json.dumps({"stage": "fits", "chunk": k, "of": len(chunks(new_ids)), "parity_max": parity}), flush=True)
    print(json.dumps({"stage": "fit_done"}), flush=True)


def own_design():
    _, payload = read_bundle(destination(RUN) / "design")
    return json.loads(payload["design.json"])


def extractor():
    import torch
    from abvid.representations.beats import extract_embeddings, load_encoder
    from abvid.representations.mfcc import mfcc_features
    d, _, _ = design(STAGE_RUN)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, _ = load_encoder(dict(d["h1_config"]["encoder"], checkpoint=str(resolve(d["checkpoint"]))))

    def extract(waves):
        beats = np.concatenate([extract_embeddings(encoder, torch.from_numpy(waves[i:i+8])).numpy()
                                for i in range(0, len(waves), 8)])
        return {"BEATs768": beats.astype(np.float32),
                "MFCC26": np.stack([mfcc_features(w.astype(np.float64), d["h1_config"]["mfcc"]) for w in waves])}
    return extract


def feature_gate(extract):
    """Gate 5: re-extracted features of reused fold-0 tv3 renders match the stored ones."""
    folder = sorted((resolve(SOURCE) / "fits" / "tv3").iterdir())[0]
    _, payload = read_bundle(folder)
    waves = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)[:FEATURE_N, 0]
    _, payload = read_bundle(resolve(SOURCE) / "features" / "tv3" / folder.name)
    with np.load(BytesIO(payload["features.npz"]), allow_pickle=False) as stored:
        control = extract(np.ascontiguousarray(waves))
        worst = {rep: float(np.abs(control[rep] - stored[rep][:FEATURE_N, 0]).max()) for rep in REPS}
    if any(v > FEATURE_TOL for v in worst.values()):
        raise ValueError(f"Gate 5 failed, feature extraction differs: {worst}")
    return worst


def features_stage():
    environment(BASE_ENV)
    check_protocol()
    extract = extractor()
    print(json.dumps({"gate5_feature_max": feature_gate(extract)}), flush=True)
    out = destination(RUN)
    for folder in sorted((out / "fits" / "tv3").iterdir()):
        target = out / "features" / "tv3" / folder.name
        if target.exists():
            verify_bundle(target)
            continue
        _, payload = read_bundle(folder)
        ids, waves = json.loads(payload["ids.json"]), np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
        per = [extract(np.ascontiguousarray(waves[:, r])) for r in (0, 1)]
        write_bundle(target, {"features.npz": npz(ids=np.array(ids), **{rep: np.stack([p[rep] for p in per], 1)
                                                                       for rep in REPS})},
                     {"kind": "cast_tv3_held_features", "environment": BASE_ENV})
        print(json.dumps({"stage": "features", "chunk": folder.name}), flush=True)


def heads(roles):
    import joblib
    path = resolve(CAST5A) / "learning" / "heads" / "manifest.json"
    if sha256(path) != HEAD_MANIFEST_SHA:
        raise ValueError("Heads manifest changed")
    entries = {e["head"]: e for e in json.loads(path.read_text())}
    models = {}
    for rid in roles:
        for rep in REPS:
            for arm in ("R", "C"):
                name = f"{rep}__{arm}__{rid}"
                file = resolve(CAST5A) / "learning" / "heads" / "models" / f"{name}.joblib"
                if sha256(file) != entries[name]["sha256"]:
                    raise ValueError(f"Head file changed: {name}")
                models[name] = joblib.load(file)
    return models


def load_features(folders):
    features = {}
    for folder in folders:
        _, payload = read_bundle(folder)
        with np.load(BytesIO(payload["features.npz"]), allow_pickle=False) as value:
            for i, fid in enumerate(value["ids"].astype(str)):
                features[fid] = {rep: value[rep][i] for rep in REPS}
    return features


def fit_losses(folders):
    out = {}
    for folder in folders:
        _, payload = read_bundle(folder)
        for line in payload["fits.jsonl"].decode().splitlines():
            r = json.loads(line)
            out[r["input_id"]] = (r["fit_loss"], r["check_loss"])
    return out


def statistics(per_head, parts, fold_site):
    from abvid.evaluation.ranking import selection_draws
    from abvid.learning.utility_stats import SEEDS, grids, summary
    from abvid.learning.utility_stats import weighted as weighted_mean
    grid, folds = grids(per_head, parts)

    def weighted(matrix, selection, fold):
        samples = weighted_mean(matrix, selection, fold)
        if samples is None:
            raise ValueError("A head is missing from the bootstrap grid")
        return samples
    sites = sorted({fold_site[f] for f in folds})
    rng = np.random.Generator(np.random.PCG64(SEED))
    sw = np.vstack([np.ones(len(SEEDS), int), selection_draws(rng, DRAWS, len(SEEDS))])
    fw = np.vstack([np.ones(len(folds), int), selection_draws(rng, DRAWS, len(folds))])
    site = np.vstack([np.ones(len(sites), int), selection_draws(rng, DRAWS, len(sites))])
    site_fw = site[:, [sites.index(fold_site[f]) for f in folds]]
    out = {"folds": folds, "means": {}, "contrasts": {}, "site_sensitivity": {}, "retained_above_chance": {}}
    for rep in REPS:
        for head in ("R", "C"):
            for key in METRICS:
                m = {(side, arm): weighted(grid[(rep, f"{head}_{side}_{arm}", key)], sw, fw)
                     for side in ("held", "train") for arm in ARMS}
                for (side, arm), samples in m.items():
                    out["means"].setdefault(rep, {})[f"{head}_{side}_{arm}:{key}"] = summary(samples)
                drop = {side: m[(side, "tv3")] - m[(side, "original")] for side in ("held", "train")}
                out["contrasts"].setdefault(rep, {})[f"{head}:{key}"] = {
                    "held_gain": summary(m[("held", "tv3")] - m[("held", "smooth8")]),
                    "train_gain": summary(m[("train", "tv3")] - m[("train", "smooth8")]),
                    "held_drop_tv3": summary(drop["held"]), "train_drop_tv3": summary(drop["train"]),
                    "memorization_gap_tv3": summary(drop["train"] - drop["held"])}
                out["site_sensitivity"].setdefault(rep, {})[f"{head}:{key}"] = summary(
                    weighted(grid[(rep, f"{head}_held_tv3", key)], sw, site_fw) -
                    weighted(grid[(rep, f"{head}_held_smooth8", key)], sw, site_fw))
        for side in ("held", "train"):
            o = out["means"][rep][f"R_{side}_original:auroc"]["estimate"]
            out["retained_above_chance"].setdefault(rep, {})[side] = {
                arm: (out["means"][rep][f"R_{side}_{arm}:auroc"]["estimate"] - .5) / (o - .5) for arm in ("smooth8", "tv3")}
    return out


def score_stage(preflight=False):
    from abvid.cast.utility_design import load_context
    from abvid.learning.utility import head_metrics
    environment(BASE_ENV)
    if not preflight:
        check_protocol()
    source_design()
    roles = load_context()["roles"]
    real = cache()
    ids = real["ids"].astype(str).tolist()
    index = {fid: k for k, fid in enumerate(ids)}
    groups = held_groups(roles, ids)
    source_root = resolve(SOURCE)
    smooth8 = load_features(sorted((source_root / "features" / "smooth8").iterdir()))
    tv3 = load_features(sorted((source_root / "features" / "tv3").iterdir()))
    if not preflight:
        tv3.update(load_features(sorted((destination(RUN) / "features" / "tv3").iterdir())))
    if len(smooth8) != 4413 or set(tv3) != (set(ids) if not preflight else set(groups[SEEN_FOLD])):
        raise ValueError("Reconstruction features are incomplete")
    models = heads(roles)
    _, payload = read_bundle(source_root / "scores")
    source_scores = json.loads(payload["results.json"])
    published = {(h["partition"], h["representation"], h["arm"]): h for h in source_scores["per_head"]}
    fold_site = {int(k): v for k, v in source_scores["bootstrap"]["fold_site"].items()}
    parts, per_head, changes, checked = [], [], [], 0
    for rid, role in sorted(roles.items()):
        fold = int(rid.rsplit("_", 1)[1])
        parts.append({"id": rid, "analysis": role.analysis, "fold": fold})
        for rep in REPS:
            for head in ("R", "C"):
                model = models[f"{rep}__{head}__{rid}"]
                for side, side_ids in (("held", list(role.held_ids)), ("train", list(role.train_ids))):
                    y = real["y"][[index[i] for i in side_ids]]
                    probs = {"original": [model.predict_proba(real[rep][[index[i] for i in side_ids]].astype(np.float64))[:, 1]]}
                    for arm, feats in (("smooth8", smooth8), ("tv3", tv3)):
                        if all(i in feats for i in side_ids):
                            x = np.stack([feats[i][rep] for i in side_ids]).astype(np.float64)
                            probs[arm] = [model.predict_proba(x[:, r])[:, 1] for r in (0, 1)]
                    for arm, ps in probs.items():
                        ms = [head_metrics(y, p) for p in ps]
                        row = {"partition": rid, "representation": rep, "arm": f"{head}_{side}_{arm}",
                               **{k: float(np.mean([m[k] for m in ms])) for k in METRICS}}
                        ref = published.get((rid, rep, row["arm"]))
                        if ref is not None:
                            if any(row[k] != ref[k] for k in METRICS):
                                raise ValueError(f"Gate 2 failed, reused metrics differ: {rid} {rep} {row['arm']}")
                            checked += 1
                        per_head.append(row)
                    if head == "R" and side == "held" and "tv3" in probs:
                        s, t = ((p[0] > .5) == (y == 1) for p in (probs["smooth8"], probs["tv3"]))
                        changes.append({"head": f"{rep}__R__{rid}", **{
                            f"{cls}_{what}": int((cond & (y == c)).sum())
                            for c, cls in ((0, "car"), (1, "truck"))
                            for what, cond in (("repaired", ~s & t), ("broken", s & ~t))}})
    print(json.dumps({"gate2_reused_rows_reproduced": checked}), flush=True)
    if preflight:
        print(json.dumps({"gate5_feature_max": feature_gate(extractor())}), flush=True)
        return
    decision_parts = [p for p in parts if p["fold"] in DECISION_FOLDS]
    decision_rows = [h for h in per_head if int(h["partition"].rsplit("_", 1)[1]) in DECISION_FOLDS]
    primary_stats = statistics(decision_rows, decision_parts, fold_site)
    all_stats = statistics(per_head, parts, fold_site)
    primary = primary_stats["contrasts"]["BEATs768"]["R:auroc"]["held_gain"]
    lo, hi = primary["ci95"]
    decision = ("helps" if lo > 0 and primary["estimate"] >= HELPS else
                "not_enough" if hi < HELPS else "ambiguous")
    per_fold = {}
    for fold in sorted(groups):
        rows = [h for h in per_head if int(h["partition"].rsplit("_", 1)[1]) == fold]
        per_fold[fold] = {"held_n": len(groups[fold]), "held_trucks": int(real["y"][[index[i] for i in groups[fold]]].sum()),
                          **{f"{rep}:{arm}": float(np.mean([h["auroc"] for h in rows if h["representation"] == rep
                                                            and h["arm"] == f"R_held_{arm}"]))
                             for rep in REPS for arm in ARMS}}
    s_loss = fit_losses(sorted((source_root / "fits" / "smooth8").iterdir()))
    t_loss = fit_losses(sorted((source_root / "fits" / "tv3").iterdir()))
    t_loss.update(fit_losses(sorted((destination(RUN) / "fits" / "tv3").iterdir())))
    losses = {kind: {"smooth8_median": float(np.median([s_loss[i][k] for i in ids])),
                     "tv3_median": float(np.median([t_loss[i][k] for i in ids])),
                     "tv3_lower_fraction": float(np.mean([t_loss[i][k] < s_loss[i][k] for i in ids]))}
              for k, kind in ((0, "fit_loss"), (1, "check_loss"))}
    result = {"id": "cast_tv3_held_v1", "protocol_sha256": PROTOCOL_SHA, "target_access": False, "new_model_fits": 0,
              "decision": decision, "decision_folds": list(DECISION_FOLDS),
              "primary_beats_r_held_auroc_gain": primary,
              "primary_site_sensitivity": primary_stats["site_sensitivity"]["BEATs768"]["R:auroc"],
              "decision_folds_statistics": primary_stats, "all_folds_statistics": all_stats,
              "per_fold": per_fold, "losses": losses, "held_verdict_changes_r0": changes, "per_head": per_head,
              "gates": {"reused_rows_reproduced": checked, "tv3_replay_parity_and_features": "checked in fit/features"},
              "bootstrap": {"draws": DRAWS, "seed": SEED, "fold_site": {str(k): v for k, v in fold_site.items()}},
              "score_code_sha256": code_hashes(), "provenance": provenance()}
    write_bundle(destination(RUN) / "scores", {"results.json": encoded(result)}, {"kind": "cast_tv3_held_scores"})
    print(json.dumps({"decision": decision, "primary": primary, "site": result["primary_site_sensitivity"],
                      "losses": losses}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check-fit", "check-score", "fit", "features", "score"))
    stage = parser.parse_args().stage
    {"check-fit": lambda: fit_stage(preflight=True), "check-score": lambda: score_stage(preflight=True),
     "fit": fit_stage, "features": features_stage, "score": score_stage}[stage]()
