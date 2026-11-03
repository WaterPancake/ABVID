"""CAST time-varying spectral weights: fit, render, features and score.

Contract: experiments/cast_utility/time_varying/PROTOCOL.md (frozen before any
fit). Training parents of role utility_s42_0 only; no held, target or sealed
data. Create-only bundles under artifact:cast/diagnostics/.

    .venv/bin/python -B experiments/cast_utility/time_varying/run.py fit
    .venv-baseline/bin/python -B experiments/cast_utility/time_varying/run.py features
    .venv-baseline/bin/python -B experiments/cast_utility/time_varying/run.py score
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from io import BytesIO
import json

import numpy as np

from abvid.cast.grouped_storage import read_bundle, write_bundle
from abvid.cast.stage_diagnostic import (BASE_ENV, CAST_ENV, REPS, ROLE, cached_role, design, destination, encoded,
                                         environment, feature_path, load_arrays, npy, npz, provenance, wave_hash)
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

STAGE_RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
RUN = "artifact:cast/diagnostics/cast_tv_weights_20261031_v1"
PROTOCOL = "repo:experiments/cast_utility/time_varying/PROTOCOL.md"
CODE = ("repo:experiments/cast_utility/time_varying/run.py", "repo:src/abvid/cast/time_varying.py")
LISTENING = "artifact:cast/listening/stage_diagnosis_20261025_v1"
KNOTS, WORKERS, PARITY = 3, 8, 1e-5
NAMES = {"smooth8": ("smooth8_check_0", "smooth8_check_1"), "tv3": ("tv3_check_0", "tv3_check_1")}


def code_hashes():
    return {ref: sha256(resolve(ref)) for ref in CODE}


def frozen():
    """This run's design; refuses a protocol changed after the freeze."""
    _, payload = read_bundle(destination(RUN) / "design")
    own = json.loads(payload["design.json"])
    if sha256(resolve(PROTOCOL)) != own["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze")
    return own


def _fit_one(job):
    import torch
    from abvid.cast import time_varying as tv
    from abvid.cast.fit import Objective
    from abvid.cast.renderer import tensors
    from abvid.data.augmentation import shape
    fid, target, smooth8, cfg = job
    target = torch.from_numpy(target)
    result = tv.fit(target, cfg, fid)
    with torch.no_grad():
        raw = tv.TimeVaryingRenderer(cfg, fid).render(tensors(result["parameters"]), "check")[0].numpy()
        waves = [shape(raw[r]).astype(np.float32) for r in (0, 1)]
        objective = Objective(target, cfg)
        losses = {arm: [float(objective(torch.from_numpy(w)[None]).mean()) for w in ws]
                  for arm, ws in (("tv3", waves), ("smooth8", smooth8))}
    return fid, result, waves, losses


def fit_stage():
    from abvid.cast import time_varying as tv
    from abvid.cast.audio import normalize, observation
    from abvid.cast.grouped_parents import _configure_cpu, read_source
    from abvid.cast.utility_design import _cast4a_context, load_context
    environment(CAST_ENV)
    _configure_cpu()
    d, parents, identity = design(STAGE_RUN)
    cfg = copy.deepcopy(d["renderer_configs"]["smooth8_v1"])
    cfg["renderer"]["weight_knots"] = KNOTS
    out = destination(RUN)
    if not (out / "design").exists():
        write_bundle(out / "design", {"design.json": encoded({
            "id": "cast_tv_weights_v1", "protocol": PROTOCOL, "protocol_sha256": sha256(resolve(PROTOCOL)),
            "stage_run": STAGE_RUN, "stage_design_identity": identity, "role": ROLE, "train_ids": d["train_ids"],
            "config": cfg, "fit_code_sha256": code_hashes(), "cast_environment": CAST_ENV, "provenance": provenance()})},
            {"kind": "cast_tv_weights_design"})
    own = frozen()
    if own["config"] != cfg or own["train_ids"] != d["train_ids"]:
        raise ValueError("Design differs from the frozen run")
    gaps = [tv.parity(p["parameters"]["smooth8"], cfg, p["file_id"]) for p in parents]
    if max(gaps) > PARITY:
        raise ValueError(f"Equal-knot parity gate failed: {max(gaps)}")
    print(json.dumps({"stage": "parity_gate", "max_abs_difference": max(gaps)}), flush=True)
    smooth8 = []
    for name in NAMES["smooth8"]:
        _, payload = read_bundle(destination(STAGE_RUN) / "render" / name)
        if json.loads(payload["ids.json"]) != d["train_ids"]:
            raise ValueError(f"Unexpected smooth8 order: {name}")
        smooth8.append(np.load(BytesIO(payload["waves.npy"]), allow_pickle=False))
    c, c4 = load_context(), _cast4a_context()
    role, base = c["roles"][ROLE], d["renderer_configs"]["smooth8_v1"]
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        expected = dict(zip(cache["ids"].astype(str), cache["wave_sha256"].astype(str)))
    jobs = []
    for i, parent in enumerate(parents):
        fid = parent["file_id"]
        role.validate_parent(parent["source"])
        raw, sr = read_source(c4, fid)
        target = normalize(observation(raw, sr, base), base)[0].numpy()
        if wave_hash(target) != expected[fid]:
            raise ValueError(f"Original profile/hash drift: {fid}")
        jobs.append((fid, target, [smooth8[0][i], smooth8[1][i]], cfg))
    results = []
    with ProcessPoolExecutor(WORKERS, initializer=_configure_cpu) as pool:
        for k, item in enumerate(pool.map(_fit_one, jobs, chunksize=2), 1):
            results.append(item)
            if k % 20 == 0 or k == len(jobs):
                print(json.dumps({"stage": "fits", "complete": k, "total": len(jobs)}), flush=True)
    ids = [fid for fid, *_ in results]
    meta = {"kind": "cast_tv_weights_fits", "design_identity": identity, "fit_code_sha256": code_hashes()}
    write_bundle(out / "fits", {"fits.jsonl": "".join(json.dumps(r, allow_nan=False) + "\n" for _, r, _, _ in results).encode(),
                                "gate.json": encoded({"max_abs_difference": max(gaps), "limit": PARITY, "parents": len(gaps)}),
                                "losses.json": encoded({fid: losses for fid, _, _, losses in results})}, meta)
    for r in (0, 1):
        waves = np.stack([w[r] for _, _, w, _ in results])
        records = [{"file_id": fid, "realization": r, "wave_sha256": wave_hash(w)} for fid, w in zip(ids, waves)]
        write_bundle(out / "render" / f"tv3_check_{r}", {"waves.npy": npy(waves), "ids.json": encoded(ids),
                                                         "records.json": encoded(records)}, meta)
    print(json.dumps({"stage": "fit_done", "parents": len(results)}), flush=True)


def features_stage():
    import torch
    from abvid.representations.beats import extract_embeddings, load_encoder
    from abvid.representations.mfcc import mfcc_features
    environment(BASE_ENV)
    d, _, identity = design(STAGE_RUN)
    frozen()
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
    for name in NAMES["tv3"]:
        _, payload = read_bundle(destination(RUN) / "render" / name)
        ids = json.loads(payload["ids.json"])
        arrays = extract(np.load(BytesIO(payload["waves.npy"]), allow_pickle=False))
        write_bundle(destination(RUN) / "features" / name,
                     {"features.npz": npz(ids=np.array(ids), **arrays),
                      "row_hashes.json": encoded([{"file_id": f, **{rep: wave_hash(arrays[rep][i]) for rep in REPS}}
                                                  for i, f in enumerate(ids)])},
                     {"kind": "cast_tv_weights_features", "design_identity": identity, "environment": BASE_ENV})
        print(json.dumps({"stage": "features", "name": name, "count": len(ids)}), flush=True)


def band_centres(n=128, low=20.0, high=8000.0):
    """Kaldi fbank bin centres: equally spaced on 1127 ln(1 + f/700)."""
    lo, hi = (1127*np.log1p(f/700) for f in (low, high))
    return 700*np.expm1((lo + (hi-lo)/(n+1)*np.arange(1, n+1))/1127)


def variation_kept(d, waves, groups):
    """Median over clips of mean(fitted / original temporal std) in BEATs fbank bands at 100-400 Hz."""
    import torch
    from abvid.representations.beats import load_encoder
    encoder, _ = load_encoder(dict(d["h1_config"]["encoder"], checkpoint=str(resolve(d["checkpoint"]))))
    with torch.inference_mode():
        mel = {k: encoder.preprocess(torch.from_numpy(w), fbank_mean=0.0, fbank_std=0.5).numpy().astype(np.float64)
               for k, w in waves.items()}
    centres = band_centres(mel["original"].shape[-1])
    band = (centres >= 100) & (centres <= 400) & (mel["original"].std(axis=(0, 1)) > 0.1)
    out = {}
    for group, index in groups.items():
        o = mel["original"][index].std(1)[:, band]
        out[group] = {arm: float(np.median((mel[arm][index].std(1)[:, band]/o).mean(1))) for arm in NAMES}
    return out


def score_stage():
    import joblib
    from abvid.learning.utility import head_metrics
    environment(BASE_ENV)
    d, parents, identity = design(STAGE_RUN)
    own = frozen()
    by_id = {p["file_id"]: p for p in parents}
    original = cached_role(d, "real")
    y, ids = original["y"], d["train_ids"]
    models = {k: joblib.load(resolve(ref)) for k, ref in d["head_refs"].items()}
    arrays = {name: load_arrays(STAGE_RUN, name, identity) for name in NAMES["smooth8"]}
    for name in NAMES["tv3"]:
        _, payload = read_bundle(destination(RUN) / "features" / name)
        with np.load(BytesIO(payload["features.npz"]), allow_pickle=False) as value:
            arrays[name] = {key: value[key] for key in value.files}
    for name, value in arrays.items():
        if value["ids"].astype(str).tolist() != ids:
            raise ValueError(f"Feature order differs: {name}")
    _, payload = read_bundle(destination(STAGE_RUN) / "scores")
    published = {(r["representation"], r["input"], r["head"]): r["metrics"]
                 for r in json.loads(payload["results.json"])["rows"] if r["scope"] == "all380"}
    groups = sorted({by_id[i]["source"]["provenance_group_id"] for i in ids})
    rows, p_by = [], {}
    for rep in REPS:
        for head in ("R", "C"):
            model = models[f"{rep}__{head}"]
            p0 = model.predict_proba(original[rep].astype(np.float64))[:, 1]
            for name, value in arrays.items():
                p = model.predict_proba(value[rep].astype(np.float64))[:, 1]
                p_by[(rep, head, name)] = p
                metrics = head_metrics(y, p)
                if name in NAMES["smooth8"] and metrics != published[(rep, name, head)]:
                    raise ValueError(f"smooth8 reference does not reproduce published metrics: {rep} {head} {name}")
                right0, right = (p0 > .5) == (y == 1), (p > .5) == (y == 1)
                rows.append({"representation": rep, "head": head, "input": name, "metrics": metrics,
                             "flipped": {cls: int((right0 & ~right & (y == c)).sum()) for c, cls in ((0, "car"), (1, "truck"))},
                             "groups": {g: head_metrics(y[m], p[m]) for g in groups
                                        for m in [np.array([by_id[i]["source"]["provenance_group_id"] == g for i in ids])]}})
    def mean(rep, head, arm, key):
        return float(np.mean([next(r for r in rows if (r["representation"], r["head"], r["input"]) == (rep, head, n))["metrics"][key]
                              for n in NAMES[arm]]))
    primary = 100*(mean("BEATs768", "R", "tv3", "auroc") - mean("BEATs768", "R", "smooth8", "auroc"))
    truck = 100*(mean("BEATs768", "R", "tv3", "positive_recall") - mean("BEATs768", "R", "smooth8", "positive_recall"))
    decision = ("promising" if primary >= 5 and truck > 0 else "no_meaningful_change" if abs(primary) <= 2
                else "small_or_ambiguous")
    _, payload = read_bundle(destination(RUN) / "fits")
    losses = json.loads(payload["losses.json"])
    tv_loss = np.array([losses[i]["tv3"] for i in ids])
    s8_loss = np.array([losses[i]["smooth8"] for i in ids])
    listening = resolve(LISTENING)
    if json.loads((listening / "originals_ids.json").read_text()) != ids:
        raise ValueError("Saved originals are in another order")
    waves = {"original": np.load(listening / "originals.npy", allow_pickle=False)}
    if any(wave_hash(w) != original_hash for w, original_hash in zip(waves["original"], cached_hashes(ids))):
        raise ValueError("Saved originals differ from the classifier inputs")
    for arm in NAMES:
        _, payload = read_bundle(destination(RUN if arm == "tv3" else STAGE_RUN) / "render" / NAMES[arm][0])
        waves[arm] = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
    p8 = p_by[("BEATs768", "R", "smooth8_check_0")]
    trucks = y == 1
    spectro_groups = {"trucks_very_wrong_at_smooth8": np.flatnonzero(trucks & (p8 < .1)),
                      "trucks_still_right_at_smooth8": np.flatnonzero(trucks & (p8 >= .9)),
                      "all": np.arange(len(ids))}
    result = {"id": "cast_tv_weights_v1", "protocol_sha256": own["protocol_sha256"], "design_identity": identity,
              "role": ROLE, "parent_count": len(ids), "target_access": False, "new_model_fits": 0,
              "primary_beats_r_auroc_points": primary, "beats_r_truck_recall_points": truck, "decision": decision,
              "check_loss": {"median": {"tv3": float(np.median(tv_loss)), "smooth8": float(np.median(s8_loss))},
                             "fraction_tv3_lower": float(np.mean(tv_loss < s8_loss))},
              "variation_kept_100_400hz": variation_kept(d, waves, spectro_groups),
              "rows": rows, "score_code_sha256": code_hashes(), "provenance": provenance()}
    write_bundle(destination(RUN) / "scores", {"results.json": encoded(result)},
                 {"kind": "cast_tv_weights_scores", "design_identity": identity})
    print(json.dumps({k: result[k] for k in ("primary_beats_r_auroc_points", "beats_r_truck_recall_points", "decision",
                                             "check_loss", "variation_kept_100_400hz")}, indent=1))
    for r in rows:
        if r["head"] == "R":
            m = r["metrics"]
            print(f"{r['representation']:9s} {r['input']:16s} AUROC {100*m['auroc']:6.2f}  BA {100*m['balanced_accuracy']:6.2f}  "
                  f"truck rec {100*m['positive_recall']:6.2f}  car rec {100*m['negative_recall']:6.2f}  flipped {r['flipped']}")


def cached_hashes(ids):
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        expected = dict(zip(cache["ids"].astype(str), cache["wave_sha256"].astype(str)))
    return [expected[i] for i in ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "features", "score"))
    {"fit": fit_stage, "features": features_stage, "score": score_stage}[parser.parse_args().stage]()
