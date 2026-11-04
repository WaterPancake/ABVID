"""B1 reference on MELAUDIS (b1_melaudis_v1): check, extract, score.

Contract: experiments/b1_melaudis/PROTOCOL.md. Baseline environment. Scores the
two frozen all-IDMT B1 heads on the 8,066 H1-admitted MELAUDIS events and
compares them with H1's saved IDMT-only MELAUDIS predictions. Nothing is fitted
and no threshold is set. MELAUDIS members are extracted from the read-only RAR
into a temporary folder, hash-checked, featurized and deleted.

    .venv-baseline/bin/python -B experiments/b1_melaudis/run.py check     # gates 1, 3-5 and a 16-member audio check; no MELAUDIS scores
    .venv-baseline/bin/python -B experiments/b1_melaudis/run.py extract   # frozen protocol required
    .venv-baseline/bin/python -B experiments/b1_melaudis/run.py score     # frozen protocol required
"""
import argparse
from importlib.metadata import version
from io import BytesIO
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import joblib
import numpy as np

from abvid.evaluation.ranking import METRICS, evaluate, group_draws, interval, selection_draws
from abvid.learning import reference as recipe
from abvid.learning.grouped_search import metrics as b1_metrics
from abvid.paths import resolve
from abvid.provenance.hashing import sha256

PROTOCOL = "repo:experiments/b1_melaudis/PROTOCOL.md"
PROTOCOL_SHA = "ed8ca4f461a95630758f8c31baf4114eb38f3a64298dd9888a032a4c17ae3a67"
CODE = ("repo:experiments/b1_melaudis/run.py", "repo:src/abvid/learning/reference.py",
        "repo:src/abvid/evaluation/ranking.py")
REFERENCE = "artifact:baseline_strength/b1_reference_20261103_v1"
REFERENCE_SHA = "7f1ebc2cd6f3b6c702ba658d05b7ed28cecefe0f55abaab1a9c5933319d748cb"
B1 = "artifact:baseline_strength/b1_20261012_v1"
B1_FEATURES_SHA = "aab4b3aac1072bbd6ede4213e224802d512b987efc41dd193354d8facbc38d93"
H1_LOCK = "archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json"
H1_CONFIG = "archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json"
H1_ADMITTED = "archive:experiments/h1/frozen/h1_common_budget_v1.3/admitted.jsonl"
H1_RESULTS = "archive:experiments/h1/results/H1_20261002"
PANEL = "artifact:target_panel/rescore_20261011_v2/results.json"
PANEL_SHA = "fb813dfe268c494f3142874ee0a435edd91c985606501022225ad64c58a51ac6"
RAR = "archive:dataset/MELAUDIS/MELAUDIS_Vehicles.rar"
RAR_SHA = "e9e9af4532f35c845661e07d89d9c3d9b319fc38542e6e2e2d52cef28a74a214"
IDMT_SOURCES = "artifact:cast/grouped_coverage/cast4a_design_20261007_v1/sources.jsonl"
IDMT_AUDIO = "archive:dataset/IDMT_Traffic/audio"
BEATS = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
OUT = "artifact:baseline_strength/b1_melaudis_20261103_v1"
ENV = {"python": "3.11.13", "numpy": "1.26.4", "scikit-learn": "1.4.2", "torch": "2.11.0", "librosa": "0.11.0"}
H1_REPS = {"beats_avg": "BEATs768", "mfcc26": "MFCC26"}
SEEDS, FOLDS = (42, 123, 456, 789, 1024), 6
SEED, DRAWS, CHECK_N, FEATURE_TOL, PRACTICAL = 20261104, 10000, 16, 1e-5, 0.03


def environment():
    actual = {"python": platform.python_version(), **{k: version(k) for k in ENV if k != "python"}}
    if actual != ENV:
        raise ValueError(f"Wrong numerical environment: {actual}")


def check_protocol():
    if PROTOCOL_SHA is None or sha256(resolve(PROTOCOL)) != PROTOCOL_SHA:
        raise ValueError("Protocol is not frozen or changed after freeze")


def pinned(path, expected, name):
    if sha256(path) != expected:
        raise ValueError(f"Gate 1 failed, pinned input changed: {name}")


def inputs():
    """Gate 1: pinned inputs. Returns the frozen-head manifest, H1 config and MELAUDIS rows in H1 target order."""
    ref = resolve(REFERENCE)
    pinned(ref / "results.json", REFERENCE_SHA, "B1 reference results")
    heads = json.loads((ref / "results.json").read_text())["all_idmt_heads"]
    for rep in recipe.REPRESENTATIONS:
        pinned(ref / heads[rep]["file"], heads[rep]["sha256"], f"head {rep}")
    lock = json.loads(resolve(H1_LOCK).read_text())
    pinned(resolve(H1_ADMITTED), lock["artifacts_sha256"]["admitted.jsonl"], "H1 admitted rows")
    pinned(resolve(H1_CONFIG), lock["artifacts_sha256"]["config.json"], "H1 config")
    pinned(resolve(PANEL), PANEL_SHA, "target panel")
    rows = {r["file_id"]: r for r in map(json.loads, resolve(H1_ADMITTED).open()) if r["dataset_id"] == "MELAUDIS"}
    first = h1_predictions("BEATs768", SEEDS[0], 0)
    ids = first["target_ids"].astype(str).tolist()
    if sorted(ids) != sorted(rows) or len(ids) != 8066:
        raise ValueError("MELAUDIS target population differs from the admitted rows")
    target = [rows[i] for i in ids]
    if any(r["class_id"] != y for r, y in zip(target, first["target_true"])):
        raise ValueError("MELAUDIS labels differ between admission and H1 predictions")
    return heads, json.loads(resolve(H1_CONFIG).read_text()), target


def h1_predictions(rep, seed, fold):
    root = resolve(H1_RESULTS)
    name = f"{rep}__real__seed{seed}__fold{fold}"
    record = json.loads((root / "models" / f"{name}.json").read_text())
    path = root / "predictions" / f"{name}.npz"
    if sha256(path) != record["predictions_sha256"]:
        raise ValueError(f"Gate 1 failed, H1 prediction file changed: {name}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def target_arrays(target):
    names = sorted({r["provenance_group_id"] for r in target})
    y = np.array([r["class_id"] for r in target])
    group = np.array([names.index(r["provenance_group_id"]) for r in target])
    return names, y, group


def h1_scores(rep, ids):
    grid = []
    for seed in SEEDS:
        row = []
        for fold in range(FOLDS):
            d = h1_predictions(rep, seed, fold)
            if d["target_ids"].astype(str).tolist() != ids:
                raise ValueError("H1 target order differs between prediction files")
            row.append(d["target_probabilities"][:, 1])
        grid.append(row)
    return np.array(grid)


def per_group(scores, y, group, n_groups):
    ones = [np.ones((1, d), int) for d in np.asarray(scores).shape[:-1]]
    return [{k: float(v[0]) for k, v in evaluate(scores, y, group, n_groups, np.eye(n_groups, dtype=int)[[g]], ones).items()}
            for g in range(n_groups)]


def scorer_replay(target):
    """Gate 4: the scorer reproduces the target panel's H1 real-arm point estimates exactly."""
    names, y, group = target_arrays(target)
    ids = [r["file_id"] for r in target]
    panel = json.loads(resolve(PANEL).read_text())["h1"]
    if panel["target"]["groups"] != names:
        raise ValueError("Target panel groups differ")
    worst = 0.0
    for rep in H1_REPS.values():
        scores = h1_scores(rep, ids)
        ones = [np.ones((1, d), int) for d in scores.shape[:-1]]
        point = evaluate(scores, y, group, len(names), np.ones((1, len(names)), int), ones)
        saved = panel[rep]["arms"]["real"]
        keys = [k for k in METRICS if k in saved["summary"]]  # the panel predates later METRICS additions
        if not {"auroc", "average_precision", "ap_lift", "recall_at_fpr_05", "balanced_accuracy"} <= set(keys):
            raise ValueError("Target panel lacks a compared metric")
        for k in keys:
            worst = max(worst, abs(point[k][0] - saved["summary"][k]["estimate"]))
        for mine, theirs in zip(per_group(scores, y, group, len(names)), saved["per_group"]):
            worst = max(worst, max(abs(mine[k] - theirs[k]) for k in keys))
    if worst != 0.0:
        raise ValueError(f"Gate 4 failed, scorer replay differs: {worst}")
    return worst


def load_heads(heads):
    return {rep: joblib.load(resolve(REFERENCE) / heads[rep]["file"]) for rep in recipe.REPRESENTATIONS}


def head_replay(heads):
    """Gate 5: frozen heads reproduce their recorded training AUROC on B1's cached IDMT features."""
    path = resolve(B1) / "features.npz"
    pinned(path, B1_FEATURES_SHA, "B1 features")
    models = load_heads(heads)
    with np.load(path, allow_pickle=False) as f:
        x = {"mfcc26": recipe.representation("mfcc26", mfcc=f["full__mfcc26"]),
             "beats_avg": recipe.representation("beats_avg", beats_layers=f["full__beats_layers"])}
        y = f["y"]
    worst = 0.0
    for rep, model in models.items():
        auroc = b1_metrics(y, model.predict_proba(x[rep].astype(np.float64))[:, 1])["auroc"]
        worst = max(worst, abs(auroc - heads[rep]["train_auroc"]))
    if worst != 0.0:
        raise ValueError(f"Gate 5 failed, frozen heads differ: {worst}")
    return worst


class Extractor:
    """The B1 full-band feature path: observation -> BEATs layer mean and MFCC26."""

    def __init__(self, cfg):
        import torch
        from abvid.representations.beats import load_encoder
        torch.set_num_threads(cfg["hardware"]["torch_threads"])
        torch.set_num_interop_threads(1)
        self.cfg, self.pre = cfg, recipe.preprocessing(cfg["preprocessing"])
        self.beats, _ = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve(BEATS))))

    def __call__(self, decoded):
        import torch
        from abvid.data.audio import observation
        from abvid.representations.beats import extract_layer_embeddings
        from abvid.representations.mfcc import mfcc_features
        waves = np.stack([observation(x, sr, self.pre) for x, sr in decoded])
        layers = extract_layer_embeddings(self.beats, torch.from_numpy(waves)).numpy()
        return {"beats_layers": layers, "mfcc26": np.stack([mfcc_features(w.astype(np.float64), self.cfg["mfcc"]) for w in waves])}


def feature_replay(extract):
    """Gate 3: the extraction path reproduces B1's cached full-route features for the first IDMT clips."""
    import soundfile as sf
    rows = {r["file_id"]: r for r in map(json.loads, resolve(IDMT_SOURCES).open())}
    with np.load(resolve(B1) / "features.npz", allow_pickle=False) as f:
        ids = f["ids"][:CHECK_N].astype(str).tolist()
        cached = {"beats_layers": f["full__beats_layers"][:CHECK_N], "mfcc26": f["full__mfcc26"][:CHECK_N]}
    decoded = []
    for i in ids:
        data = (resolve(IDMT_AUDIO) / Path(rows[i]["source_path"]).name).read_bytes()
        if hashlib.sha256(data).hexdigest() != rows[i]["source_file_sha256"]:
            raise ValueError(f"IDMT source changed: {i}")
        decoded.append(sf.read(BytesIO(data), dtype="float64", always_2d=True))
    fresh = extract(decoded)
    worst = {k: float(np.abs(fresh[k] - cached[k]).max()) for k in cached}
    if any(v > FEATURE_TOL for v in worst.values()):
        raise ValueError(f"Gate 3 failed, feature path differs: {worst}")
    return worst


def decoded_hash(x, rate):
    h = hashlib.sha256(f"abvid-decoded-f64le-v1|{rate}|{x.shape[1]}|{len(x)}\n".encode())
    h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    return h.hexdigest()


def extract_members(target, work):
    """Extract the admitted members from the read-only RAR into ``work``."""
    listing = work / "members.txt"
    listing.write_text("".join(r["relative_path"] + "\n" for r in target))
    subprocess.run(["/usr/bin/tar", "-xf", str(resolve(RAR)), "-C", str(work), "-T", str(listing),
                    "--no-same-owner", "--no-same-permissions"], check=True, capture_output=True)


def decode_checked(rows, work):
    """Gate 2: each extracted member matches both admission hashes; returns decoded audio."""
    import soundfile as sf
    decoded = []
    for r in rows:
        data = (work / r["relative_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["source_file_sha256"]:
            raise ValueError(f"Gate 2 failed, member bytes differ: {r['file_id']}")
        x, rate = sf.read(BytesIO(data), dtype="float64", always_2d=True)
        if decoded_hash(x, rate) != r["decoded_audio_sha256"]:
            raise ValueError(f"Gate 2 failed, decoded audio differs: {r['file_id']}")
        decoded.append((x, rate))
    return decoded


def provenance():
    git = lambda *a: subprocess.check_output(["git", *a], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree": git("status", "--short").splitlines(), "platform": platform.platform(),
            "environment": ENV, "code_sha256": {ref: sha256(resolve(ref)) for ref in CODE}}


def check():
    environment()
    heads, cfg, target = inputs()
    print(json.dumps({"gate1_inputs": "ok"}), flush=True)
    print(json.dumps({"gate5_head_replay_max": head_replay(heads)}), flush=True)
    print(json.dumps({"gate4_scorer_replay_max": scorer_replay(target)}), flush=True)
    extract = Extractor(cfg)
    print(json.dumps({"gate3_feature_replay_max": feature_replay(extract)}), flush=True)
    pinned(resolve(RAR), RAR_SHA, "MELAUDIS RAR")
    with tempfile.TemporaryDirectory() as tmp:  # audio integrity only: no features, no scores
        extract_members(target[:CHECK_N], Path(tmp))
        decode_checked(target[:CHECK_N], Path(tmp))
    print(json.dumps({"gate2_audio_sample": f"{CHECK_N} members match both admission hashes"}), flush=True)


def extract_stage():
    environment()
    check_protocol()
    heads, cfg, target = inputs()
    head_replay(heads)
    extract = Extractor(cfg)
    replay = feature_replay(extract)
    out = resolve(OUT, must_exist=False)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work_audio"
    work.mkdir()
    try:
        pinned(resolve(RAR), RAR_SHA, "MELAUDIS RAR")
        extract_members(target, work)
        parts = []
        for i in range(0, len(target), 16):
            parts.append(extract(decode_checked(target[i:i+16], work)))
            if i % 1600 == 0:
                print(json.dumps({"stage": "features", "done": i + 16, "of": len(target)}), flush=True)
    finally:
        shutil.rmtree(work)
    beats = np.concatenate([p["beats_layers"] for p in parts])
    arrays = {"ids": np.array([r["file_id"] for r in target]), "y": np.array([r["class_id"] for r in target]),
              "beats_avg": recipe.representation("beats_avg", beats_layers=beats).astype(np.float32),
              "mfcc26": np.concatenate([p["mfcc26"] for p in parts]).astype(np.float32)}
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k not in ("ids", "y")):
        raise ValueError("Nonfinite features")
    with (out / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    record = {"protocol_sha256": PROTOCOL_SHA, "rar_sha256": RAR_SHA, "members": len(target),
              "audio_gate": "all members matched source_file_sha256 and decoded_audio_sha256; extracted copies deleted",
              "feature_replay_max": replay, "features_sha256": sha256(out / "features.npz"), "provenance": provenance()}
    (out / "features_record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(json.dumps({k: record[k] for k in ("members", "feature_replay_max", "features_sha256")}), flush=True)


def summarize(samples):
    return {k: {"estimate": float(v[0]), "ci95": interval(v[1:])} for k, v in samples.items()}


def score_stage():
    environment()
    check_protocol()
    heads, _, target = inputs()
    head_replay(heads)
    replay = scorer_replay(target)
    out = resolve(OUT)
    record = json.loads((out / "features_record.json").read_text())
    pinned(out / "features.npz", record["features_sha256"], "MELAUDIS features")
    with np.load(out / "features.npz", allow_pickle=False) as f:
        features = {k: f[k] for k in f.files}
    ids = [r["file_id"] for r in target]
    names, y, group = target_arrays(target)
    if features["ids"].astype(str).tolist() != ids or not np.array_equal(features["y"], y):
        raise ValueError("Feature rows differ from the target population")
    support = np.stack([np.bincount(group[y == c], minlength=len(names)) for c in (0, 1)], 1)
    rng = np.random.Generator(np.random.PCG64(SEED))
    gw, rejected = group_draws(rng, DRAWS, support)
    gw = np.vstack([np.ones(len(names), int), gw])
    sw = np.vstack([np.ones(len(SEEDS), int), selection_draws(rng, DRAWS, len(SEEDS))])
    fw = np.vstack([np.ones(FOLDS, int), selection_draws(rng, DRAWS, FOLDS)])
    ref = json.loads((resolve(REFERENCE) / "results.json").read_text())["representations"]
    models = load_heads(heads)
    result_reps = {}
    for rep, h1_rep in H1_REPS.items():
        p_b1 = models[rep].predict_proba(features[rep].astype(np.float64))[:, 1]
        s_h1 = h1_scores(h1_rep, ids)
        b1 = evaluate(p_b1, y, group, len(names), gw, [])
        h1 = evaluate(s_h1, y, group, len(names), gw, [sw, fw])
        idmt = ref[rep]["mean"]["auroc"]["estimate"]
        result_reps[rep] = {
            "b1": summarize(b1), "h1_real": summarize(h1),
            "b1_minus_h1": {k: {"estimate": float(b1[k][0] - h1[k][0]), "ci95": interval(b1[k][1:] - h1[k][1:])}
                            for k in ("auroc", "average_precision", "ap_lift", "recall_at_fpr_05", "balanced_accuracy")},
            "per_group": {"groups": names, "support": support.tolist(), "b1": per_group(p_b1, y, group, len(names)),
                          "h1_real": per_group(s_h1, y, group, len(names))},
            "idmt_within_auroc": idmt, "idmt_minus_melaudis_auroc": idmt - float(b1["auroc"][0]),
            "b1_probabilities_sha256": hashlib.sha256(np.ascontiguousarray(p_b1, "<f8").tobytes()).hexdigest()}
    primary = result_reps["beats_avg"]["b1_minus_h1"]["auroc"]
    lo, hi = primary["ci95"]
    decision = "positive" if lo > 0 else "no_practical_gain" if hi < PRACTICAL else "ambiguous"
    result = {"id": "b1_melaudis_v1", "protocol_sha256": PROTOCOL_SHA, "decision": decision,
              "primary_beats_auroc_b1_minus_h1": primary, "representations": result_reps,
              "gates": {"scorer_replay_max": replay, "head_replay": "exact", "audio": record["audio_gate"],
                        "feature_replay_max": record["feature_replay_max"]},
              "bootstrap": {"draws": DRAWS, "seed": SEED, "rejected_group_draws": int(rejected)},
              "features_sha256": record["features_sha256"], "provenance": provenance()}
    with (out / "results.json").open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True, allow_nan=False)
    print(json.dumps({"decision": decision, "primary": primary}, indent=1))
    for rep, r in result_reps.items():
        b, h = r["b1"], r["h1_real"]
        print(f"{rep:9s} AUROC B1 {b['auroc']['estimate']:.4f} {b['auroc']['ci95']} vs H1 {h['auroc']['estimate']:.4f}"
              f" | diff {r['b1_minus_h1']['auroc']['estimate']:+.4f} {r['b1_minus_h1']['auroc']['ci95']}"
              f" | AP lift {b['ap_lift']['estimate']:.2f} vs {h['ap_lift']['estimate']:.2f}"
              f" | IDMT-MELAUDIS {r['idmt_minus_melaudis_auroc']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "extract", "score"))
    {"check": check, "extract": extract_stage, "score": score_stage}[parser.parse_args().stage]()
