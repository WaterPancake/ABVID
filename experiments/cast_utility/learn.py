"""CAST-5A learning stage (baseline environment): features, heads, scores.

Thin protocol layer over abvid.learning.utility. Stages are create-only:
  features-base  real shape-only observations, RA copies, RU banks
  features-rc    CAST RC/C banks (after the CAST environment wrote them)
  heads          420 heads; every model saved and hashed before any scoring
  score          held predictions, per-head metrics, paired bootstrap, contrasts
Run: UV_PROJECT_ENVIRONMENT=.venv-baseline uv run --extra baseline python experiments/cast_utility/learn.py <stage>
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import torch

from abvid.cast.grouped_storage import read_bundle, verify_bundle
from abvid.learning.linear import fixed_fit
from abvid.learning.utility_stats import REPS, statistics
from abvid.learning.utility import (ARMS, Backgrounds, background_pool, head_metrics, json_bytes,
                                    ra_copy, real_observation, training_rows)
from abvid.paths import resolve
from abvid.provenance.hashing import sha256
from abvid.representations.beats import extract_embeddings, load_encoder
from abvid.representations.mfcc import mfcc_features

RUN = "artifact:cast/utility/cast5a_run_20261011_v1"
H1_CONFIG = "archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json"
H1_LOCK = "archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json"
CHECKPOINT = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
AUDIO = "archive:dataset/IDMT_Traffic/audio"
SEEDS = (42, 123, 456, 789, 1024)
DRAWS, BOOT_SEED = 10000, 20261013


def h1_config():
    lock = json.loads(resolve(H1_LOCK).read_text())
    path = resolve(H1_CONFIG)
    if sha256(path) != lock["artifacts_sha256"]["config.json"]:
        raise ValueError("Frozen H1 config changed")
    return json.loads(path.read_text())


def design():
    folder = resolve(RUN) / "design"
    verify_bundle(folder)
    _, payload = read_bundle(folder)
    sources = [json.loads(line) for line in payload["inputs/sources.jsonl"].splitlines()]
    return json.loads(payload["partitions.json"]), {r["file_id"]: r for r in sources}


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256(path)


def provenance(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": len(git("status", "--porcelain").splitlines()),
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "script_sha256": sha256(Path(__file__))}


class Featurizer:
    def __init__(self, cfg):
        torch.set_num_threads(cfg["hardware"]["torch_threads"])
        torch.set_num_interop_threads(1)
        self.cfg = cfg
        self.encoder, self.provenance = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve(CHECKPOINT))))
        t = np.arange(32000) / 16000
        toy = torch.from_numpy((.1 * np.sin(2 * np.pi * 123 * t)).astype(np.float32))[None]
        a, b = extract_embeddings(self.encoder, toy), extract_embeddings(self.encoder, toy)
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=0)

    def __call__(self, waves):
        waves = np.asarray(waves, dtype=np.float32)
        batch = self.cfg["hardware"]["batch_size"]
        beats = np.concatenate([extract_embeddings(self.encoder, torch.from_numpy(waves[i:i + batch])).numpy()
                                for i in range(0, len(waves), batch)])
        mfcc = np.stack([mfcc_features(w.astype(np.float64), self.cfg["mfcc"]) for w in waves])
        if not (np.isfinite(beats).all() and np.isfinite(mfcc).all()):
            raise ValueError("Nonfinite features")
        return beats.astype(np.float32), mfcc.astype(np.float32)


def features_base():
    started, cfg = time.perf_counter(), h1_config()
    parts, sources = design()
    out = resolve(RUN) / "learning" / "features_base"
    out.mkdir(parents=True, exist_ok=False)
    featurize = Featurizer(cfg)
    audio = resolve(AUDIO)
    ids = sorted(sources)
    real = {}
    for start in range(0, len(ids), 64):
        chunk = ids[start:start + 64]
        real.update({i: real_observation(audio / Path(sources[i]["source_path"]).name,
                                         sources[i]["source_file_sha256"], cfg["preprocessing"]) for i in chunk})
    beats, mfcc = featurize([real[i] for i in ids])
    hashes = {"real.npz": save_npz(out / "real.npz", ids=np.array(ids), BEATs768=beats, MFCC26=mfcc,
                                   y=np.array([sources[i]["class_id"] for i in ids]),
                                   wave_sha256=np.array([sha256_bytes(real[i]) for i in ids]))}
    print(f"real features {len(ids)} in {time.perf_counter() - started:.0f}s", flush=True)
    pool, backgrounds = background_pool(audio, sources.values()), Backgrounds(cfg["preprocessing"])
    records, ra_ids, ra_beats, ra_mfcc = [], [], [], []
    for part in parts:
        waves = []
        for file_id in part["train_ids"]:
            wave, record = ra_copy(part["id"], file_id, real[file_id], part["training_group_counts"], pool, backgrounds)
            waves.append(wave)
            ra_ids.append(f"{part['id']}|{file_id}")
            records.append(record)
        b, m = featurize(waves)
        ra_beats.append(b)
        ra_mfcc.append(m)
    hashes["ra.npz"] = save_npz(out / "ra.npz", ids=np.array(ra_ids), BEATs768=np.concatenate(ra_beats),
                                MFCC26=np.concatenate(ra_mfcc),
                                y=np.array([sources[i.split("|")[1]]["class_id"] for i in ra_ids]))
    (out / "ra_records.json").write_bytes(json_bytes(records))
    print(f"RA features {len(ra_ids)} in {time.perf_counter() - started:.0f}s", flush=True)
    ru_ids, ru_waves, ru_y = [], [], []
    for seed in SEEDS:
        folder = resolve(RUN) / "arms" / "ru" / f"s{seed}"
        _, payload = read_bundle(folder)
        recs = json.loads(payload["records.json"])
        w = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
        ru_ids += [f"s{seed}|{r['input_id']}" for r in recs]
        ru_y += [0 if r["class"] == "car" else 1 for r in recs]
        ru_waves.append(w)
    beats, mfcc = featurize(np.concatenate(ru_waves))
    hashes["ru.npz"] = save_npz(out / "ru.npz", ids=np.array(ru_ids), BEATs768=beats, MFCC26=mfcc, y=np.array(ru_y))
    summary = {"stage": "features_base", "elapsed_s": time.perf_counter() - started, "hashes": hashes,
               "encoder_provenance": featurize.provenance, "provenance": provenance(resolve("repo:."))}
    (out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps({"elapsed_s": summary["elapsed_s"], "hashes": hashes}))


def sha256_bytes(array):
    import hashlib
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def features_rc():
    started, cfg = time.perf_counter(), h1_config()
    parts, _ = design()
    out = resolve(RUN) / "learning" / "features_rc"
    out.mkdir(parents=True, exist_ok=False)
    featurize = Featurizer(cfg)
    ids, ys, waves, bundles = [], [], [], {}
    for part in parts:
        folder = resolve(RUN) / "arms" / "rc" / part["id"] / "samples"
        _, payload = read_bundle(folder)
        bundles[part["id"]] = verify_bundle(folder)["metadata"]
        recs = json.loads(payload["records.json"])
        waves.append(np.load(BytesIO(payload["waves.npy"]), allow_pickle=False))
        ids += [f"{part['id']}|{r['planned_id']}" for r in recs]
        ys += [0 if r["class"] == "car" else 1 for r in recs]
    beats, mfcc = featurize(np.concatenate(waves))
    digest = save_npz(out / "rc.npz", ids=np.array(ids), BEATs768=beats, MFCC26=mfcc, y=np.array(ys))
    (out / "summary.json").write_bytes(json_bytes({"stage": "features_rc", "rc_sha256": digest, "bundles": bundles,
                                                   "elapsed_s": time.perf_counter() - started,
                                                   "provenance": provenance(resolve("repo:."))}))
    print(json.dumps({"rc_sha256": digest, "elapsed_s": time.perf_counter() - started}))


def load_features():
    base = resolve(RUN) / "learning"
    return {name: np.load(base / folder / f"{name}.npz", allow_pickle=False)
            for name, folder in (("real", "features_base"), ("ra", "features_base"),
                                 ("ru", "features_base"), ("rc", "features_rc"))}


def role_inputs(part, feats, rep):
    real_index = {i: k for k, i in enumerate(feats["real"]["ids"])}
    pick = [real_index[i] for i in part["train_ids"]]
    real = (feats["real"][rep][pick], feats["real"]["y"][pick])
    def select(name, prefix):
        mask = np.char.startswith(feats[name]["ids"].astype(str), prefix)
        return feats[name][rep][mask], feats[name]["y"][mask]
    seed = part["analysis"].removeprefix("utility_s")
    return real, select("ra", part["id"] + "|"), select("ru", f"s{seed}|"), select("rc", part["id"] + "|")


def heads():
    started, cfg = time.perf_counter(), h1_config()
    parts, _ = design()
    feats = load_features()
    out = resolve(RUN) / "learning" / "heads"
    (out / "models").mkdir(parents=True, exist_ok=False)
    manifest = []
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for rep in REPS:
            for part in parts:
                real, ra, ru, rc = role_inputs(part, feats, rep)
                for arm in ARMS:
                    x, y = training_rows(arm, real, ra, ru, rc)
                    model = fixed_fit(x.astype(np.float64), y, cfg["classifier"])
                    name = f"{rep}__{arm}__{part['id']}"
                    path = out / "models" / f"{name}.joblib"
                    joblib.dump(model, path, compress=3)
                    manifest.append({"head": name, "representation": rep, "arm": arm, "partition": part["id"],
                                     "rows": int(len(y)), "n_iter": model[1].n_iter_.tolist(), "sha256": sha256(path)})
    (out / "manifest.json").write_bytes(json_bytes(manifest))
    (out / "summary.json").write_bytes(json_bytes({"heads": len(manifest), "elapsed_s": time.perf_counter() - started,
                                                   "manifest_sha256": sha256(out / "manifest.json"),
                                                   "provenance": provenance(resolve("repo:."))}))
    print(json.dumps({"heads": len(manifest), "elapsed_s": time.perf_counter() - started}))


def score():
    started = time.perf_counter()
    parts, sources = design()
    feats = load_features()
    base = resolve(RUN) / "learning"
    manifest = json.loads((base / "heads" / "manifest.json").read_text())
    if sha256(base / "heads" / "manifest.json") != json.loads((base / "heads" / "summary.json").read_text())["manifest_sha256"]:
        raise ValueError("Head manifest changed after fitting")
    out = base / "scores"
    out.mkdir(parents=True, exist_ok=False)
    real_index = {i: k for k, i in enumerate(feats["real"]["ids"])}
    by_part = {p["id"]: p for p in parts}
    per_head, predictions = [], {}
    for entry in manifest:
        path = base / "heads" / "models" / f"{entry['head']}.joblib"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Head changed: {entry['head']}")
        part = by_part[entry["partition"]]
        held = [real_index[i] for i in part["held_ids"]]
        model = joblib.load(path)
        p = model.predict_proba(feats["real"][entry["representation"]][held].astype(np.float64))[:, 1]
        y = feats["real"]["y"][held]
        predictions[entry["head"]] = p
        per_head.append({**entry, "held_group": part["held_groups"][0], "held_n": int(len(y)),
                         "held_trucks": int(y.sum()), **head_metrics(y, p)})
    np.savez_compressed(out / "predictions.npz", **predictions)
    group_site = {r["provenance_group_id"]: r["site_id"] for r in sources.values()}
    fold_site = {p["fold"]: group_site[p["held_groups"][0]] for p in parts}
    result = {"per_head": per_head, "statistics": statistics(per_head, parts, fold_site, DRAWS, BOOT_SEED),
              "elapsed_s": time.perf_counter() - started, "provenance": provenance(resolve("repo:."))}
    (out / "results.json").write_bytes(json_bytes(result))
    print(json.dumps({"heads_scored": len(per_head), "results_sha256": sha256(out / "results.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features-base", "features-rc", "heads", "score"))
    stage = parser.parse_args().stage
    {"features-base": features_base, "features-rc": features_rc, "heads": heads, "score": score}[stage]()


if __name__ == "__main__":
    main()
