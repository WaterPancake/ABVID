"""Bounded, create-only source-training reconstruction diagnosis; no fitting.

Contract: experiments/cast_utility/stage_diagnostic/PROTOCOL.md.
Numerical code is loaded lazily in its pinned CAST or baseline environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from abvid.cast.grouped_storage import read_bundle, verify_bundle, write_bundle
from abvid.paths import repository_root, resolve, roots
from abvid.provenance.hashing import canonical, sha256


PROTOCOL = "repo:experiments/cast_utility/stage_diagnostic/PROTOCOL.md"
SCRIPT = "repo:src/abvid/cast/stage_diagnostic.py"
INPUT = "artifact:cast/utility/cast5a_run_20261011_v1"
ROLE = "utility_s42_0"
REPS = ("BEATs768", "MFCC26")
CAST_ENV = {"python": "3.11.13", "numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.13.0", "soundfile": "0.14.0"}
BASE_ENV = {"python": "3.11.13", "numpy": "1.26.4", "scipy": "1.17.1", "torch": "2.11.0", "soundfile": "0.13.1", "scikit-learn": "1.4.2", "librosa": "0.11.0"}
FEATURE_HASHES = {"real": "6e1a4b84b86c56389afec7ac95032b37ab6a1df0891ac8bff979f53a534e5fd2",
                  "ra": "da0368174cdb8e76c5550e391b586cc09503ae7ff74fa8c91122f070f97935be",
                  "rc": "27e2a4a6bec915c1a6c36b3350e90118c0a9805aacc42cda9092ab68b8166413"}
HEAD_MANIFEST_SHA = "9315e73b7f74f1bce0685df298982bc490e74a8c1492613efcdfb1ee0242db61"


def encoded(x):
    return json.dumps(x, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def npy(x):
    stream = BytesIO()
    np.save(stream, np.asarray(x), allow_pickle=False)
    return stream.getvalue()


def npz(**values):
    stream = BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def wave_hash(x):
    return hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest()


def environment(expected):
    actual = {"python": platform.python_version(), **{k: version(k) for k in expected if k != "python"}}
    if actual != expected:
        raise ValueError(f"Wrong numerical environment: {actual}")
    return actual


def destination(reference):
    path = resolve(reference, must_exist=False)
    if (not reference.startswith("artifact:cast/diagnostics/") or
            path == roots()["artifact"] or path.is_relative_to(roots()["repo"]) or
            path.is_relative_to(roots()["archive"])):
        raise ValueError("Use a new external artifact:cast/diagnostics/ run")
    return path


def provenance():
    return {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree": subprocess.check_output(["git", "status", "--short"], text=True).splitlines()}


def feature_path(name):
    folder = "features_rc" if name == "rc" else "features_base"
    return resolve(INPUT) / "learning" / folder / f"{name}.npz"


def check_refs(refs):
    for ref, expected in refs.items():
        if sha256(resolve(ref)) != expected:
            raise ValueError(f"Pinned input changed: {ref}")


def design(run):
    folder = destination(run) / "design"
    metadata, payload = read_bundle(folder)
    d = json.loads(payload["design.json"])
    if d["id"] != "cast_stage_diagnosis_v1" or d["role"] != ROLE:
        raise ValueError("Unexpected diagnosis design")
    check_refs(d["references_sha256"])
    if payload["protocol.md"] != resolve(PROTOCOL).read_bytes():
        raise ValueError("Diagnosis protocol changed")
    parents = [json.loads(line) for line in payload["parents.jsonl"].splitlines()]
    if [p["file_id"] for p in parents] != d["train_ids"] or len(parents) != 380:
        raise ValueError("Training parent inventory changed")
    return d, parents, verify_bundle(folder)["identity_sha256"]


def freeze(run):
    environment(CAST_ENV)
    from abvid.protocol import load
    from abvid.cast.utility_design import load_context, _cast4a_context
    from abvid.cast.grouped_parents import _read_parent
    c = load_context()
    c4 = _cast4a_context()
    part = next(p for p in c["partitions"] if p["id"] == ROLE)
    role = c["roles"][ROLE]
    if part["train_counts"] != {"car": 190, "truck": 190} or len(part["training_group_counts"]) != 5:
        raise ValueError("Wrong training budget/groups")
    if not set(part["train_ids"]).issubset(c["parents"]["from_cast4a_run"]):
        raise ValueError("This bounded role requires its already completed 4A parents")
    if set(part["train_ids"]) & set(part["held_ids"]):
        raise ValueError("Training and held IDs intersect")
    refs = {}
    for ref in (PROTOCOL, SCRIPT, "repo:src/abvid/cast/utility_gap_audit.py",
                "repo:src/abvid/representations/beats.py", "repo:src/abvid/representations/mfcc.py",
                "repo:src/abvid/learning/utility.py", "repo:src/abvid/data/augmentation.py",
                "archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json",
                "archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json",
                INPUT + "/design/_bundle.json",
                "repo:experiments/cast_coverage/grouped_coverage/execution/config.json"):
        refs[ref] = sha256(resolve(ref))
    for name in ("h1_baselines", "cast_grouped_coverage"):
        ref = f"repo:configs/experiments/{name}.json"
        load(resolve(ref), check_files=True)
        refs[ref] = sha256(resolve(ref))
    manifests = json.loads((resolve(INPUT) / "learning/heads/manifest.json").read_text())
    head_ref = INPUT + "/learning/heads/manifest.json"
    if sha256(resolve(head_ref)) != HEAD_MANIFEST_SHA:
        raise ValueError("Original head manifest changed")
    refs[head_ref] = HEAD_MANIFEST_SHA
    head_refs = {}
    for h in manifests:
        if h["partition"] == ROLE and h["arm"] in ("R", "C"):
            ref = INPUT + f"/learning/heads/models/{h['head']}.joblib"
            refs[ref] = h["sha256"]
            head_refs[f"{h['representation']}__{h['arm']}"] = ref
    if len(head_refs) != 4:
        raise ValueError("Missing frozen heads")
    for name in ("real", "ra", "rc"):
        path = feature_path(name)
        expected = FEATURE_HASHES[name]
        if sha256(path) != expected:
            raise ValueError("Feature cache differs from the published 5A identity")
        refs[INPUT + "/learning/" + path.relative_to(resolve(INPUT) / "learning").as_posix()] = sha256(path)
    cfg = json.loads(resolve("archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json").read_text())
    h1_lock = json.loads(resolve("archive:experiments/h1/frozen/h1_common_budget_v1.3/lock.json").read_text())
    if refs["archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json"] != h1_lock["artifacts_sha256"]["config.json"]:
        raise ValueError("H1 configuration hash mismatch")
    checkpoint = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
    refs[checkpoint] = cfg["encoder"]["checkpoint_sha256"]
    prov_ref = "artifact:models/beats/provenance.json"
    refs[prov_ref] = sha256(resolve(prov_ref))
    for name, entry in json.loads(resolve(prov_ref).read_text())["code"].items():
        refs["artifact:models/beats/upstream/" + name] = entry["sha256"]
    parents = []
    for fid in part["train_ids"]:
        role.validate_parent(c["sources"][fid])
        _, payload = _read_parent(c4, fid, c4["roles"]["group_0"])
        parent_ref = "artifact:" + (c4["output"] / "parents" / fid / "_bundle.json").relative_to(roots()["artifact"]).as_posix()
        refs[parent_ref] = sha256(resolve(parent_ref))
        stages = {stage: json.loads(payload[stage + ".json"]) for stage in ("smooth8", "v7", "v8", "v9", "v10", "v11")}
        audit = c4["readiness_records"][fid]
        comp_ref = c4["chain"]["stages"]["smooth8"] + f"/fits/{fid}/components.npz"
        refs[comp_ref] = audit["checked_hashes"][comp_ref]
        parents.append({"file_id": fid, "source": c["sources"][fid], "parent_manifest": parent_ref,
                        "smooth8_components": comp_ref,
                        "parameters": {stage: row["parameters"] for stage, row in stages.items()},
                        "flags": {stage: {k: v if k == "flags" else v.get("flags", [])
                                          for k, v in row.items() if k == "flags" or k.endswith("_calibration")}
                                  for stage, row in stages.items()},
                        "stage_row_sha256": {stage: digest(row) for stage, row in stages.items()}})
    subset, controls = [], []
    for group in sorted(part["training_group_counts"]):
        for label in ("car", "truck"):
            ids = sorted(p["file_id"] for p in parents if p["source"]["provenance_group_id"] == group and p["source"]["canonical_class"] == label)
            if len(ids) < 5:
                raise ValueError("Insufficient source-group/class support")
            subset += ids[:5]
            controls += ids[:1]
    configs = c["numerical"]["renderer_configs"]
    d = {"id": "cast_stage_diagnosis_v1", "role": ROLE, "input_run": INPUT,
         "dataset_version": "IDMT_V1", "domain": "source_training_only", "train_ids": part["train_ids"],
         "held_ids_for_exclusion_only": part["held_ids"], "secondary_ids": subset, "control_ids": controls,
         "expected_new_renders": 2680, "new_fits": 0, "new_heads": 0,
         "head_refs": head_refs, "checkpoint": checkpoint, "references_sha256": refs,
         "cast_environment": CAST_ENV, "baseline_environment": BASE_ENV, "h1_config": cfg,
         "renderer_configs": configs, "group_counts": part["training_group_counts"],
         "provenance": provenance()}
    check_refs(refs)
    payload = {"design.json": encoded(d), "parents.jsonl": b"".join(encoded_line(p) for p in parents),
               "protocol.md": resolve(PROTOCOL).read_bytes()}
    for ref in refs:
        if ref.startswith("repo:") and resolve(ref).suffix in (".py", ".md", ".json"):
            payload["snapshot/" + ref.removeprefix("repo:")] = resolve(ref).read_bytes()
    out = write_bundle(destination(run) / "design", payload,
                       {"kind": "cast_stage_diagnostic_design", "scope": "source_training_only_no_new_numerical_scores"})
    print(json.dumps({"stage": "freeze", "identity": out["identity_sha256"], "parents": len(parents), "secondary": len(subset), "new_renders": 2680}), flush=True)


def encoded_line(x):
    return canonical(x) + b"\n"


def stage_plan(d):
    plan = []
    for stage in ("smooth8", "v11"):
        for realization in (0, 1):
            plan.append((f"{stage}_check_{realization}", stage, "check", realization, d["train_ids"]))
    for realization in (0, 1):
        plan.append((f"v11_fresh_{realization}", "v11", "fresh", realization, d["train_ids"]))
    for stage in ("v7", "v8", "v9", "v10"):
        for realization in (0, 1):
            plan.append((f"{stage}_check_{realization}", stage, "check", realization, d["secondary_ids"]))
    return plan


def get_renderer(stage, d):
    from abvid.cast.population.engine import SmoothRenderer
    from abvid.cast.population.resolution_model import ResolutionRenderer
    key = "spectrum16_temporal41_v11" if stage == "v11" else "smooth8_temporal41_v10" if stage == "v10" else "smooth8_v1"
    return (ResolutionRenderer if stage == "v11" else SmoothRenderer), d["renderer_configs"][key]


def shape_array(x):
    from abvid.data.augmentation import shape
    return shape(np.asarray(x)).astype(np.float32)


def render_one(parent, stage, purpose, realization, d):
    import torch
    from abvid.cast.renderer import tensors, seed_for
    cls, cfg = get_renderer(stage, d)
    fid = parent["file_id"]
    rid = fid if purpose == "check" else f"cast_stage_diagnosis_v1:{fid}:fresh:{realization}"
    renderer = cls(cfg, rid, sampling=(purpose == "fresh"))
    noise_purpose = "check" if purpose == "check" else "sampling"
    with torch.no_grad():
        raw = renderer.render(tensors(parent["parameters"][stage]), noise_purpose)[0].numpy()
    index = realization if purpose == "check" else 0
    wave = shape_array(raw[index])
    if wave.shape != (32000,) or not np.isfinite(wave).all():
        raise ValueError("Invalid rendered observation")
    seeds = {"phase": seed_for(cfg, rid, "phase", "shared" if purpose == "check" else "sampling"),
             "noise": seed_for(cfg, rid, "noise", noise_purpose, index), "renderer_input_id": rid}
    return wave, raw, seeds


def render(run):
    environment(CAST_ENV)
    import torch
    from abvid.cast.grouped_parents import _configure_cpu, read_source
    from abvid.cast.utility_design import _cast4a_context, load_context
    from abvid.cast.audio import normalize, observation
    d, parents, identity = design(run)
    _configure_cpu()
    by_id = {p["file_id"]: p for p in parents}
    c, c4 = load_context(), _cast4a_context()
    role = c["roles"][ROLE]
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        wave_shas = dict(zip(ids, cache["wave_sha256"].astype(str)))
    originals, original_records = [], []
    for i, parent in enumerate(parents):
        fid = parent["file_id"]
        role.validate_parent(parent["source"])
        raw, sr = read_source(c4, fid)
        target, _ = normalize(observation(raw, sr, d["renderer_configs"]["smooth8_v1"]), d["renderer_configs"]["smooth8_v1"])
        wave = target.numpy()
        if wave_hash(wave) != wave_shas[fid]:
            raise ValueError(f"Original profile/hash drift: {fid}")
        original_records.append({"file_id": fid, "wave_sha256": wave_hash(wave), "source_sha256": parent["source"]["source_file_sha256"]})
        if fid in d["control_ids"]:
            originals.append((fid, wave))
    control_map = dict(originals)
    write_bundle(destination(run) / "render" / "original_controls",
                 {"waves.npy": npy(np.stack([control_map[i] for i in d["control_ids"]])),
                  "ids.json": encoded(d["control_ids"]), "original_hash_checks.json": encoded(original_records)},
                 {"design_identity": identity, "kind": "original_profile_replay_380_controls_10"})
    for name, stage, purpose, realization, ids in stage_plan(d):
        started = time.perf_counter()
        waves, records = [], []
        for fid in ids:
            parent = by_id[fid]
            role.validate_parent(parent["source"])
            wave, raw, seeds = render_one(parent, stage, purpose, realization, d)
            if stage == "smooth8":
                with np.load(resolve(parent["smooth8_components"]), allow_pickle=False) as saved:
                    expected = saved["check_waveforms"]
                if not np.array_equal(raw, expected[0]):
                    raise ValueError(f"Smooth8 raw checking replay differs: {fid}")
            waves.append(wave)
            records.append({"file_id": fid, "stage": stage, "purpose": purpose, "realization": realization,
                            "source_class": parent["source"]["canonical_class"], "group": parent["source"]["provenance_group_id"],
                            "wave_sha256": wave_hash(wave), "stage_row_sha256": parent["stage_row_sha256"][stage], "seeds": seeds})
        write_bundle(destination(run) / "render" / name,
                     {"waves.npy": npy(np.stack(waves)), "ids.json": encoded(ids), "records.json": encoded(records)},
                     {"design_identity": identity, "kind": "conditional_parent_diagnostic", "name": name})
        print(json.dumps({"stage": "render", "name": name, "waves": len(ids), "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    check_refs(d["references_sha256"])
    print(json.dumps({"stage": "render_complete", "new_renders": 2680}), flush=True)


def features(run):
    environment(BASE_ENV)
    import torch
    from abvid.representations.beats import load_encoder, extract_embeddings
    from abvid.representations.mfcc import mfcc_features
    d, _, identity = design(run)
    cfg = d["h1_config"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, encoder_provenance = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve(d["checkpoint"]))))
    plan = [("original_controls", None, None, None, d["control_ids"]), *stage_plan(d)]
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        index = {str(i): k for k, i in enumerate(cache["ids"])}
        real_features = {rep: cache[rep][[index[i] for i in d["control_ids"]]] for rep in REPS}
    for name, *_ in plan:
        started = time.perf_counter()
        meta, payload = read_bundle(destination(run) / "render" / name)
        if meta["design_identity"] != identity:
            raise ValueError("Rendered stage belongs to another design")
        ids = json.loads(payload["ids.json"])
        waves = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
        b = []
        for offset in range(0, len(waves), 8):
            b.append(extract_embeddings(encoder, torch.from_numpy(waves[offset:offset + 8])).numpy())
            if offset % 64 == 0:
                print(json.dumps({"stage": "features", "name": name, "complete": min(offset + 8, len(waves)), "total": len(waves)}), flush=True)
        arrays = {"BEATs768": np.concatenate(b).astype(np.float32),
                  "MFCC26": np.stack([mfcc_features(w.astype(np.float64), cfg["mfcc"]) for w in waves])}
        if any(not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("Nonfinite new features")
        control_checks = {}
        if name == "original_controls":
            for rep in REPS:
                delta = np.abs(arrays[rep] - real_features[rep])
                control_checks[rep] = {"max_abs_error": float(delta.max()), "exact": bool(np.array_equal(arrays[rep], real_features[rep])),
                                       "atol": 1e-5, "rtol": 1e-5}
                if not np.allclose(arrays[rep], real_features[rep], atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Original feature control failed: {rep}, {control_checks[rep]}")
        write_bundle(destination(run) / "features" / name,
                     {"features.npz": npz(ids=np.array(ids), **arrays), "control_checks.json": encoded(control_checks),
                      "row_hashes.json": encoded([{ "file_id": fid, **{rep: wave_hash(arrays[rep][i]) for rep in REPS}}
                                                  for i, fid in enumerate(ids)])},
                     {"design_identity": identity, "render_identity": verify_bundle(destination(run) / "render" / name)["identity_sha256"],
                      "kind": "frozen_source_features", "name": name, "encoder_provenance_sha256": digest(encoder_provenance),
                      "environment": BASE_ENV})
        print(json.dumps({"stage": "features_done", "name": name, "count": len(waves), "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    check_refs(d["references_sha256"])


def load_arrays(run, name, identity):
    metadata, payload = read_bundle(destination(run) / "features" / name)
    if metadata["design_identity"] != identity:
        raise ValueError("Feature bundle belongs to another frozen design")
    with np.load(BytesIO(payload["features.npz"]), allow_pickle=False) as value:
        arrays = {key: value[key] for key in value.files}
    records = json.loads(payload["row_hashes.json"])
    if [r["file_id"] for r in records] != arrays["ids"].tolist():
        raise ValueError("Feature IDs differ from row-hash inventory")
    for i, record in enumerate(records):
        if any(wave_hash(arrays[rep][i]) != record[rep] for rep in REPS):
            raise ValueError("Feature row hash differs")
    return arrays


def cached_role(d, name):
    with np.load(feature_path(name), allow_pickle=False) as value:
        all_ids = value["ids"].astype(str)
        if name == "real":
            index = {i: k for k, i in enumerate(all_ids)}
            picks = np.array([index[i] for i in d["train_ids"]])
        elif name == "ra":
            index = {i: k for k, i in enumerate(all_ids)}
            picks = np.array([index[ROLE + "|" + i] for i in d["train_ids"]])
        else:
            picks = np.flatnonzero(np.char.startswith(all_ids, ROLE + "|"))
        arrays = {key: value[key][picks] for key in (*REPS, "y")}
    if np.bincount(arrays["y"], minlength=2).tolist() != [190, 190]:
        raise ValueError(f"Wrong cached-role budget: {name}")
    return arrays


def calculate(run):
    import joblib
    from abvid.learning.utility import head_metrics
    from abvid.cast.utility_gap_audit import geometry
    d, parents, identity = design(run)
    by_id = {p["file_id"]: p for p in parents}
    original, ra, population = (cached_role(d, name) for name in ("real", "ra", "rc"))
    real_index = {i: k for k, i in enumerate(d["train_ids"])}
    y = original["y"]
    expected = np.array([by_id[i]["source"]["class_id"] for i in d["train_ids"]])
    if not np.array_equal(y, expected) or not np.array_equal(y, ra["y"]):
        raise ValueError("Source/cache label order mismatch")
    models = {key: joblib.load(resolve(ref)) for key, ref in d["head_refs"].items()}
    stages = {name: load_arrays(run, name, identity) for name, *_ in stage_plan(d)}
    rows = []
    scopes = {"all380": d["train_ids"], "balanced50": d["secondary_ids"]}
    for rep in REPS:
        for scope, wanted in scopes.items():
            pick = np.array([real_index[i] for i in wanted])
            reference, labels = original[rep][pick].astype(np.float64), y[pick]
            standardized = models[rep + "__R"][0].transform(reference)
            inputs = {"original": (reference, labels, True, wanted), "ra": (ra[rep][pick].astype(np.float64), labels, True, wanted)}
            if scope == "all380":
                inputs["population"] = (population[rep].astype(np.float64), population["y"], False, None)
            for name, values in stages.items():
                index = {str(i): k for k, i in enumerate(values["ids"])}
                if not set(wanted).issubset(index):
                    continue
                inputs[name] = (values[rep][[index[i] for i in wanted]].astype(np.float64), labels, True, wanted)
            for name, (x, label, paired, ids) in inputs.items():
                scaled = models[rep + "__R"][0].transform(x)
                geom = geometry(standardized, labels, scaled, label)
                if paired:
                    geom["paired_standardized_feature_rmse"] = float(np.sqrt(np.mean((scaled - standardized) ** 2)))
                group_names = sorted({by_id[i]["source"]["provenance_group_id"] for i in wanted}) if paired else []
                for arm in ("R", "C"):
                    model = models[rep + "__" + arm]
                    p = model.predict_proba(x)[:, 1]
                    metrics = head_metrics(label, p)
                    signed = (2 * label - 1) * model.decision_function(x)
                    record = {"representation": rep, "scope": scope, "input": name, "head": arm,
                              "count": len(label), "class_counts": np.bincount(label, minlength=2).tolist(),
                              "metrics": metrics, "signed_margin_mean": float(signed.mean()), "geometry": geom,
                              "class_signed_margin_mean": {str(c): float(signed[label == c].mean()) for c in (0, 1)}, "groups": []}
                    if paired:
                        op = model.predict_proba(reference)[:, 1]
                        om = (2 * labels - 1) * model.decision_function(reference)
                        record["original_prediction_agreement"] = float(np.mean((p > .5) == (op > .5)))
                        record["paired_signed_margin_change"] = float(np.mean(signed - om))
                        for group in group_names:
                            mask = np.array([by_id[i]["source"]["provenance_group_id"] == group for i in wanted])
                            record["groups"].append({"group": group, "class_counts": np.bincount(label[mask], minlength=2).tolist(),
                                                     "metrics": head_metrics(label[mask], p[mask])})
                    rows.append(record)
    _, control_payload = read_bundle(destination(run) / "features" / "original_controls")
    result = {"id": "cast_stage_diagnosis_v1", "design_identity": identity, "scope": "source_training_only_post_hoc",
              "dataset_version": "IDMT_V1", "role": ROLE, "selection_seed": 42, "parent_count": 380,
              "checkpoint_sha256": d["h1_config"]["encoder"]["checkpoint_sha256"],
              "original_feature_controls": json.loads(control_payload["control_checks.json"]), "rows": rows,
              "new_render_count": 2680, "new_model_fits": 0, "target_access": False,
              "limits": ["Training diagnostics; no independent generalization or physical attribution.",
                         "Checking realizations and five conservative groups are not independent sessions.",
                         "Fresh randomness changes phase and noise jointly.",
                         "Population comparison changes parameter sampling and sample composition; no parent pairing."],
              "references_sha256": d["references_sha256"]}
    return result


def score(run):
    environment(BASE_ENV)
    result = calculate(run)
    write_bundle(destination(run) / "scores", {"results.json": encoded(result)},
                 {"kind": "fixed_head_source_training_diagnostics", "environment": BASE_ENV, "provenance": provenance()})
    print(json.dumps({"stage": "score", "rows": len(result["rows"]), "output": run + "/scores/results.json"}), flush=True)


def verify(run):
    environment(CAST_ENV)
    from abvid.cast.grouped_parents import _configure_cpu
    d, parents, identity = design(run)
    _configure_cpu()
    by_id = {p["file_id"]: p for p in parents}
    checked_waves = 0
    for name, stage, purpose, realization, ids in stage_plan(d):
        metadata, payload = read_bundle(destination(run) / "render" / name)
        if metadata["design_identity"] != identity or json.loads(payload["ids.json"]) != ids:
            raise ValueError("Render design/order mismatch")
        waves = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
        records = json.loads(payload["records.json"])
        for k, fid in enumerate(ids):
            actual, _, seeds = render_one(by_id[fid], stage, purpose, realization, d)
            if not np.array_equal(actual, waves[k]) or wave_hash(actual) != records[k]["wave_sha256"] or seeds != records[k]["seeds"]:
                raise ValueError(f"Waveform replay differs: {name}/{fid}")
            checked_waves += 1
        _, feats = read_bundle(destination(run) / "features" / name)
        with np.load(BytesIO(feats["features.npz"]), allow_pickle=False) as a:
            if a["ids"].tolist() != ids or any(not np.isfinite(a[rep]).all() for rep in REPS):
                raise ValueError("Feature order/finite check failed")
        print(json.dumps({"stage": "verify", "name": name, "checked": checked_waves}), flush=True)
    read_bundle(destination(run) / "render/original_controls")
    read_bundle(destination(run) / "features/original_controls")
    read_bundle(destination(run) / "scores")
    check_refs(d["references_sha256"])
    result = {"waveforms_exact": checked_waves, "all_input_hashes_unchanged": True,
              "feature_containers_verified": 15, "numerical_score_recompute": "requires baseline verify-scores"}
    write_bundle(destination(run) / "verification", {"waveforms.json": encoded(result)},
                 {"kind": "exact_waveform_replay", "environment": CAST_ENV})
    print(json.dumps(result), flush=True)


def verify_scores(run):
    environment(BASE_ENV)
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    import joblib
    actual = calculate(run)
    _, payload = read_bundle(destination(run) / "scores")
    if encoded(actual) != payload["results.json"]:
        raise ValueError("Numerical result replay differs")
    d, parents, identity = design(run)
    labels = np.array([p["source"]["class_id"] for p in parents])
    by_id = {p["file_id"]: i for i, p in enumerate(parents)}
    checks = 0
    for name, *_ in stage_plan(d):
        arrays = load_arrays(run, name, identity)
        y = labels[[by_id[str(i)] for i in arrays["ids"]]]
        for rep in REPS:
            model = joblib.load(resolve(d["head_refs"][rep + "__R"]))
            p = model.predict_proba(arrays[rep].astype(np.float64))[:, 1]
            row = next(r for r in actual["rows"] if r["scope"] == ("all380" if len(y) == 380 else "balanced50") and r["input"] == name and r["representation"] == rep and r["head"] == "R")
            if abs(roc_auc_score(y, p) - row["metrics"]["auroc"]) > 1e-12 or abs(balanced_accuracy_score(y, p > .5) - row["metrics"]["balanced_accuracy"]) > 1e-12:
                raise ValueError("Independent metric calculation differs")
            checks += 1
    result = {"numerical_results_exact": True, "independent_sklearn_metric_pairs": checks,
              "input_hashes_unchanged": True}
    write_bundle(destination(run) / "verification_scores", {"scores.json": encoded(result)},
                 {"kind": "numerical_score_recompute_and_independent_metrics", "environment": BASE_ENV})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "render", "features", "score", "verify", "verify-scores"))
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    # A missing stage is resumable; already published stages are never replaced.
    action = {"freeze": freeze, "render": render, "features": features, "score": score,
              "verify": verify, "verify-scores": verify_scores}[args.stage]
    try:
        action(args.run)
    except Exception as error:
        root = destination(args.run)
        if root.exists():
            failure = root / f"failure_{args.stage}.json"
            if not failure.exists():
                failure.write_bytes(encoded({"stage": args.stage, "type": type(error).__name__, "message": str(error),
                                             "provenance": provenance(), "replacement_allowed": False}))
        raise


if __name__ == "__main__":
    main()
