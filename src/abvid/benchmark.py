"""Runnable real-data ABVID prototype consuming the frozen H1 contract.

This reproduces the existing real/BEATs fits, exports numeric heads (no pickle),
and provides WAV inference. It does not select models, tune settings, or run CAST.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.special import expit

from abvid.data.audio import observation
from abvid.data.splits import assert_separation
from abvid.evaluation.baselines import cm_metrics, metrics
from abvid.paths import repository_root, resolve, roots
from abvid.protocol import load as load_registry
from abvid.provenance.hashing import sha256


PROTOTYPE_ID = "abvid_real_beats_v0_1"
H1 = "archive:experiments/h1/"
FREEZE = H1 + "frozen/h1_common_budget_v1.3/"
CACHE = H1 + "cache/h1_common_budget_v1.3/"
RESULTS = H1 + "results/H1_20261002/"
LOCK_SHA = "3e427a2f4ee084ecdd4102dd3a9863a55cc06496abdb9688258ec26cbaaf901e"
RESULTS_SHA = "71f2901a56ed49297e81353885f611dcdbe4623e7ee808b86dc2cc4c475a5bd3"
CHECKPOINT = "artifact:models/beats/BEATs_iter3_plus_AS2M.pt"
DEFAULT_CONFIG = "repo:configs/experiments/h1_baselines.json"
DEFAULT_OUTPUT = "artifact:benchmarks/abvid_real_beats_v0_1"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked(reference, expected):
    path = resolve(reference)
    if sha256(path) != expected:
        raise ValueError(f"Changed frozen input: {reference}")
    return path


def parameters(model):
    """Export only the fitted scaler and binary linear decision function."""
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected class order car=0, truck=1")
    return dict(mean=model[0].mean_.copy(), scale=model[0].scale_.copy(),
                coef=model[1].coef_[0].copy(), intercept=model[1].intercept_[0].copy())


def probabilities(head, features):
    x = np.asarray(features, dtype=np.float64)
    mean, scale, coef = (np.asarray(head[k]) for k in ("mean", "scale", "coef"))
    bias = np.asarray(head["intercept"])
    if (x.ndim != 2 or x.shape[1] != 768 or not np.isfinite(x).all()
            or any(a.shape != (768,) for a in (mean, scale, coef))
            or bias.shape != () or not np.isfinite(bias)
            or not all(np.isfinite(a).all() for a in (mean, scale, coef))
            or not (scale > 0).all()):
        raise ValueError("Invalid 768D features or linear head")
    truck = expit(((x - mean) / scale) @ coef + bias)
    return np.column_stack((1 - truck, truck))


def validate_roles(train_ids, test_ids, rows, held_out_group, budget):
    """Fail before fitting if IDs, known provenance, roles, or counts disagree."""
    for ids in (train_ids, test_ids):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate split IDs")
        if any(rows[i]["dataset_id"] != "IDMT" or rows[i]["class_id"] not in (0, 1)
               or not rows[i]["admitted_for_h1"] for i in ids):
            raise ValueError("Only admitted IDMT car/truck data are allowed")
    if any(not rows[i]["admitted_for_training"] for i in train_ids):
        raise ValueError("Unadmitted training record")
    if {rows[i]["provenance_group_id"] for i in test_ids} != {held_out_group}:
        raise ValueError("Test records do not match the frozen held-out group")
    assert_separation(train_ids, test_ids, rows)
    unknown = (None, "", "unknown")
    sessions = lambda ids: {rows[i].get("recording_session") for i in ids
                            if rows[i].get("recording_session") not in unknown}
    if sessions(train_ids) & sessions(test_ids):
        raise ValueError("recording_session leakage")
    if np.bincount([rows[i]["class_id"] for i in train_ids], minlength=2).tolist() != [budget, budget]:
        raise ValueError("Training budget differs from the frozen contract")


def frozen_inputs(config_reference):
    """Validate the canonical registry, then its already-executed H1 freeze."""
    registry_path = resolve(config_reference)
    registry = load_registry(registry_path, check_files=True)
    if registry["id"] != "h1_baselines" or registry["phase"] != "baseline":
        raise ValueError("This prototype only consumes the frozen H1 baseline")
    lock_path = checked(FREEZE + "lock.json", LOCK_SHA)
    lock = read(lock_path)
    for name, digest in lock["artifacts_sha256"].items():
        checked(FREEZE + name, digest)
    config = read(resolve(FREEZE + "config.json"))
    original = read(resolve(registry["contract"]["path"]))
    original["admission"]["sha256"] = lock["artifacts_sha256"]["admitted.jsonl"]
    if config != original:
        raise ValueError("Registry and executed scientific contract disagree")
    if config["classes"] != ["car", "truck"] or not config["encoder"]["frozen"]:
        raise ValueError("Expected frozen BEATs car/truck protocol")
    versions = {name: version(name) for name in lock["versions"]}
    if versions != lock["versions"]:
        raise ValueError(f"Use the pinned baseline environment; found {versions}")
    cache_summary = read(resolve(CACHE + "summary.json"))
    if cache_summary["lock_sha256"] != LOCK_SHA:
        raise ValueError("Feature cache belongs to another split freeze")
    summary = read(checked(RESULTS + "summary.json", RESULTS_SHA))
    if summary["lock_sha256"] != LOCK_SHA:
        raise ValueError("Reference results belong to another split freeze")
    feature_path = checked(CACHE + "features.npz", summary["feature_cache_sha256"])
    if cache_summary["artifacts_sha256"]["features.npz"] != summary["feature_cache_sha256"]:
        raise ValueError("Cache provenance mismatch")
    checked(RESULTS + "metrics.json", summary["artifacts_sha256"]["metrics.json"])
    rows = {r["file_id"]: r for r in map(json.loads, resolve(FREEZE + "admitted.jsonl").read_text().splitlines())}
    with np.load(feature_path, allow_pickle=False) as data:
        ids = data["file_ids"].tolist()
        if ids != lock["feature_ids"] or len(ids) != len(set(ids)):
            raise ValueError("Cache order/IDs differ from the frozen manifest")
        features = data["BEATs768"].copy()
        wave_hashes = data["waveform_sha256"].tolist()
    if features.shape != (len(ids), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid BEATs cache")
    return config, lock, rows, ids, features, wave_hashes, summary, versions


def audio_features(paths, config):
    """Run the saved 2-second observation and frozen encoder on local audio."""
    import soundfile as sf
    import torch
    from abvid.representations.beats import extract_embeddings, load_encoder

    torch.set_num_threads(config["hardware"]["torch_threads"])
    torch.manual_seed(42)
    encoder_config = dict(config["encoder"], checkpoint=str(resolve(CHECKPOINT)))
    encoder, provenance = load_encoder(encoder_config)
    waves, windows = [], []
    for path in paths:
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        waves.append(observation(audio, rate, config["preprocessing"]))
        windows.append(dict(input_sample_rate_hz=rate, input_channels=audio.shape[1],
                            input_duration_s=len(audio) / rate,
                            window_start_s=max(0., (len(audio) / rate - 2.) / 2),
                            window_duration_s=2.0))
    batches = []
    size = config["hardware"]["batch_size"]
    for start in range(0, len(waves), size):
        batch = torch.from_numpy(np.stack(waves[start:start + size]))
        batches.append(extract_embeddings(encoder, batch).numpy())
    return np.concatenate(batches), waves, windows, provenance


def load_bundle(folder):
    folder = Path(folder)
    meta = read(folder / "model.json")
    if meta["prototype_id"] != PROTOTYPE_ID or meta["classes"] != ["car", "truck"]:
        raise ValueError("Unsupported model bundle")
    if sha256(folder / "heads.npz") != meta["heads_sha256"]:
        raise ValueError("Model parameters changed")
    with np.load(folder / "heads.npz", allow_pickle=False) as data:
        heads = {k: data[k].copy() for k in ("mean", "scale", "coef", "intercept")}
    n = len(meta["models"])
    if any(heads[k].shape != (n, 768) for k in ("mean", "scale", "coef")) or heads["intercept"].shape != (n,):
        raise ValueError("Model bundle dimensions disagree")
    for i in range(n):
        probabilities({k: v[i] for k, v in heads.items()}, np.zeros((1, 768)))
    return meta, heads


def predict(bundle_reference, audio_path, model_id=None):
    folder = resolve(bundle_reference)
    meta, heads = load_bundle(folder)
    selected = model_id or meta["default_model"]
    index = [m["model_id"] for m in meta["models"]].index(selected)
    path = resolve(audio_path) if audio_path.startswith(("repo:", "data:", "artifact:", "archive:")) else Path(audio_path).expanduser().resolve()
    features, _, windows, _ = audio_features([path], meta["scientific_config"])
    p = probabilities({k: v[index] for k, v in heads.items()}, features)[0]
    return dict(prototype_id=PROTOTYPE_ID, model_id=selected, audio=str(path),
                audio_sha256=sha256(path), prediction=meta["classes"][int(p.argmax())],
                probabilities=dict(zip(meta["classes"], p.tolist())),
                window=windows[0], task="closed_set_car_vs_truck",
                probability_calibration="not established",
                model_bundle_sha256=sha256(folder / "model.json"))


def build(config_reference=DEFAULT_CONFIG, output_reference=DEFAULT_OUTPUT):
    from abvid.learning.linear import fixed_fit
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    output = resolve(output_reference, must_exist=False)
    if not output_reference.startswith("artifact:") or output.is_relative_to(roots()["archive"]):
        raise ValueError("New bundles must be outside the checkout and frozen archive, under artifact:")
    if output.exists():
        raise FileExistsError(f"Choose a new output; existing bundles are never overwritten: {output}")
    config, lock, rows, ids, features, wave_hashes, reference, versions = frozen_inputs(config_reference)
    positions = {fid: i for i, fid in enumerate(ids)}
    folds, seeds = lock["folds"], config["seeds"]
    records, exported, fold_scores = [], [], []
    maximum_probability_error = 0.
    tests = [fid for fold in folds for fid in fold["test_ids"]]
    admitted_idmt = {i for i, r in rows.items() if r["dataset_id"] == "IDMT" and r["admitted_for_h1"]}
    if len(tests) != len(set(tests)) or set(tests) != admitted_idmt:
        raise ValueError("Held-out folds must partition all admitted IDMT records")
    with threadpool_limits(limits=config["hardware"]["blas_threads"]):
        for fold in folds:
            test_ids = fold["test_ids"]
            test_pos = [positions[i] for i in test_ids]
            ytest = np.array([rows[i]["class_id"] for i in test_ids])
            for seed in seeds:
                train_ids = fold["train_ids"][str(seed)]
                validate_roles(train_ids, test_ids, rows, fold["held_out_group"], config["budget"]["examples_per_class"])
                assert_separation(train_ids, lock["target_ids"], rows)
                train_pos = [positions[i] for i in train_ids]
                ytrain = np.array([rows[i]["class_id"] for i in train_ids])
                model = fixed_fit(features[train_pos].astype(np.float64), ytrain, config["classifier"])
                head = parameters(model)
                p = probabilities(head, features[test_pos])
                np.testing.assert_allclose(p, model.predict_proba(features[test_pos].astype(np.float64)), rtol=1e-10, atol=1e-10)
                model_id = f'BEATs768__real__seed{seed}__fold{fold["fold"]}'
                pred_name = f"predictions/{model_id}.npz"
                pred_path = checked(RESULTS + pred_name, reference["artifacts_sha256"][pred_name])
                with np.load(pred_path, allow_pickle=False) as old:
                    if old["source_ids"].tolist() != test_ids or not np.array_equal(old["source_true"], ytest):
                        raise ValueError("Reference prediction roles changed")
                    old_p = old["source_probabilities"]
                    np.testing.assert_allclose(p, old_p, rtol=1e-10, atol=1e-10)
                    maximum_probability_error = max(maximum_probability_error, float(np.abs(p - old_p).max()))
                scores = metrics(ytest, p)
                records.append(dict(model_id=model_id, seed=seed, fold=fold["fold"],
                                    held_out_group=fold["held_out_group"], train_ids=train_ids,
                                    test_ids=test_ids, train_class_counts=np.bincount(ytrain).tolist(),
                                    metrics=scores))
                exported.append(head)
                fold_scores.append(scores)
            print(f'Checked all {len(seeds)} heads for held-out group {fold["fold"] + 1}/{len(folds)}', flush=True)

    # Fixed by ID order, never by correctness: one held-out car and truck/group.
    sample_ids = [min(i for i in f["test_ids"] if rows[i]["class_id"] == label)
                  for f in folds for label in (0, 1)]
    paths = [checked("archive:" + rows[i]["source_path"], rows[i]["source_file_sha256"]) for i in sample_ids]
    observed, waves, _, encoder_provenance = audio_features(paths, config)
    raw_checks = []
    for n, fid in enumerate(sample_ids):
        pos = positions[fid]
        wave_sha = hashlib.sha256(waves[n].astype("<f4").tobytes()).hexdigest()
        if wave_sha != wave_hashes[pos]:
            raise ValueError("Raw preprocessing differs from the cached observation")
        np.testing.assert_allclose(observed[n], features[pos], rtol=1e-4, atol=1e-4)
        model_number = (n // 2) * len(seeds)  # first predeclared seed, matching held-out fold
        fresh = probabilities(exported[model_number], observed[n:n + 1])
        cached = probabilities(exported[model_number], features[pos:pos + 1])
        np.testing.assert_allclose(fresh, cached, rtol=0, atol=1e-4)
        if not np.array_equal(fresh.argmax(1), cached.argmax(1)):
            raise ValueError("Raw-audio prediction differs from cached prediction")
        raw_checks.append(dict(file_id=fid, audio_reference="archive:" + rows[fid]["source_path"],
                               model_id=records[model_number]["model_id"], waveform_sha256=wave_sha,
                               max_embedding_error=float(np.abs(observed[n] - features[pos]).max()),
                               max_probability_error=float(np.abs(fresh - cached).max()),
                               prediction=int(fresh.argmax(1)[0]), true_label=rows[fid]["class_id"]))

    headline = {k: float(np.mean([s[k] for s in fold_scores]))
                for k in ("macro_f1", "balanced_accuracy", "accuracy")}
    h1_estimate = read(resolve(RESULTS + "metrics.json"))["inference"]["BEATs768"]["estimates"]["source"]
    np.testing.assert_allclose(headline["macro_f1"], h1_estimate["estimate"], rtol=0, atol=1e-15)
    controls = []
    for f in folds:
        counts = np.bincount([rows[i]["class_id"] for i in f["test_ids"]], minlength=2)
        controls.append(cm_metrics([[int(counts[0]), 0], [int(counts[1]), 0]]))
    control = {k: float(np.mean([s[k] for s in controls])) for k in headline}
    repo = repository_root()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).splitlines()
    code = {str(p.relative_to(repo)): sha256(p) for p in sorted((repo / "src/abvid").rglob("*.py"))}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "heads.npz", **{k: np.stack([h[k] for h in exported]) for k in exported[0]})
    meta = dict(prototype_id=PROTOTYPE_ID, classes=config["classes"],
                default_model=records[0]["model_id"], default_selection="first frozen fold and first declared seed; no score selection",
                scientific_config=config, encoder_checkpoint=CHECKPOINT, encoder_provenance=encoder_provenance,
                heads_sha256=sha256(output / "heads.npz"), models=records,
                dataset="IDMT_V1", domain="IDMT source development", evaluation_role="exposed_development",
                validation=None, final_test=None, target_scored=False, encoder_trained=False,
                components_fitted=["StandardScaler", "LogisticRegression"],
                grouping="six conservative provenance groups; original sessions unknown",
                completed_utc=datetime.now(timezone.utc).isoformat(), hardware=platform.platform(),
                versions=versions, git_commit=commit, git_status=status, code_sha256=code,
                registry_reference=config_reference, registry_sha256=sha256(resolve(config_reference)),
                frozen_lock_sha256=LOCK_SHA, reference_results_summary_sha256=RESULTS_SHA,
                frozen_config_sha256=reference["frozen_config_sha256"],
                feature_cache_sha256=reference["feature_cache_sha256"])
    save(output / "model.json", meta)
    # Load from disk and check every saved head, not only the in-memory fit.
    _, loaded_heads = load_bundle(output)
    for number, record in enumerate(records):
        test_pos = [positions[i] for i in record["test_ids"]]
        actual = probabilities({k: v[number] for k, v in loaded_heads.items()}, features[test_pos])
        np.testing.assert_array_equal(actual, probabilities(exported[number], features[test_pos]))
    evaluation = dict(prototype_id=PROTOTYPE_ID, headline=headline, always_car=control,
                      aggregation="equal mean over six held-out groups and five training-selection seeds",
                      macro_f1_ci95=h1_estimate["ci95"],
                      interval_origin="existing H1 paired group/selection bootstrap, conditional on fits; all source predictions reproduced",
                      uncertainty=config["uncertainty"], unique_evaluation_files=len(tests),
                      provenance_groups=len(folds), fits=len(records), seeds=seeds,
                      maximum_archived_probability_error=maximum_probability_error,
                      raw_audio_checks=raw_checks, saved_head_roundtrip="exact",
                      target_scored=False, final_test=None, runtime_s=time.perf_counter() - started,
                      model_sha256=sha256(output / "model.json"))
    save(output / "evaluation.json", evaluation)
    report = render_report(meta, evaluation, output_reference)
    (output / "EVALUATION.md").write_text(report)
    print(report, flush=True)
    return output


def render_report(meta, result, output_reference):
    score = result["headline"]
    ci = result["macro_f1_ci95"]
    return f"""# ABVID real-data benchmark prototype v0.1

Frozen BEATs (768D mean-pooled features) + a training-only StandardScaler and
L2 logistic regression. Task: broad car/truck categories, not vehicle identity.
The classifier was fitted and saved; BEATs was not fine-tuned.

| Evaluation | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| IDMT grouped development benchmark | {score['macro_f1']:.2%} | {score['balanced_accuracy']:.2%} | {score['accuracy']:.2%} |
| Always-car control, identical aggregation | {result['always_car']['macro_f1']:.2%} | {result['always_car']['balanced_accuracy']:.2%} | {result['always_car']['accuracy']:.2%} |

Macro-F1 95% interval: [{ci[0]:.2%}, {ci[1]:.2%}]. This is the existing H1
group/selection bootstrap interval, conditional on the fitted models. All 30
source prediction sets were reproduced before retaining that interval. It is
not an uncertainty estimate across independently sampled vehicle populations.

Each fit uses 190 cars and 190 trucks from five groups and evaluates the sixth.
All {result['unique_evaluation_files']} admitted IDMT files are evaluated once per seed.
The headline averages group/seed scores equally; it is not pooled clip accuracy.
Seeds: {result['seeds']}. Original recording sessions remain unknown; the six
connected provenance groups conservatively retain known recording/duplicate links.

There is no validation/search stage, augmentation, threshold tuning, or encoder
fine-tuning. Preprocessing preserves H1: channel mean, 8 kHz intermediate
resampling, 16 kHz model input, centered 2 s window, no gain normalization or padding.

All saved heads passed exact serialization roundtrip checks. Maximum probability
difference from archived H1 predictions: {result['maximum_archived_probability_error']:.3g}.
Raw WAV preprocessing, embeddings and predictions were checked on
{len(result['raw_audio_checks'])} predeclared examples (one car/truck per group).
This raw-audio check is a parity sample; full evaluation uses the verified cache.

Default demo model: `{meta['default_model']}`, chosen by frozen order, not score.
Bundle: `{output_reference}`. `model.json` records all train/test IDs, config,
checkpoint, code hashes, commit, package versions and per-model results;
`heads.npz` stores numeric heads without pickle; `evaluation.json` records checks.
Build and verification runtime: {result['runtime_s']:.1f} seconds on {meta['hardware']}.

These are existing, exposed development splits. No new MELAUDIS or CAST transfer
evaluation was run. This prototype supplies a runnable baseline, not evidence of
unseen-domain robustness or source-model dominance. Its closed-set predictions
always choose car or truck; probability calibration and non-vehicle rejection
are not established. The BEATs mirror hash is verified; identity against an
official upstream checkpoint checksum remains unverified.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("build", help="Reproduce frozen real-data heads, save bundle and verify benchmark")
    fit.add_argument("--baseline-config", default=DEFAULT_CONFIG)
    fit.add_argument("--output", default=DEFAULT_OUTPUT)
    infer = commands.add_parser("predict", help="Classify the center 2 seconds of a vehicle WAV")
    infer.add_argument("--bundle", default=DEFAULT_OUTPUT)
    infer.add_argument("--audio", required=True)
    infer.add_argument("--model-id", default=None)
    args = parser.parse_args()
    if args.command == "build":
        build(args.baseline_config, args.output)
    else:
        print(json.dumps(predict(args.bundle, args.audio, args.model_id), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
