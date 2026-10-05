"""Create an H2 metadata freeze; never generates audio or trains a model."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import shutil
import subprocess
import tempfile

from metadata import HERE, ROOT, TABLES, build, compact, jsonl, read, rows, save, sha, require, validate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=__import__("pathlib").Path, required=True)
    args = ap.parse_args(); cfg = read(HERE/"config.json")
    out = args.output.resolve(); require(not out.exists(), "Never overwrite a freeze")
    hp = ROOT/cfg["target"]["h1_lock"]
    require(sha(hp) == cfg["target"]["h1_lock_sha256"], "H1 lock changed")
    hl = read(hp)
    for name, digest in hl["artifacts_sha256"].items():
        require(sha(hp.parent/name) == digest, f"H1 frozen artifact changed: {name}")
    hc = read(hp.parent/"config.json")
    for key in ["encoder", "preprocessing", "mfcc", "seeds", "representations"]:
        require(cfg[key] == hc[key], f"H1 control changed: {key}")
    require({k: cfg["classifier"][k] for k in hc["classifier"]} == hc["classifier"], "H1 fixed classifier changed")
    checkpoint = ROOT/cfg["encoder"]["checkpoint"]
    require(sha(checkpoint) == cfg["encoder"]["checkpoint_sha256"], "Encoder checkpoint changed")
    provenance = read(checkpoint.parent/"provenance.json")
    require(provenance["upstream_commit"] == cfg["encoder"]["upstream_commit"], "Encoder revision changed")
    for name, entry in provenance["code"].items():
        require(sha(checkpoint.parent/"upstream"/name) == entry["sha256"], "Encoder upstream source changed")
    originals = {r["file_id"]: r for r in rows(hp.parent/"admitted.jsonl")}
    keys = ["file_id", "dataset_id", "canonical_class", "class_id", "source_path", "source_file_sha256",
            "decoded_audio_sha256", "provenance_group_id", "recording_session", "original_media_id",
            "native_sample_rate_hz", "native_channels", "duration_s", "licence", "split_role", "grouping_basis"]
    target = [{**{k: originals[i].get(k) for k in keys}, "h2_role": "exposed_development_evaluation_only"} for i in hl["target_ids"]]
    tables = build(cfg); validation = validate(cfg, tables, target)
    protected = []
    for name in ["h1", "h1_diagnostics", "h1_transfer_sensitivity"]:
        for p in sorted((ROOT/"experiments"/name).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                protected.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    for p in [checkpoint, checkpoint.parent/"provenance.json"] + [checkpoint.parent/"upstream"/n for n in provenance["code"]]:
        protected.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    references = []
    for p in sorted((HERE/"evidence").rglob("*")):
        if p.is_file(): references.append(dict(path=str(p.relative_to(HERE)), sha256=sha(p)))
    tree = read(HERE/"evidence/pyroadacoustics/tree.json")
    import hashlib
    for e in tree["tree"]:
        if e["type"] != "blob" or e["path"] == ".gitignore": continue
        p = HERE/"evidence/pyroadacoustics"/e["path"]; b = p.read_bytes()
        require(hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest() == e["sha"], f"Upstream Git blob mismatch: {e['path']}")
    require(sha(HERE/cfg["source"]["reference"]) == cfg["source"]["reference_sha256"], "P02 source changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = __import__("pathlib").Path(tempfile.mkdtemp(prefix=".h2-stage-", dir=out.parent))
    try:
        for name in ["config.json", "PROTOCOL.md", "BACKEND_AUDIT.md", "metadata.py", "freeze_design.py", "verify_design.py", "test_design.py"]:
            shutil.copy2(HERE/name, stage/name)
        shutil.copytree(HERE/"evidence", stage/"evidence")
        for name in TABLES: (stage/f"{name}.jsonl").write_text(jsonl(tables[name]))
        (stage/"target_manifest.jsonl").write_text(jsonl(target))
        save(stage/"protected_artifacts.json", protected)
        save(stage/"reference_files.json", references)
        save(stage/"metadata_checks.json", validation)
        save(stage/"execution_gate.json", dict(design_frozen=True, ready_for_target_evaluation=False,
             implementation_status=cfg["implementation_status"],
             unresolved=["ground_angle_table_indexing", "reflected_spreading", "common_speed_of_sound",
                         "source_and_exact_trajectory_adapter", "physical_and_manipulation_validation",
                         "pinned_execution_environment", "generated_corpus_and_fitted_model_lock"]))
        lock = dict(protocol_id=cfg["protocol_id"], freeze_id=cfg["freeze_id"],
                    frozen_utc=datetime.now(timezone.utc).isoformat(), metadata_only=True,
                    design_ready=True, ready_for_target_evaluation=False,
                    waveform_generation=False, model_fitting=False, target_inference=False,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    dirty_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                    versions={"numpy": importlib.metadata.version("numpy"), "python": __import__("platform").python_version()},
                    counts=validation["counts"], protected_artifact_count=len(protected),
                    artifacts_sha256={str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()})
        save(stage/"lock.json", lock)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(compact(dict(freeze=str(out.relative_to(ROOT)), lock_sha256=sha(out/"lock.json"),
                       counts=lock["counts"], protected_files=len(protected), status="design_only")))


if __name__ == "__main__": main()
