"""Read-only verification of an H2 design; no audio/model imports."""
import argparse
from pathlib import Path
from metadata import HERE, ROOT, TABLES, build, jsonl, read, rows, save, sha, require, validate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    folder = args.freeze.resolve(); lock = read(folder/"lock.json"); cfg = read(folder/"config.json")
    require(not args.output.exists(), "Do not overwrite verification")
    for name, digest in lock["artifacts_sha256"].items(): require(sha(folder/name) == digest, f"Frozen artifact changed: {name}")
    for name in ["metadata.py", "verify_design.py"]:
        require(sha(HERE/name) == lock["artifacts_sha256"][name], f"Verifier code changed: {name}")
    tables = {name: rows(folder/f"{name}.jsonl") for name in TABLES}
    target = rows(folder/"target_manifest.jsonl"); checks = validate(cfg, tables, target)
    regenerated = build(cfg)
    for name in TABLES:
        require(jsonl(regenerated[name]) == (folder/f"{name}.jsonl").read_text(), f"Nondeterministic manifest: {name}")
    hp = ROOT/cfg["target"]["h1_lock"]
    require(sha(hp) == cfg["target"]["h1_lock_sha256"], "H1 lock changed")
    require([r["file_id"] for r in target] == read(hp)["target_ids"], "Different H1 target order/population")
    original = {r["file_id"]: r for r in rows(hp.parent/"admitted.jsonl")}
    for r in target:
        require(all(v == original[r["file_id"]].get(k) for k, v in r.items() if k != "h2_role"), "Target metadata changed")
    protected = read(folder/"protected_artifacts.json")
    for row in protected: require(sha(ROOT/row["path"]) == row["sha256"], f"Historical artifact changed: {row['path']}")
    gate = read(folder/"execution_gate.json")
    require(not gate["ready_for_target_evaluation"] and not lock["ready_for_target_evaluation"], "False execution admission")
    require(all(v is None for k, v in gate["implementation_status"].items() if k.endswith("sha256")), "Unimplemented artifact claimed")
    checks.update(lock_sha256=sha(folder/"lock.json"), frozen_artifacts_checked=len(lock["artifacts_sha256"]),
                  protected_artifacts_unchanged=len(protected), deterministic_manifests=list(TABLES),
                  target_ids_and_metadata_match_h1=True, new_target_scores=False)
    args.output.parent.mkdir(parents=True, exist_ok=True); save(args.output, checks)
    print(__import__("json").dumps(checks, sort_keys=True))


if __name__ == "__main__": main()
