"""CAST-5A role design and parent preparation over the frozen CAST-4A sources.

Thirty roles (six H1 folds x five H1 training selections) are built from the
unchanged CAST-4A source rows with the unchanged ``grouped_design.partition``
validation and ``FoldRole`` guards. Parents already prepared by the CAST-4A
run are read through the 4A context's validated loader; the remaining parents
are fitted here with the unchanged per-parent chain and numerical contract.
No CAST-4A pinned module is modified.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from types import MappingProxyType

from abvid.cast import grouped_design
from abvid.cast.grouped_storage import read_bundle, verify_bundle, write_bundle
from abvid.cast.roles import FoldRole
from abvid.paths import resolve
from abvid.provenance.hashing import canonical, sha256

PROTOCOL = "repo:experiments/cast_utility/PROTOCOL.md"
DESIGN_4A = "artifact:cast/grouped_coverage/cast4a_design_20261007_v1"
CHAIN = "repo:experiments/cast_coverage/grouped_coverage/readiness/chain_contract.json"
OUTPUT = "artifact:cast/utility/cast5a_run_20261011_v1"
SEEDS = (42, 123, 456, 789, 1024)
ID = "cast5a_utility_v1"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def partition_id(seed, fold):
    return f"utility_s{seed}_{fold}"


def partitions(rows, h1_lock):
    """Thirty validated partitions in (seed, fold) order; no file access."""
    folds = h1_lock["folds"]
    if len(folds) != 6 or {f["held_out_group"] for f in folds} != {r["provenance_group_id"] for r in rows}:
        raise ValueError("Expected all six frozen H1 conservative groups")
    out = []
    for seed in SEEDS:
        for f in folds:
            part = grouped_design.partition(rows, f["train_ids"][str(seed)], f["test_ids"],
                                            analysis=f"utility_s{seed}", fold=f["fold"],
                                            held_label=f["held_out_group"], held_groups={f["held_out_group"]})
            if part["id"] != partition_id(seed, f["fold"]):
                raise ValueError("Unexpected partition identity")
            out.append(part)
    return out


def build_roles(rows, parts, identity):
    """FoldRole objects with the same held-link and class guards as CAST-4A."""
    source = grouped_design.source_rows(json.loads(canonical(rows)))
    by_id = {r["file_id"]: r for r in source}
    source_bytes = MappingProxyType({r["file_id"]: canonical(r) for r in source})
    roles = {}
    for part in parts:
        rebuilt = grouped_design.partition(source, part["train_ids"], part["held_ids"], analysis=part["analysis"],
                                           fold=part["fold"], held_label=part["held_label"],
                                           held_groups=set(part["held_groups"]))
        if canonical(rebuilt) != canonical(part):
            raise ValueError("Partition differs from its metadata reconstruction")
        held = [by_id[i] for i in part["held_ids"]]
        links = MappingProxyType({key: frozenset(r[key] for r in held if grouped_design.known(r.get(key)))
                                  for key in grouped_design.LINKS})
        classes = MappingProxyType({label: tuple(i for i in part["train_ids"] if by_id[i]["canonical_class"] == label)
                                    for label in ("car", "truck")})
        roles[part["id"]] = FoldRole(part["id"], part["analysis"], tuple(part["train_ids"]), tuple(part["held_ids"]),
                                     tuple(part["held_groups"]), tuple(sorted(part["training_group_counts"])),
                                     source_bytes, links, classes, identity)
    return roles


def read_bundle_lock(reference):
    """Read the CAST-4A design freeze (lock.json + payload hashes); metadata only."""
    folder = resolve(reference)
    lock = json.loads((folder / "lock.json").read_text())
    if lock["status"] != "metadata_only_complete_scientific_execution_unrun":
        raise ValueError("Unexpected CAST-4A design status")
    if lock["payload_identity_sha256"] != grouped_design.digest(lock["payload_sha256"]):
        raise ValueError("CAST-4A design identity differs")
    payload = {}
    for name, expected in lock["payload_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"CAST-4A design payload changed: {name}")
        payload[name] = (folder / name).read_bytes()
    return lock, payload


def design_payload():
    lock_4a, payload_4a = read_bundle_lock(DESIGN_4A)
    rows = [json.loads(line) for line in payload_4a["sources.jsonl"].splitlines()]
    h1_lock = json.loads(payload_4a["inputs/h1_lock.json"])
    parts = partitions(rows, h1_lock)
    union = sorted({i for p in parts for i in p["train_ids"]})
    union_4a = sorted({i for p in json.loads(payload_4a["folds.json"]) for i in p["train_ids"]})
    from_4a, new = sorted(set(union) & set(union_4a)), sorted(set(union) - set(union_4a))
    protocol = resolve(PROTOCOL).read_bytes()
    design = {"schema_version": 1, "id": ID, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
              "cast4a_design_identity_sha256": lock_4a["payload_identity_sha256"],
              "seeds": list(SEEDS), "partition_ids": [p["id"] for p in parts],
              "selection": "exact H1 lock train_ids[seed] in original order; held = H1 test_ids of the fold",
              "union_parents": len(union), "from_cast4a_run": len(from_4a), "new_fits": len(new)}
    return {"design.json": canonical(design), "partitions.json": canonical(parts),
            "parents.json": canonical({"union": union, "from_cast4a_run": from_4a, "new_fits": new}),
            "inputs/protocol.md": protocol, "inputs/sources.jsonl": payload_4a["sources.jsonl"],
            "inputs/h1_lock.json": payload_4a["inputs/h1_lock.json"],
            "inputs/numerical_contract.json": payload_4a["inputs/numerical_contract.json"],
            "inputs/chain_contract.json": resolve(CHAIN).read_bytes()}


def freeze():
    folder = resolve(OUTPUT, must_exist=False) / "design"
    manifest = write_bundle(folder, design_payload(), {"kind": "cast5a_design_v1",
                                                       "status": "metadata_only_no_scientific_execution"})
    return manifest


def load_context():
    """Rebuild the 5A execution context from the frozen design bundle only."""
    output = resolve(OUTPUT, must_exist=False)
    manifest = verify_bundle(output / "design")
    _, payload = read_bundle(output / "design")
    if payload != design_payload():
        raise ValueError("Frozen CAST-5A design differs from its current reconstruction")
    rows = [json.loads(line) for line in payload["inputs/sources.jsonl"].splitlines()]
    parts = json.loads(payload["partitions.json"])
    identity = digest(manifest)
    roles = build_roles(rows, parts, identity)
    return {"roles": roles, "sources": {r["file_id"]: r for r in rows},
            "numerical": json.loads(payload["inputs/numerical_contract.json"]),
            "chain": json.loads(payload["inputs/chain_contract.json"]), "readiness_records": {},
            "config_sha256": identity, "config": {"max_workers": 4}, "output": output,
            "parents": json.loads(payload["parents.json"]), "design_identity_sha256": identity,
            "partitions": parts}


_WORKER = None


def _initialize():
    global _WORKER
    from abvid.cast import grouped_parents
    grouped_parents._configure_cpu()
    _WORKER = load_context()


def _fit(file_id):
    from abvid.cast import grouped_parents
    return file_id, grouped_parents._fit_parent(_WORKER, file_id)


def fit_new_parents(workers=4, limit=None):
    """Fit each not-yet-published new parent once; failures stay terminal."""
    context = load_context()
    todo = [i for i in context["parents"]["new_fits"]
            if not (context["output"] / "parents" / i).exists()][:limit]
    results = {"complete": 0, "technical_failure": 0}
    if not todo:
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize) as pool:
        futures = [pool.submit(_fit, i) for i in todo]
        for future in as_completed(futures):
            _, status = future.result()
            results[status] += 1
    return results


def load_parent_rows(context, file_id):
    """Return (bank row, observation) for a role parent from 4A or 5A storage."""
    from abvid.cast import grouped_parents
    if file_id in set(context["parents"]["from_cast4a_run"]):
        # Build the pinned 4A context once; it re-verifies every pin when constructed.
        if "_cast4a" not in context:
            context["_cast4a"] = _cast4a_context()
        context_4a = context["_cast4a"]
        return grouped_parents.load_parent(context_4a, file_id, grouped_parents._witness(context_4a, file_id))
    return grouped_parents.load_parent(context, file_id, grouped_parents._witness(context, file_id))


def _cast4a_context():
    from abvid.cast.grouped_execution_config import load_context as load_4a
    return load_4a()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "plan", "fit"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze()["metadata"]))
    elif args.command == "plan":
        context = load_context()
        done = sum((context["output"] / "parents" / i).exists() for i in context["parents"]["new_fits"])
        print(json.dumps({"roles": len(context["roles"]), "union": len(context["parents"]["union"]),
                          "from_cast4a_run": len(context["parents"]["from_cast4a_run"]),
                          "new_fits": len(context["parents"]["new_fits"]), "new_fits_published": done}))
    else:
        print(json.dumps(fit_new_parents(args.workers, args.limit)))


if __name__ == "__main__":
    main()
