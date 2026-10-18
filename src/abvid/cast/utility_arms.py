"""CAST-5A generated banks: per-role CAST v15 samples (RC/C) and the RU prior.

RC/C: the unchanged CAST-4A ``GroupedPopulation`` built from one role's 380
parents, frozen and persisted before drawing, sampled with the frozen v15 joint
schedule entries for that partition. RU: the same renderer with parameters
drawn from a declared manual prior; it learns no statistics. Waveforms are
shape-normalized (zero mean, unit RMS) float32 at 16 kHz, 32,000 samples.
"""
from __future__ import annotations

from io import BytesIO
import json

import numpy as np

from abvid.cast import grouped_design
from abvid.cast.config import digest
from abvid.cast.grouped_population import build_population
from abvid.cast.grouped_storage import read_bundle, write_bundle
from abvid.cast.population.resolution_evaluation import render
from abvid.cast.utility_design import SEEDS, load_context, load_parent_rows
from abvid.data.augmentation import shape
from abvid.provenance.hashing import canonical

RC_ARM = "v15_joint"
PER_CLASS = 190
RU_ID = "cast5a_ru_v1"
# Firing-frequency ranges (Hz) = engine speed x firing events per revolution.
# Car: 800-3000 r/min (P11 Goksu, sec. II), four-cylinder four-stroke (rpm/30).
# Truck: 500-2200 r/min (industry article, not peer-reviewed), six-cylinder four-stroke (rpm/20).
RU_SPACING_HZ = {"car": (800 / 30, 3000 / 30), "truck": (500 / 20, 2200 / 20)}
SPACING_BOUNDS = (10.0, 400.0)
ENVELOPE_BOUNDS = (0.1, 3.0)


def rc_schedule(partition_id):
    """First 190 frozen v15-joint schedule entries per class, seed-major order."""
    entries = [e for e in grouped_design.planned_schedule([{"id": partition_id}]) if e["arm"] == RC_ARM]
    out = []
    for label in ("car", "truck"):
        chosen = sorted((e for e in entries if e["class"] == label),
                        key=lambda e: (SEEDS.index(e["seed"]), e["index"]))[:PER_CLASS]
        if len(chosen) != PER_CLASS:
            raise ValueError("Frozen schedule lacks 190 v15-joint entries per class")
        out += chosen
    return out


def ru_parameters(label, seed, index):
    """One RU parameter dict and its renderer input ID; deterministic."""
    rng = np.random.Generator(np.random.PCG64(grouped_design.hash_seed([RU_ID, label, seed, index])))
    lo, hi = RU_SPACING_HZ[label]
    centre = rng.uniform(lo, hi)
    spacing = np.clip(centre * np.exp(rng.uniform(-0.1, 0.1, 3)), *SPACING_BOUNDS)
    anchors = rng.uniform(*ENVELOPE_BOUNDS, 5)
    envelope = np.interp(np.linspace(0, 4, 41), np.arange(5), anchors)
    params = {"spacing_hz": spacing.tolist(), "harmonic_weights": rng.dirichlet(np.ones(8)).tolist(),
              "harmonic_fraction": float(rng.uniform(0, 1)), "noise_weights": rng.dirichlet(np.ones(16)).tolist(),
              "envelope_knots": envelope.tolist()}
    return f"{RU_ID}:{label}:{seed}:{index:04d}", params


def _npy(array):
    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _renderer_config(context):
    return context["numerical"]["renderer_configs"]["spectrum16_temporal41_v11"]


def ru_bank(context, seed):
    folder = context["output"] / "arms" / "ru" / f"s{seed}"
    if not folder.exists():
        cfg, waves, records = _renderer_config(context), [], []
        for label in ("car", "truck"):
            for index in range(PER_CLASS):
                input_id, params = ru_parameters(label, seed, index)
                wave, streams = render({"input_id": input_id, "parameters": params}, cfg, "spectrum16")
                waves.append(shape(wave).astype(np.float32))
                records.append({"input_id": input_id, "class": label, "parameters": params, "streams": streams})
        write_bundle(folder, {"waves.npy": _npy(np.stack(waves)), "records.json": canonical(records)},
                     {"kind": "cast5a_ru_bank_v1", "seed": seed, "per_class": PER_CLASS,
                      "design_identity_sha256": context["design_identity_sha256"]})
    return read_bundle(folder)


def rc_bank(context, partition_id):
    """Freeze the role population, then draw and persist its 380 RC/C samples."""
    role = context["roles"][partition_id]
    base = context["output"] / "arms" / "rc" / partition_id
    bank, observed = [], []
    for file_id in role.train_ids:
        row, obs = load_parent_rows(context, file_id)
        bank.append(row)
        observed.append(obs)
    role.validate_bank([r["parent"] for r in bank])
    population = build_population(bank, observed, role, context["numerical"])
    write_bundle(base / "population", {"statistics.json": canonical(population.statistics),
                                       "scales.json": canonical(population.scales)},
                 {"kind": "cast5a_rc_population_v1", "partition": partition_id,
                  "fingerprint": population.fingerprint,
                  "design_identity_sha256": context["design_identity_sha256"]})
    if not (base / "samples").exists():
        waves, records = [], []
        for planned in rc_schedule(partition_id):
            record, wave, descriptors = population.draw(planned)
            waves.append(shape(wave).astype(np.float32))
            records.append({"planned_id": planned["id"], "class": planned["class"],
                            "parent_ids": record["parent_ids"], "record_sha256": digest(record),
                            "descriptor_sha256": record["descriptor_sha256"]})
        write_bundle(base / "samples", {"waves.npy": _npy(np.stack(waves)), "records.json": canonical(records)},
                     {"kind": "cast5a_rc_bank_v1", "partition": partition_id,
                      "population_fingerprint": population.fingerprint,
                      "design_identity_sha256": context["design_identity_sha256"]})
    return read_bundle(base / "samples")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ru", "rc", "status"))
    parser.add_argument("--partition", action="append")
    args = parser.parse_args(argv)
    context = load_context()
    if args.command == "ru":
        for seed in SEEDS:
            meta, _ = ru_bank(context, seed)
            print(json.dumps(meta))
    elif args.command == "rc":
        for pid in args.partition or [p["id"] for p in context["partitions"]]:
            meta, _ = rc_bank(context, pid)
            print(json.dumps(meta), flush=True)
    else:
        done = [p["id"] for p in context["partitions"]
                if (context["output"] / "arms" / "rc" / p["id"] / "samples").exists()]
        print(json.dumps({"rc_complete": len(done), "ru_complete": sum(
            (context["output"] / "arms" / "ru" / f"s{s}").exists() for s in SEEDS)}))


if __name__ == "__main__":
    main()
