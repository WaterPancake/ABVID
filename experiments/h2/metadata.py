"""H2 metadata only: no audio decoding, simulator imports, fitting or inference."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = ("templates", "dry_source_plan", "geometries", "events", "render_plan", "fit_plan", "timing_slots")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(rows):
    return "".join(compact(row) + "\n" for row in rows)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def derived_seed(cfg, seed, role, stream, label, template, trajectory=None):
    key = [cfg["protocol_id"], seed, role, stream, label, template, trajectory]
    return int(hashlib.sha256(compact(key).encode()).hexdigest()[:16], 16)


def build(cfg):
    """Instantiate latents/planned IDs, not waveforms. No hidden random draws."""
    require(np.__version__ == cfg["rng"]["numpy_version"], "Use pinned metadata NumPy")
    result = {name: [] for name in TABLES}
    source, path, budget = cfg["source"], cfg["propagation"], cfg["budget"]
    fs = source["rate_hz"]
    c = 331.3 * math.sqrt(1 + path["temperature_C"] / 273.15)
    for seed in cfg["seeds"]:
        for role, count in [("train", budget["train_templates_per_class"]),
                            ("validation", budget["validation_templates_per_class"])]:
            for ti in range(count):
                group = f"h2.r{seed}.{role}.t{ti:03d}"
                geometry_ids = []
                for gi in range(budget["trajectories_per_template"]):
                    latent_seed = derived_seed(cfg, seed, role, "geometry", None, ti, gi)
                    rng = np.random.Generator(np.random.PCG64(latent_seed))
                    distance = float(rng.uniform(*path["range_m"]))
                    speed = float(rng.uniform(*path["speed_kmh"]))
                    direction = int(rng.choice(path["directions"]))
                    gid = f"{group}.g{gi:02d}"
                    center = path["closest_source_time_s"] + math.hypot(distance, .7) / c
                    start = math.floor((center - 1) * fs + .5)
                    geom = dict(geometry_id=gid, replicate_seed=seed, role=role, template_slot=ti,
                                trajectory_slot=gi, geometry_seed=latent_seed, range_m=distance,
                                speed_kmh=speed, direction=direction, source_height_m=path["source_height_m"],
                                mic_xyz_m=path["mic_xyz_m"], sample_count=source["samples"],
                                speed_of_sound_m_s=c, crop_start_sample_8k=start,
                                crop_end_sample_8k=start + path["crop_samples"],
                                ideal_crop_center_s=center,
                                max_image_path_delay_s=math.sqrt(distance**2 + (5*speed/3.6)**2 + 1.7**2)/c)
                    result["geometries"].append(geom)
                    geometry_ids.append(gid)
                    if role == "validation" and ti < 3 and gi < 2:
                        result["timing_slots"].append(dict(geometry_id=gid, role=role, replicate_seed=seed,
                            purpose="later_synthetic_only_QC_and_timing;both_classes_all_cells"))
                for label, class_id in cfg["classes"].items():
                    cp = source[label]
                    streams = {stream: derived_seed(cfg, seed, role, stream, label, ti)
                               for stream in cfg["rng"]["stream_separation"] if stream != "geometry"}
                    rng = np.random.Generator(np.random.PCG64(streams["base_parameters"]))
                    rpm = float(rng.uniform(*cp["rpm"]))
                    base = dict(rpm=rpm, cylinders=cp["cylinders"], stroke=cp["stroke"],
                                firing_hz=rpm*cp["cylinders"]/120,
                                harmonics=cp["harmonics"], engine_rolloff_db_per_order=cp["rolloff_db_per_order"],
                                engine_am_depth=float(rng.uniform(*source["am_depth"])),
                                tire_center_hz=float(rng.uniform(*cp["tire_center_hz"])),
                                tire_width_hz=float(rng.uniform(*cp["tire_width_hz"])))
                    if label == "car":
                        u = (rpm - 1500) / 2500
                        base["weights"] = {"engine": .3 + .4*(1-u), "tire": .3 + .4*u}
                    else:
                        base["weights"] = {k: float(rng.uniform(*v)) for k, v in cp["weights"].items()}
                    base["engine_phases_rad"] = np.random.Generator(np.random.PCG64(streams["engine_phases"])).uniform(0, 2*math.pi, cp["harmonics"]).tolist()
                    base["exhaust_phases_rad"] = (np.random.Generator(np.random.PCG64(streams["exhaust_phases"])).uniform(0, 2*math.pi, 5).tolist() if label == "truck" else [])
                    base["tire_noise_seed"] = streams["tire_noise"]
                    base["exhaust_noise_seed"] = streams["exhaust_noise"] if label == "truck" else None
                    enrichment = source["S1"]
                    rng = np.random.Generator(np.random.PCG64(streams["S1_parameters"]))
                    s1 = dict(engine_rolloff_offset_db_per_order=float(rng.uniform(*enrichment["engine_rolloff_offset_db_per_order"])),
                              frequency_fractional_amplitude=float(rng.uniform(*enrichment["frequency_fractional_amplitude"])),
                              frequency_modulation_hz=float(rng.uniform(*enrichment["frequency_modulation_hz"])),
                              phase_rad=float(rng.uniform(*enrichment["phase_rad"])),
                              component_gain_db={k: float(rng.uniform(*enrichment["component_gain_db"])) for k in base["weights"]})
                    tid = f"{group}.{label}"
                    result["templates"].append(dict(template_id=tid, source_parent_group=tid,
                        role=role, replicate_seed=seed, template_slot=ti, canonical_class=label, class_id=class_id,
                        streams=streams, base=base, S1=s1, physical_vehicle_id=None,
                        base_parameters_sha256=hashlib.sha256(compact(base).encode()).hexdigest()))
                    for source_level in ["S0", "S1"]:
                        result["dry_source_plan"].append(dict(dry_source_id=f"{tid}.{source_level}", template_id=tid,
                            role=role, replicate_seed=seed, class_id=class_id, source_level=source_level,
                            waveform_sha256=None, status="planned_not_generated"))
                    for gid in geometry_ids:
                        eid = f"{gid}.{label}"
                        result["events"].append(dict(event_id=eid, geometry_id=gid, template_id=tid,
                            source_parent_group=tid, canonical_class=label, class_id=class_id,
                            replicate_seed=seed, role=role))
                        for cell, levels in cfg["cells"].items():
                            result["render_plan"].append(dict(job_id=f"{eid}.{cell}", event_id=eid,
                                template_id=tid, source_parent_group=tid, geometry_id=gid, cell=cell,
                                class_id=class_id, replicate_seed=seed, role=role,
                                source_level=levels["source"], path_level=levels["path"],
                                dry_source_id=f"{tid}.{levels['source']}",
                                source_waveform_sha256=None, rendered_waveform_sha256=None,
                                observation_sha256=None, status="planned_not_generated"))
    for seed in cfg["seeds"]:
        for cell in cfg["cells"]:
            rr = [r for r in result["render_plan"] if r["replicate_seed"] == seed and r["cell"] == cell]
            for rep in cfg["representations"]:
                result["fit_plan"].append(dict(fit_id=f"h2.r{seed}.{cell}.{rep}", replicate_seed=seed,
                    cell=cell, representation=rep,
                    train_job_ids=sorted(r["job_id"] for r in rr if r["role"] == "train"),
                    validation_job_ids=sorted(r["job_id"] for r in rr if r["role"] == "validation"),
                    target_manifest="target_manifest.jsonl", classifier=cfg["classifier"],
                    threshold=cfg["threshold"], positive_rule=cfg["positive_rule"],
                    selection=None, trainable=["StandardScaler", "LogisticRegression"],
                    encoder_frozen=True, checkpoint_sha256=None, status="planned_not_fitted"))
    return result


def validate(cfg, tables, target):
    """Check leakage/pairing independently of manifest generation."""
    require(cfg["classes"] == {"car": 0, "truck": 1}, "Class mapping changed")
    require(cfg["cells"] == {"F00": {"source": "S0", "path": "P0"}, "F10": {"source": "S1", "path": "P0"},
                             "F01": {"source": "S0", "path": "P1"}, "F11": {"source": "S1", "path": "P1"}}, "Factor mapping changed")
    require(cfg["implementation_status"]["ready_for_target_evaluation"] is False, "Design is not executable")
    index = {}
    id_keys = {"templates": "template_id", "dry_source_plan": "dry_source_id", "geometries": "geometry_id",
               "events": "event_id", "render_plan": "job_id", "fit_plan": "fit_id", "timing_slots": "geometry_id"}
    expected = dict(templates=240, dry_source_plan=480, geometries=1200, events=2400, render_plan=9600, fit_plan=40, timing_slots=30)
    for name, count in expected.items():
        require(len(tables[name]) == count, f"Incorrect {name} count")
        index[name] = {r[id_keys[name]]: r for r in tables[name]}
        require(len(index[name]) == count, f"Duplicate ID in {name}")
    parents = defaultdict(set)
    for t in tables["templates"]:
        parents[t["source_parent_group"]].add(t["role"])
        cp = cfg["source"][t["canonical_class"]]; b = t["base"]; s = t["S1"]
        require(cp["rpm"][0] <= b["rpm"] < cp["rpm"][1], "RPM out of range")
        require(b["firing_hz"] == b["rpm"]*b["cylinders"]/120, "Firing formula changed")
        require(len(b["engine_phases_rad"]) == cp["harmonics"], "Harmonic phase count")
        require(set(s["component_gain_db"]) == set(b["weights"]), "S1 adds absent component")
        require(t["base_parameters_sha256"] == hashlib.sha256(compact(b).encode()).hexdigest(), "Base parameters changed")
    require(all(len(roles) == 1 for roles in parents.values()), "Template parent crosses roles")
    for g in tables["geometries"]:
        require(5 <= g["range_m"] < 50 and 30 <= g["speed_kmh"] < 90 and g["direction"] in [-1, 1], "Geometry range")
        require(g["crop_end_sample_8k"] - g["crop_start_sample_8k"] == 16000, "Invalid crop length")
        require(g["crop_start_sample_8k"]/8000 > g["max_image_path_delay_s"] + .1, "Insufficient lead-in")
        require(g["crop_end_sample_8k"]/8000 + g["max_image_path_delay_s"] + .1 < 10, "Insufficient tail")
    by_geometry = defaultdict(list); by_event = defaultdict(list)
    for r in tables["render_plan"]:
        t = index["templates"][r["template_id"]]; g = index["geometries"][r["geometry_id"]]
        e = index["events"][r["event_id"]]; dry = index["dry_source_plan"][r["dry_source_id"]]
        require(r["role"] == t["role"] == g["role"] == e["role"] == dry["role"], "Role leakage")
        require(r["replicate_seed"] == t["replicate_seed"] == g["replicate_seed"] == e["replicate_seed"], "Bank mismatch")
        require(r["class_id"] == t["class_id"] == e["class_id"] == dry["class_id"], "Class mismatch")
        require(r["source_parent_group"] == t["source_parent_group"] == e["source_parent_group"], "Source lineage mismatch")
        require(e["geometry_id"] == g["geometry_id"] and e["template_id"] == t["template_id"], "Broken event pairing")
        require(t["template_slot"] == g["template_slot"], "Template/geometry slot mismatch")
        require(dry["template_id"] == t["template_id"] and dry["source_level"] == r["source_level"], "Dry source pairing mismatch")
        require(cfg["cells"][r["cell"]] == {"source": r["source_level"], "path": r["path_level"]}, "Factor toggle mismatch")
        require(all(r[k] is None for k in ["source_waveform_sha256", "rendered_waveform_sha256", "observation_sha256"]), "False waveform completion")
        by_geometry[r["geometry_id"]].append(r); by_event[r["event_id"]].append(r)
    for rr in by_geometry.values():
        require({(r["class_id"], r["cell"]) for r in rr} == {(c, f) for c in [0, 1] for f in cfg["cells"]} and len(rr) == 8, "Class/cell geometry not paired")
    for rr in by_event.values():
        require(len(rr) == 4 and len({r["geometry_id"] for r in rr}) == 1, "Event missing cell")
        for level in ["S0", "S1"]:
            require(len({r["dry_source_id"] for r in rr if r["source_level"] == level}) == 1, "Dry source varies with P")
    for fit in tables["fit_plan"]:
        train = [index["render_plan"][i] for i in fit["train_job_ids"]]
        val = [index["render_plan"][i] for i in fit["validation_job_ids"]]
        require(len(train) == len(set(fit["train_job_ids"])) == 380 and len(val) == len(set(fit["validation_job_ids"])) == 100, "Fit budget or duplicate")
        require(Counter(r["class_id"] for r in train) == {0: 190, 1: 190}, "Unbalanced training budget")
        require(Counter(r["class_id"] for r in val) == {0: 50, 1: 50}, "Unbalanced validation budget")
        require(not ({r["source_parent_group"] for r in train} & {r["source_parent_group"] for r in val}), "Train/validation source leakage")
        require(all(r["role"] == "train" for r in train) and all(r["role"] == "validation" for r in val), "Wrong fit role")
        require(all(r["cell"] == fit["cell"] and r["replicate_seed"] == fit["replicate_seed"] for r in train+val), "Cross-cell/seed fitting")
        require(fit["classifier"] == cfg["classifier"] and fit["classifier"]["C"] == 1 and fit["threshold"] == .5 and fit["selection"] is None, "Common head changed")
    require(len(target) == len({r["file_id"] for r in target}) == 8066, "Wrong target IDs/count")
    require(Counter(r["class_id"] for r in target) == {0: 7810, 1: 256}, "Wrong target classes")
    require(len({r["provenance_group_id"] for r in target}) == 4, "Wrong target groups")
    require(all(r["dataset_id"] == "MELAUDIS" and r["h2_role"] == "exposed_development_evaluation_only" for r in target), "Real target role violation")
    return dict(passed=True, metadata_only=True, counts={**expected, "target": len(target)},
                train_observations=7600, validation_observations=2000, waveform_generation=False,
                model_fitting=False, target_inference=False, ready_for_target_evaluation=False)
