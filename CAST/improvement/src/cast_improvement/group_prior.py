"""Class-conditional recombination of source group effects and parent residuals."""
from collections import Counter
from copy import deepcopy
import hashlib
import numpy as np

from cast.config import canonical, digest
from .evaluate import transformed
from .resolution_bank import draw as previous_draw, validate_ancestry
from .resolution_model import validate_controls
from .resolution_sampling import pack, unpack

MODES = ("spectrum16", "group_recombined", "group_predictive")


def inverse(z, cfg, envelope_log_gauge):
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (69,) or not np.isfinite(z).all() or not np.isfinite(envelope_log_gauge):
        raise ValueError("Invalid group-effect vector")
    def softmax(x):
        e = np.exp(x-x.max()); return (e/e.sum()).tolist()
    spacing_bounds = cfg["renderer"]["spacing_hz_bounds"]
    envelope_bounds = cfg["renderer"]["envelope_bounds"]
    spacing_logs = np.clip(z[:3], *np.log(spacing_bounds))
    envelope_logs = z[28:]-z[28:].mean()+envelope_log_gauge
    bounded_envelope_logs = np.clip(envelope_logs, *np.log(envelope_bounds))
    mix_logit = np.clip(z[27], -15, 15)
    def bounded_exp(original, bounded, bounds):
        # Saturated coordinates must equal the declared endpoints exactly.
        lower, upper = np.log(bounds)
        return np.where(original <= lower, bounds[0],
                        np.where(original >= upper, bounds[1], np.exp(bounded))).tolist()
    p = {"spacing_hz": bounded_exp(z[:3], spacing_logs, spacing_bounds),
         "harmonic_weights": softmax(z[3:11]), "noise_weights": softmax(z[11:27]),
         "harmonic_fraction": float(1/(1+np.exp(-mix_logit))),
         "envelope_knots": bounded_exp(envelope_logs, bounded_envelope_logs, envelope_bounds)}
    validate_controls(p, cfg)
    flags = {"spacing_coordinates_clipped": int(np.count_nonzero(spacing_logs != z[:3])),
             "envelope_coordinates_clipped": int(np.count_nonzero(bounded_envelope_logs != envelope_logs)),
             "mixture_logit_clipped": bool(mix_logit != z[27])}
    return p, flags


class GroupPrior:
    def __init__(self, bank, cfg, metric, variant, fold):
        if variant not in MODES or len({r["file_id"] for r in bank}) != len(bank):
            raise ValueError("Undeclared group prior or duplicate parent")
        forbidden = {metric["held_group"]}
        if fold != "outer": forbidden.add(fold)
        if any(r["group"] in forbidden for r in bank):
            raise ValueError("Group-prior ancestry crosses excluded group boundary")
        if {r["class"] for r in bank} != set(metric["class_order"]):
            raise ValueError("Missing or undeclared prior class")
        self.bank = sorted(bank, key=lambda r:r["file_id"])
        self.cfg, self.metric, self.variant, self.fold = cfg, metric, variant, fold
        for row in self.bank:
            validate_controls(row["parameters"], cfg); validate_ancestry(row)
        self.children, self.statistics, self.prototypes = {}, {}, {}
        if variant == "spectrum16": return
        for label in metric["class_order"]:
            rows = [r for r in self.bank if r["class"] == label]
            groups = sorted({r["group"] for r in rows})
            if len(groups) < 3:
                raise ValueError("At least three training groups per class required")
            values = {r["file_id"]: transformed(r["parameters"]) for r in rows}
            centers = {g:np.mean([values[r["file_id"]] for r in rows if r["group"]==g], axis=0) for g in groups}
            center = np.mean(list(centers.values()), axis=0)
            factor = 1. if variant == "group_recombined" else float(np.sqrt((len(groups)+1)/(len(groups)-1)))
            signs = [1.] if variant == "group_recombined" else [-1.,1.]
            flags = Counter(); child_records = []
            for row in rows:
                residual = values[row["file_id"]]-centers[row["group"]]
                gauge = float(np.log(row["parameters"]["envelope_knots"]).mean())
                children = []
                for g in groups:
                    for sign in signs:
                        z = center + residual + sign*factor*(centers[g]-center)
                        p, clipped = inverse(z, cfg, gauge)
                        child = {"parameters":p, "center_group":g, "sign":sign, "flags":clipped}
                        children.append(child)
                        flags.update({k:int(v) for k,v in clipped.items()})
                self.children[row["file_id"]] = children
                child_records.append({"file_id":row["file_id"], "children":children})
            # Same equal-group / equal-parent / equal-child population for all arms.
            means = [np.mean([pack(child["parameters"]) for row in rows if row["group"]==g
                              for child in self.children[row["file_id"]]],axis=0) for g in groups]
            self.prototypes[label] = unpack(np.mean(means,axis=0))
            self.statistics[label] = {"class":label, "groups":groups, "parent_ids":[r["file_id"] for r in rows],
                "parent_parameter_sha256":{r["file_id"]:digest(r["parameters"]) for r in rows},
                "parent_calibration_sha256":{r["file_id"]:r["resolution_calibration_sha256"] for r in rows},
                "class_center":center.tolist(), "group_centers":{g:x.tolist() for g,x in centers.items()},
                "factor":factor, "signs":signs, "virtual_children_per_parent":len(groups)*len(signs),
                "virtual_population_size":len(rows)*len(groups)*len(signs),
                "children_sha256":digest(child_records), "clipping_counts":dict(flags),
                "prototype_parameters":self.prototypes[label],
                "envelope_gauge_rule":"retain each residual parent's original mean log envelope",
                "weighting":"equal group, equal original parent within group, equal virtual child"}

    def draw(self, label, seed, index, arm):
        original = previous_draw(self.bank,self.cfg,self.metric,label,seed,index,arm,"spectrum16",self.fold)
        if self.variant == "spectrum16": return original
        stats = self.statistics[label]
        child_seed = int.from_bytes(hashlib.sha256(canonical([
            f"cast_improvement_sampling_v1:{self.fold}","group_effect_v12_children",label,seed,index,arm])).digest()[:8],"big")
        rng = np.random.default_rng(child_seed)
        choices = []
        def child(parent):
            k = int(rng.integers(len(self.children[parent])))
            row = self.children[parent][k]
            choices.append({"residual_parent_id":parent,"child_index":k,
                            "center_group":row["center_group"],"sign":row["sign"],"flags":row["flags"]})
            return row["parameters"]
        if arm == "joint":
            parameters = deepcopy(child(original["parent_ids"][0]))
        elif arm == "marginals":
            parameters = unpack([pack(child(parent))[j] for j,parent in enumerate(original["coordinate_donors"])])
        else:
            parameters = deepcopy(self.prototypes[label])
        validate_controls(parameters,self.cfg)
        return {**original,"parameters":parameters,"group_effect_model":self.variant,
            "group_effect_statistics_sha256":digest(stats),"group_effect_child_seed":child_seed,
            "group_effect_choices":choices,"prior_calibration_parent_ids":stats["parent_ids"],
            "prior_calibration_groups":stats["groups"],
            "calibration_role":"all class/group centers and finite virtual population; direct residual donors retained separately"}
