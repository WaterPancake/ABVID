"""Balanced finite draws from the unchanged v12 group-predictive population."""
from copy import deepcopy
import hashlib
import numpy as np

from cast.config import canonical, digest
from .group_prior import GroupPrior
from .resolution_model import validate_controls
from .resolution_sampling import KEYS, SIZES, pack, unpack

MODES = ("group_predictive", "group_balanced")


def balanced_indices(categories, count, rng):
    """Randomized order and remainder; each category occurs floor/ceil(n/k) times."""
    if categories < 1 or count < 0:
        raise ValueError("Invalid balanced categorical dimensions")
    full, remainder = divmod(count, categories)
    values = np.concatenate((np.tile(np.arange(categories), full), rng.permutation(categories)[:remainder]))
    return rng.permutation(values)


class BalancedPrior:
    def __init__(self, bank, cfg, metric, variant, fold):
        if variant not in MODES: raise ValueError("Undeclared balanced prior")
        self.base = GroupPrior(bank, cfg, metric, "group_predictive", fold)
        self.bank, self.cfg, self.metric, self.variant, self.fold = self.base.bank, cfg, metric, variant, fold
        self.by_id = {r["file_id"]:r for r in self.bank}
        self.plans = {}
        if variant == "group_predictive":
            self.statistics = self.base.statistics
            return
        for label in metric["class_order"]:
            for seed in metric["seeds"]:
                for arm in metric["arms"]:
                    if arm != "prototype":
                        self.plans[self.key(label,seed,arm)] = self.make_plan(label,seed,arm)
        self.statistics = {"population_statistics":self.base.statistics,
                           "sampling_plans":self.plans,
                           "rule":"balanced groups, balanced conditional parents, independently balanced virtual children; independent plans per marginal coordinate"}

    @staticmethod
    def key(label, seed, arm):
        return f"{label}:{seed}:{arm}"

    def make_plan(self, label, seed, arm):
        rows = [r for r in self.bank if r["class"] == label]
        groups = sorted({r["group"] for r in rows})
        grouped = [[r for r in rows if r["group"] == g] for g in groups]
        count = self.metric["samples_per_class_per_seed_per_arm"]
        dimensions = 1 if arm == "joint" else 69
        seed_value = int.from_bytes(hashlib.sha256(canonical([
            f"cast_improvement_sampling_v1:{self.fold}", "balanced_population_v13", label, seed, arm])).digest()[:8],"big")
        rng = np.random.default_rng(seed_value)
        parents = [[] for _ in range(count)]
        children = [[] for _ in range(count)]
        child_count = len(groups)*2
        for coordinate in range(dimensions):
            group_indices = balanced_indices(len(groups), count, rng)
            donor_ids = [None]*count
            for g, group in enumerate(grouped):
                positions = np.flatnonzero(group_indices == g)
                parent_indices = balanced_indices(len(group), len(positions), rng)
                for position, index in zip(positions, parent_indices):
                    donor_ids[position] = group[index]["file_id"]
            child_indices = balanced_indices(child_count, count, rng)
            for i in range(count):
                parents[i].append(donor_ids[i]); children[i].append(int(child_indices[i]))
        return {"seed":seed_value, "class":label, "arm":arm, "count":count,
                "parent_ids_by_index_coordinate":parents, "child_indices_by_index_coordinate":children,
                "groups":groups, "coordinates":dimensions, "children_per_parent":child_count}

    def draw(self, label, seed, index, arm):
        if (label not in self.metric["class_order"] or seed not in self.metric["seeds"]
            or arm not in self.metric["arms"] or not 0 <= index < self.metric["samples_per_class_per_seed_per_arm"]):
            raise ValueError("Outside frozen balanced sampling schedule")
        if self.variant == "group_predictive": return self.base.draw(label,seed,index,arm)
        if arm == "prototype":
            return {**self.base.draw(label,seed,index,arm), "sampling_strategy":"balanced_population_v13", "prototype_unchanged":True}
        plan = self.plans[self.key(label,seed,arm)]
        ids = plan["parent_ids_by_index_coordinate"][index]
        indices = plan["child_indices_by_index_coordinate"][index]
        selected, choices = [], []
        for parent, child_index in zip(ids,indices):
            child = self.base.children[parent][child_index]
            selected.append(child["parameters"])
            choices.append({"residual_parent_id":parent, "child_index":child_index,
                            "center_group":child["center_group"], "sign":child["sign"], "flags":child["flags"]})
        parameters = deepcopy(selected[0]) if arm == "joint" else unpack([pack(p)[j] for j,p in enumerate(selected)])
        validate_controls(parameters,self.cfg)
        parents = sorted(set(ids))
        ancestry = {p:{"resolution":self.by_id[p]["resolution_calibration_sha256"],
                       "prior_bank_row":self.by_id[p]["resolution_calibration"]["initial_bank_row_sha256"]} for p in parents}
        stats = self.base.statistics[label]
        return {"class":label, "seed":seed, "index":index, "arm":arm,
            "origin":"real_calibrated_observation_synthesis", "parameters":parameters,
            "input_id":f"cast_improvement_sampling_v1:{self.fold}:sampling:{label}:{seed}:{index:04d}",
            "selection_seed":plan["seed"], "coordinate_order":list(zip(KEYS,SIZES)),
            "coordinate_donors":ids*69 if arm == "joint" else ids,
            "parent_ids":parents, "parent_groups":sorted({self.by_id[p]["group"] for p in parents}),
            "calibration_ancestors":ancestry, "group_effect_model":"group_predictive",
            "group_effect_statistics_sha256":digest(stats), "group_effect_choices":choices,
            "prior_calibration_parent_ids":stats["parent_ids"], "prior_calibration_groups":stats["groups"],
            "calibration_role":"unchanged v12 predictive population; all class/group means and direct residual donors retained",
            "sampling_strategy":"balanced_population_v13", "balanced_plan_sha256":digest(plan)}
