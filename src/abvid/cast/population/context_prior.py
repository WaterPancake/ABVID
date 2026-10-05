"""Numeric components extracted from CAST/improvement/src/cast_improvement/context_prior.py."""
from collections import Counter


from copy import deepcopy


import numpy as np


from abvid.cast.config import digest


from abvid.cast.population.group_prior import GroupPrior


from abvid.cast.population.resolution_model import validate_controls


from abvid.cast.population.resolution_sampling import KEYS, pack, unpack


MODES = ("group_predictive", "spectral_context", "spectrotemporal_context")


RESTORED = {
    "spectral_context": ("spacing_hz", "harmonic_weights", "envelope_knots"),
    "spectrotemporal_context": ("spacing_hz", "harmonic_weights"),
}


class ContextPrior:
    def __init__(self, bank, cfg, metric, variant, fold):
        if variant not in MODES: raise ValueError("Undeclared group-effect scope")
        self.base = GroupPrior(bank, cfg, metric, "group_predictive", fold)
        self.bank, self.cfg, self.metric, self.variant, self.fold = self.base.bank, cfg, metric, variant, fold
        if variant == "group_predictive":
            self.statistics = self.base.statistics
            return
        restored = RESTORED[variant]
        for label in metric["class_order"]:
            rows = [r for r in self.bank if r["class"] == label]
            groups = sorted({r["group"] for r in rows})
            previous_statistics_sha256 = digest(self.base.statistics[label])
            flags, discarded = Counter(), Counter()
            for row in rows:
                for child in self.base.children[row["file_id"]]:
                    for key in restored:
                        child["parameters"][key] = deepcopy(row["parameters"][key])
                    discarded["spacing_coordinates_clipped"] += child["flags"]["spacing_coordinates_clipped"]
                    child["flags"]["spacing_coordinates_clipped"] = 0
                    if "envelope_knots" in restored:
                        discarded["envelope_coordinates_clipped"] += child["flags"]["envelope_coordinates_clipped"]
                        child["flags"]["envelope_coordinates_clipped"] = 0
                    flags.update({k:int(v) for k,v in child["flags"].items()})
                    validate_controls(child["parameters"],cfg)
            means = [np.mean([pack(child["parameters"]) for row in rows if row["group"] == group
                              for child in self.base.children[row["file_id"]]],axis=0) for group in groups]
            prototype = unpack(np.mean(means,axis=0))
            stats = deepcopy(self.base.statistics[label])
            stats.update(previous_full_scope_statistics_sha256=previous_statistics_sha256,
                group_effect_scope=variant, restored_parent_controls=list(restored),
                group_transformed_controls=[key for key in KEYS if key not in restored],
                restored_control_rule="bit-exact original residual-parent controls; original boundary/ambiguity flags remain in parent bank",
                children_sha256=digest([{"file_id":r["file_id"],"children":self.base.children[r["file_id"]]} for r in rows]),
                prototype_parameters=prototype, clipping_counts=dict(flags),
                discarded_full_scope_projection_counts=dict(discarded))
            self.base.statistics[label] = stats
            self.base.prototypes[label] = prototype
        self.statistics = self.base.statistics

    def draw(self, label, seed, index, arm):
        record = self.base.draw(label,seed,index,arm)
        if self.variant == "group_predictive": return record
        return {**record, "group_effect_model":self.variant,
                "restored_parent_controls":list(RESTORED[self.variant]),
                "calibration_role":"scoped v12 group means and finite virtual population; restored original controls and all center/residual ancestors retained"}
