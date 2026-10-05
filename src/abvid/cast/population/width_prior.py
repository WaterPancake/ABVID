"""Numeric components extracted from CAST/improvement/src/cast_improvement/width_prior.py."""
from collections import Counter


from copy import deepcopy


import numpy as np


from abvid.cast.config import digest


from abvid.cast.population.context_prior import ContextPrior


from abvid.cast.population.evaluate import transformed


from abvid.cast.population.group_prior import inverse


from abvid.cast.population.resolution_model import validate_controls


from abvid.cast.population.resolution_sampling import pack, unpack


VARIANT = "spectrotemporal_context"


TEMPERATURES = (1., 1.1, 1.25, 1.5)


class WidthPrior:
    def __init__(self, bank, cfg, metric, variant, fold, temperature=1.):
        if variant != VARIANT:
            raise ValueError("Undeclared width prior")
        if isinstance(temperature, bool) or temperature not in TEMPERATURES:
            raise ValueError("Undeclared or non-finite width")
        self.temperature = float(temperature)
        self.base = ContextPrior(bank, cfg, metric, variant, fold)
        self.bank, self.cfg, self.metric, self.variant, self.fold = self.base.bank, cfg, metric, variant, fold
        population = self.base.base
        if temperature == 1.:
            self.statistics = self.base.statistics
            return
        for label in metric["class_order"]:
            rows = [r for r in self.bank if r["class"] == label]
            groups = sorted({r["group"] for r in rows})
            previous = deepcopy(population.statistics[label])
            # Equal groups, equal parents within group, equal existing children.
            means = [np.mean([transformed(child["parameters"])
                              for r in rows if r["group"] == group
                              for child in population.children[r["file_id"]]], axis=0)
                     for group in groups]
            center = np.mean(means, axis=0)
            projections = Counter()
            for row in rows:
                gauge = float(np.log(row["parameters"]["envelope_knots"]).mean())
                for child in population.children[row["file_id"]]:
                    original = deepcopy(child["parameters"])
                    z = transformed(original)
                    z[11:] = center[11:] + temperature * (z[11:] - center[11:])
                    parameters, flags = inverse(z, cfg, gauge)
                    for key in ("spacing_hz", "harmonic_weights"):
                        parameters[key] = original[key]
                    flags["spacing_coordinates_clipped"] = 0
                    child["upstream_projection_flags"] = child["flags"]
                    child["parameters"], child["flags"] = parameters, flags
                    projections.update({k: int(v) for k, v in flags.items()})
                    validate_controls(parameters, cfg)
            means = [np.mean([pack(child["parameters"])
                              for r in rows if r["group"] == group
                              for child in population.children[r["file_id"]]], axis=0)
                     for group in groups]
            prototype = unpack(np.mean(means, axis=0))
            validate_controls(prototype, cfg)
            stats = deepcopy(previous)
            stats.update(population_width=temperature,
                width_center=center.tolist(),
                width_controls=["noise_weights", "harmonic_fraction", "envelope_knots"],
                width_rule="mu + T*(z-mu) on context coordinates only; equal group/parent/child center; original residual-parent envelope gauge",
                upstream_scoped_statistics_sha256=digest(previous),
                upstream_clipping_counts=previous["clipping_counts"],
                clipping_counts=dict(projections),
                projection_scope="width inverse projections; upstream projections and original parent flags retained separately",
                children_sha256=digest([{"file_id":r["file_id"], "children":population.children[r["file_id"]]} for r in rows]),
                prototype_parameters=prototype)
            population.statistics[label] = stats
            population.prototypes[label] = prototype
        self.statistics = population.statistics

    def draw(self, label, seed, index, arm):
        record = self.base.draw(label, seed, index, arm)
        if self.temperature == 1.:
            return record
        return {**record, "population_width":self.temperature,
                "group_effect_model":"spectrotemporal_context_width_v15",
                "calibration_role":"v14 scoped population with frozen source-only width center; all group centers, residual donors and width-center ancestors retained"}
