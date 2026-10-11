"""Training-role-bound CAST-4A populations and the frozen six-arm sampler.

No storage, fitting, held-observation access, width selection or scoring occurs
here. The runner must persist ``statistics`` and ``fingerprint`` before drawing.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import numpy as np

from abvid.cast import grouped_design
from abvid.cast.config import canonical, digest
from abvid.cast.coverage import describe, scales, stack
from abvid.cast.population.resolution_evaluation import render
from abvid.cast.population.resolution_sampling import pack, unpack
from abvid.cast.population.width_prior import WidthPrior, VARIANT
from abvid.cast.roles import FoldRole


WIDTHS = {"v14": 1.0, "v15": 1.1}
DIMENSIONS = {"log_spectrum": 64, "bands": 8, "envelope": 40, "modulation": 20}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(row):
    return {key: row[key] for key in ("file_id", "class", "group")}


def _validate_rows(bank, observations, role):
    require(isinstance(role, FoldRole), "A verified FoldRole is required")
    require(len(role.train_ids) == 380 and all(len(role.class_train_ids(c)) == 190
                                             for c in ("car", "truck")), "Exact 190/class training budget required")
    role.validate_bank([_identity(row) for row in bank])
    role.validate_bank([_identity(row) for row in observations])
    obs = {row["file_id"]: row for row in observations}
    for row in bank:
        observed = obs[row["file_id"]]
        ancestry = [row["parent"], _identity(row)]
        for key, value in row.items():
            if key.endswith("_calibration"):
                require(isinstance(value, dict), "Calibration metadata must be a mapping")
                ancestry.append({k: value[k] for k in ("file_id", "class", "class_label", "group", "input_id") if k in value})
                require(digest(value) == row[key+"_sha256"], "Changed calibration record")
        role.validate_parent(row["file_id"], ancestry=ancestry)
        role.validate_parent(observed["file_id"], ancestry=[observed["parent"], _identity(observed)])
        require(all(row[k] == observed[k] for k in ("file_id", "class", "group", "parent")), "Observation and parameter ancestry differ")
        require(row["parent"] == role.source_metadata(row["file_id"]), "Incomplete source metadata")
        require(set(observed["descriptors"]) == set(DIMENSIONS), "Expected exactly four descriptor families")
        for family, width in DIMENSIONS.items():
            values = np.asarray(observed["descriptors"][family])
            require(values.shape == (width,) and np.isfinite(values).all(), f"Invalid training {family} descriptors")


class GroupedPopulation:
    """One fold's fresh T=1/T=1.1 populations, controls and training scales."""

    def __init__(self, bank, observations, role, numerical):
        # Validate the entire role before constructing either numerical prior.
        _validate_rows(bank, observations, role)
        require(numerical["population"]["width_v15"]["fixed_widths"] == [1.0, 1.1], "Only fixed v14/v15 widths are permitted")
        self.role = role
        self._bank = sorted(deepcopy(bank), key=lambda row: row["file_id"])
        self._by_id = {row["file_id"]: row for row in self._bank}
        obs = sorted(deepcopy(observations), key=lambda row: row["file_id"])
        self._cfg = deepcopy(numerical["renderer_configs"]["spectrum16_temporal41_v11"])
        self._metric = deepcopy(numerical["metric_config"])
        require(self._metric["class_order"] == ["car", "truck"]
                and self._metric["arms"] == ["joint", "prototype", "marginals"]
                and self._metric["seeds"] == grouped_design.SEEDS
                and self._metric["samples_per_class_per_seed_per_arm"] == 50,
                "Changed CAST-4A sample schedule")
        require(bool(role.held_groups), "Missing excluded groups")
        # Legacy code accepts one excluded group. Its scalar is only a second
        # guard; FoldRole above and below excludes the complete group/site set.
        self._metric["held_group"] = role.held_groups[0]
        self._metric["version"] = "cast_improvement_sampling_v1:outer"
        self._priors = {name: WidthPrior(self._bank, self._cfg, self._metric, VARIANT,
                                          "outer", temperature)
                        for name, temperature in WIDTHS.items()}
        self._scales = {label: scales(stack([r["descriptors"] for r in obs if r["class"] == label],
                                           self._metric), self._metric)
                        for label in self._metric["class_order"]}
        self._statistics = {
            "schema_version": 1, "partition": role.partition_id,
            "design_identity_sha256": role.design_identity_sha256,
            "held_groups": list(role.held_groups), "training_groups": list(role.training_groups),
            "training_ids": list(role.train_ids), "bank_sha256": digest(self._bank),
            "observations_sha256": digest(obs), "numerical_contract_sha256": digest(numerical),
            "renderer_config_sha256": digest(self._cfg), "metric_config_sha256": digest(self._metric),
            "widths": dict(WIDTHS), "statistics": {name: deepcopy(prior.statistics)
                                                    for name, prior in self._priors.items()},
            "training_scales": deepcopy(self._scales),
            "scale_parent_ids": {label: sorted(role.class_train_ids(label)) for label in ("car", "truck")},
            "parent_flags": {r["file_id"]: {"original_fit": deepcopy(r["flags"]),
                                            "calibrations": {key: deepcopy(value.get("flags", []))
                                                for key, value in r.items() if key.endswith("_calibration")}}
                             for r in self._bank},
            "population_statistics_reused": False,
            "scale_rule": "equal observation population SD; file-ID order; training class only; frozen family floors",
            "rng_fold_argument": "outer", "width_selection_performed": False,
            "execution_scope": "source-domain effective-observation coverage; no classification or transfer"}
        for name, prior in self._priors.items():
            for label, stats in prior.statistics.items():
                role.validate_bank(stats["parent_ids"], class_name=label,
                                   calibration_ids=stats["parent_ids"])
                require(set(stats["groups"]).isdisjoint(role.held_groups), "Held group entered population statistics")
                require(stats["groups"] == sorted({self._by_id[p]["group"] for p in stats["parent_ids"]}),
                        "Population group ancestry differs")
        self._fingerprint = digest(self._statistics)
        self._schedule = {row["id"]: canonical(row) for row in
                          grouped_design.planned_schedule([{"id": role.partition_id}])}

    @property
    def priors(self):
        """Read-only registry; callers must not mutate its numerical objects."""
        return MappingProxyType(self._priors)

    @property
    def scales(self):
        return deepcopy(self._scales)

    @property
    def statistics(self):
        """Serializable payload to persist and hash before any generation."""
        return deepcopy(self._statistics)

    @property
    def fingerprint(self):
        return self._fingerprint

    def _record(self, planned):
        require(isinstance(planned, dict) and self._schedule.get(planned.get("id")) == canonical(planned),
                "Sample differs from the exact frozen fold schedule")
        name = planned["arm"].split("_", 1)[0]
        prior = self._priors[name]
        label = planned["class"]
        require(prior.statistics == self._statistics["statistics"][name], "Population statistics changed after freezing")
        record = prior.draw(label, planned["seed"], planned["index"], planned["sampler"])
        require(record["input_id"] == planned["input_id"]
                and record["selection_seed"] == planned["selection_seed"]
                and record["group_effect_child_seed"] == planned["child_seed"], "Unpaired sampling streams")
        self.role.validate_donors(record["parent_ids"], class_name=label,
                                  calibration_ids=record["prior_calibration_parent_ids"])
        require(record["prior_calibration_groups"] == prior.statistics[label]["groups"]
                and set(record["prior_calibration_groups"]).isdisjoint(self.role.held_groups),
                "Incomplete or held group-center ancestry")
        require(record["group_effect_statistics_sha256"] == digest(prior.statistics[label]),
                "Sample population fingerprint differs")
        require(record["parent_groups"] == sorted({self._by_id[p]["group"] for p in record["parent_ids"]}),
                "Direct donor group ancestry differs")
        require(set(record["calibration_ancestors"]) == set(record["parent_ids"]), "Direct calibration ancestry incomplete")
        for parent, declared in record["calibration_ancestors"].items():
            row = self._by_id[parent]
            require(declared == {"resolution": row["resolution_calibration_sha256"],
                                 "prior_bank_row": row["resolution_calibration"]["initial_bank_row_sha256"]},
                    "Direct calibration hashes differ")
        choices = record["group_effect_choices"]
        require(len(choices) == {"joint": 1, "prototype": 0, "marginals": 69}[planned["sampler"]],
                "Unexpected virtual-child count")
        if planned["sampler"] == "prototype":
            require(set(record["parent_ids"]) == set(self.role.class_train_ids(label)), "Prototype ancestry is incomplete")
            require(record["coordinate_donors"] == {"rule": "equal_group_mean_then_class_mean", "coordinates": "all"},
                    "Prototype coordinate ancestry differs")
        else:
            donors = record["coordinate_donors"]
            require(isinstance(donors, list) and len(donors) == 69 and set(donors) == set(record["parent_ids"]),
                    "Coordinate donor ancestry differs")
            expected = [donors[0]] if planned["sampler"] == "joint" else donors
            require([choice["residual_parent_id"] for choice in choices] == expected,
                    "Virtual-child residual donors differ from coordinates")
        upstream, selected_children = [], []
        for choice in choices:
            parent = choice["residual_parent_id"]
            self.role.validate_donors([parent], class_name=label,
                                      calibration_ids=record["prior_calibration_parent_ids"])
            require(choice["center_group"] in record["prior_calibration_groups"], "Unknown virtual center group")
            children = prior.base.base.children[parent]
            require(type(choice["child_index"]) is int and 0 <= choice["child_index"] < len(children), "Invalid virtual child index")
            child = children[choice["child_index"]]
            require(all(choice[k] == child[k] for k in ("center_group", "sign", "flags")), "Virtual-child ancestry differs")
            upstream.append(deepcopy(child.get("upstream_projection_flags")))
            selected_children.append(child["parameters"])
        expected_parameters = (prior.statistics[label]["prototype_parameters"] if planned["sampler"] == "prototype"
                               else selected_children[0] if planned["sampler"] == "joint"
                               else unpack([pack(child)[j] for j, child in enumerate(selected_children)]))
        require(record["parameters"] == expected_parameters, "Sample parameters differ from declared virtual population")
        record.update(partition=self.role.partition_id, arm_id=planned["arm"],
                      population_width=planned["temperature"], planned_sample_id=planned["id"],
                      planned_sample_sha256=digest(planned), population_fingerprint=self.fingerprint,
                      held_groups=list(self.role.held_groups),
                      scale_parent_ids=list(self._statistics["scale_parent_ids"][label]),
                      scale_sha256=digest(self._scales[label]),
                      parent_flags={p: deepcopy(self._statistics["parent_flags"][p]) for p in record["parent_ids"]},
                      child_upstream_projection_flags=upstream,
                      projection_statistics_sha256=digest(self._statistics["statistics"][name][label]),
                      role_guard="complete current-fold training population and all known links; whole held group set",
                      population_statistics_reused=False)
        return record

    def draw(self, planned):
        """Return a checked record, one 2-s waveform and four descriptors."""
        record = self._record(planned)
        wave, observed_streams = render(record, self._cfg, "spectrum16")
        require(observed_streams == {"phase": planned["phase_seed"], "noise": [planned["noise_seed"]],
                                     "purpose": "sampling"}, "Renderer streams differ from frozen schedule")
        require(wave.shape == (32000,) and np.isfinite(wave).all(), "Invalid generated waveform")
        descriptors = {key: value.tolist() for key, value in describe(wave, self._cfg, self._metric).items()}
        require(all(np.asarray(descriptors[key]).shape == (width,) for key, width in DIMENSIONS.items()),
                "Generated descriptor dimensions differ")
        return {**record, "streams": observed_streams, "descriptor_sha256": digest(descriptors)}, wave, descriptors


def build_population(bank, observations, role, numerical):
    return GroupedPopulation(bank, observations, role, numerical)
