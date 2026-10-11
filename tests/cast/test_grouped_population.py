"""Synthetic metadata/controls only; production budgets, two rendered fixtures."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from abvid.cast import grouped_design, grouped_population as gp
from abvid.cast.config import digest
from abvid.cast.population.resolution_sampling import pack, unpack
from abvid.cast.roles import build_roles
from abvid.data.splits import select
from abvid.paths import resolve


@pytest.fixture(scope="module")
def fixture():
    rows = []
    for g in range(6):
        for c, label in enumerate(("car", "truck")):
            for i in range(50):
                file_id = f"fixture_{g}_{label}_{i:02d}"
                rows.append({"file_id": file_id, "dataset_id": "IDMT", "release_id": "IDMT_V1",
                    "canonical_class": label, "class_id": c, "provider_channel_pair": "CH34",
                    "original_channel_ids": [3, 4], "intended_channel_operation": "mean(all_channels)",
                    "admitted_for_h1": True, "admitted_for_training": True, "integrity_pass": True,
                    "integrity_and_class_pass": True, "metadata_parse_ok": True,
                    "provenance_group_id": f"g{g}", "site_id": f"site{g//2}",
                    "recording_session": None, "original_media_id": None, "physical_vehicle_id": "unknown",
                    "source_file_sha256": hashlib.sha256(file_id.encode()).hexdigest(),
                    "paired_event_id": file_id})
    lock = {"folds": [{"fold": g, "held_out_group": f"g{g}",
              "train_ids": {"42": select([r for r in rows if r["provenance_group_id"] != f"g{g}"],
                                         190, 42, f"grouped_population_fixture|{g}")},
              "test_ids": [r["file_id"] for r in rows if r["provenance_group_id"] == f"g{g}"]}
                      for g in range(6)]}
    parts = grouped_design.partitions(rows, lock)
    roles = build_roles(rows, parts, design_identity_sha256="synthetic_fixture")
    numerical = json.loads(resolve("repo:experiments/cast_coverage/grouped_coverage/numerical_contract.json").read_text())
    bank, obs = {}, {}
    for row in rows:
        file_id = row["file_id"]
        _, g, _, i = file_id.split("_")
        g, i = int(g), int(i)
        weights = np.exp(np.linspace(-1, 1, 16)*(.02*g+.001*i)); weights /= weights.sum()
        parameters = {"spacing_hz": [100+5*g+.1*i]*3, "harmonic_weights": [.125]*8,
                      "noise_weights": weights.tolist(), "harmonic_fraction": .2+.01*g,
                      "envelope_knots": np.exp(np.linspace(-1, 1, 41)*(.01*g+.001*i)).tolist()}
        calibration = {"file_id": file_id, "class_label": row["canonical_class"], "group": f"g{g}",
                       "input_id": file_id, "initial_parameters": deepcopy(parameters),
                       "parameters": deepcopy(parameters), "initial_bank_row_sha256": digest(row),
                       "flags": ["fixture_flag"]}
        bank[file_id] = {"file_id": file_id, "class": row["canonical_class"], "group": f"g{g}",
                        "parent": deepcopy(row), "parameters": parameters,
                        "flags": {"scientific_flags": ["fixture_ambiguity"], "numerical_failure": False},
                        "resolution_calibration": calibration, "resolution_calibration_sha256": digest(calibration)}
        obs[file_id] = {"file_id": file_id, "class": row["canonical_class"], "group": f"g{g}",
                       "parent": deepcopy(row), "descriptors": {
                           family: (np.arange(width)/10+g+i/100+row["class_id"]*100).tolist()
                           for family, width in gp.DIMENSIONS.items()}}
    return roles, numerical, bank, obs


def selected(fixture, partition="site_0"):
    roles, numerical, bank, obs = fixture
    role = roles[partition]
    return ([deepcopy(bank[p]) for p in role.train_ids],
            [deepcopy(obs[p]) for p in role.train_ids], role, deepcopy(numerical))


@pytest.fixture(scope="module")
def population(fixture):
    return gp.build_population(*selected(fixture))


def schedule(population, arm="v15_joint", index=0, label="car", seed=42):
    return next(row for row in grouped_design.planned_schedule([{"id": population.role.partition_id}])
                if row["arm"] == arm and row["index"] == index and row["class"] == label and row["seed"] == seed)


def test_exact_budget_train_only_scales_and_unknown_metadata(fixture, population):
    _, obs, role, _ = selected(fixture)
    assert len(population.statistics["training_ids"]) == 380
    assert population.statistics["held_groups"] == ["g0", "g1"]
    for label in ("car", "truck"):
        rows = [r for r in obs if r["class"] == label]
        values = np.array([r["descriptors"]["envelope"][0] for r in rows])
        expected = np.sqrt(sum((values-values.mean())**2)/190)
        assert population.scales[label]["envelope"][0] == pytest.approx(expected, abs=1e-12)
        assert set(population.statistics["scale_parent_ids"][label]) == set(role.class_train_ids(label))
        assert all(role.source_metadata(p)["recording_session"] is None for p in role.class_train_ids(label))
    assert population.fingerprint == digest(population.statistics)


def test_width_specific_matched_prototypes_and_complete_centers(population):
    for name, prior in population.priors.items():
        for label in ("car", "truck"):
            stats = population.statistics["statistics"][name][label]
            assert len(stats["parent_ids"]) == 190
            assert set(stats["groups"]) == {"g2", "g3", "g4", "g5"}
            means = [np.mean([pack(child["parameters"]) for r in prior.bank
                              if r["class"] == label and r["group"] == group
                              for child in prior.base.base.children[r["file_id"]]], axis=0)
                     for group in stats["groups"]]
            assert stats["prototype_parameters"] == unpack(np.mean(means, axis=0))
            assert population._record(schedule(population, name+"_prototype", label=label))["parameters"] == stats["prototype_parameters"]
    assert population.statistics["statistics"]["v14"]["car"]["children_sha256"] != population.statistics["statistics"]["v15"]["car"]["children_sha256"]


@pytest.mark.parametrize("sampler", ["joint", "prototype", "marginals"])
def test_widths_share_streams_and_every_class_center_ancestor(population, sampler):
    a = population._record(schedule(population, "v14_"+sampler))
    b = population._record(schedule(population, "v15_"+sampler))
    for key in ("parent_ids", "coordinate_donors", "input_id", "selection_seed", "group_effect_child_seed",
                "prior_calibration_parent_ids", "prior_calibration_groups", "calibration_ancestors"):
        assert a[key] == b[key]
    assert len(b["prior_calibration_parent_ids"]) == 190
    assert not set(b["prior_calibration_groups"]) & {"g0", "g1"}
    assert set(b["parent_flags"]) == set(b["parent_ids"])
    assert all(v["original_fit"]["scientific_flags"] == ["fixture_ambiguity"] for v in b["parent_flags"].values())
    assert len(b["child_upstream_projection_flags"]) == {"joint": 1, "prototype": 0, "marginals": 69}[sampler]


@pytest.mark.parametrize("group", ["g0", "g1"])
def test_both_site_groups_are_rejected_before_population_construction(fixture, monkeypatch, group):
    bank, obs, role, numeric = selected(fixture)
    held = next(r for r in fixture[2].values() if r["group"] == group)
    bank[0] = deepcopy(held)
    def forbidden(*args, **kwargs):
        pytest.fail("A contaminated bank reached WidthPrior")
    monkeypatch.setattr(gp, "WidthPrior", forbidden)
    with pytest.raises(ValueError, match="Held ancestor"):
        gp.build_population(bank, obs, role, numeric)


@pytest.mark.parametrize("bad", ["partial_bank", "extra_observation", "held_observation", "wrong_calibration_parent", "nonfinite_descriptor"])
def test_incomplete_or_cross_role_inputs_fail_before_population(fixture, monkeypatch, bad):
    bank, obs, role, numeric = selected(fixture)
    if bad == "partial_bank":
        bank.pop()
    elif bad == "extra_observation":
        obs.append(deepcopy(obs[0]))
    elif bad == "held_observation":
        obs[0] = deepcopy(fixture[3][role.held_ids[0]])
    elif bad == "wrong_calibration_parent":
        bank[0]["resolution_calibration"]["input_id"] = bank[1]["file_id"]
        bank[0]["resolution_calibration_sha256"] = digest(bank[0]["resolution_calibration"])
    else:
        obs[0]["descriptors"]["bands"][0] = float("nan")
    monkeypatch.setattr(gp, "WidthPrior", lambda *a, **kw: pytest.fail("Invalid input reached numerical prior"))
    with pytest.raises(ValueError):
        gp.build_population(bank, obs, role, numeric)


@pytest.mark.parametrize("field,value", [("partition", "group_0"), ("seed", 99), ("temperature", 1.25),
                                         ("child_seed", 0), ("phase_seed", 0), ("input_id", "different-fold-key")])
def test_changed_plan_rejected_before_renderer(population, monkeypatch, field, value):
    planned = schedule(population)
    planned[field] = value
    monkeypatch.setattr(gp, "render", lambda *a, **kw: pytest.fail("Modified schedule reached renderer"))
    with pytest.raises(ValueError, match="frozen fold schedule"):
        population.draw(planned)


def test_two_fixture_renders_repeat_exactly_and_keep_declared_streams(population):
    planned = schedule(population)
    a, wave_a, desc_a = population.draw(planned)
    b, wave_b, desc_b = population.draw(planned)
    assert a == b and desc_a == desc_b and np.array_equal(wave_a, wave_b)
    assert wave_a.shape == (32000,) and np.isfinite(wave_a).all()
    assert a["streams"] == {"phase": planned["phase_seed"], "noise": [planned["noise_seed"]], "purpose": "sampling"}
    assert a["population_fingerprint"] == population.fingerprint


@pytest.mark.parametrize("mutation", ["missing_calibration_parent", "held_direct_parent", "other_class_donor", "held_center_group"])
def test_sample_ancestry_failures_never_reach_renderer(population, monkeypatch, mutation):
    prior = population.priors["v15"]
    original = prior.draw
    def altered(*args, **kwargs):
        record = original(*args, **kwargs)
        if mutation == "missing_calibration_parent":
            record["prior_calibration_parent_ids"] = record["prior_calibration_parent_ids"][:-1]
        elif mutation == "held_direct_parent":
            record["parent_ids"] = [population.role.held_ids[0]]
        elif mutation == "other_class_donor":
            record["parent_ids"] = [population.role.class_train_ids("truck")[0]]
        else:
            record["prior_calibration_groups"] = [*record["prior_calibration_groups"], "g1"]
        return record
    monkeypatch.setattr(prior, "draw", altered)
    monkeypatch.setattr(gp, "render", lambda *a, **kw: pytest.fail("Contaminated ancestry reached renderer"))
    with pytest.raises(ValueError):
        population.draw(schedule(population))


def test_public_metadata_copies_cannot_mutate_frozen_scales_or_statistics(population):
    fingerprint = population.fingerprint
    copied = population.statistics
    copied["training_scales"]["car"]["envelope"][0] = -1
    exposed = population.scales
    exposed["car"]["envelope"][0] = -1
    assert population.scales["car"]["envelope"][0] > 0
    assert digest(population.statistics) == fingerprint
