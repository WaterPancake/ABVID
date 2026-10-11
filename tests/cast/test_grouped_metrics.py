"""Synthetic score schedules exercise CAST-4A's frozen aggregation contract."""
from copy import deepcopy
import json

import numpy as np
import pytest

from abvid.cast import grouped_metrics as metrics
from abvid.cast.coverage import compare


@pytest.fixture(scope="module")
def config():
    return {"class_order": ["car", "truck"], "seeds": [42, 123, 456, 789, 1024],
            "primary_families": list(metrics.FAMILIES), "arms": ["joint", "prototype", "marginals"],
            "samples_per_class_per_seed_per_arm": 50, "coverage_quantiles": [.05, .95],
            "criteria": {"relative_W1_gain_over_each_control_each_class": .05}}


@pytest.fixture(scope="module")
def partitions():
    return [{"id": f"{analysis}_{fold}", "analysis": analysis, "fold": fold,
             "held_counts": {"car": 100 + 100 * fold, "truck": 10 + fold}}
            for analysis, count in (("group", 6), ("site", 3)) for fold in range(count)]


def score(w1, coverage, spread, held_count):
    return {"W1": w1, "coverage": coverage, "generated_count": 50, "held_count": held_count,
            "families": {family: {"W1": w1, "coverage": coverage, "spread_ratio": spread + index * .1,
                                    "per_coordinate_W1": [w1],
                                    "generated_quantiles_05_50_95": [[-.5], [0.], [.5]],
                                    "held_quantiles_05_50_95": [[-.5], [0.], [.5]]}
                         for index, family in enumerate(metrics.FAMILIES)}}


@pytest.fixture(scope="module")
def records(partitions):
    rows = []
    for part in partitions:
        fold = part["fold"]
        for class_index, label in enumerate(metrics.CLASSES):
            for seed_index, seed in enumerate(metrics.SEEDS):
                for arm in metrics.ARMS:
                    width, sampler = arm.split("_")
                    joint = sampler == "joint"
                    w1 = (1. if joint else 2.) + fold * .1 + class_index * .2 + seed_index * .07
                    coverage = .82 + fold * .01 + class_index * .02 + seed_index * .001
                    if width == "v15":
                        w1 -= .1 + fold * .005
                        coverage += .06 + fold * .002
                    rows.append({"partition": part["id"], "class": label, "seed": seed, "arm": arm,
                                 "score": score(w1, coverage, 1. + seed_index * .01, part["held_counts"][label]),
                                 "gates": dict.fromkeys(metrics.GATES, True), "failures": []})
    return rows


@pytest.fixture(scope="module")
def summarized(records, partitions, config):
    return metrics.summarize(records, partitions, config)


def find(rows, *, partition="group_0", label="car", seed=42, arm="v15_joint"):
    return next(row for row in rows if (row["partition"], row["class"], row["seed"], row["arm"])
                == (partition, label, seed, arm))


def fold(summary, name="group_0"):
    return next(row for row in summary["folds"] if row["partition"] == name)


def replace_metric(row, name, value):
    row["score"][name] = value
    for family in metrics.FAMILIES:
        row["score"]["families"][family][name] = value


def test_complete_schedule_keeps_all_seeds_and_separates_group_and_site(summarized):
    assert summarized["status"] == "complete"
    assert summarized["schedule"]["expected_rows"] == summarized["schedule"]["valid_rows"] == 540
    assert len(summarized["seed_records"]) == 540
    assert summarized["primary_v15_all_groups_passed"] is True
    assert summarized["primary_v15_falsified_on_complete_fold_class"] is False
    assert summarized["criteria"]["relative_W1_gain_min_each_matched_control"] == .025
    assert summarized["criteria"]["legacy_0_05_threshold_active"] is False
    for analysis, count in (("group", 6), ("site", 3)):
        report = summarized["analyses"][analysis]
        assert report["complete"] and report["all_folds_passed_by_width"] == {"v14": True, "v15": True}
        assert len(report["partitions"]) == count
        assert report["uncertainty"]["draws"] == 10000
        assert report["uncertainty"]["seed_resampling"] is False
    json.dumps(summarized, allow_nan=False)


def test_five_seed_means_gain_ratio_and_seed_variation_use_matched_width_controls(records, partitions, config):
    rows = deepcopy(records)
    joint = [1., 1., 1., 1., 2.]
    prototype = [1., 2., 3., 4., 5.]
    for index, seed in enumerate(metrics.SEEDS):
        replace_metric(find(rows, seed=seed), "W1", joint[index])
        replace_metric(find(rows, seed=seed, arm="v15_prototype"), "W1", prototype[index])
        replace_metric(find(rows, seed=seed, arm="v14_prototype"), "W1", 1000.)
    report = fold(metrics.summarize(rows, partitions, config))["classes"]["car"]["v15"]
    assert report["arms"]["joint"]["means"]["W1"] == np.mean(joint)
    gain = report["relative_W1_gains"]["prototype"]
    assert gain == 1 - np.mean(joint) / np.mean(prototype)
    assert gain != np.mean(1 - np.asarray(joint) / prototype)
    assert report["arms"]["joint"]["seed_variation"]["W1"] == {
        "values": joint, "min": 1., "max": 2., "all_seeds_numeric": True}


def test_all_fold_requirement_cannot_be_rescued_by_pooled_or_site_success(records, partitions, config):
    rows = deepcopy(records)
    for seed in metrics.SEEDS:
        replace_metric(find(rows, partition="group_2", label="truck", seed=seed), "coverage", .79)
    report = metrics.summarize(rows, partitions, config)
    assert report["status"] == "complete"
    assert report["primary_v15_all_groups_passed"] is False
    assert report["primary_v15_falsified_on_complete_fold_class"] is True
    assert report["analyses"]["site"]["all_folds_passed_by_width"]["v15"] is True
    assert report["primary_v15_criterion_failures"] == [{
        "partition": "group_2", "class": "truck", "width": "v15", "failed_criteria": ["coverage"]}]
    truck = report["analyses"]["group"]["classes"]["truck"]["v15"]["arms"]["joint"]
    assert truck["equal_fold_means"]["coverage"] > .8
    assert truck["worst_observed_fold"]["coverage"]["partitions"] == ["group_2"]
    assert report["analyses"]["group"]["contrasts"]["coverage_v15_minus_v14"]["truck"]["ci95"] is not None


def test_effective_gain_025_and_inclusive_spread_coverage_bounds(records, partitions, config):
    rows = deepcopy(records)
    for seed in metrics.SEEDS:
        row = find(rows, seed=seed)
        replace_metric(row, "W1", .975)
        replace_metric(row, "coverage", .8)
        for index, family in enumerate(metrics.FAMILIES):
            row["score"]["families"][family]["spread_ratio"] = .5 if index % 2 else 2.
        for control in ("prototype", "marginals"):
            replace_metric(find(rows, seed=seed, arm=f"v15_{control}"), "W1", 1.)
    outcome = fold(metrics.summarize(rows, partitions, config))["classes"]["car"]["v15"]
    assert .025 <= outcome["relative_W1_gains"]["prototype"] < .05
    assert outcome["passed"] is True


def test_equal_fold_and_equal_class_weighting_ignores_unequal_held_support(summarized):
    report = summarized["analyses"]["group"]
    car = [fold(summarized, f"group_{i}")["classes"]["car"]["v15"]["arms"]["joint"]["means"]["coverage"]
           for i in range(6)]
    truck = [fold(summarized, f"group_{i}")["classes"]["truck"]["v15"]["arms"]["joint"]["means"]["coverage"]
             for i in range(6)]
    observed = report["equal_class"]["v15"]["arms"]["joint"]["equal_fold_equal_class_means"]["coverage"]
    assert observed == np.mean([np.mean(car), np.mean(truck)])
    assert report["classes"]["car"]["v15"]["arms"]["joint"]["equal_fold_means"]["coverage"] != np.average(
        car, weights=[100 + 100 * i for i in range(6)])


def test_bootstrap_matches_whole_fold_paired_reference_and_resets_per_analysis(summarized):
    for analysis, count in (("group", 6), ("site", 3)):
        report = summarized["analyses"][analysis]
        indices = np.random.Generator(np.random.PCG64(314159)).integers(0, count, size=(10000, count))
        for name, contrasts in report["contrasts"].items():
            car, truck = (np.asarray(contrasts[label]["per_fold"]) for label in metrics.CLASSES)
            assert contrasts["equal_class"]["per_fold"] == list((car + truck) / 2)
            for label in (*metrics.CLASSES, "equal_class"):
                item = contrasts[label]
                values = np.asarray(item["per_fold"])
                assert item["equal_fold_mean"] == np.mean(values)
                assert item["ci95"] == np.quantile(values[indices].mean(axis=1), [.025, .975], method="linear").tolist()
                assert item["ci_status"] == "complete_schedule_paired_whole_fold"
        assert report["contrasts"]["W1_v14_minus_v15"]["car"]["equal_fold_mean"] > 0
        assert report["contrasts"]["coverage_v15_minus_v14"]["car"]["equal_fold_mean"] > 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "failed_gate", "reported_failure", "unknown_gate",
                                    "nonfinite_spread", "nonfinite_auxiliary", "null_spread", "wrong_held_count"])
def test_incomplete_analysis_keeps_evidence_and_never_bootstraps_selected_folds(records, partitions, config, defect):
    rows = deepcopy(records)
    row = find(rows)
    if defect == "missing":
        rows.remove(row)
    elif defect == "duplicate":
        rows.append(deepcopy(row))
    elif defect == "failed_gate":
        row["gates"]["replay"] = False
    elif defect == "reported_failure":
        row["failures"].append("replay waveform changed")
    elif defect == "unknown_gate":
        row["gates"]["ancestry"] = None
    elif defect == "nonfinite_spread":
        row["score"]["families"]["bands"]["spread_ratio"] = float("nan")
    elif defect == "null_spread":
        row["score"]["families"]["bands"]["spread_ratio"] = None
    elif defect == "nonfinite_auxiliary":
        row["score"]["families"]["bands"]["generated_quantiles_05_50_95"][0][0] = float("inf")
    else:
        row["score"]["held_count"] -= 1
    result = metrics.summarize(rows, partitions, config)
    assert result["status"] == "technical_incompletion"
    assert result["primary_v15_all_groups_passed"] is False
    assert result["primary_v15_falsified_on_complete_fold_class"] is False
    group = result["analyses"]["group"]
    assert not group["complete"] and group["uncertainty"]["draws"] == 0
    assert group["criterion_failures"] == []
    assert group["technical_failures"] == [{"partition": "group_0", "class": "car", "width": "v15"}]
    assert all(item["ci95"] is None for contrast in group["contrasts"].values() for item in contrast.values())
    assert result["analyses"]["site"]["complete"]
    assert len(result["seed_records"]) == len(rows)
    assert any(item["validation_failures"] for item in result["seed_records"]) or defect == "missing"
    json.dumps(result, allow_nan=False)


def test_no_results_means_unassessed_not_scientific_falsification(partitions, config):
    result = metrics.summarize([], partitions, config)
    assert result["primary_v15_falsified_on_complete_fold_class"] is None
    assert not result["primary_v15_all_groups_passed"]
    assert len(result["schedule"]["failures"]) == 540
    assert result["analyses"]["group"]["criterion_failures"] == []
    assert result["analyses"]["site"]["uncertainty"]["draws"] == 0


def test_missing_reference_arm_blocks_primary_claim_even_when_all_v15_rows_pass(records, partitions, config):
    rows = deepcopy(records)
    rows.remove(find(rows, arm="v14_marginals"))
    result = metrics.summarize(rows, partitions, config)
    assert not result["primary_v15_all_groups_passed"]
    assert result["primary_v15_falsified_on_complete_fold_class"] is False
    assert not result["analyses"]["group"]["all_folds_passed_by_width"]["v15"]
    assert result["analyses"]["group"]["criterion_failures"] == []
    assert result["analyses"]["group"]["technical_failures"] == [
        {"partition": "group_0", "class": "car", "width": "v14"}]


def test_nonpositive_gain_denominator_remains_undefined_and_fails_criterion(records, partitions, config):
    rows = deepcopy(records)
    for seed in metrics.SEEDS:
        replace_metric(find(rows, seed=seed, arm="v15_prototype"), "W1", 0.)
    result = metrics.summarize(rows, partitions, config)
    outcome = fold(result)["classes"]["car"]["v15"]
    assert outcome["relative_W1_gains"]["prototype"] is None
    assert outcome["passes"]["gain"] is False
    assert result["primary_v15_falsified_on_complete_fold_class"] is True
    interval = result["analyses"]["group"]["contrasts"]["relative_W1_gain_v15_prototype"]["car"]
    assert interval["equal_fold_mean"] is None and interval["ci95"] is None
    assert interval["ci_status"] == "undefined_nonfinite_contrast"
    json.dumps(result, allow_nan=False)


def test_input_records_are_unchanged_and_aggregation_is_order_independent(records, partitions, config, summarized):
    original = deepcopy(records)
    result = metrics.summarize(list(reversed(records)), list(reversed(partitions)), config)
    assert result["folds"] == summarized["folds"]
    assert result["analyses"] == summarized["analyses"]
    assert result["seed_records"] == summarized["seed_records"]
    assert records == original


@pytest.mark.parametrize("field,value", [("partition", "group_99"), ("class", "motorcycle"),
                                       ("seed", 5), ("arm", "v16_joint")])
def test_undeclared_score_identity_is_rejected(records, partitions, config, field, value):
    rows = deepcopy(records)
    rows[0][field] = value
    with pytest.raises(ValueError, match="outside the frozen"):
        metrics.summarize(rows, partitions, config)


def test_partial_partition_contract_and_modified_effective_criteria_are_rejected(records, partitions, config):
    with pytest.raises(ValueError, match="All six group"):
        metrics.summarize(records, partitions[:-1], config)
    cfg = deepcopy(config)
    cfg["criteria"]["relative_W1_gain_over_each_matched_width_control_each_class_each_fold"] = .05
    with pytest.raises(ValueError, match="Changed effective"):
        metrics.summarize(records, partitions, cfg)


def test_existing_compare_output_is_consumed_without_redefining_metrics(records, partitions, config):
    rows = deepcopy(records)
    row = find(rows)
    generated = {family: np.linspace(-2, 2, 50)[:, None] for family in metrics.FAMILIES}
    held = {family: np.linspace(-1, 1, 100)[:, None] for family in metrics.FAMILIES}
    scale = {family: [1.] for family in metrics.FAMILIES}
    original = compare(generated, held, scale, config)
    row["score"] = original
    result = metrics.summarize(rows, partitions, config)
    retained = next(item for item in result["seed_records"] if item["record"] == row)
    assert retained["valid"] and retained["record"]["score"] == original
    assert retained["values"]["W1"] == original["W1"]
    assert retained["values"]["spread:bands"] == original["families"]["bands"]["spread_ratio"]
