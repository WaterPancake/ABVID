"""Role and ancestry checks use source metadata only; no fitting or audio IO."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

from abvid.cast import grouped_design as design
from abvid.cast import roles
from abvid.data.splits import select
from abvid.paths import resolve


@pytest.fixture(scope="module")
def inventory():
    rows = []
    for number in range(6):
        group = f"source_group_{number}"
        for class_id, label in enumerate(("car", "truck")):
            for index in range(100):
                parent = f"{group}_{label}_{index:03d}"
                rows.append({
                    "file_id": parent, "dataset_id": "IDMT", "release_id": "IDMT_V1",
                    "canonical_class": label, "class_id": class_id,
                    "provider_channel_pair": "CH34", "original_channel_ids": [3, 4],
                    "intended_channel_operation": "mean(all_channels)",
                    "admitted_for_h1": True, "admitted_for_training": True,
                    "integrity_pass": True, "integrity_and_class_pass": True,
                    "metadata_parse_ok": True, "provenance_group_id": group,
                    "site_id": f"site_{number // 2}", "calendar_date": f"2020-01-{number + 1:02d}",
                    "conservative_parent_proxy": group,
                    "candidate_recording_id": f"recording_{number}",
                    "source_time_axis": f"axis_{number}",
                    "paired_event_id": parent, "duplicate_group_id": parent,
                    "source_file_sha256": hashlib.sha256(parent.encode()).hexdigest(),
                    "decoded_audio_sha256": hashlib.sha256((parent + "decoded").encode()).hexdigest(),
                    "recording_session": None, "original_media_id": None,
                    "physical_vehicle_id": "unknown", "vehicle_speed": None,
                    "unknown_reason": {"recording_session": "unknown"},
                })
    rows = design.source_rows(rows)
    lock = {"folds": [
        {"fold": number, "held_out_group": group,
         "train_ids": {"42": select([row for row in rows if row["provenance_group_id"] != group],
                                      190, 42, f"role_fixture|{number}")},
         "test_ids": [row["file_id"] for row in rows if row["provenance_group_id"] == group]}
        for number, group in enumerate(sorted({row["provenance_group_id"] for row in rows}))
    ]}
    return rows, design.partitions(rows, lock)


@pytest.fixture(scope="module")
def declared(inventory):
    return roles.build_roles(*inventory, design_identity_sha256="fixture_identity")


def test_complete_role_inventory_preserves_exact_order_and_site_holdouts(inventory, declared):
    rows, partitions = inventory
    by_id = {row["file_id"]: row for row in rows}
    assert set(declared) == {part["id"] for part in partitions}
    for part in partitions:
        role = declared[part["id"]]
        assert role.partition_id == part["id"]
        assert role.train_ids == tuple(part["train_ids"])
        assert role.held_ids == tuple(part["held_ids"])
        assert role.held_groups == tuple(part["held_groups"])
        assert role.design_identity_sha256 == "fixture_identity"
        assert len(role.train_ids) == 380
        assert len(role.class_train_ids("car")) == len(role.class_train_ids("truck")) == 190
        assert not set(role.held_groups) & set(role.training_groups)
        if role.analysis == "site":
            assert len(role.held_groups) == 2
            assert len({by_id[parent]["calendar_date"] for parent in role.held_ids}) == 2
            assert all(by_id[parent]["site_id"] != part["held_label"] for parent in role.train_ids)


def test_roles_are_immutable_and_metadata_is_independent(inventory, declared):
    role = declared["group_0"]
    parent = role.train_ids[0]
    expected = deepcopy(role.source_metadata(parent))
    exposed = role.source_metadata(parent)
    exposed["recording_session"] = "invented_session"
    exposed["unknown_reason"]["recording_session"] = "changed"
    assert role.source_metadata(parent) == expected
    assert expected["recording_session"] is None
    assert expected["physical_vehicle_id"] == "unknown"
    assert "source_template_id" not in expected
    with pytest.raises(FrozenInstanceError):
        role.train_ids = ()
    with pytest.raises(TypeError):
        role._source[parent] = b"{}"
    rows, partitions = deepcopy(inventory)
    independent = roles.build_roles(rows, partitions)["group_0"]
    rows[0]["recording_session"] = "later_mutation"
    partitions[0]["train_ids"].clear()
    assert independent.train_ids == role.train_ids
    assert independent.source_metadata(inventory[0][0]["file_id"])["recording_session"] is None


def test_same_parent_chain_is_validated_but_does_not_approve_fit_reuse(declared):
    role = declared["group_0"]
    parent = role.train_ids[0]
    result = role.validate_parent(parent, ancestry=[parent, role.source_metadata(parent)])
    assert result["file_id"] == parent
    assert result["ancestry_ids"] == [parent, parent]
    assert result["source_metadata"]["recording_session"] is None
    assert result["role_validation_only"] is True
    assert result["fit_reuse_approved"] is False


def test_parent_chain_rejects_another_selected_parent(declared):
    role = declared["group_0"]
    with pytest.raises(ValueError, match="same original parent"):
        role.validate_parent(role.train_ids[0], ancestry=[role.train_ids[1]])


@pytest.mark.parametrize("value", [None, "unknown", "held_unrelated", 123, {}])
def test_unknown_parent_cannot_enter_role(declared, value):
    with pytest.raises(ValueError):
        declared["group_0"].validate_parent(value)


def test_unselected_parent_cannot_enter_training_role(inventory, declared):
    role = declared["group_0"]
    parent = next(row["file_id"] for row in inventory[0]
                  if row["file_id"] not in set(role.train_ids + role.held_ids))
    with pytest.raises(ValueError, match="selected training role"):
        role.validate_parent(parent)
    with pytest.raises(ValueError):
        role.validate_parent(role.held_ids[0])


@pytest.mark.parametrize("link", design.LINKS)
def test_known_held_link_is_rejected_even_with_selected_parent(inventory, declared, link):
    role = declared["site_0"]
    held = role.source_metadata(role.held_ids[0])
    if not design.known(held.get(link)):
        # Synthetic fixture with an actual known session/media/vehicle ancestor.
        # Real frozen IDMT unknown values remain untouched.
        rows, partitions = deepcopy(inventory)
        next(row for row in rows if row["file_id"] == held["file_id"])[link] = "known_held_ancestor"
        role = roles.build_roles(rows, partitions)["site_0"]
        held = role.source_metadata(role.held_ids[0])
    supplied = {"file_id": role.train_ids[0], link: held[link]}
    with pytest.raises(ValueError, match="Held ancestor"):
        role.validate_parent(supplied)


@pytest.mark.parametrize("alias,key", list(roles.ALIASES.items()))
def test_historical_aliases_must_match_pinned_metadata(declared, alias, key):
    role = declared["group_0"]
    parent = role.train_ids[0]
    metadata = role.source_metadata(parent)
    assert role.validate_parent({"file_id": parent, alias: metadata[key]})["file_id"] == parent
    with pytest.raises(ValueError):
        role.validate_parent({"file_id": parent, alias: "incorrect"})


@pytest.mark.parametrize("metadata", [
    {"class": "car", "canonical_class": "truck"},
    {"recording_session": "invented_session"},
    {"recording_session": "unknown"},  # Null and an explicit unknown string are not interchangeable.
    {"unknown_field": None}, {"source_file_sha256": ["not_scalar"]},
])
def test_parent_metadata_cannot_change_or_invent_ancestry(declared, metadata):
    role = declared["group_0"]
    with pytest.raises(ValueError):
        role.validate_parent({"file_id": role.train_ids[0], **metadata})


def test_bank_requires_full_current_population_and_retains_order(declared):
    role = declared["group_0"]
    supplied = list(reversed(role.train_ids))
    bank = role.validate_bank(supplied, calibration_ids=role.train_ids)
    assert bank["parent_ids"] == supplied
    assert bank["role_validation_only"] is True
    assert bank["population_statistics_reused"] is False
    for label in ("car", "truck"):
        ids = role.class_train_ids(label)
        bank = role.validate_bank(ids, class_name=label, calibration_ids=ids)
        assert bank["parent_ids"] == list(ids)
        assert len(bank["calibration_ids"]) == 190


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_class", "held", "empty", "unknown_class"])
def test_bank_rejects_incomplete_or_contaminated_population(declared, defect):
    role = declared["group_0"]
    ids = list(role.class_train_ids("car"))
    label = "car"
    if defect == "missing":
        ids.pop()
    elif defect == "duplicate":
        ids[0] = ids[1]
    elif defect == "wrong_class":
        ids[0] = role.class_train_ids("truck")[0]
    elif defect == "held":
        ids[0] = role.held_ids[0]
    elif defect == "empty":
        ids = []
    else:
        label = "motorcycle"
    with pytest.raises(ValueError):
        role.validate_bank(ids, class_name=label)


def test_donor_subset_requires_complete_class_calibration(declared):
    role = declared["group_0"]
    car = role.class_train_ids("car")
    result = role.validate_donors(car[:3], class_name="car", calibration_ids=car)
    assert result["parent_ids"] == list(car[:3])
    assert result["calibration_ids"] == list(car)
    for wrong in ([], car[:-1], [*car[:-1], car[0]], role.train_ids, role.class_train_ids("truck")):
        with pytest.raises(ValueError):
            role.validate_donors(car[:1], class_name="car", calibration_ids=wrong)
        with pytest.raises(ValueError):
            role.validate_bank(car, class_name="car", calibration_ids=wrong)
    for wrong in ([], [car[0], car[0]], role.class_train_ids("truck")[:1], role.held_ids[:1]):
        with pytest.raises(ValueError):
            role.validate_donors(wrong, class_name="car", calibration_ids=car)


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_population_statistics_reuse_is_never_approved(declared, value):
    role = declared["group_0"]
    car = role.class_train_ids("car")
    with pytest.raises(ValueError, match="cannot be reused"):
        role.validate_bank(car, class_name="car", population_statistics_reused=value)
    with pytest.raises(ValueError, match="cannot be reused"):
        role.validate_donors(car[:1], class_name="car", calibration_ids=car,
                             population_statistics_reused=value)


@pytest.mark.parametrize("value", ["single_parent", {"parent": "mapping"}, None, set()])
def test_ancestry_populations_require_explicit_sequences(declared, value):
    role = declared["group_0"]
    with pytest.raises(ValueError, match="explicit sequence"):
        role.validate_parent(role.train_ids[0], ancestry=value)
    with pytest.raises(ValueError, match="explicit sequence"):
        role.validate_bank(value)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "MELAUDIS"), ("class_id", False), ("class_id", None),
    ("file_id", None), ("provenance_group_id", "unknown"), ("site_id", None),
    ("source_file_sha256", ["alias"]), ("recording_session", {"id": "session"}),
])
def test_malformed_source_metadata_fails_closed(inventory, field, value):
    rows, partitions = deepcopy(inventory)
    rows[0][field] = value
    with pytest.raises(ValueError):
        roles.build_roles(rows, partitions)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "missing_field", "held_omission",
                                     "held_group_omission", "altered_count", "held_in_train", "extra_field"])
def test_partition_metadata_cannot_drift(inventory, defect):
    rows, partitions = deepcopy(inventory)
    if defect == "missing":
        partitions.pop()
    elif defect == "duplicate":
        partitions[-1] = deepcopy(partitions[0])
    elif defect == "missing_field":
        del partitions[0]["analysis"]
    elif defect == "held_omission":
        partitions[-1]["held_ids"].pop()
    elif defect == "held_group_omission":
        partitions[-1]["held_groups"].pop()
    elif defect == "altered_count":
        partitions[0]["train_counts"]["car"] = 189
    elif defect == "held_in_train":
        partitions[0]["train_ids"][0] = partitions[0]["held_ids"][0]
    else:
        partitions[0]["invented_session"] = "claimed_independent"
    with pytest.raises(ValueError):
        roles.build_roles(rows, partitions)


def test_role_loader_rejects_alternate_configuration_before_io(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("An alternate config must not be loaded")
    monkeypatch.setattr(roles, "load", fail)
    with pytest.raises(ValueError, match="exact canonical"):
        roles.load_roles(tmp_path / "other.json")


@pytest.mark.archive
def test_current_verified_design_loads_without_audio_arrays_or_fit_access(monkeypatch):
    root = resolve("artifact:cast/grouped_coverage/cast4a_design_20261007_v1", must_exist=False)
    if not root.is_dir():
        pytest.skip("Frozen CAST-4A metadata artifact is unavailable")
    original = Path.open
    opened = []
    def metadata_only(path, *args, **kwargs):
        assert path.suffix not in {".wav", ".flac", ".mp3", ".npy", ".npz", ".pt", ".pth"}
        assert not any(part in {"parameter_bank.json", "bank.json", "fit_records.jsonl"}
                       for part in path.parts)
        opened.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", metadata_only)
    declared = roles.load_roles()
    assert len(declared) == 9
    assert opened
    identities = {role.design_identity_sha256 for role in declared.values()}
    assert len(identities) == 1 and len(next(iter(identities))) == 64
    for role in declared.values():
        assert len(role.train_ids) == 380
        assert all(role.source_metadata(parent)["recording_session"] is None for parent in role.train_ids)
