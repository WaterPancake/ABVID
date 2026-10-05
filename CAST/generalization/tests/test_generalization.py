import copy
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from cast.config import configuration, digest
from cast.provenance import read_frozen
from cast.renderer import seed_for, validate_controls
from cast_generalization import pipeline
from cast_generalization.method import compare, describe, pack, render, sample, scales, unpack


@pytest.fixture
def cfg():
    return pipeline.read(pipeline.EXT/"config.json")


@pytest.fixture
def base():
    torch.set_num_threads(1)
    return configuration()


@pytest.fixture
def bank():
    rows = []
    for c in ("car", "truck"):
        for g in range(5):
            for i in range(5):
                v = g*5+i
                rows.append({"class": c, "group": f"group_{g}", "file_id": f"{c}_{v:02d}",
                             "flags": {"scientific_flags": ["ambiguous"]},
                             "parameters": {"spacing_hz": [80+v, 90+v, 100+v],
                                            "harmonic_weights": [1/8]*8, "noise_weights": [1/8]*8,
                                            "harmonic_fraction": .1+v/50, "envelope_knots": [1, 1.1, 1.2, 1.1, 1]}})
    return rows


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_sampler_is_deterministic_bounded_and_has_complete_donors(bank, cfg, base, arm):
    a = sample(bank, cfg, "car", 42, 0, arm, base)
    assert a == sample(bank, cfg, "car", 42, 0, arm, base)
    assert all(p.startswith("car_") for p in a["parent_ids"])
    assert set(a["parent_flags"]) == set(a["parent_ids"])
    assert validate_controls(a["parameters"], base)
    if arm != "prototype":
        assert len(a["coordinate_donors"]) == 25
        assert set(a["coordinate_donors"]) == set(a["parent_ids"])
    if arm == "joint":
        parent = next(r for r in bank if r["file_id"] == a["parent_ids"][0])
        assert parent["parameters"] == a["parameters"]
    if arm == "prototype":
        assert len(a["parent_ids"]) == 25


def test_marginals_each_coordinate_has_correct_donor_before_simplex_normalization(bank, cfg, base):
    r = sample(bank, cfg, "car", 123, 2, "marginals", base)
    lookup = {r["file_id"]: r for r in bank}
    v = [pack(lookup[parent]["parameters"])[i] for i, parent in enumerate(r["coordinate_donors"])]
    assert unpack(v) == r["parameters"]
    assert len(set(r["coordinate_donors"])) > 1


def test_prototype_equal_group_weight_even_with_unequal_counts(bank, cfg, base):
    subset = [r for r in bank if r["group"] != "group_0" or r["file_id"].endswith("00")]
    r = sample(subset, cfg, "car", 42, 0, "prototype", base)
    expected = np.mean([80, 87, 92, 97, 102])
    assert r["parameters"]["spacing_hz"][0] == expected


def test_private_selection_stream_does_not_depend_on_global_rng(bank, cfg, base):
    a = sample(bank, cfg, "car", 42, 0, "joint", base)
    np.random.seed(823)
    np.random.randn(100)
    assert sample(bank, cfg, "car", 42, 0, "joint", base) == a
    # Deterministic broad check: no group is accidentally ignored by sampler.
    seen = {g for s in cfg["seeds"] for i in range(50) for g in sample(bank, cfg, "car", s, i, "joint", base)["parent_groups"]}
    assert seen == {f"group_{g}" for g in range(5)}


@pytest.mark.parametrize("changes", [{"label": "motorcycle"}, {"seed": 999}, {"index": 50}, {"arm": "unknown"}])
def test_sampler_refuses_unfrozen_schedule(bank, cfg, base, changes):
    args = dict(label="car", seed=42, index=0, arm="joint")
    args.update(changes)
    with pytest.raises(ValueError):
        sample(bank, cfg, renderer_cfg=base, **args)


def test_sampler_refuses_held_parent(bank, cfg, base):
    bank[0]["group"] = cfg["held_group"]
    with pytest.raises(ValueError, match="contaminated"):
        sample(bank, cfg, "car", 42, 0, "joint", base)


def test_sampling_wave_replays_and_streams_are_disjoint_matched(bank, cfg, base):
    record = sample(bank, cfg, "car", 42, 0, "joint", base)
    a, streams = render(record, base)
    b, repeated = render(record, base)
    assert np.array_equal(a, b) and repeated == streams
    assert np.isfinite(a).all() and a.shape == (32000,)
    assert abs(float(np.sqrt(np.mean(a*a)))-1) < 1e-6
    assert abs(float(a.mean())) < 1e-6
    other = sample(bank, cfg, "car", 42, 0, "prototype", base)
    _, shared = render(other, base)
    assert streams == shared
    assert streams["phase"] != seed_for(base, record["input_id"], "phase", "shared")
    assert streams["noise"][0] not in [seed_for(base, record["input_id"], "noise", p, 0) for p in ("fit", "check")]
    different = copy.deepcopy(record)
    different["input_id"] += ":different"
    x, _ = render(different, base)
    assert not np.array_equal(a, x)
    assert "sampling_realizations" not in base["seeds"]


def test_descriptors_ignore_level_and_dc_and_have_declared_shape(cfg, base):
    x = np.sin(2*np.pi*200*np.arange(32000)/16000)+.03*np.random.default_rng(1).normal(size=32000)
    a, b = describe(x, base, cfg), describe(x*3+10, base, cfg)
    assert {k: len(v) for k, v in a.items()} == {"log_spectrum": 64, "bands": 8, "envelope": 40, "modulation": 20}
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-10, atol=1e-10)


def test_wasserstein_known_shift_and_permutation(cfg):
    a = {f: np.linspace(0, 1, 100)[:, None].repeat(2, axis=1) for f in cfg["primary_families"]}
    b = {f: x+2 for f, x in a.items()}
    unit = {f: [1, 1] for f in a}
    score = compare(a, b, unit, cfg)
    assert score["W1"] == pytest.approx(2)
    assert score["coverage"] == 0
    assert compare({f: x[::-1] for f, x in a.items()}, b, unit, cfg) == score
    identical = compare(a, a, unit, cfg)
    assert identical["W1"] == 0
    assert identical["coverage"] == pytest.approx(.9)
    assert all(s["spread_ratio"] == pytest.approx(1) for s in identical["families"].values())


def test_scale_floors_and_constant_distribution_are_finite(cfg):
    a = {f: np.zeros((25, 2)) for f in cfg["primary_families"]}
    scale = scales(a, cfg)
    for f, value in scale.items():
        assert value == [cfg["scale_floors"][f]]*2
    result = compare(a, a, scale, cfg)
    assert result["W1"] == 0 and result["coverage"] == 1
    assert all(s["spread_ratio"] is None for s in result["families"].values())


def test_marginal_score_explicitly_does_not_certify_joint_support(cfg):
    a = {f: np.array([[0, 0], [1, 1]]) for f in cfg["primary_families"]}
    b = {f: np.array([[0, 1], [1, 0]]) for f in cfg["primary_families"]}
    assert compare(a, b, {f: [1, 1] for f in a}, cfg)["W1"] == 0


@pytest.mark.parametrize("field", pipeline.ANCESTRY_FIELDS)
def test_cross_split_linked_ancestry_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        pipeline.disjoint([{field: "shared"}], [{field: "shared"}])


@pytest.fixture
def held_row(cfg, base):
    # Source-only metadata access; no held audio bytes are read by these tests.
    rows = pipeline.readl(pipeline.ROOT/base["data"]["manifest"])
    return next(r for r in rows if r["provenance_group_id"] == cfg["held_group"])


@pytest.mark.parametrize("field,value", [("dataset_id", "MELAUDIS"), ("provenance_group_id", "reserved"), ("canonical_class", "motorcycle"), ("native_channels", 1), ("raw_audio_modified", True), ("integrity_pass", False), ("source_path", "../secret.wav"), ("source_path", "/tmp/x_SE_CH34.wav"), ("source_path", "dataset/other/x_SE_CH34.wav")])
def test_held_allowlist_rejects_disallowed_metadata_and_paths(held_row, cfg, base, field, value):
    row = copy.deepcopy(held_row)
    row[field] = value
    with pytest.raises(ValueError):
        pipeline.validate_held(row, [row], base, cfg)


def test_held_exact_metadata_and_id_required(held_row, cfg, base):
    row = copy.deepcopy(held_row)
    row["calendar_date"] = "different"
    with pytest.raises(ValueError, match="exact frozen"):
        pipeline.validate_held(row, [held_row], base, cfg)
    with pytest.raises(ValueError, match="exact frozen"):
        pipeline.validate_held(held_row, [], base, cfg)


def test_held_symlink_rejected_before_audio_read(tmp_path, held_row, cfg, base):
    root = tmp_path/"repo"
    allowed = root/base["data"]["allowed_audio_root"]
    allowed.mkdir(parents=True)
    external = tmp_path/"protected.wav"
    external.write_bytes(b"not audio")
    row = copy.deepcopy(held_row)
    row["source_path"] = str((allowed/"link_SE_CH34.wav").relative_to(root))
    (root/row["source_path"]).symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        pipeline.validate_held(row, [row], base, cfg, root)


def test_incomplete_generation_gate_blocks_decode(tmp_path, monkeypatch, cfg, base):
    monkeypatch.setattr(pipeline, "frozen", lambda out: (cfg, base, {"generated_count": 1500}))
    (tmp_path/"generation.complete.json").write_text(json.dumps({"count": 1499}))
    (tmp_path/"tests.json").write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(pipeline.sf, "read", lambda *a, **k: pytest.fail("Must not decode before generation gate"))
    with pytest.raises(ValueError, match="complete before held access"):
        pipeline.evaluate(tmp_path)
    assert not (tmp_path/"held_access_receipt.json").exists()


def test_artifact_inventory_ignores_only_finder_metadata(tmp_path):
    audio = tmp_path/"shape.wav"
    audio.write_bytes(b"experiment bytes")
    metadata = tmp_path/".DS_Store"
    metadata.write_bytes(b"finder state one")
    before = pipeline.file_hashes(tmp_path)
    assert set(before) == {"shape.wav"}
    metadata.write_bytes(b"finder state two")
    assert pipeline.file_hashes(tmp_path) == before
    audio.write_bytes(b"changed experiment bytes")
    assert pipeline.file_hashes(tmp_path) != before
    (tmp_path/"unexpected.json").write_text("{}")
    assert "unexpected.json" in pipeline.file_hashes(tmp_path)


def test_pilot_remains_frozen_and_original_guard_still_excludes_held(cfg, base, held_row):
    parent_cfg, train, lock = read_frozen(pipeline.ROOT/cfg["parent_run"])
    assert digest(parent_cfg) == lock["config_sha256"]
    assert len(train) == 50
    from cast.provenance import validate_row
    with pytest.raises(ValueError, match="Disallowed"):
        validate_row(held_row, base, {held_row["file_id"]})
