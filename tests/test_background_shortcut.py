"""Background-shortcut audit, data side: roles, site folds, separation and the feature pin."""
import json
from pathlib import Path

import numpy as np
import pytest

from abvid.audit import background_shortcut as bs
from abvid.audit.run_record import write_once
from abvid.provenance.hashing import sha256

CONFIG = Path(__file__).resolve().parents[1] / "experiments/background_shortcut/config.json"
STAMP = "2019-10-22-08-40"


def clip(site, kind, position, mic="SE", channel="CH34"):
    """An IDMT clip name in the provider's convention (kind: C, T, M or BG)."""
    if kind == "BG":
        return f"{STAMP}_{site}_50Kmh_{position}_{mic}_{channel}-BG.wav"
    return f"{STAMP}_{site}_50Kmh_{position}_M_D_{kind}R_{mic}_{channel}.wav"


def corpus(cars=3, trucks=3, background=2):
    names = []
    for site in ("Site-A", "Site-B", "Site-C"):
        position = iter(range(1000, 10 ** 6, 1000))
        names += [clip(site, "C", next(position)) for _ in range(cars)]
        names += [clip(site, "T", next(position)) for _ in range(trucks)]
        names += [clip(site, "BG", next(position)) for _ in range(background)]
        names += [clip(site, "C", next(position), "ME", "CH12"),
                  clip(site, "BG", next(position), "ME", "CH12"),
                  clip(site, "M", next(position), channel="CH12")]
    return bs.clip_table(names)


def test_roles_follow_admission_rules():
    rows = corpus()
    assert [sum(r["role"] == role for r in rows) for role in ("vehicle", "background", "excluded")] \
        == [18, 6, 9]
    assert {r["provider_class"] for r in rows if r["role"] == "excluded"} \
        == {"car", "background", "motorcycle"}


def test_unparsed_name_fails_loud():
    with pytest.raises(ValueError, match="unparsed"):
        bs.clip_table(["not-an-idmt-clip.wav"])


def test_folds_hold_out_whole_sites_with_a_balanced_budget():
    rows = corpus()
    folds = bs.leave_one_site_out(rows, budget=2, seed=42)
    assert [f["held_out_site"] for f in folds] == ["Site-A", "Site-B", "Site-C"]
    for f in folds:
        assert f["held_out_site"] not in {rows[i]["site_id"] for i in f["train"]}
        assert {rows[i]["site_id"] for i in f["test_vehicle"] + f["test_background"]} \
            == {f["held_out_site"]}
        assert sorted(rows[i]["provider_class"] for i in f["train"]) == ["car"] * 2 + ["truck"] * 2
        assert len(f["test_vehicle"]) == 6 and len(f["test_background"]) == 2


def test_folds_are_reproducible_per_seed():
    rows = corpus(cars=6, trucks=6)
    folds = bs.leave_one_site_out(rows, budget=3, seed=7)
    assert folds == bs.leave_one_site_out(rows, budget=3, seed=7)
    assert folds != bs.leave_one_site_out(rows, budget=3, seed=8)


def test_budget_larger_than_the_training_pool_raises():
    with pytest.raises(ValueError, match="budget 7"):
        bs.leave_one_site_out(corpus(), budget=7, seed=42)


@pytest.mark.parametrize("key", bs.SEPARATION_KEYS)
def test_induced_leak_is_caught(key):
    rows = corpus()
    fold = bs.leave_one_site_out(rows, budget=2, seed=42)[0]
    first_train, first_test = fold["train"][0], fold["test_vehicle"][0]
    rows[first_train] = dict(rows[first_train], **{key: rows[first_test][key]})
    with pytest.raises(ValueError, match=f"{key} leakage"):
        bs.assert_separated(rows, fold["train"], fold["test_vehicle"], keys=(key,))


def test_unknown_provenance_is_not_treated_as_separated():
    rows = corpus()
    fold = bs.leave_one_site_out(rows, budget=2, seed=42)[0]
    first_test = fold["test_vehicle"][0]
    rows[first_test] = dict(rows[first_test], candidate_group_id=None)
    with pytest.raises(ValueError, match="unknown"):
        bs.assert_separated(rows, fold["train"], fold["test_vehicle"])


def test_feature_pin(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, ids=np.array(["a.wav", "b.wav"]), X=np.ones((2, 3), dtype=np.float32))
    ids, X = bs.load_features(path, sha256(path))
    assert ids == ["a.wav", "b.wav"] and X.shape == (2, 3)
    with pytest.raises(ValueError, match="does not match pin"):
        bs.load_features(path, "0" * 64)
    np.savez(path, ids=np.array(["a.wav", "a.wav"]), X=np.ones((2, 3)))
    with pytest.raises(ValueError, match="duplicate"):
        bs.load_features(path, sha256(path))


def test_shipped_config_validates_and_unknown_keys_are_rejected(tmp_path):
    config = bs.load_config(CONFIG)
    altered = tmp_path / "config.json"
    altered.write_text(json.dumps(dict(config, threshold=0.5)))
    with pytest.raises(ValueError, match="unknown"):
        bs.load_config(altered)


def test_write_once_refuses_to_overwrite(tmp_path):
    write_once(tmp_path, "result.json", {"a": 1})
    with pytest.raises(FileExistsError):
        write_once(tmp_path, "result.json", {"a": 2})
