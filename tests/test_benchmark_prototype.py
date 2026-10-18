"""Prototype-specific leakage and numeric model-bundle boundaries."""
import copy
import json

import numpy as np
import pytest

from abvid.benchmark import PROTOTYPE_ID, load_bundle, parameters, probabilities, validate_roles
from abvid.learning.linear import fixed_fit
from abvid.provenance.hashing import sha256


def example_head():
    return dict(mean=np.zeros(768), scale=np.ones(768), coef=np.ones(768), intercept=np.array(.3))


def test_numeric_export_preserves_fitted_pipeline():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 768))
    y = np.tile([0, 1], 20)
    cfg = dict(C=1., solver="lbfgs", tol=1e-6, max_iter=5000, class_weight=None, fit_intercept=True)
    model = fixed_fit(x, y, cfg)
    held = rng.normal(loc=2., scale=1.5, size=(11, 768))
    head = parameters(model)
    before = head["mean"].copy()
    np.testing.assert_allclose(probabilities(head, held), model.predict_proba(held), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(head["mean"], before)


@pytest.mark.parametrize("change", ["wrong_width", "nonfinite", "zero_scale", "bad_intercept"])
def test_invalid_feature_or_head_rejected(change):
    head = example_head()
    x = np.ones((2, 768))
    if change == "wrong_width": x = x[:, :26]
    if change == "nonfinite": x[0, 0] = np.nan
    if change == "zero_scale": head["scale"][0] = 0
    if change == "bad_intercept": head["intercept"] = np.array([0.])
    with pytest.raises(ValueError): probabilities(head, x)


def test_bundle_tampering_rejected(tmp_path):
    head = example_head()
    np.savez_compressed(tmp_path / "heads.npz", **{k: np.stack([v]) for k, v in head.items()})
    meta = dict(prototype_id=PROTOTYPE_ID, classes=["car", "truck"], models=[{}],
                heads_sha256=sha256(tmp_path / "heads.npz"))
    (tmp_path / "model.json").write_text(json.dumps(meta))
    _, loaded = load_bundle(tmp_path)
    for k in head: np.testing.assert_array_equal(loaded[k][0], head[k])
    with (tmp_path / "heads.npz").open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="parameters changed"): load_bundle(tmp_path)


@pytest.fixture
def roles():
    rows = {}
    for i, (group, label) in enumerate((("train", 0), ("train", 1), ("test", 0), ("test", 1))):
        rows[str(i)] = dict(dataset_id="IDMT", class_id=label, admitted_for_h1=True,
                            admitted_for_training=True, provenance_group_id=group,
                            source_file_sha256=f"sha{i}", duplicate_group_id=f"dup{i}",
                            recording_session=None)
    return rows


def test_unknown_sessions_preserved_without_inventing_independence(roles):
    before = copy.deepcopy(roles)
    validate_roles(["0", "1"], ["2", "3"], roles, "test", 1)
    assert roles == before


@pytest.mark.parametrize("field", ["recording_session", "source_file_sha256", "duplicate_group_id", "provenance_group_id"])
def test_known_links_cannot_cross_roles(roles, field):
    roles["0"][field] = roles["2"][field] = "shared"
    with pytest.raises(ValueError): validate_roles(["0", "1"], ["2", "3"], roles, "test", 1)


def test_budget_and_admission_enforced(roles):
    with pytest.raises(ValueError, match="budget"):
        validate_roles(["0", "1"], ["2", "3"], roles, "test", 2)
    roles["1"]["admitted_for_training"] = False
    with pytest.raises(ValueError, match="Unadmitted"):
        validate_roles(["0", "1"], ["2", "3"], roles, "test", 1)
