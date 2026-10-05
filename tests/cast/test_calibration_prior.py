from copy import deepcopy
import numpy as np
import pytest
from abvid.cast.synthetic import fixtures
from abvid.cast.population.engine import model_config
from abvid.cast.population.block_prior import transform_bank as original_transform
from abvid.cast.population.calibration_prior import transform_bank


def data():
    bank, pairs = [], []
    for c in ("car", "truck"):
        for g in ("a", "b"):
            p = deepcopy(fixtures()["mixture"])
            row = {"file_id": c+g, "class": c, "group": g, "parameters": p}
            bank.append(row)
            observed = np.array([2., 2., 1., 1., 1., 1., 1., 1.]); observed/=observed.sum()
            pairs.append({"file_id": c+g, "class": c, "group": g, "observed_bands": observed.tolist(),
                          "reconstructed_bands": [.125]*8})
    return bank, pairs, model_config()


@pytest.mark.parametrize("spectral", [.5, .65, .8, 1.])
def test_zero_correction_preserves_original_bank_exactly(spectral):
    bank, pairs, cfg = data()
    actual, _ = transform_bank(bank, cfg, spectral, 0., pairs)
    expected, _ = original_transform(bank, cfg, spectral, 1.)
    assert actual == expected


def test_correction_has_known_direction_bounds_and_preserves_other_controls():
    bank, pairs, cfg = data()
    actual, stats = transform_bank(bank, cfg, 1., 1., pairs)
    for before, after in zip(sorted(bank,key=lambda r:r['file_id']), actual):
        p, q = before["parameters"], after["parameters"]
        assert all(p[k] == q[k] for k in p if k != "noise_weights")
        assert q["noise_weights"][0]/q["noise_weights"][2] == pytest.approx(2*p["noise_weights"][0]/p["noise_weights"][2])
        assert sum(q["noise_weights"]) == pytest.approx(1.)
    assert max(abs(x) for x in stats["car"]["clipped_log_noise_correction"]) <= np.log(2)
    assert bank[0]["parameters"]["noise_weights"] == fixtures()["mixture"]["noise_weights"]


@pytest.mark.parametrize("alteration", ["extra", "missing", "duplicate", "group", "nonfinite"])
def test_calibration_rejects_contaminated_or_incomplete_parent_sets(alteration):
    bank, pairs, cfg = data()
    if alteration == "extra": pairs.append({**pairs[0], "file_id": "validation_parent"})
    elif alteration == "missing": pairs.pop()
    elif alteration == "duplicate": pairs.append(deepcopy(pairs[0]))
    elif alteration == "group": pairs[0]["group"] = "validation_group"
    else: pairs[0]["observed_bands"][0] = float("nan")
    with pytest.raises(ValueError): transform_bank(bank, cfg, 1., 1., pairs)
