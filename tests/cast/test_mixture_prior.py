from copy import deepcopy
import numpy as np
import pytest
from abvid.cast.synthetic import fixtures
from abvid.cast.population.engine import model_config
from abvid.cast.population.block_prior import transform_bank as original
from abvid.cast.population.mixture_prior import transform_bank


def bank(m):
    return [{"file_id": c+g, "class": c, "group": g,
             "parameters": {**deepcopy(fixtures()["mixture"]), "harmonic_fraction": m}}
            for c in ("car", "truck") for g in ("a", "b")]


@pytest.mark.parametrize("spectral", [.65, .8, 1.])
def test_zero_offset_bit_exact_prior_identity(spectral):
    rows, cfg = bank(.4), model_config()
    assert transform_bank(rows, cfg, spectral, 0)[0] == original(rows, cfg, spectral, 1)[0]


@pytest.mark.parametrize("m", [0., .00001, .4, .99999, 1.])
def test_mixture_odds_direction_endpoints_and_no_other_parameter_change(m):
    rows, cfg = bank(m), model_config()
    before = deepcopy(rows)
    after, _ = transform_bank(rows, cfg, 1., -1.)
    assert rows == before
    for row in after:
        p = row["parameters"]
        assert all(p[k] == before[0]["parameters"][k] for k in p if k != "harmonic_fraction")
        if m in (0.,1.): assert p["harmonic_fraction"] == m
        else:
            a = p["harmonic_fraction"]
            assert 0 < a < m
            assert a/(1-a) == pytest.approx(np.exp(-1)*m/(1-m))


def test_unregistered_offset_rejected():
    with pytest.raises(ValueError): transform_bank(bank(.4), model_config(), 1., -2.)
