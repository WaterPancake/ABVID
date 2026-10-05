from copy import deepcopy
import pytest
from test_alternative_prior import setup
from cast_improvement.alternative_prior import sample
from cast_improvement.alternative_replay import ReplaySampler


@pytest.mark.parametrize("mode", ["winner", "equivalent_fits"])
def test_replay_prototype_cache_preserves_all_fields_and_schedule(mode):
    rows, cfg, renderer = setup()
    cfg["seeds"] = [42, 123]
    replay = ReplaySampler(rows, cfg, renderer, mode)
    for seed, index in ((42, 0), (123, 9), (42, 49)):
        for arm in cfg["arms"]:
            assert replay.draw("car", seed, index, arm) == sample(rows, cfg, "car", seed, index, arm, renderer, mode)
    with pytest.raises(ValueError): replay.draw("car", 42, 50, "prototype")
    # The original mutable caller bank cannot silently change a cached fold.
    expected = replay.draw("car", 42, 0, "prototype")
    rows[0]["parameters"]["spacing_hz"] = [399.]*3
    assert replay.draw("car", 42, 0, "prototype") == expected
