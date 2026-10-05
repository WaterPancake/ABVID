from copy import deepcopy
import numpy as np
import pytest

from cast.config import digest
from cast_improvement.context_prior import ContextPrior
from cast_improvement.evaluate import transformed
from cast_improvement.resolution_evaluation import render
from cast_improvement.resolution_sampling import pack, unpack
from cast_improvement.width_prior import WidthPrior, VARIANT
from test_group_prior import fixture


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_width_one_is_bit_exact_v14_reference(arm):
    cfg, metric, bank = fixture()
    reference = ContextPrior(bank, cfg, metric, VARIANT, "held")
    prior = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.)
    assert prior.statistics == reference.statistics
    a = prior.draw("car", 42, 3, arm); b = reference.draw("car", 42, 3, arm)
    assert a == b
    x, sx = render(a, cfg, "spectrum16"); y, sy = render(b, cfg, "spectrum16")
    assert np.array_equal(x, y) and sx == sy


def test_interior_context_log_ratios_have_known_width_and_harmonics_are_exact():
    cfg, metric, bank = fixture()
    for row in bank:
        group = int(row["group"][-1])-1
        w = np.exp(.2*group*np.linspace(-1,1,16))
        row["parameters"]["noise_weights"] = (w/w.sum()).tolist()
        row["parameters"]["harmonic_fraction"] = float(1/(1+np.exp(-.2*group)))
        row["resolution_calibration"]["parameters"] = deepcopy(row["parameters"])
        row["resolution_calibration"]["initial_parameters"] = deepcopy(row["parameters"])
        row["resolution_calibration_sha256"] = digest(row["resolution_calibration"])
    original = deepcopy(bank)
    ref = ContextPrior(bank, cfg, metric, VARIANT, "held")
    prior = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.5)
    # The fixture has symmetric, interior context trajectories. Its known
    # transformed center is zero and population variance scales by T squared.
    before, after = [], []
    for row in bank:
        for a, b in zip(ref.base.children[row["file_id"]], prior.base.base.children[row["file_id"]]):
            for key in ("spacing_hz", "harmonic_weights"):
                assert a["parameters"][key] == b["parameters"][key]
            z0, z1 = transformed(a["parameters"]), transformed(b["parameters"])
            np.testing.assert_allclose(z1[11:], 1.5*z0[11:], rtol=1e-11, atol=1e-12)
            assert not any(b["flags"].values())
            before.append(z0[28]); after.append(z1[28])
    assert np.var(after) == pytest.approx(2.25*np.var(before), abs=1e-13)
    assert bank == original


def test_unequal_groups_keep_equal_weight_center_and_matched_prototype():
    cfg, metric, bank = fixture()
    extra = deepcopy(bank[0]); extra["file_id"] += "_extra"
    extra["resolution_calibration"]["file_id"] = extra["file_id"]
    extra["resolution_calibration_sha256"] = digest(extra["resolution_calibration"])
    bank.append(extra)
    ref = ContextPrior(bank, cfg, metric, VARIANT, "held")
    prior = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.25)
    for label in metric["class_order"]:
        groups = sorted({r["group"] for r in bank if r["class"] == label})
        centers, means = [], []
        for group in groups:
            rows = [r for r in prior.bank if r["class"] == label and r["group"] == group]
            centers.append(np.mean([transformed(c["parameters"]) for r in rows for c in ref.base.children[r["file_id"]]], axis=0))
            means.append(np.mean([pack(c["parameters"]) for r in rows for c in prior.base.base.children[r["file_id"]]], axis=0))
        assert prior.statistics[label]["width_center"] == np.mean(centers, axis=0).tolist()
        assert prior.base.base.prototypes[label] == unpack(np.mean(means, axis=0))


@pytest.mark.parametrize("arm", ["joint", "prototype", "marginals"])
def test_width_changes_no_donor_or_waveform_stream_and_replays_exactly(arm):
    cfg, metric, bank = fixture()
    ref = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.)
    a = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.1)
    b = WidthPrior(list(reversed(bank)), cfg, metric, VARIANT, "held", 1.1)
    base = ref.draw("truck", 123, 6, arm); row = a.draw("truck", 123, 6, arm)
    assert row == b.draw("truck", 123, 6, arm)
    for key in ("parent_ids", "parent_groups", "coordinate_donors", "selection_seed", "input_id", "group_effect_child_seed", "prior_calibration_parent_ids", "prior_calibration_groups", "calibration_ancestors"):
        assert row[key] == base[key]
    assert [{k:v for k,v in c.items() if k != "flags"} for c in row["group_effect_choices"]] == [{k:v for k,v in c.items() if k != "flags"} for c in base["group_effect_choices"]]
    for key in ("spacing_hz", "harmonic_weights"):
        assert row["parameters"][key] == base["parameters"][key]
    x, streams = render(row, cfg, "spectrum16")
    y, again = render(b.draw("truck",123,6,arm), cfg, "spectrum16")
    _, old = render(base, cfg, "spectrum16")
    assert np.array_equal(x,y) and streams == again == old and np.isfinite(x).all()
    assert row["group_effect_statistics_sha256"] == digest(a.statistics["truck"])


def test_width_projects_extreme_envelopes_and_retains_upstream_flags():
    cfg, metric, bank = fixture()
    for row in bank:
        group = int(row["group"][-1])
        row["parameters"]["envelope_knots"] = [.1 if (j+group)%2 else 3. for j in range(41)]
        row["resolution_calibration"]["parameters"] = deepcopy(row["parameters"])
        row["resolution_calibration"]["initial_parameters"] = deepcopy(row["parameters"])
        row["resolution_calibration_sha256"] = digest(row["resolution_calibration"])
    prior = WidthPrior(bank, cfg, metric, VARIANT, "held", 1.5)
    for label in metric["class_order"]:
        assert prior.statistics[label]["clipping_counts"]["envelope_coordinates_clipped"] > 0
        assert "upstream_clipping_counts" in prior.statistics[label]
    for row in bank:
        for child in prior.base.base.children[row["file_id"]]:
            assert "upstream_projection_flags" in child
            assert all(.1 <= x <= 3 for x in child["parameters"]["envelope_knots"])


@pytest.mark.parametrize("width", [0., .9, 1.01, 2., float('nan'), float('inf'), True])
def test_undeclared_widths_rejected(width):
    cfg,metric,bank = fixture()
    with pytest.raises(ValueError, match="width"):
        WidthPrior(bank,cfg,metric,VARIANT,"held",width)


def test_bad_scope_and_ancestry_fail_before_width_calibration():
    cfg,metric,bank=fixture()
    with pytest.raises(ValueError,match="prior"):
        WidthPrior(bank,cfg,metric,"unknown","held",1.1)
    with pytest.raises(ValueError,match="excluded"):
        WidthPrior(bank,cfg,metric,VARIANT,"g0",1.1)
    bank[0]["parameters"]["spacing_hz"][0]+=1
    with pytest.raises(ValueError,match="ancestry"):
        WidthPrior(bank,cfg,metric,VARIANT,"held",1.1)
