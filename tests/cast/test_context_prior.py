from copy import deepcopy
import math
import numpy as np
import pytest

from abvid.cast.config import digest
from abvid.cast.population.context_prior import ContextPrior
from abvid.cast.population.group_prior import GroupPrior
from abvid.cast.population.resolution_evaluation import render
from abvid.cast.population.resolution_sampling import pack, unpack
from test_group_prior import fixture


@pytest.mark.parametrize("mode",["spectral_context","spectrotemporal_context"])
def test_scoped_children_restore_exact_parent_blocks_and_keep_transformed_blocks(mode):
    cfg,metric,bank=fixture();original=deepcopy(bank)
    ref=GroupPrior(bank,cfg,metric,"group_predictive","held")
    scoped=ContextPrior(bank,cfg,metric,mode,"held")
    for row in bank:
        for old,new in zip(ref.children[row["file_id"]],scoped.base.children[row["file_id"]]):
            for key in ("spacing_hz","harmonic_weights"):
                assert new["parameters"][key]==row["parameters"][key]
            for key in ("noise_weights","harmonic_fraction"):
                assert new["parameters"][key]==old["parameters"][key]
            expected=row["parameters"] if mode=="spectral_context" else old["parameters"]
            assert new["parameters"]["envelope_knots"]==expected["envelope_knots"]
            assert (new["center_group"],new["sign"])==(old["center_group"],old["sign"])
    assert bank==original


@pytest.mark.parametrize("mode",["spectral_context","spectrotemporal_context"])
def test_prototype_is_same_weighted_scoped_population_and_known_restored_spacing(mode):
    cfg,metric,bank=fixture();prior=ContextPrior(bank,cfg,metric,mode,"held")
    expected=100*math.cosh(.015)*(1+2*math.cosh(.08))/3
    assert prior.base.prototypes["car"]["spacing_hz"][0]==pytest.approx(expected,abs=1e-11)
    # Unequal group sizes must not silently change equal-group weighting.
    extra=deepcopy(bank[0]);extra["file_id"]+="_extra"
    extra["resolution_calibration"]["file_id"]=extra["file_id"]
    extra["resolution_calibration_sha256"]=digest(extra["resolution_calibration"])
    prior=ContextPrior(bank+[extra],cfg,metric,mode,"held")
    weights=[]
    for group in ("g0","g1","g2"):
        children=[pack(ch["parameters"]) for r in prior.bank if r["class"]=="car" and r["group"]==group for ch in prior.base.children[r["file_id"]]]
        weights.append(np.mean(children,axis=0))
    assert prior.base.prototypes["car"]==unpack(np.mean(weights,axis=0))


@pytest.mark.parametrize("mode",["spectral_context","spectrotemporal_context"])
@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_identical_selection_and_waveform_streams_complete_ancestry_and_determinism(mode,arm):
    cfg,metric,bank=fixture();ref=GroupPrior(bank,cfg,metric,"group_predictive","held")
    a=ContextPrior(bank,cfg,metric,mode,"held");b=ContextPrior(list(reversed(bank)),cfg,metric,mode,"held")
    old=ref.draw("car",42,4,arm);row=a.draw("car",42,4,arm)
    assert row==b.draw("car",42,4,arm)
    for key in ("parent_ids","parent_groups","coordinate_donors","selection_seed","group_effect_child_seed","input_id","prior_calibration_parent_ids","prior_calibration_groups","calibration_ancestors"):
        assert row[key]==old[key]
    assert [{k:v for k,v in c.items() if k!="flags"} for c in row["group_effect_choices"]]==[{k:v for k,v in c.items() if k!="flags"} for c in old["group_effect_choices"]]
    x,sx=render(row,cfg,"spectrum16");y,sy=render(b.draw("car",42,4,arm),cfg,"spectrum16")
    _,so=render(old,cfg,"spectrum16")
    assert np.array_equal(x,y) and sx==sy==so and np.isfinite(x).all()
    assert row["group_effect_statistics_sha256"]==digest(a.statistics["car"])


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_reference_is_exact_group_predictive(arm):
    cfg,metric,bank=fixture()
    assert ContextPrior(bank,cfg,metric,"group_predictive","held").draw("truck",123,7,arm)==GroupPrior(bank,cfg,metric,"group_predictive","held").draw("truck",123,7,arm)


@pytest.mark.parametrize("mode",["spectral_context","spectrotemporal_context"])
def test_discarded_projections_are_not_counted_as_applied(mode):
    cfg,metric,bank=fixture()
    for row in bank:
        group=int(row["group"][-1])
        row["parameters"]["spacing_hz"]=[10.,100.,400.] if group==0 else [400.,100.,10.]
        row["parameters"]["envelope_knots"]=[.1 if (j+group)%2 else 3. for j in range(41)]
        row["resolution_calibration"]["parameters"]=deepcopy(row["parameters"])
        row["resolution_calibration"]["initial_parameters"]=deepcopy(row["parameters"])
        row["resolution_calibration_sha256"]=digest(row["resolution_calibration"])
    prior=ContextPrior(bank,cfg,metric,mode,"held")
    for label in metric["class_order"]:
        stats=prior.statistics[label]
        assert stats["discarded_full_scope_projection_counts"]["spacing_coordinates_clipped"]>0
        assert stats["clipping_counts"]["spacing_coordinates_clipped"]==0
        if mode=="spectral_context":
            assert stats["discarded_full_scope_projection_counts"]["envelope_coordinates_clipped"]>0
            assert stats["clipping_counts"]["envelope_coordinates_clipped"]==0


def test_excluded_tampered_and_unknown_scope_rejected():
    cfg,metric,bank=fixture()
    with pytest.raises(ValueError,match="excluded"):ContextPrior(bank,cfg,metric,"spectral_context","g0")
    bad=deepcopy(bank);bad[0]["parameters"]["spacing_hz"][0]+=1
    with pytest.raises(ValueError,match="ancestry"):ContextPrior(bad,cfg,metric,"spectral_context","held")
    with pytest.raises(ValueError,match="scope"):ContextPrior(bank,cfg,metric,"undeclared","held")
