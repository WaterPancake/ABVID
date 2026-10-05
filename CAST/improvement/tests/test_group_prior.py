from copy import deepcopy
import math
import numpy as np
import pytest

from cast.config import digest
from cast.synthetic import fixtures
from cast_improvement.group_prior import GroupPrior, inverse
from cast_improvement.evaluate import transformed
from cast_improvement.resolution_model import model_config, lift_controls
from cast_improvement.temporal_prior import lift
from cast_improvement.resolution_bank import draw as previous_draw
from cast_improvement.resolution_evaluation import render


def fixture():
    cfg = model_config()
    metric = {"class_order":["car","truck"],"seeds":[42,123,456,789,1024],
              "arms":["joint","prototype","marginals"],"samples_per_class_per_seed_per_arm":50,
              "held_group":"outer"}
    bank = []
    for label in metric["class_order"]:
        for g in (-1,0,1):
            for j in (-.5,.5):
                p = lift_controls(lift(deepcopy(fixtures()["mixture"])),cfg)
                p["spacing_hz"] = [100*math.exp(.08*g+.03*j)]*3
                p["envelope_knots"] = np.exp((.06*g+.02*j)*np.linspace(-1,1,41)).tolist()
                file_id = f"{label}_{g+1}_{j+.5:g}"
                row = {"file_id":file_id,"class":label,"group":f"g{g+1}","parameters":p,"flags":{}}
                record = {"file_id":file_id,"class_label":label,"group":row["group"],
                          "initial_parameters":deepcopy(p),"parameters":deepcopy(p),
                          "initial_bank_row_sha256":"known fixture"}
                row.update(resolution_calibration=record,resolution_calibration_sha256=digest(record))
                bank.append(row)
    return cfg, metric, bank


def test_transform_inverse_preserves_valid_interior_controls_and_gauge():
    cfg, _, bank = fixture();p=bank[0]["parameters"]
    result, flags=inverse(transformed(p),cfg,float(np.log(p["envelope_knots"]).mean()))
    for key in p:np.testing.assert_allclose(result[key],p[key],rtol=1e-12,atol=1e-12)
    assert not any(flags.values())
    with pytest.raises(ValueError):inverse(np.zeros(68),cfg,0.)
    with pytest.raises(ValueError):inverse(np.full(69,np.nan),cfg,0.)


@pytest.mark.parametrize("mode",["group_recombined","group_predictive"])
def test_known_group_and_residual_variances_and_prototype(mode):
    cfg, metric, bank = fixture();prior=GroupPrior(bank,cfg,metric,mode,"held")
    values=np.array([transformed(child["parameters"])[0] for row in bank if row["class"]=="car"
                     for child in prior.children[row["file_id"]]])
    a=1 if mode=="group_recombined" else math.sqrt(2)
    assert values.mean()==pytest.approx(math.log(100),abs=1e-14)
    assert values.var()==pytest.approx(.03**2*.25+a*a*.08**2*2/3,abs=1e-14)
    expected=100*math.cosh(.015)*(1+2*math.cosh(a*.08))/3
    assert prior.prototypes["car"]["spacing_hz"][0]==pytest.approx(expected,abs=1e-11)
    assert prior.statistics["car"]["virtual_children_per_parent"]==(3 if mode=="group_recombined" else 6)


@pytest.mark.parametrize("mode",["group_recombined","group_predictive"])
@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_deterministic_matching_donors_bounds_and_complete_statistics_ancestry(mode,arm):
    cfg, metric, bank=fixture();original=deepcopy(bank)
    a=GroupPrior(bank,cfg,metric,mode,"held");b=GroupPrior(list(reversed(bank)),cfg,metric,mode,"held")
    row=a.draw("car",42,3,arm);again=b.draw("car",42,3,arm)
    assert row==again and bank==original
    base=previous_draw(bank,cfg,metric,"car",42,3,arm,"spectrum16","held")
    for k in ("parent_ids","parent_groups","coordinate_donors","input_id","selection_seed"):
        assert row[k]==base[k]
    assert len(row["group_effect_choices"])=={"joint":1,"prototype":0,"marginals":69}[arm]
    assert set(row["prior_calibration_parent_ids"])=={r["file_id"] for r in bank if r["class"]=="car"}
    assert row["prior_calibration_groups"]==["g0","g1","g2"]
    x, streams=render(row,cfg,"spectrum16");y,replayed=render(again,cfg,"spectrum16")
    assert np.array_equal(x,y) and streams==replayed and np.isfinite(x).all()


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_reference_is_exact_original_sampler(arm):
    cfg, metric, bank=fixture()
    assert GroupPrior(bank,cfg,metric,"spectrum16","held").draw("car",42,4,arm)==previous_draw(bank,cfg,metric,"car",42,4,arm,"spectrum16","held")


def test_excluded_duplicate_missing_class_and_tampered_parents_fail_before_statistics():
    cfg, metric, bank=fixture()
    with pytest.raises(ValueError,match="excluded"):GroupPrior(bank,cfg,metric,"group_predictive","g0")
    contaminated=deepcopy(bank);contaminated[0]["group"]="outer"
    with pytest.raises(ValueError,match="excluded"):GroupPrior(contaminated,cfg,metric,"group_predictive","held")
    with pytest.raises(ValueError,match="duplicate"):GroupPrior(bank+[bank[0]],cfg,metric,"group_predictive","held")
    with pytest.raises(ValueError,match="class"):GroupPrior(bank[:6],cfg,metric,"group_predictive","held")
    changed=deepcopy(bank);changed[0]["parameters"]["spacing_hz"][0]+=1
    with pytest.raises(ValueError,match="ancestry"):GroupPrior(changed,cfg,metric,"group_predictive","held")


def test_projection_is_finite_bounded_and_records_every_clipped_coordinate():
    cfg,_,_=fixture();z=np.zeros(69);z[:3]=1000;z[27]=-1000;z[28:]=np.linspace(-1000,1000,41)
    p,flags=inverse(z,cfg,0.)
    assert p["spacing_hz"]==[400.]*3
    assert all(.1<=x<=3 for x in p["envelope_knots"])
    assert flags["spacing_coordinates_clipped"]==3 and flags["envelope_coordinates_clipped"]==40
    assert flags["mixture_logit_clipped"]
