from collections import Counter
from copy import deepcopy
import numpy as np
import pytest

from cast_improvement.balanced_prior import BalancedPrior, balanced_indices
from cast_improvement.group_prior import GroupPrior
from cast_improvement.resolution_evaluation import render
from test_group_prior import fixture


@pytest.mark.parametrize("categories,count",[(1,50),(3,50),(4,50),(5,50),(8,50),(10,50),(100,50),(3,0)])
def test_balanced_finite_categorical_schedule(categories,count):
    x=balanced_indices(categories,count,np.random.default_rng(42))
    assert np.array_equal(x,balanced_indices(categories,count,np.random.default_rng(42)))
    counts=np.bincount(x,minlength=categories)
    assert len(x)==count and counts.sum()==count
    assert counts.min()==count//categories and counts.max()==(count+categories-1)//categories


@pytest.mark.parametrize("arm",["joint","marginals"])
def test_all_coordinate_plans_balance_groups_conditional_parents_and_children(arm):
    cfg,metric,bank=fixture();prior=BalancedPrior(bank,cfg,metric,"group_balanced","held")
    group={r["file_id"]:r["group"] for r in bank}
    for label in metric["class_order"]:
        for seed in metric["seeds"]:
            plan=prior.plans[prior.key(label,seed,arm)]
            for j in range(plan["coordinates"]):
                ids=[row[j] for row in plan["parent_ids_by_index_coordinate"]]
                groups=Counter(group[p] for p in ids)
                assert set(groups.values())=={16,17}
                for g in groups:
                    counts=[ids.count(r["file_id"]) for r in bank if r["class"]==label and r["group"]==g]
                    assert max(counts)-min(counts)<=1
                children=Counter(row[j] for row in plan["child_indices_by_index_coordinate"])
                assert len(children)==6 and set(children.values())=={8,9}


@pytest.mark.parametrize("arm",["joint","prototype","marginals"])
def test_exact_reference_and_matched_waveform_streams(arm):
    cfg,metric,bank=fixture();old=GroupPrior(bank,cfg,metric,"group_predictive","held")
    ref=BalancedPrior(bank,cfg,metric,"group_predictive","held")
    balanced=BalancedPrior(bank,cfg,metric,"group_balanced","held")
    assert ref.draw("car",42,0,arm)==old.draw("car",42,0,arm)
    a=ref.draw("car",42,0,arm);b=balanced.draw("car",42,0,arm)
    assert a["input_id"]==b["input_id"]
    x,sx=render(a,cfg,"spectrum16");y,sy=render(b,cfg,"spectrum16")
    assert sx==sy and np.isfinite(y).all()
    if arm=="prototype":assert a["parameters"]==b["parameters"] and np.array_equal(x,y)
    assert a["prior_calibration_parent_ids"]==b["prior_calibration_parent_ids"]
    assert a["group_effect_statistics_sha256"]==b["group_effect_statistics_sha256"]


def test_determinism_reordering_distinct_marginal_plans_and_unchanged_input():
    cfg,metric,bank=fixture();original=deepcopy(bank)
    a=BalancedPrior(bank,cfg,metric,"group_balanced","held")
    b=BalancedPrior(list(reversed(bank)),cfg,metric,"group_balanced","held")
    assert a.statistics==b.statistics and bank==original
    rows=[a.draw("truck",123,i,"joint") for i in range(50)]
    assert rows==[b.draw("truck",123,i,"joint") for i in range(50)]
    p=a.plans[a.key("car",42,"marginals")]
    assert any(x[0]!=x[1] for x in p["parent_ids_by_index_coordinate"])
    x,_=render(rows[3],cfg,"spectrum16");y,_=render(b.draw("truck",123,3,"joint"),cfg,"spectrum16")
    assert np.array_equal(x,y)


def test_provenance_and_sampling_boundary_rejection():
    cfg,metric,bank=fixture()
    with pytest.raises(ValueError,match="excluded"):BalancedPrior(bank,cfg,metric,"group_balanced","g0")
    bad=deepcopy(bank);bad[0]["parameters"]["spacing_hz"][0]+=1
    with pytest.raises(ValueError,match="ancestry"):BalancedPrior(bad,cfg,metric,"group_balanced","held")
    a=BalancedPrior(bank,cfg,metric,"group_balanced","held")
    for label,seed,index,arm in [("other",42,0,"joint"),("car",999,0,"joint"),("car",42,50,"joint"),("car",42,-1,"joint"),("car",42,0,"other")]:
        with pytest.raises(ValueError,match="schedule"):a.draw(label,seed,index,arm)
    with pytest.raises(ValueError):balanced_indices(0,50,np.random.default_rng(42))
