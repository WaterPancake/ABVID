from copy import deepcopy
from itertools import product
import pytest

from cast_improvement.width_selection import aggregate
from cast_improvement.width_prior import VARIANT, TEMPERATURES


def scores():
    metric={"class_order":["car","truck"],"arms":["joint","prototype","marginals"],
            "seeds":[42,123,456,789,1024],"primary_families":["a","b","c","d"],"held_group":"outer"}
    folds=[f"g{i}" for i in range(5)]; rows=[]
    for fold,t,c,arm,seed in product(folds,TEMPERATURES,metric["class_order"],metric["arms"],metric["seeds"]):
        cov=.79 if t==1 and c=="car" and fold=="g0" else .9
        w1=.5+(t-1)*.1 if arm=="joint" else 1.
        rows.append({"fold":fold,"variant":VARIANT,"temperature":t,"class":c,"arm":arm,"seed":seed,
                     "score":{"W1":w1,"coverage":cov,"families":{f:{"spread_ratio":1.} for f in metric["primary_families"]}}})
    return metric,folds,rows


def test_stronger_gate_rejects_passing_mean_and_keeps_original_ranking_among_eligible():
    metric,folds,rows=scores();result=aggregate(rows,metric,folds)
    reference=result[f"{VARIANT}_T1"]
    assert reference["original_mean_criteria_passed"] and not reference["passes"]
    assert reference["minimum_source_fold_coverage_by_class"]["car"]==pytest.approx(.79)
    selected=min(result,key=lambda k:result[k]["selection_key"])
    assert selected==f"{VARIANT}_T1.1"
    assert result[selected]["passes"] and result[selected]["source_fold_coverage_passed"]
    assert result[selected]["selection_key"][1:]==result[selected]["original_selection_key"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra_seed", "missing_fold"])
def test_incomplete_or_selected_score_schedule_cannot_pass(defect):
    metric,folds,rows=scores()
    if defect=="missing":rows.pop()
    elif defect=="duplicate":rows.append(deepcopy(rows[0]))
    elif defect=="extra_seed":rows[0]["seed"]=999
    else:rows=[r for r in rows if r["fold"]!="g0"]
    with pytest.raises(ValueError,match="schedule"):aggregate(rows,metric,folds)


def test_extra_fold_gate_cannot_replace_original_margin_or_spread_checks():
    metric,folds,rows=scores()
    for row in rows:
        if row["arm"]=="joint":row["score"]["W1"] = 1.
    result=aggregate(rows,metric,folds)
    assert all(not c["passes"] for c in result.values())
    assert result[f"{VARIANT}_T1.1"]["source_fold_coverage_passed"]
    assert not result[f"{VARIANT}_T1.1"]["original_mean_criteria_passed"]


def test_forbidden_folds_and_nonfinite_scores_are_rejected():
    metric,folds,rows=scores()
    with pytest.raises(ValueError,match="folds"):aggregate(rows,metric,["outer"]+folds[1:])
    rows[0]["score"]["coverage"]=float('nan')
    with pytest.raises(ValueError,match="score"):aggregate(rows,metric,folds)
