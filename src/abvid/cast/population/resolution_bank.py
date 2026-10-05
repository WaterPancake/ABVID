"""Numeric components extracted from CAST/improvement/src/cast_improvement/resolution_bank.py."""
from copy import deepcopy


from abvid.cast.config import digest


from abvid.cast.population.resolution_sampling import sample


from abvid.cast.population.resolution_model import lift_controls


from abvid.cast.population.resolution_calibration import fit


from abvid.cast.population.temporal_prior import draw as previous_draw


MODES=("temporal41","spectrum16")


def fit_record(row,observation,cfg,metric,input_id):
    if any(row[k]!=observation[k] for k in ("file_id","class","group","parent")) or row["group"]==metric["held_group"]:
        raise ValueError("Spectral calibration ancestry crosses input boundary")
    target={k:observation["descriptors"][k] for k in ("log_spectrum","bands")}
    result=fit(lift_controls(row["parameters"],cfg),target,cfg,input_id,metric)
    result.update(file_id=row["file_id"],group=row["group"],class_label=row["class"],
                  initial_bank_row_sha256=digest(row),observation_sha256=digest(observation),
                  original_fit_sha256=row["fit_sha256"])
    derived=deepcopy(row);derived["parameters"]=result["parameters"]
    derived["resolution_calibration"]=result;derived["resolution_calibration_sha256"]=digest(result)
    return derived


def validate_ancestry(row):
    result=row["resolution_calibration"]
    if (any(result[a]!=row[b] for a,b in (("file_id","file_id"),("class_label","class"),("group","group")))
        or digest(result)!=row["resolution_calibration_sha256"] or result["parameters"]!=row["parameters"]
        or any(result["initial_parameters"][k]!=row["parameters"][k] for k in ("spacing_hz","harmonic_weights","envelope_knots"))):
        raise ValueError("Changed spectrum calibration parameters or ancestry")


def draw(bank,cfg,metric,label,seed,index,arm,variant,fold):
    if variant not in MODES or any(r["group"] in (fold,metric["held_group"]) for r in bank):
        raise ValueError("Undeclared spectrum variant or excluded source ancestor")
    if variant=="temporal41":return previous_draw(bank,cfg,metric,label,seed,index,arm,"temporal41",fold)
    record=sample(bank,{**metric,"version":f"cast_improvement_sampling_v1:{fold}"},label,seed,index,arm,cfg)
    record.pop("parent_flags");by_id={r["file_id"]:r for r in bank};ancestry={}
    for p in record["parent_ids"]:
        row=by_id[p];validate_ancestry(row)
        ancestry[p]={"resolution":row["resolution_calibration_sha256"],"prior_bank_row":row["resolution_calibration"]["initial_bank_row_sha256"]}
    return {**record,"calibration_ancestors":ancestry}
