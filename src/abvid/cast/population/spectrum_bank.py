"""Numeric components extracted from CAST/improvement/src/cast_improvement/spectrum_bank.py."""
from copy import deepcopy


from abvid.cast.config import digest


from abvid.cast.coverage import sample


from abvid.cast.population.spectrum_calibration import fit


from abvid.cast.population.envelope_prior import draw as previous_draw


MODES=("v8_reference","spectrum_calibrated")


def fit_record(row,observation,cfg,metric,input_id):
    if any(row[k]!=observation[k] for k in ("file_id","class","group","parent")) or row["group"]==metric["held_group"]:
        raise ValueError("Spectral calibration ancestry crosses input boundary")
    target={k:observation["descriptors"][k] for k in ("log_spectrum","bands")}
    result=fit(row["parameters"],target,cfg,input_id,metric)
    result.update(file_id=row["file_id"],group=row["group"],class_label=row["class"],
                  initial_bank_row_sha256=digest(row),observation_sha256=digest(observation),
                  original_fit_sha256=row["fit_sha256"])
    derived=deepcopy(row);derived["parameters"]=result["parameters"]
    derived["spectral_calibration"]=result;derived["spectral_calibration_sha256"]=digest(result)
    return derived


def validate_ancestry(row):
    result=row["spectral_calibration"]
    if (any(result[a]!=row[b] for a,b in (("file_id","file_id"),("class_label","class"),("group","group")))
        or digest(result)!=row["spectral_calibration_sha256"] or result["parameters"]!=row["parameters"]
        or any(result["initial_parameters"][k]!=row["parameters"][k] for k in ("spacing_hz","harmonic_weights","envelope_knots"))):
        raise ValueError("Changed spectrum calibration parameters or ancestry")


def draw(bank,cfg,metric,label,seed,index,arm,variant,fold):
    if variant not in MODES or any(r["group"] in (fold,metric["held_group"]) for r in bank):
        raise ValueError("Undeclared spectrum variant or excluded source ancestor")
    if variant=="v8_reference":return previous_draw(bank,cfg,metric,label,seed,index,arm,"mixture_envelope",fold)
    record=sample(bank,{**metric,"version":f"cast_improvement_sampling_v1:{fold}"},label,seed,index,arm,cfg)
    record.pop("parent_flags");by_id={r["file_id"]:r for r in bank};ancestry={}
    for p in record["parent_ids"]:
        row=by_id[p];validate_ancestry(row)
        ancestry[p]={"spectral":row["spectral_calibration_sha256"],"earlier_mixture":row["mixture_calibration_sha256"],"envelope":row["envelope_calibration_sha256"]}
    return {**record,"calibration_ancestors":ancestry}
