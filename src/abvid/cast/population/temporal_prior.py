"""Numeric components extracted from CAST/improvement/src/cast_improvement/temporal_prior.py."""
from copy import deepcopy


import numpy as np


from scipy.optimize import minimize


from abvid.cast.config import digest


from abvid.cast.population.spectrum_bank import draw as previous_draw


from abvid.cast.population.engine import model_config as old_config


COUNT=41


SMOOTHNESS=.01


MODES=("v8_reference","spectrum_calibrated","temporal41")


def model_config(base=None):
    cfg=deepcopy(old_config() if base is None else base)
    cfg["renderer"].update(version="cast_smooth8_temporal41_v10",envelope_control_count=COUNT)
    return cfg


def validate_controls(values,cfg):
    expected={"spacing_hz":(3,),"harmonic_weights":(8,),"noise_weights":(8,),"harmonic_fraction":(),"envelope_knots":(COUNT,)}
    if set(values)!=set(expected) or cfg["renderer"].get("envelope_control_count")!=COUNT:
        raise ValueError("Temporal control schema mismatch")
    for key,dims in expected.items():
        a=np.asarray(values[key])
        if a.shape!=dims or not np.isfinite(a).all():raise ValueError("Invalid temporal control tensor")
    for key,bounds in (("spacing_hz",cfg["renderer"]["spacing_hz_bounds"]),("envelope_knots",cfg["renderer"]["envelope_bounds"]),("harmonic_fraction",[0,1])):
        a=np.asarray(values[key])
        if ((a<bounds[0]-1e-6)|(a>bounds[1]+1e-6)).any():raise ValueError("Temporal control bounds violated")
    for key in ("harmonic_weights","noise_weights"):
        a=np.asarray(values[key])
        if (a<0).any() or abs(a.sum()-1)>1e-6:raise ValueError("Invalid temporal simplex")
    return values


def lift(parameters):
    result=deepcopy(parameters)
    if np.asarray(result["envelope_knots"]).shape!=(5,):raise ValueError("Expected original five envelope knots")
    result["envelope_knots"]=np.interp(np.linspace(0,1,COUNT),np.linspace(0,1,5),result["envelope_knots"]).tolist()
    return result


def frame_basis():
    t=np.linspace(0,1,16000);knots=np.linspace(0,1,COUNT)
    basis=np.stack([np.interp(t,knots,np.eye(COUNT)[i]) for i in range(COUNT)],axis=1)
    return basis.reshape(40,400,COUNT).mean(axis=1)


def calibrate(original,target):
    original=np.asarray(original,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    if (original.shape!=(COUNT,) or target.shape!=(40,) or not np.isfinite(original).all() or not np.isfinite(target).all()
        or ((original<.1)|(original>3)).any() or (target<0).any() or target.mean()<=1e-12):
        raise ValueError("Invalid temporal calibration input")
    b=frame_basis();d=np.diff(np.eye(COUNT),n=2,axis=0);gauge=b.mean(axis=0)
    amplitude=float(gauge@original);scale=amplitude/float(target.mean());y=target*scale
    def fun(x):return float(np.square(b@x-y).mean()+SMOOTHNESS*np.square(d@x).mean())
    def jac(x):return 2*b.T@(b@x-y)/40+2*SMOOTHNESS*d.T@(d@x)/(COUNT-2)
    solved=minimize(fun,original,jac=jac,method="SLSQP",bounds=[(.1,3.)]*COUNT,
                    constraints=[{"type":"eq","fun":lambda x:float(gauge@x-amplitude),"jac":lambda x:gauge}],
                    options={"ftol":1e-14,"maxiter":300})
    x=solved.x;flags=[]
    if (not solved.success or not np.isfinite(x).all() or ((x<.1-1e-12)|(x>3+1e-12)).any()
        or abs(gauge@x-amplitude)>1e-8 or fun(x)>fun(original)+1e-12):
        x=original.copy();flags.append("temporal_solver_failure_lifted_original_retained")
    x=np.clip(x,.1,3.)
    if ((x<=.10000001)|(x>=2.99999999)).any():flags.append("calibrated_temporal_boundary")
    return {"lifted_initial_envelope_knots":original.tolist(),"envelope_knots":x.tolist(),"observed_envelope":target.tolist(),
            "target_scale":scale,"preserved_mean_amplitude":amplitude,"result_mean_amplitude":float(gauge@x),
            "initial_objective":fun(original),"calibrated_objective":fun(x),"smoothness":SMOOTHNESS,"flags":flags,
            "solver":{"success":bool(solved.success),"status":int(solved.status),"message":str(solved.message),
                      "iterations":int(solved.nit),"function_evaluations":int(solved.nfev)}}


def derive_bank(bank,observations,cfg,held_group):
    observed={r["file_id"]:r for r in observations};ids=[r["file_id"] for r in bank]
    if len(ids)!=len(set(ids)) or len(observed)!=len(observations) or set(ids)!=set(observed) or any(r["group"]==held_group for r in bank):
        raise ValueError("Temporal calibration requires exactly the non-outer training parents")
    result=[]
    for row in bank:
        obs=observed[row["file_id"]]
        if any(obs[k]!=row[k] for k in ("file_id","class","group","parent")):raise ValueError("Temporal observation ancestry differs")
        p=lift(row["parameters"])
        record={"file_id":row["file_id"],"class":row["class"],"group":row["group"],"initial_bank_row_sha256":digest(row),
                "observation_sha256":digest(obs),"initial_parameters":deepcopy(row["parameters"]),
                **calibrate(p["envelope_knots"],obs["descriptors"]["envelope"])}
        p["envelope_knots"]=record["envelope_knots"];validate_controls(p,cfg)
        copied=deepcopy(row);copied["parameters"]=p;copied["temporal_calibration"]=record;copied["temporal_calibration_sha256"]=digest(record)
        result.append(copied)
    return result


def draw(bank,cfg,metric,label,seed,index,arm,variant,fold):
    if variant not in MODES or any(r["group"] in (fold,metric["held_group"]) for r in bank):
        raise ValueError("Undeclared temporal variant or excluded source ancestor")
    if variant!="temporal41":return previous_draw(bank,cfg,metric,label,seed,index,arm,variant,fold)
    from abvid.cast.population.temporal_sampling import sample
    record=sample(bank,{**metric,"version":f"cast_improvement_sampling_v1:{fold}"},label,seed,index,arm,cfg)
    record.pop("parent_flags");parents={r["file_id"]:r for r in bank};ancestors={}
    for parent in record["parent_ids"]:
        row=parents[parent];cal=row["temporal_calibration"]
        if (any(cal[k]!=row[k] for k in ("file_id","class","group")) or digest(cal)!=row["temporal_calibration_sha256"]
            or cal["envelope_knots"]!=row["parameters"]["envelope_knots"]
            or any(cal["initial_parameters"][k]!=row["parameters"][k] for k in ("spacing_hz","harmonic_weights","noise_weights","harmonic_fraction"))):
            raise ValueError("Changed temporal calibration parameters or ancestry")
        ancestors[parent]={"temporal":row["temporal_calibration_sha256"],"prior_bank_row":cal["initial_bank_row_sha256"]}
    return {**record,"calibration_ancestors":ancestors}
