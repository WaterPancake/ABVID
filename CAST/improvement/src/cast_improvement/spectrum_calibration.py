"""Expected-power calibration of the existing smooth8 noise and mixture."""
from copy import deepcopy
import numpy as np
import torch

from cast.audio import shape
from cast.config import digest
from cast.renderer import interpolate, tensors, validate_controls
from .engine import SmoothRenderer

SETTINGS = {"steps":150,"lr":.03,"betas":[.9,.999],"epsilon":1e-8,"gradient_clip":100.}


def welch_power(wave):
    if wave.shape[-1] != 32000 or not torch.isfinite(wave).all():
        raise ValueError("Expected finite two-second 16 kHz waveform")
    frames = wave.unfold(-1,2048,1024)
    frames = frames-frames.mean(dim=-1,keepdim=True)
    window = torch.hann_window(2048,periodic=True,dtype=wave.dtype,device=wave.device)
    power = torch.fft.rfft(frames*window).abs().square().mean(dim=-2)
    factors = torch.full((1025,),2.,dtype=wave.dtype,device=wave.device)
    factors[0]=1; factors[-1]=1
    return power*factors/(16000*window.square().sum())


class ExpectedSpectrum:
    def __init__(self, params, cfg, input_id, floor):
        validate_controls(params,cfg)
        if cfg["seeds"]["check_realizations"] != 2 or not 0<floor<1:
            raise ValueError("Changed expected-spectrum calibration configuration")
        self.cfg,self.floor=cfg,floor
        self.renderer=SmoothRenderer(cfg,input_id)
        pure=deepcopy(params);pure["harmonic_fraction"]=1.
        with torch.no_grad():
            h=self.renderer.render(tensors(pure),"check")[0]
            self.harmonic=welch_power(h).mean(dim=0)
            self.envelope=interpolate(tensors(params)["envelope_knots"],self.renderer.n)
            self.envelope=self.envelope/self.envelope.square().mean(dim=-1,keepdim=True).sqrt()
        self.freq=torch.fft.rfftfreq(2048,1/16000)
        self.band_masks=[(self.freq>=lo)&(self.freq<hi) for lo,hi in zip(cfg["renderer"]["noise_edges_hz"][:-1],cfg["renderer"]["noise_edges_hz"][1:])]

    def noise_power(self,weights):
        gain=self.renderer.spectral_gain(weights[None])
        noise=shape(torch.fft.irfft(self.renderer.white_spectra("check")[None]*gain[:,None],n=self.renderer.n))
        wave=self.renderer.upsample(noise*self.envelope[:,None])[0]
        return welch_power(wave).mean(dim=0)

    def probabilities(self,weights,mix):
        power=mix*self.harmonic+(1-mix)*self.noise_power(weights)
        spectrum=power[:512].reshape(64,8).sum(dim=1)
        bands=torch.stack([power[m].sum() for m in self.band_masks])
        return spectrum/spectrum.sum().clamp_min(1e-12),bands/bands.sum().clamp_min(1e-12)

    def logs(self,weights,mix):
        return tuple(p.clamp_min(self.floor).log() for p in self.probabilities(weights,mix))


def fit(params,target,cfg,input_id,metric):
    validate_controls(params,cfg)
    spectrum=np.asarray(target["log_spectrum"],dtype=np.float64)
    bands=np.asarray(target["bands"],dtype=np.float64)
    if spectrum.shape!=(64,) or bands.shape!=(8,) or not np.isfinite(spectrum).all() or not np.isfinite(bands).all() or (bands<0).any() or abs(bands.sum()-1)>1e-6:
        raise ValueError("Invalid same-parent spectral target")
    expected=ExpectedSpectrum(params,cfg,input_id,metric["spectral_probability_floor"])
    targets=(torch.tensor(spectrum,dtype=torch.float32),torch.tensor(bands,dtype=torch.float32).clamp_min(expected.floor).log())
    logits=torch.nn.Parameter(torch.tensor(params["noise_weights"],dtype=torch.float32).clamp_min(1e-12).log())
    mix=torch.nn.Parameter(torch.tensor(params["harmonic_fraction"],dtype=torch.float32))
    opt=torch.optim.Adam([logits,mix],lr=SETTINGS["lr"],betas=tuple(SETTINGS["betas"]),eps=SETTINGS["epsilon"])
    history=[];best=float("inf");best_p=None;best_step=None;max_grad=0.
    for step in range(SETTINGS["steps"]+1):
        opt.zero_grad(set_to_none=True)
        weights=logits.softmax(dim=0)
        logs=expected.logs(weights,mix)
        loss=sum((a-b).square().mean() for a,b in zip(logs,targets))/2
        value=float(loss.detach())
        if not np.isfinite(value): raise FloatingPointError("Nonfinite spectral calibration objective")
        history.append(value)
        if value<best:
            best=value;best_step=step;best_p=deepcopy(params)
            best_p.update(noise_weights=weights.detach().tolist(),harmonic_fraction=float(mix.detach()))
        if step==SETTINGS["steps"]:break
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (logits,mix)):
            raise FloatingPointError("Nonfinite spectral calibration gradient")
        norm=float(torch.nn.utils.clip_grad_norm_([logits,mix],SETTINGS["gradient_clip"]))
        max_grad=max(max_grad,norm)
        opt.step()
        with torch.no_grad():mix.clamp_(0,1)
    validate_controls(best_p,cfg)
    flags=[]
    if best_p["harmonic_fraction"] in (0.,1.):flags.append("calibrated_mixture_boundary")
    if best_p["harmonic_fraction"]<.05:flags.append("spacing_unidentified_after_noise_dominant_calibration")
    if best_p["harmonic_fraction"]>.95:flags.append("noise_weights_weakly_identified_after_harmonic_dominant_calibration")
    if min(best_p["noise_weights"])<1e-5:flags.append("calibrated_noise_weight_near_boundary")
    return {"input_id":input_id,"initial_parameters":deepcopy(params),"parameters":best_p,
            "initial_objective":history[0],"best_objective":best,"best_step":best_step,"history":history,
            "maximum_gradient_norm_before_clip":max_grad,"flags":flags,"settings":SETTINGS,
            "target_sha256":digest(target),"streams":expected.renderer.streams(),
            "checking_stream_role":"calibration only; no independent checking-loss claim"}
