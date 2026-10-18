"""Offline caption-hidden review packets and native-frame interval export."""
from __future__ import annotations

import hashlib
import json
import math
import secrets
from copy import deepcopy
from html import escape

import numpy as np
import soundfile as sf

from abvid.data.overlap import mono8k, spectrogram
from abvid.provenance.hashing import canonical, sha256

from . import PROTOCOL_ID, TOOL_VERSION
from .audio import read_asset
from .config import external_path
from .manifest import CONTAMINANTS, MOTION, REASONS, validate


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _write_json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _wave_svg(x):
    mono = x.mean(axis=1)
    bounds = np.linspace(0, len(mono), min(800, len(mono)) + 1, dtype=int)
    peak = max(float(np.max(np.abs(mono))), 1e-12)
    strokes = []
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = mono[lo:hi]
        a, b = 50 - 47 * part.max() / peak, 50 - 47 * part.min() / peak
        strokes.append(f"M{i},{a:.2f}V{b:.2f}")
    return (f'<svg viewBox="0 0 {len(strokes)} 100" preserveAspectRatio="none" aria-label="Waveform">'
            f'<path d="{"".join(strokes)}" stroke="#166b7c" stroke-width="1"/></svg>')


def _spectrum_svg(x, rate):
    s = spectrogram(mono8k(x, rate))
    if not len(s):
        return "<p>Insufficient duration for a spectrogram.</p>"
    # Decimation affects the display only. Audio, trim times and deduplication
    # use their specified native/fingerprint routes without these display bins.
    ti = np.linspace(0, len(s) - 1, min(256, len(s)), dtype=int)
    fi = np.linspace(0, s.shape[1] - 1, 64, dtype=int)
    pixels = s[np.ix_(ti, fi)]
    top = float(s.max())
    shades = np.clip((pixels - (top - 80)) / 80, 0, 1)
    rects = []
    for t in range(len(ti)):
        for f in range(len(fi)):
            c = round(255 * shades[t, f])
            rects.append(f'<rect x="{t}" y="{63-f}" width="1" height="1" fill="rgb({c},{c},{c})"/>')
    return (f'<svg viewBox="0 0 {len(ti)} 64" preserveAspectRatio="none" aria-label="Spectrogram 0 to 4 kHz">'
            + "".join(rects) + "</svg>")


def make_packet(manifest, config, blind_reference, private_reference, *, seed=17):
    validate(manifest, config)
    blind = external_path(blind_reference)
    private = external_path(private_reference)
    if private == blind or private.is_relative_to(blind) or blind.is_relative_to(private):
        raise ValueError("Private mapping must be outside the blind review directory")
    if blind.exists() or private.exists():
        raise FileExistsError("Review outputs are immutable; choose new output paths")
    # Verify all inputs before publishing any packet.
    loaded = [(a, *read_asset(a, config)) for a in manifest["assets"]]
    ordering = np.random.default_rng(seed).permutation(len(loaded))
    public, mapping, cards = [], {}, []
    blind.mkdir(parents=True)
    for ordinal, i in enumerate(ordering):
        asset, x, rate = loaded[int(i)]
        key = "review-" + _digest([seed, ordinal, asset["raw_sha256"]])[:16]
        name = key + ".wav"
        # Re-encoding drops source filenames and embedded title/artist tags.
        sf.write(blind / name, x, rate, subtype="FLOAT")
        public.append({"blind_id": key, "audio_file": name, "duration_seconds": len(x) / rate,
                       "audio_sha256": sha256(blind / name)})
        mapping[key] = asset["asset_id"]
        cards.append(f'<section data-key="{key}"><h2>Recording {ordinal+1}</h2>'
                     f'<audio controls preload="metadata" src="{name}"></audio>'
                     f'<p>Duration {len(x)/rate:.3f} s. Waveform uses relative display amplitude. '
                     'Spectrogram spans 0 to 4 kHz and 80 dB below the recording maximum.</p>'
                     + _wave_svg(x) + _spectrum_svg(x, rate) +
                     '<div class="interval"><label>Start seconds <input class="start" type="number" step="0.001" min="0" value="0"></label>'
                     '<label>End seconds <input class="end" type="number" step="0.001" min="0" value="2"></label>'
                     '<button class="at-start">Use playback time as start</button><button class="at-end">Use playback time as end</button>'
                     '<button class="play-selection">Play interval</button></div>'
                     '<label>Decision <select class="kind"><option>foreground</option><option>background</option><option>rejected</option></select></label>'
                     '<label>Audible vehicles <select class="count"><option>1</option><option>0</option><option>2</option><option>unknown</option></select></label>'
                     '<label>Motion <select class="motion">' + "".join(f'<option{ " selected" if m == "unknown" else ""}>{escape(m)}</option>' for m in MOTION) + '</select></label>'
                     '<fieldset><legend>Contaminants</legend>' + "".join(f'<label><input class="contaminant" type="checkbox" value="{c}">{c}</label>' for c in CONTAMINANTS) + '</fieldset>'
                     '<label>Rejection reason <select class="reason"><option value="">Choose if rejected</option>' +
                     "".join(f'<option>{r}</option>' for r in REASONS) + '</select></label>'
                     '<button class="add">Record interval</button><ul class="saved"></ul></section>')
    private_data = {"manifest_sha256": _digest(manifest), "protocol_sha256": _digest(config),
                    "mapping": mapping, "nonce": secrets.token_hex(16)}
    batch = {"protocol_id": PROTOCOL_ID, "tool_version": TOOL_VERSION, "recordings": public,
             "private_commitment_sha256": _digest(private_data)}
    batch_hash = _digest(batch)
    _write_json_new(blind / "review_batch.json", batch)
    _write_json_new(private, {**private_data, "batch_sha256": batch_hash})
    payload = json.dumps({"batch_sha256": batch_hash, "tool_version": TOOL_VERSION}, allow_nan=False).replace("<", "\\u003c")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acoustic recording review</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#182b32;background:#f8fafb}
section{background:white;padding:1.4rem;margin:1.5rem 0;border:1px solid #ccd9df;border-radius:8px}svg{display:block;width:100%;height:120px;margin:.7rem 0;background:#edf2f4}audio{width:100%}
label{display:inline-block;margin:.5rem .8rem .5rem 0}input,select,button{font:inherit;padding:.4rem}input[type=number]{width:7rem}button{margin:.3rem;border:1px solid #849aa6;border-radius:4px;background:#eef6f8;cursor:pointer}fieldset{border:1px solid #ccd9df}li{margin:.4rem 0}#status{white-space:pre-line;color:#a32727}</style>
<h1>Acoustic recording review</h1><p>Review audio without captions. Record separate vehicle and background intervals, and retain rejected intervals with reasons. Save annotations when every recording has been reviewed.</p>
<label>Reviewer ID <input id="reviewer" autocomplete="off" pattern="[A-Za-z0-9_.-]+" required></label><button id="save">Save annotations</button><p id="status" role="status"></p>
''' + "".join(cards) + '<script>const meta=' + payload + '; const trimming=' + json.dumps(config["trimming"]) + '''; const annotations=[];
const status=document.querySelector('#status');
document.querySelectorAll('section').forEach(section=>{
 const q=c=>section.querySelector(c), audio=q('audio');
 q('.at-start').onclick=()=>q('.start').value=audio.currentTime.toFixed(3);
 q('.at-end').onclick=()=>q('.end').value=audio.currentTime.toFixed(3);
 let stopAt=null; audio.ontimeupdate=()=>{if(stopAt!==null&&audio.currentTime>=stopAt){audio.pause();stopAt=null;}};
 const bounds=()=>{const lo=Number(q('.start').value),hi=Number(q('.end').value);return Number.isFinite(audio.duration)&&Number.isFinite(lo)&&Number.isFinite(hi)&&lo>=0&&hi>lo&&hi<=audio.duration;};
 q('.play-selection').onclick=()=>{if(!bounds()){status.textContent='Enter an interval inside the loaded recording.';return;}audio.currentTime=Number(q('.start').value);stopAt=Number(q('.end').value);audio.play().catch(e=>status.textContent=e.message);};
 q('.kind').onchange=()=>{q('.count').value=q('.kind').value==='background'?'0':q('.kind').value==='foreground'?'1':'unknown';q('.motion').value='unknown';};
 q('.add').onclick=()=>{
  const lo=Number(q('.start').value), hi=Number(q('.end').value), kind=q('.kind').value;
  if(!bounds()){status.textContent='Wait for the recording to load, then enter an interval inside it.';return;}
  if(kind==='rejected'&&!q('.reason').value){status.textContent='Choose a rejection reason.';return;}
  const count=q('.count').value;
  const item={blind_id:section.dataset.key,start_seconds:lo,end_seconds:hi,kind,audible_vehicle_count:count==='unknown'?count:Number(count),motion_state:q('.motion').value,contaminants:Array.from(section.querySelectorAll('.contaminant:checked'),e=>e.value),rejection_reasons:kind==='rejected'?[q('.reason').value]:[]};
  if(kind!=='rejected'){
   const minimum=trimming['minimum_'+kind+'_seconds'];
   if(hi-lo<minimum||hi-lo>trimming.maximum_segment_seconds){status.textContent='Accepted intervals must be '+minimum+' to '+trimming.maximum_segment_seconds+' seconds.';return;}
   if(item.audible_vehicle_count!==(kind==='foreground'?1:0)){status.textContent='Foreground requires one audible vehicle; background requires zero.';return;}
   if(kind==='foreground'&&item.contaminants.some(c=>trimming.forbidden_foreground_contaminants.includes(c))){status.textContent='Record this contaminated interval as rejected.';return;}
   if(annotations.some(a=>a.blind_id===item.blind_id&&a.kind!=='rejected'&&a.start_seconds<hi&&lo<a.end_seconds)){status.textContent='Accepted intervals cannot overlap.';return;}
  }
  annotations.push(item);const li=document.createElement('li');li.textContent=kind+' '+lo.toFixed(3)+'–'+hi.toFixed(3)+' s';
  const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{annotations.splice(annotations.indexOf(item),1);li.remove();};li.append(remove);q('.saved').append(li);status.textContent='';
 };
});
document.querySelector('#save').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();
 if(!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(reviewer)){status.textContent='Enter a reviewer ID.';return;}
 const missing=Array.from(document.querySelectorAll('section'),s=>s.dataset.key).filter(k=>!annotations.some(a=>a.blind_id===k));
 if(missing.length){status.textContent='Review each recording before saving.';return;}
 const payload={...meta,trimmer_id:reviewer,intervals:annotations};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{type:'application/json'});
 const url=URL.createObjectURL(blob), a=document.createElement('a');a.href=url;a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};</script></html>'''
    (blind / "index.html").write_text(document)
    return {"blind_reference": blind_reference, "private_reference": private_reference,
            "batch_sha256": batch_hash, "recordings": len(public)}


def import_annotations(manifest, config, batch, private, annotations, *, role="development"):
    validate(manifest, config)
    if private.get("manifest_sha256") != _digest(manifest) or private.get("protocol_sha256") != _digest(config):
        raise ValueError("Private mapping belongs to different inputs")
    commitment = {key: value for key, value in private.items() if key != "batch_sha256"}
    if batch.get("private_commitment_sha256") != _digest(commitment):
        raise ValueError("Private mapping commitment mismatch")
    if private.get("batch_sha256") != _digest(batch) or annotations.get("batch_sha256") != _digest(batch):
        raise ValueError("Annotation/review batch hash mismatch")
    if set(annotations) != {"batch_sha256", "tool_version", "trimmer_id", "intervals"}:
        raise ValueError("Unexpected annotation fields")
    if annotations["tool_version"] != TOOL_VERSION:
        raise ValueError("Changed annotation tool version")
    public_ids = {r["blind_id"] for r in batch["recordings"]}
    if batch.get("protocol_id") != PROTOCOL_ID or batch.get("tool_version") != TOOL_VERSION:
        raise ValueError("Changed review protocol/tool version")
    if public_ids != set(private["mapping"]):
        raise ValueError("Public/private recording identities differ")
    result = deepcopy(manifest)
    interval_keys = {"blind_id", "start_seconds", "end_seconds", "kind", "audible_vehicle_count",
                     "motion_state", "contaminants", "rejection_reasons"}
    reviewed = set()
    if not isinstance(annotations["intervals"], list):
        raise ValueError("Annotation intervals must be an array")
    for interval in annotations["intervals"]:
        if not isinstance(interval, dict) or set(interval) != interval_keys or interval["blind_id"] not in public_ids:
            raise ValueError("Invalid blind interval identity or fields")
        reviewed.add(interval["blind_id"])
        row = {key: deepcopy(value) for key, value in interval.items() if key != "blind_id"}
        row.update({"segment_id": "segment-" + _digest([interval, annotations["trimmer_id"]])[:20],
                    "asset_id": private["mapping"][interval["blind_id"]],
                    "role": "quarantine" if interval["kind"] == "rejected" else role,
                    "trimmer_id": annotations["trimmer_id"], "tool_version": TOOL_VERSION,
                    "blind_review": True, "distance_m": "unknown", "mic_type": "unknown"})
        result["segments"].append(row)
    if reviewed != public_ids:
        raise ValueError("Every recording requires a reviewed interval or a logged rejection")
    validate(result, config)
    return result


def export_segments(manifest, config, output_reference):
    validate(manifest, config)
    output = external_path(output_reference)
    if output.exists():
        raise FileExistsError("Segment output is immutable; choose a new directory")
    assets = {a["asset_id"]: a for a in manifest["assets"]}
    loaded = {aid: read_asset(a, config) for aid, a in assets.items()}
    records = []
    output.mkdir(parents=True)
    for segment in manifest["segments"]:
        if segment["kind"] == "rejected":
            continue
        aid = segment["asset_id"]
        x, rate = loaded[aid]
        start = math.floor(segment["start_seconds"] * rate)
        end = math.ceil(segment["end_seconds"] * rate)
        if not 0 <= start < end <= len(x):
            raise ValueError("Rounded interval exceeds raw audio")
        path = output / (segment["segment_id"] + ".wav")
        sf.write(path, x[start:end], rate, subtype="FLOAT")
        records.append({"segment_id": segment["segment_id"], "asset_id": aid,
                        "raw_sha256": assets[aid]["raw_sha256"], "start_frame": start,
                        "end_frame": end, "sample_rate_hz": rate, "channels": x.shape[1],
                        "clip_sha256": sha256(path), "clip_ref": output_reference + "/" + path.name})
    result = {"manifest_sha256": _digest(manifest), "protocol_sha256": _digest(config),
              "clips": records, "rejections": [s for s in manifest["segments"] if s["kind"] == "rejected"]}
    _write_json_new(output / "export.json", result)
    return result
