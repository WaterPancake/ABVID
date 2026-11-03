"""Render the stage-diagnosis waveforms to MP3 with a local listening page.

Listening aid only: no fitting, features or scores. Run in the CAST environment:

    .venv/bin/python -B experiments/cast_utility/stage_diagnostic/listen.py

Originals are rebuilt from the role's training parents and must match the
CAST-5A cached wave hashes before anything is written. The full-band column
(native -> 16 kHz) is NOT what the frozen heads saw; it shows what the 8 kHz
route removed. Audio and manifest are create-only, outside the checkout;
`--page` rebuilds only the derived index.html (e.g. after verdicts.py), and
`--originals` saves the hash-verified originals for spectrograms.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import gcd
import json
import subprocess

import numpy as np
from scipy.signal import resample_poly

from abvid.cast.grouped_storage import read_bundle
from abvid.cast.stage_diagnostic import CAST_ENV, ROLE, design, destination, environment, feature_path, stage_plan, wave_hash
from abvid.paths import resolve

RUN = "artifact:cast/diagnostics/cast_stage_diagnosis_20261025_v1"
OUT = "artifact:cast/listening/stage_diagnosis_20261025_v1"
SR, GAIN, PEAK = 16000, 0.1, 0.98
COLUMNS = [("original", "Original (classifier input, ≤4 kHz)"), ("full_band", "Full band (≤8 kHz, not seen)"),
           ("smooth8", "Fitted (smooth8)"), ("v7", "v7"), ("v8", "v8"), ("v9", "v9"), ("v10", "v10"),
           ("v11", "Calibrated (v11)"), ("v11_fresh", "v11, fresh noise")]


def full_band(raw, sr, cfg):
    x = np.asarray(raw, dtype=np.float64).mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR//g, sr//g, window=("kaiser", cfg["audio"]["kaiser_beta"]), padtype="constant")
    n = cfg["audio"]["samples"]
    return x[(len(x)-n)//2:(len(x)-n)//2+n].astype(np.float32)


def originals(d, parents):
    from abvid.cast.audio import normalize, observation
    from abvid.cast.grouped_parents import _configure_cpu, read_source
    from abvid.cast.utility_design import _cast4a_context, load_context
    _configure_cpu()
    c, c4 = load_context(), _cast4a_context()
    role, cfg = c["roles"][ROLE], d["renderer_configs"]["smooth8_v1"]
    with np.load(feature_path("real"), allow_pickle=False) as cache:
        expected = dict(zip(cache["ids"].astype(str), cache["wave_sha256"].astype(str)))
    waves = {}
    for parent in parents:
        fid = parent["file_id"]
        role.validate_parent(parent["source"])
        raw, sr = read_source(c4, fid)
        target = normalize(observation(raw, sr, cfg), cfg)[0].numpy()
        if wave_hash(target) != expected[fid]:
            raise ValueError(f"Original profile/hash drift: {fid}")
        waves[(fid, "original", 0)] = target
        waves[(fid, "full_band", 0)] = normalize(full_band(raw, sr, cfg), cfg)[0].numpy()
    return waves


def staged(d):
    waves, labels = {}, {}
    for name, stage, purpose, realization, ids in stage_plan(d):
        _, payload = read_bundle(destination(RUN) / "render" / name)
        array = np.load(BytesIO(payload["waves.npy"]), allow_pickle=False)
        records = json.loads(payload["records.json"])
        if (json.loads(payload["ids.json"]) != ids or array.shape != (len(ids), 32000) or
                [r["file_id"] for r in records] != ids or any(r["wave_sha256"] != wave_hash(w) for r, w in zip(records, array))):
            raise ValueError(f"Unexpected render bundle: {name}")
        column = "v11_fresh" if purpose == "fresh" else stage
        waves.update({(fid, column, realization): wave for fid, wave in zip(ids, array)})
        labels.update({r["file_id"]: (r["source_class"], r["group"]) for r in records})
    return waves, labels


def encode(path, wave):
    peak = float(np.max(np.abs(wave)))*GAIN
    gain = GAIN if peak <= PEAK else GAIN*PEAK/peak
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                    "-i", "pipe:0", "-c:a", "libmp3lame", "-q:a", "2", str(path)],
                   input=(wave*gain).astype("<f4").tobytes(), check=True)
    return round(20*np.log10(gain/GAIN), 2)


def labels_only():
    _, payload = read_bundle(destination(RUN) / "render" / "smooth8_check_0")
    return {r["file_id"]: (r["source_class"], r["group"]) for r in json.loads(payload["records.json"])}


def page(out, d, parents, labels, files):
    rows = [{"id": p["file_id"][:12], "class": labels[p["file_id"]][0], "group": labels[p["file_id"]][1],
             "traced": p["file_id"] in d["secondary_ids"]} for p in parents]
    html = (resolve("repo:experiments/cast_utility/stage_diagnostic/listen.html").read_text()
            .replace("__ROWS__", json.dumps(rows)).replace("__COLUMNS__", json.dumps(COLUMNS))
            .replace("__FILES__", json.dumps(sorted({(f["file_id"][:12], f["column"], f["realization"]) for f in files}))))
    (out / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", action="store_true", help="rebuild only index.html from the existing manifest")
    parser.add_argument("--originals", action="store_true", help="save the verified originals as originals.npy")
    args = parser.parse_args()
    environment(CAST_ENV)
    d, parents, identity = design(RUN)
    out = resolve(OUT, must_exist=False)
    if args.originals:
        waves = originals(d, parents)
        with (out / "originals.npy").open("xb") as handle:
            np.save(handle, np.stack([waves[(i, "original", 0)] for i in d["train_ids"]]), allow_pickle=False)
        (out / "originals_ids.json").write_text(json.dumps(d["train_ids"]) + "\n")
        print(f"saved {len(d['train_ids'])} verified originals -> {out / 'originals.npy'}")
        return
    if args.page:
        page(out, d, parents, labels_only(), json.loads((out / "manifest.json").read_text())["files"])
        print(f"rebuilt {out / 'index.html'}")
        return
    out.mkdir(parents=True, exist_ok=False)
    rendered, labels = staged(d)
    waves = {**originals(d, parents), **rendered}
    jobs = []
    for (fid, column, realization), wave in waves.items():
        (out / "audio" / fid[:12]).mkdir(parents=True, exist_ok=True)
        jobs.append((f"audio/{fid[:12]}/{column}_{realization}.mp3", fid, column, realization, wave))
    with ThreadPoolExecutor(8) as pool:
        trims = list(pool.map(lambda j: encode(out / j[0], j[4]), jobs))
    files = [{"path": j[0], "file_id": j[1], "column": j[2], "realization": j[3], "wave_sha256": wave_hash(j[4]),
              "peak_trim_db": t} for j, t in zip(jobs, trims)]
    (out / "manifest.json").write_text(json.dumps({"run": RUN, "design_identity": identity, "gain": GAIN,
                                                   "sample_rate_hz": SR, "files": files}, indent=1) + "\n")
    page(out, d, parents, labels, files)
    print(f"{len(files)} MP3s, {sum(t < 0 for t in trims)} peak-trimmed -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
