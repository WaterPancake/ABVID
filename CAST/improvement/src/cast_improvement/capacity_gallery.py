"""Local audio/diagnostic review of an audited training reconstruction bank."""
import argparse
from collections import defaultdict
from html import escape
import json
import os
from pathlib import Path
import re

from cast.config import ROOT, save, sha
from cast_generalization.pipeline import read
from .capacity_data import HERE
from .capacity_evaluation import load_banks


def build(out, scope, dest):
    _, banks, _, _ = load_banks(out, scope)
    audit_path = out/("fit_"+scope+"_verification.json")
    audit = read(audit_path)
    if not audit["passed"] or audit["fit_summary_sha256"] != sha(out/("fit_"+scope+"_summary.json")):
        raise ValueError("A matching successful fit replay audit is required")
    dest.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in banks["capacity16x9"]:
        folder = out/"fits"/row["file_id"]
        fit = read(folder/"fit.json")
        rows.append({**row, "fit": fit, "folder": folder})
    examples = defaultdict(list)
    for label in ("car", "truck"):
        ranked = sorted([r for r in rows if r["class"] == label], key=lambda r: (r["fit"]["check_loss"], r["file_id"]))
        for rank, row in (("best checking loss", ranked[0]), ("median checking loss", ranked[len(ranked)//2]), ("worst checking loss", ranked[-1])):
            examples[row["file_id"]].append(rank)
        for group in sorted({r["group"] for r in ranked}):
            first = min((r for r in ranked if r["group"] == group), key=lambda r: r["file_id"])
            examples[first["file_id"]].append("first ID in class/group; selection independent of loss")

    def link(path):
        if not path.is_file():
            raise ValueError(f"Missing linked artifact: {path}")
        return escape(os.path.relpath(path, dest), quote=True)

    def player(row, name, label):
        return f'<label>{escape(label)}<audio controls preload="none" src="{link(row["folder"]/(name+".wav"))}"></audio></label>'

    cards = []
    for row in sorted((r for r in rows if r["file_id"] in examples), key=lambda r: (r["class"], r["file_id"])):
        fit = row["fit"]
        flags = ", ".join(fit["diagnostics"]["flags"]) or "none"
        cards.append(f'''<article id="clip-{row['file_id']}">
<h3>{escape(row['class'].title())} · {row['file_id'][:12]}</h3>
<p class="selection">{escape('; '.join(examples[row['file_id']]))}</p>
<p>Checking loss {fit['check_loss']:.6f}; best initial loss {fit['baseline_check_loss']:.6f}; optimizer time {fit['fit_seconds']:.2f} s.</p>
<div class="players">{player(row, 'original_playback', 'Original observation')}{player(row, 'reconstructed_playback', 'Fitted reconstruction')}</div>
<p><strong>Scientific flags:</strong> {escape(flags)}</p>
<details><summary>Diagnostic plot and provenance</summary>
<img loading="lazy" alt="Original, reconstructed and initial spectra, envelopes, optimizer histories and spectrograms" src="{link(row['folder']/'diagnostics.png')}">
<p>Source group: <code>{escape(row['group'])}</code></p>
<p>Complete input ID: <code>{row['file_id']}</code></p>
<p><a href="{link(row['folder']/'fit.json')}">Parameters, ancestry and all optimizer starts</a> · <a href="{link(row['folder']/'failure.json')}">Failure record</a> · <a href="{link(row['folder']/'hashes.json')}">Artifact hashes</a></p>
</details></article>''')
    table = []
    for row in sorted(rows, key=lambda r: (r["class"], r["group"], r["file_id"])):
        fit = row["fit"]
        table.append(f'''<tr><td>{row['class']}</td><td><code>{row['file_id'][:12]}</code><br>{row['group'][-8:]}</td>
<td>{fit['check_loss']:.6f}</td><td>{100*(1-fit['check_loss']/fit['baseline_check_loss']):.2f}%</td>
<td><a href="{link(row['folder']/'original_playback.wav')}">Original</a> · <a href="{link(row['folder']/'reconstructed_playback.wav')}">Reconstructed</a> · <a href="{link(row['folder']/'diagnostics.png')}">Plot</a> · <a href="{link(row['folder']/'fit.json')}">Parameters</a></td>
<td>{escape(', '.join(fit['diagnostics']['flags']) or 'none')}</td></tr>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAST training reconstruction review</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#16343b;background:#f4f7f7;margin:0}main{max-width:1180px;margin:auto;padding:30px}
h1,h2,h3{line-height:1.2}article,.intro{background:white;padding:24px;border:1px solid #d6e0e2;border-radius:8px;margin:20px 0}
.players{display:flex;gap:24px;flex-wrap:wrap}.players label{display:grid;gap:8px;font-weight:600}audio{max-width:100%}
img{max-width:100%;height:auto}.selection{color:#4b6571}code{font-size:.85em;overflow-wrap:anywhere}a{color:#076b72}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #cbd9dc}
.scroll{overflow:auto}details summary{cursor:pointer;font-weight:600}</style><main>
<h1>CAST training reconstruction review</h1><div class="intro">
<p>Audited, source-only IDMT observation fits. These clips were used for calibration. Listening here measures reconstruction quality; it does not establish held-group coverage or classification transfer.</p>
<p>Original and reconstructed playback share a gain, with the reconstruction restored to the original AC RMS. The plots use unit-RMS shape normalization. Unknown physical vehicle identity, RPM, load and distance remain unknown. Alternative-start ambiguities are retained.</p>
<p>Local research derivatives; source licence is CC BY-NC-ND 4.0. This page does not authorize redistribution.</p>'''
    document += f'<p>Run: <code>{escape(out.name)}</code>; scope: {scope}; {len(rows)} fitted parents. <a href="{link(audit_path)}">Replay audit</a>.</p></div>'
    document += '<h2>Systematic examples</h2><p>Best, median and worst checking-loss cases, plus the first input ID in each class/group. Every fit is linked in the table below.</p>'+"\n".join(cards)
    document += '<h2>All fits</h2><div class="scroll"><table><thead><tr><th>Class</th><th>Input / group suffix</th><th>Checking loss</th><th>Gain over initialization</th><th>Artifacts</th><th>Scientific flags</th></tr></thead><tbody>'+"\n".join(table)+'</tbody></table></div></main></html>'
    (dest/"index.html").write_text(document)
    save(dest/"manifest.json", {"fit_run": str(out.relative_to(ROOT)), "scope": scope, "count": len(rows),
         "fit_audit_sha256": sha(audit_path), "examples": dict(examples),
         "selection": "class best/upper-median/worst checking loss plus first sorted ID in every class/group; all fits linked",
         "fit_hashes": {r["file_id"]: sha(r["folder"]/"fit.json") for r in rows},
         "gallery_sha256": sha(dest/"index.html"), "code_sha256": sha(Path(__file__)), "outer_held_access": False})
    print(json.dumps({"gallery": str((dest/"index.html").relative_to(ROOT)), "fits": len(rows), "systematic_examples": len(examples)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=["pilot", "full"], required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", x) for x in (args.run_id, args.output_id)):
        parser.error("Use simple IDs")
    build(HERE/"runs"/args.run_id, args.scope, HERE/"diagnostics"/args.output_id)


if __name__ == "__main__":
    main()
