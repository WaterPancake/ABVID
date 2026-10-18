"""Render CAST-5A results.json into Markdown tables (no computation of new results).
Run: .venv-baseline/bin/python experiments/cast_utility/report_tables.py > tables.md
"""
import json

from abvid.paths import resolve
from abvid.provenance.hashing import sha256

RESULTS = "artifact:cast/utility/cast5a_run_20261011_v1/learning/scores/results.json"
ARMS = ("R", "RR", "RA", "RU", "RC", "U", "C")
KEYS = (("macro_f1", "Macro-F1"), ("balanced_accuracy", "BA"), ("auroc", "AUROC"),
        ("average_precision", "AP"), ("positive_recall", "Truck recall"), ("negative_recall", "Car recall"))


def cell(v):
    if v is None:
        return "missing"
    lo, hi = v["ci95"]
    return f"{100 * v['estimate']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"


def main():
    path = resolve(RESULTS)
    r = json.loads(path.read_text())
    s = r["statistics"]
    print(f"Source: `{RESULTS}` (SHA256 `{sha256(path)}`).\n")
    for rep in ("BEATs768", "MFCC26"):
        print(f"### {rep}: arm means over 30 heads\n")
        print("| Arm | " + " | ".join(h for _, h in KEYS) + " |")
        print("|---|" + "---:|" * len(KEYS))
        for arm in ARMS:
            print(f"| {arm} | " + " | ".join(cell(s["arms"][rep][arm][k]) for k, _ in KEYS) + " |")
        print()
        print(f"### {rep}: paired contrasts (percentage points)\n")
        print("| Contrast | " + " | ".join(h for _, h in KEYS) + " | Site-block macro-F1 |")
        print("|---|" + "---:|" * (len(KEYS) + 1))
        for name, row in s["contrasts"][rep].items():
            print(f"| {name} | " + " | ".join(cell(row[k]) for k, _ in KEYS)
                  + f" | {cell(s['site_sensitivity'][rep][name]['macro_f1'])} |")
        print()
    print("### Decision flags (BEATs768 RC − RA)\n")
    for k, v in s["flags"].items():
        print(f"- `{k}`: {v}")
    print("\n### Per-fold BEATs768 macro-F1 (mean over five selections)\n")
    folds = sorted({h["held_group"] for h in r["per_head"]})
    print("| Held group | Trucks | " + " | ".join(ARMS) + " |")
    print("|---|---:|" + "---:|" * len(ARMS))
    for g in folds:
        heads = [h for h in r["per_head"] if h["held_group"] == g and h["representation"] == "BEATs768"]
        trucks = heads[0]["held_trucks"]
        vals = []
        for arm in ARMS:
            xs = [h["macro_f1"] for h in heads if h["arm"] == arm]
            vals.append(f"{100 * sum(xs) / len(xs):.2f}")
        print(f"| {g} | {trucks} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
