#!/bin/sh
set -eu
cast_width_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_width_root=$(CDPATH= cd -- "$cast_width_dir/../../.." && pwd)
cast_width_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_width_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_width_dir/../.cache"
export PYTHONPATH="$cast_width_dir/../src:$cast_width_root/CAST/generalization/src:$cast_width_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_width_python=${CAST_PYTHON:-$cast_width_root/.venv/bin/python}
cd "$cast_width_root"
"$cast_width_python" - "$cast_width_id" <<'PY'
import re
import sys
from pathlib import Path
name = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
    raise SystemExit("Use a simple fresh ID of at most 80 characters")
base = Path("CAST/improvement/width_v15")
paths = [base/"evaluations"/name] + [base/"diagnostics"/(name+s) for s in ("_figures", "_audio")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose a fresh ID")
PY
sh "$cast_width_dir/run.sh" source --eval-id "$cast_width_id"
sh "$cast_width_dir/run.sh" audit-source --eval-id "$cast_width_id"
"$cast_width_python" -m cast_improvement.width_figures --eval-id "$cast_width_id" --output-id "${cast_width_id}_figures"
"$cast_width_python" -m cast_improvement.width_audio --eval-id "$cast_width_id" --output-id "${cast_width_id}_audio"
"$cast_width_python" - "$cast_width_id" <<'PY'
import sys
from cast.config import save, sha
from cast_improvement.width_evaluation import HERE, check_source
out = HERE/"evaluations"/sys.argv[1]
_, summary, selected = check_source(out)
passed = selected["passes"] and selected["temperature"] != 1.
save(out/"progression_decision.json", {"selected": summary["selected"],
    "source_thresholds_passed": selected["passes"], "new_candidate_selected": selected["temperature"] != 1.,
    "original_mean_criteria_passed": selected["original_mean_criteria_passed"], "source_fold_coverage_passed": selected["source_fold_coverage_passed"],
    "outer_access_permitted_by_source": passed, "all_results_retained": True,
    "summary_sha256": sha(out/"summary.json"), "audit_sha256": sha(out/"audit_verification.json"),
    "reason": "New passing candidate selected" if passed else "Source failure or unchanged reference retained; preserve its existing outer result"})
print("New source candidate selected and passing:", "PASS" if passed else "FAIL or unchanged reference retained; no outer access")
raise SystemExit(0 if passed else 3)
PY
