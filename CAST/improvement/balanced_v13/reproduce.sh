#!/bin/sh
set -eu
cast_balanced_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_balanced_root=$(CDPATH= cd -- "$cast_balanced_dir/../../.." && pwd)
cast_balanced_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_balanced_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_balanced_dir/../.cache"
export PYTHONPATH="$cast_balanced_dir/../src:$cast_balanced_root/CAST/generalization/src:$cast_balanced_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_balanced_python=${CAST_PYTHON:-$cast_balanced_root/.venv/bin/python}
cd "$cast_balanced_root"
"$cast_balanced_python" - "$cast_balanced_id" <<'PY'
import re
import sys
from pathlib import Path
name = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
    raise SystemExit("Use a simple fresh ID of at most 80 characters")
base = Path("CAST/improvement/balanced_v13")
paths = [base/"evaluations"/name] + [base/"diagnostics"/(name+s) for s in ("_figures", "_audio")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose a fresh ID")
PY
sh "$cast_balanced_dir/run.sh" source --eval-id "$cast_balanced_id"
sh "$cast_balanced_dir/run.sh" audit-source --eval-id "$cast_balanced_id"
"$cast_balanced_python" -m cast_improvement.balanced_figures --eval-id "$cast_balanced_id" --output-id "${cast_balanced_id}_figures"
"$cast_balanced_python" -m cast_improvement.balanced_audio --eval-id "$cast_balanced_id" --output-id "${cast_balanced_id}_audio"
"$cast_balanced_python" - "$cast_balanced_id" <<'PY'
import sys
from cast.config import save, sha
from cast_improvement.balanced_evaluation import HERE, check_source
out = HERE/"evaluations"/sys.argv[1]
_, summary, selected = check_source(out)
passed = selected["passes"] and selected["variant"] != "group_predictive"
save(out/"progression_decision.json", {"selected": summary["selected"],
    "source_thresholds_passed": selected["passes"], "new_candidate_selected": selected["variant"] != "group_predictive",
    "outer_access_permitted_by_source": passed, "all_results_retained": True,
    "summary_sha256": sha(out/"summary.json"), "audit_sha256": sha(out/"audit_verification.json"),
    "reason": "New passing candidate selected" if passed else "Source failure or unchanged reference retained; preserve its existing outer result"})
print("New source candidate selected and passing:", "PASS" if passed else "FAIL or unchanged reference retained; no outer access")
raise SystemExit(0 if passed else 3)
PY
