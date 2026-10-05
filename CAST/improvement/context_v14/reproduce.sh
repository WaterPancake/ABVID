#!/bin/sh
set -eu
cast_context_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_context_root=$(CDPATH= cd -- "$cast_context_dir/../../.." && pwd)
cast_context_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_context_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_context_dir/../.cache"
export PYTHONPATH="$cast_context_dir/../src:$cast_context_root/CAST/generalization/src:$cast_context_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_context_python=${CAST_PYTHON:-$cast_context_root/.venv/bin/python}
cd "$cast_context_root"
"$cast_context_python" - "$cast_context_id" <<'PY'
import re
import sys
from pathlib import Path
name = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
    raise SystemExit("Use a simple fresh ID of at most 80 characters")
base = Path("CAST/improvement/context_v14")
paths = [base/"evaluations"/name] + [base/"diagnostics"/(name+s) for s in ("_figures", "_audio")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose a fresh ID")
PY
sh "$cast_context_dir/run.sh" source --eval-id "$cast_context_id"
sh "$cast_context_dir/run.sh" audit-source --eval-id "$cast_context_id"
"$cast_context_python" -m cast_improvement.context_figures --eval-id "$cast_context_id" --output-id "${cast_context_id}_figures"
"$cast_context_python" -m cast_improvement.context_audio --eval-id "$cast_context_id" --output-id "${cast_context_id}_audio"
"$cast_context_python" - "$cast_context_id" <<'PY'
import sys
from cast.config import save, sha
from cast_improvement.context_evaluation import HERE, check_source
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
