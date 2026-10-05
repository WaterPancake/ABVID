#!/bin/sh
set -eu
cast_resolution_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_resolution_root=$(CDPATH= cd -- "$cast_resolution_dir/../../.." && pwd)
cast_resolution_id=${1:?Usage: reproduce.sh FRESH_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_resolution_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_resolution_dir/../.cache"
export PYTHONPATH="$cast_resolution_dir/../src:$cast_resolution_root/CAST/generalization/src:$cast_resolution_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_resolution_python=${CAST_PYTHON:-$cast_resolution_root/.venv/bin/python}
cd "$cast_resolution_root"
"$cast_resolution_python" - "$cast_resolution_id" <<'PY'
import re
import sys
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",sys.argv[1]):
    raise SystemExit("Use a simple ID of at most 80 characters")
PY
"$cast_resolution_python" -m cast_improvement.resolution_evaluation calibrate --run-id "$cast_resolution_id" --workers 4
"$cast_resolution_python" -m cast_improvement.resolution_evaluation audit-bank --run-id "$cast_resolution_id" --workers 4
"$cast_resolution_python" -m cast_improvement.resolution_evaluation source --run-id "$cast_resolution_id" --eval-id "${cast_resolution_id}_source"
"$cast_resolution_python" -m cast_improvement.resolution_evaluation audit-source --eval-id "${cast_resolution_id}_source"
"$cast_resolution_python" -m cast_improvement.resolution_figures --eval-id "${cast_resolution_id}_source" --output-id "${cast_resolution_id}_figures"
"$cast_resolution_python" -m cast_improvement.resolution_diagnostics --eval-id "${cast_resolution_id}_source" --output-id "${cast_resolution_id}_audio"
"$cast_resolution_python" - "${cast_resolution_id}_source" <<'PY'
import sys
from cast_improvement.resolution_evaluation import HERE,check_source
_,_,selected=check_source(HERE/"evaluations"/sys.argv[1])
passed=selected["passes"] and selected["variant"]=="spectrum16"
print("New source candidate selected and passing:","PASS" if passed else "FAIL or unchanged reference retained; no outer access")
raise SystemExit(0 if passed else 3)
PY
