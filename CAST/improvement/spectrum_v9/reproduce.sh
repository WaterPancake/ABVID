#!/bin/sh
set -eu
cast_spectrum_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_spectrum_root=$(CDPATH= cd -- "$cast_spectrum_dir/../../.." && pwd)
cast_spectrum_id=${1:?Usage: reproduce.sh FRESH_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_spectrum_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_spectrum_dir/../.cache"
export PYTHONPATH="$cast_spectrum_dir/../src:$cast_spectrum_root/CAST/generalization/src:$cast_spectrum_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_spectrum_python=${CAST_PYTHON:-$cast_spectrum_root/.venv/bin/python}
cd "$cast_spectrum_root"
"$cast_spectrum_python" - "$cast_spectrum_id" <<'PY'
import re
import sys
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}",sys.argv[1]):
    raise SystemExit("Use a simple ID of at most 80 characters")
PY
"$cast_spectrum_python" -m cast_improvement.spectrum_evaluation calibrate --run-id "$cast_spectrum_id" --workers 4
"$cast_spectrum_python" -m cast_improvement.spectrum_evaluation audit-bank --run-id "$cast_spectrum_id" --workers 4
"$cast_spectrum_python" -m cast_improvement.spectrum_evaluation source --run-id "$cast_spectrum_id" --eval-id "${cast_spectrum_id}_source"
"$cast_spectrum_python" -m cast_improvement.spectrum_evaluation audit-source --eval-id "${cast_spectrum_id}_source"
"$cast_spectrum_python" -m cast_improvement.spectrum_figures --eval-id "${cast_spectrum_id}_source" --output-id "${cast_spectrum_id}_figures"
"$cast_spectrum_python" -m cast_improvement.spectrum_diagnostics --eval-id "${cast_spectrum_id}_source" --output-id "${cast_spectrum_id}_audio"
"$cast_spectrum_python" - "${cast_spectrum_id}_source" <<'PY'
import sys
from cast_improvement.spectrum_evaluation import HERE,check_source
_,_,selected=check_source(HERE/"evaluations"/sys.argv[1])
print("Scientific source criteria:","PASS" if selected["passes"] else "FAIL; all artifacts retained; no outer access")
raise SystemExit(0 if selected["passes"] else 3)
PY
