#!/bin/sh
set -eu
cast_envelope_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_envelope_root=$(CDPATH= cd -- "$cast_envelope_dir/../../.." && pwd)
cast_envelope_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_envelope_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_envelope_dir/../.cache"
export PYTHONPATH="$cast_envelope_dir/../src:$cast_envelope_root/CAST/generalization/src:$cast_envelope_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_envelope_python=${CAST_PYTHON:-$cast_envelope_root/.venv/bin/python}
cd "$cast_envelope_root"
"$cast_envelope_python" - "$cast_envelope_id" <<'PY'
import re
import sys
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", sys.argv[1]):
    raise SystemExit("Use a simple source ID of at most 80 characters, leaving room for artifact suffixes")
PY
"$cast_envelope_python" -m cast_improvement.envelope_evaluation source --eval-id "$cast_envelope_id"
"$cast_envelope_python" -m cast_improvement.envelope_evaluation audit-source --eval-id "$cast_envelope_id"
"$cast_envelope_python" -m cast_improvement.envelope_figures --eval-id "$cast_envelope_id" --output-id "${cast_envelope_id}_figures"
"$cast_envelope_python" -m cast_improvement.envelope_diagnostics --eval-id "$cast_envelope_id" --output-id "${cast_envelope_id}_audio"
"$cast_envelope_python" - "$cast_envelope_id" <<'PY'
import sys
from cast_improvement.envelope_evaluation import HERE, check_source
_, _, selected = check_source(HERE/"evaluations"/sys.argv[1])
print("Scientific criteria:", "PASS" if selected["passes"] else "FAIL; all failures retained; no outer evaluation")
raise SystemExit(0 if selected["passes"] else 3)
PY
