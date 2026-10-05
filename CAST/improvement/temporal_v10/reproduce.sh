#!/bin/sh
set -eu
cast_temporal_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_temporal_root=$(CDPATH= cd -- "$cast_temporal_dir/../../.." && pwd)
cast_temporal_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_temporal_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_temporal_dir/../.cache"
export PYTHONPATH="$cast_temporal_dir/../src:$cast_temporal_root/CAST/generalization/src:$cast_temporal_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_temporal_python=${CAST_PYTHON:-$cast_temporal_root/.venv/bin/python}
cd "$cast_temporal_root"
"$cast_temporal_python" - "$cast_temporal_id" <<'PY'
import re
import sys
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", sys.argv[1]):
    raise SystemExit("Use a simple source ID of at most 80 characters, leaving room for artifact suffixes")
PY
"$cast_temporal_python" -m cast_improvement.temporal_evaluation source --eval-id "$cast_temporal_id"
"$cast_temporal_python" -m cast_improvement.temporal_evaluation audit-source --eval-id "$cast_temporal_id"
"$cast_temporal_python" -m cast_improvement.temporal_figures --eval-id "$cast_temporal_id" --output-id "${cast_temporal_id}_figures"
"$cast_temporal_python" -m cast_improvement.temporal_diagnostics --eval-id "$cast_temporal_id" --output-id "${cast_temporal_id}_audio"
"$cast_temporal_python" - "$cast_temporal_id" <<'PY'
import sys
from cast_improvement.temporal_evaluation import HERE, check_source
_, _, selected = check_source(HERE/"evaluations"/sys.argv[1])
print("Scientific criteria:", "PASS" if selected["passes"] else "FAIL; all failures retained; no outer evaluation")
raise SystemExit(0 if selected["passes"] else 3)
PY
