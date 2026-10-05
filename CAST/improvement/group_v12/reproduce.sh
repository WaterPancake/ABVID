#!/bin/sh
set -eu
cast_group_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_group_root=$(CDPATH= cd -- "$cast_group_dir/../../.." && pwd)
cast_group_id=${1:?Usage: reproduce.sh FRESH_SOURCE_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_group_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_group_dir/../.cache"
export PYTHONPATH="$cast_group_dir/../src:$cast_group_root/CAST/generalization/src:$cast_group_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_group_python=${CAST_PYTHON:-$cast_group_root/.venv/bin/python}
cd "$cast_group_root"
"$cast_group_python" - "$cast_group_id" <<'PY'
import re
import sys
from pathlib import Path
name = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
    raise SystemExit("Use a simple fresh ID of at most 80 characters")
base = Path("CAST/improvement/group_v12")
paths = [base/"evaluations"/name] + [base/"diagnostics"/(name+s) for s in ("_figures", "_audio")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose a fresh ID")
PY
sh "$cast_group_dir/run.sh" source --eval-id "$cast_group_id"
sh "$cast_group_dir/run.sh" audit-source --eval-id "$cast_group_id"
"$cast_group_python" -m cast_improvement.group_figures --eval-id "$cast_group_id" --output-id "${cast_group_id}_figures"
"$cast_group_python" -m cast_improvement.group_audio --eval-id "$cast_group_id" --output-id "${cast_group_id}_audio"
"$cast_group_python" - "$cast_group_id" <<'PY'
import sys
from cast_improvement.group_evaluation import HERE, check_source
_, _, selected = check_source(HERE/"evaluations"/sys.argv[1])
passed = selected["passes"] and selected["variant"] != "spectrum16"
print("New source candidate selected and passing:", "PASS" if passed else "FAIL or unchanged reference retained; no outer access")
raise SystemExit(0 if passed else 3)
PY
