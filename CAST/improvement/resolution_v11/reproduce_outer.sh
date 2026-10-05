#!/bin/sh
set -eu
cast_resolution_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_resolution_root=$(CDPATH= cd -- "$cast_resolution_dir/../../.." && pwd)
cast_resolution_run=${1:?Usage: reproduce_outer.sh FRESH_RUN_ID FRESH_OUTER_ID}
cast_resolution_outer=${2:?Usage: reproduce_outer.sh FRESH_RUN_ID FRESH_OUTER_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_resolution_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_resolution_dir/../.cache"
export PYTHONPATH="$cast_resolution_dir/../src:$cast_resolution_root/CAST/generalization/src:$cast_resolution_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_resolution_python=${CAST_PYTHON:-$cast_resolution_root/.venv/bin/python}
cd "$cast_resolution_root"
"$cast_resolution_python" - "$cast_resolution_run" "$cast_resolution_outer" <<'PY'
import re
import sys
from pathlib import Path
run, outer = sys.argv[1:]
if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", v) for v in (run, outer)) or outer == run+"_source":
    raise SystemExit("Use simple fresh IDs of at most 80 characters; outer ID must differ from RUN_source")
base = Path("CAST/improvement/resolution_v11")
paths = [base/"runs"/run, base/"evaluations"/(run+"_source"), base/"evaluations"/outer]
paths += [base/"diagnostics"/(run+suffix) for suffix in ("_figures", "_audio")]
paths += [base/"diagnostics"/(outer+"_review")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose fresh IDs")
PY
sh "$cast_resolution_dir/reproduce.sh" "$cast_resolution_run"
sh "$cast_resolution_dir/outer.sh" run --source-eval-id "${cast_resolution_run}_source" --eval-id "$cast_resolution_outer"
sh "$cast_resolution_dir/outer.sh" verify --eval-id "$cast_resolution_outer"
"$cast_resolution_python" -m cast_improvement.resolution_outer_report --eval-id "$cast_resolution_outer" --output-id "${cast_resolution_outer}_review"
"$cast_resolution_python" - "$cast_resolution_outer" <<'PY'
import json
import sys
from pathlib import Path
out = Path("CAST/improvement/resolution_v11/evaluations")/sys.argv[1]
verified = json.loads((out/"verification.json").read_text())
passed = verified["passed"] and verified["goal_passed"]
print("Exposed outer development criteria:", "PASS" if passed else "FAIL; all results retained")
raise SystemExit(0 if passed else 3)
PY
