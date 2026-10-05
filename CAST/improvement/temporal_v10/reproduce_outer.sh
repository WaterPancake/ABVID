#!/bin/sh
set -eu
cast_temporal_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_temporal_root=$(CDPATH= cd -- "$cast_temporal_dir/../../.." && pwd)
cast_temporal_source=${1:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
cast_temporal_outer=${2:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_temporal_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_temporal_dir/../.cache"
export PYTHONPATH="$cast_temporal_dir/../src:$cast_temporal_root/CAST/generalization/src:$cast_temporal_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_temporal_python=${CAST_PYTHON:-$cast_temporal_root/.venv/bin/python}
cd "$cast_temporal_root"
"$cast_temporal_python" - "$cast_temporal_source" "$cast_temporal_outer" <<'PY'
import re
import sys
from pathlib import Path
if len(set(sys.argv[1:])) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", v) for v in sys.argv[1:]):
    raise SystemExit("Use two distinct simple IDs of at most 80 characters")
base = Path("CAST/improvement/temporal_v10")
paths = [base/"evaluations"/v for v in sys.argv[1:]]
paths += [base/"diagnostics"/(sys.argv[1]+suffix) for suffix in ("_figures", "_audio")]
paths += [base/"diagnostics"/(sys.argv[2]+"_review")]
if any(p.exists() for p in paths):
    raise SystemExit("Preserve existing artifacts; choose fresh IDs")
PY
sh "$cast_temporal_dir/reproduce.sh" "$cast_temporal_source"
sh "$cast_temporal_dir/outer.sh" run --source-eval-id "$cast_temporal_source" --eval-id "$cast_temporal_outer"
sh "$cast_temporal_dir/outer.sh" verify --eval-id "$cast_temporal_outer"
"$cast_temporal_python" -m cast_improvement.temporal_outer_report --eval-id "$cast_temporal_outer" --output-id "${cast_temporal_outer}_review"
"$cast_temporal_python" - "$cast_temporal_outer" <<'PY'
import json
import sys
from pathlib import Path
out = Path("CAST/improvement/temporal_v10/evaluations")/sys.argv[1]
verified = json.loads((out/"verification.json").read_text())
passed = verified["passed"] and verified["goal_passed"]
print("Exposed outer development criteria:", "PASS" if passed else "FAIL; all results retained")
raise SystemExit(0 if passed else 3)
PY
