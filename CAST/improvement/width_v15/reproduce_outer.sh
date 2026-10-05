#!/bin/sh
set -eu
cast_width_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_width_root=$(CDPATH= cd -- "$cast_width_dir/../../.." && pwd)
cast_width_source=${1:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
cast_width_outer=${2:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_width_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_width_dir/../.cache"
export PYTHONPATH="$cast_width_dir/../src:$cast_width_root/CAST/generalization/src:$cast_width_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_width_python=${CAST_PYTHON:-$cast_width_root/.venv/bin/python}
cd "$cast_width_root"
"$cast_width_python" - "$cast_width_source" "$cast_width_outer" <<'PY'
import re
import sys
from pathlib import Path
source, outer = sys.argv[1:]
if source == outer or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", x) for x in (source,outer)):
    raise SystemExit("Use distinct simple fresh IDs of at most 80 characters")
base = Path("CAST/improvement/width_v15")
if (base/"evaluations"/outer).exists() or (base/"diagnostics"/(outer+"_review")).exists():
    raise SystemExit("Preserve existing outer artifacts; choose a fresh ID")
PY
sh "$cast_width_dir/reproduce.sh" "$cast_width_source"
sh "$cast_width_dir/outer.sh" run --source-eval-id "$cast_width_source" --eval-id "$cast_width_outer"
sh "$cast_width_dir/outer.sh" verify --eval-id "$cast_width_outer"
"$cast_width_python" -m cast_improvement.width_outer_report --eval-id "$cast_width_outer" --output-id "${cast_width_outer}_review"
"$cast_width_python" - "$cast_width_outer" <<'PY'
import json
import sys
from pathlib import Path
out = Path("CAST/improvement/width_v15/evaluations")/sys.argv[1]
verified = json.loads((out/"verification.json").read_text())
passed = verified["passed"] and verified["goal_passed"]
print("Exposed outer development criteria:", "PASS" if passed else "FAIL; all results retained")
raise SystemExit(0 if passed else 3)
PY
