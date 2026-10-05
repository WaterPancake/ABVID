#!/bin/sh
set -eu
cast_group_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_group_root=$(CDPATH= cd -- "$cast_group_dir/../../.." && pwd)
cast_group_source=${1:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
cast_group_outer=${2:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_group_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_group_dir/../.cache"
export PYTHONPATH="$cast_group_dir/../src:$cast_group_root/CAST/generalization/src:$cast_group_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_group_python=${CAST_PYTHON:-$cast_group_root/.venv/bin/python}
cd "$cast_group_root"
"$cast_group_python" - "$cast_group_source" "$cast_group_outer" <<'PY'
import re
import sys
from pathlib import Path
source, outer = sys.argv[1:]
if source == outer or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", x) for x in (source,outer)):
    raise SystemExit("Use distinct simple fresh IDs of at most 80 characters")
base = Path("CAST/improvement/group_v12")
if (base/"evaluations"/outer).exists() or (base/"diagnostics"/(outer+"_review")).exists():
    raise SystemExit("Preserve existing outer artifacts; choose a fresh ID")
PY
sh "$cast_group_dir/reproduce.sh" "$cast_group_source"
sh "$cast_group_dir/outer.sh" run --source-eval-id "$cast_group_source" --eval-id "$cast_group_outer"
sh "$cast_group_dir/outer.sh" verify --eval-id "$cast_group_outer"
"$cast_group_python" -m cast_improvement.group_outer_report --eval-id "$cast_group_outer" --output-id "${cast_group_outer}_review"
"$cast_group_python" - "$cast_group_outer" <<'PY'
import json
import sys
from pathlib import Path
out = Path("CAST/improvement/group_v12/evaluations")/sys.argv[1]
verified = json.loads((out/"verification.json").read_text())
passed = verified["passed"] and verified["goal_passed"]
print("Exposed outer development criteria:", "PASS" if passed else "FAIL; all results retained")
raise SystemExit(0 if passed else 3)
PY
