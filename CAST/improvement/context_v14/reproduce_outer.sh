#!/bin/sh
set -eu
cast_context_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_context_root=$(CDPATH= cd -- "$cast_context_dir/../../.." && pwd)
cast_context_source=${1:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
cast_context_outer=${2:?Usage: reproduce_outer.sh FRESH_SOURCE_ID FRESH_OUTER_ID}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_context_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_context_dir/../.cache"
export PYTHONPATH="$cast_context_dir/../src:$cast_context_root/CAST/generalization/src:$cast_context_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_context_python=${CAST_PYTHON:-$cast_context_root/.venv/bin/python}
cd "$cast_context_root"
"$cast_context_python" - "$cast_context_source" "$cast_context_outer" <<'PY'
import re
import sys
from pathlib import Path
source, outer = sys.argv[1:]
if source == outer or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", x) for x in (source,outer)):
    raise SystemExit("Use distinct simple fresh IDs of at most 80 characters")
base = Path("CAST/improvement/context_v14")
if (base/"evaluations"/outer).exists() or (base/"diagnostics"/(outer+"_review")).exists():
    raise SystemExit("Preserve existing outer artifacts; choose a fresh ID")
PY
sh "$cast_context_dir/reproduce.sh" "$cast_context_source"
sh "$cast_context_dir/outer.sh" run --source-eval-id "$cast_context_source" --eval-id "$cast_context_outer"
sh "$cast_context_dir/outer.sh" verify --eval-id "$cast_context_outer"
"$cast_context_python" -m cast_improvement.context_outer_report --eval-id "$cast_context_outer" --output-id "${cast_context_outer}_review"
"$cast_context_python" - "$cast_context_outer" <<'PY'
import json
import sys
from pathlib import Path
out = Path("CAST/improvement/context_v14/evaluations")/sys.argv[1]
verified = json.loads((out/"verification.json").read_text())
passed = verified["passed"] and verified["goal_passed"]
print("Exposed outer development criteria:", "PASS" if passed else "FAIL; all results retained")
raise SystemExit(0 if passed else 3)
PY
