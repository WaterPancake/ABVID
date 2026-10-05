#!/bin/sh
set -eu
cast_cap_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_cap_root=$(CDPATH= cd -- "$cast_cap_dir/../../.." && pwd)
cd "$cast_cap_root"
cast_cap_id=${1:?Provide a fresh run ID}
cast_cap_workers=${2:-4}
case "$cast_cap_id" in
  ''|*[!A-Za-z0-9_-]*) echo "Use a simple run ID" >&2; exit 2 ;;
esac
if [ -e "$cast_cap_dir/runs/$cast_cap_id" ]; then
  echo "Preserve existing runs; choose a fresh ID" >&2
  exit 2
fi
sh "$cast_cap_dir/run.sh" prepare --run-id "$cast_cap_id"
sh "$cast_cap_dir/run.sh" synthetic --run-id "$cast_cap_id" --workers "$cast_cap_workers"
sh "$cast_cap_dir/run.sh" fit --run-id "$cast_cap_id" --scope pilot --workers "$cast_cap_workers"
sh "$cast_cap_dir/run.sh" audit-fits --run-id "$cast_cap_id" --scope pilot
sh "$cast_cap_dir/run.sh" source --run-id "$cast_cap_id" --scope pilot --eval-id "${cast_cap_id}_pilot_source"
sh "$cast_cap_dir/run.sh" audit-source --eval-id "${cast_cap_id}_pilot_source"
sh "$cast_cap_dir/run.sh" diagnostics paired --id "$cast_cap_id" --scope pilot --output-id "${cast_cap_id}_paired"
sh "$cast_cap_dir/run.sh" diagnostics fits --id "$cast_cap_id" --scope pilot --output-id "${cast_cap_id}_residuals"
sh "$cast_cap_dir/run.sh" diagnostics source --id "${cast_cap_id}_pilot_source" --output-id "${cast_cap_id}_source_figures"
sh "$cast_cap_dir/run.sh" diagnostics gallery --id "$cast_cap_id" --scope pilot --output-id "${cast_cap_id}_audio"
"${CAST_PYTHON:-$cast_cap_root/.venv/bin/python}" - "$cast_cap_dir/evaluations/${cast_cap_id}_pilot_source/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    r = json.load(f)
passed = r["candidates"][r["selected"]]["passes"]
print(json.dumps({"source_passed": passed, "outer_access": False,
                  "next_step": "Review paired and source evidence before full-bank expansion"}))
sys.exit(0 if passed else 3)
PY
