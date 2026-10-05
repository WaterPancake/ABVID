#!/bin/sh
# Fresh-run reproduction. Existing IDs are intentionally never overwritten.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash CAST/improvement/reproduce.sh NEW_RUN_ID [WORKERS=4]" >&2
  exit 2
fi
cast_repro_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_repro_id=$1
cast_repro_workers=${2:-4}
case "$cast_repro_id" in
  ''|*[!A-Za-z0-9_-]*) echo "Use a simple new run ID" >&2; exit 2 ;;
esac
case "$cast_repro_workers" in
  1|2|3|4) ;;
  *) echo "Use 1 to 4 workers" >&2; exit 2 ;;
esac
if [ "${#cast_repro_id}" -gt 70 ]; then
  echo "Keep run ID to 70 characters or fewer" >&2
  exit 2
fi
bash "$cast_repro_dir/run.sh" prepare --run-id "$cast_repro_id"
bash "$cast_repro_dir/run.sh" synthetic --run-id "$cast_repro_id" --workers "$cast_repro_workers"
bash "$cast_repro_dir/run.sh" fit --run-id "$cast_repro_id" --scope pilot --workers "$cast_repro_workers"
bash "$cast_repro_dir/analyze.sh" audit-fits --run-id "$cast_repro_id" --scope pilot
bash "$cast_repro_dir/analyze.sh" source --run-id "$cast_repro_id" --eval-id "${cast_repro_id}_pilot_source" --scope pilot --temperatures 1 --include-v0
bash "$cast_repro_dir/analyze.sh" audit-source --eval-id "${cast_repro_id}_pilot_source"
bash "$cast_repro_dir/run.sh" fit --run-id "$cast_repro_id" --scope full --workers "$cast_repro_workers"
bash "$cast_repro_dir/analyze.sh" audit-fits --run-id "$cast_repro_id" --scope full
bash "$cast_repro_dir/analyze.sh" source --run-id "$cast_repro_id" --eval-id "${cast_repro_id}_full_source" --scope full --temperatures 1 1.1 1.25 1.5
bash "$cast_repro_dir/analyze.sh" audit-source --eval-id "${cast_repro_id}_full_source"
bash "$cast_repro_dir/analyze.sh" diagnostics source --id "${cast_repro_id}_full_source" --output-id "${cast_repro_id}_source_figures"
bash "$cast_repro_dir/analyze.sh" diagnostics fits --id "$cast_repro_id" --scope full --output-id "${cast_repro_id}_fit_figures"
bash "$cast_repro_dir/analyze.sh" gallery --run-id "$cast_repro_id" --scope full --output-id "${cast_repro_id}_audio_review"
# The outer runner refuses a failed source candidate before any held access.
# A scientific failure therefore exits nonzero, preserving all evidence above.
bash "$cast_repro_dir/analyze.sh" outer run --source-eval-id "${cast_repro_id}_full_source" --eval-id "${cast_repro_id}_outer"
bash "$cast_repro_dir/analyze.sh" outer verify --eval-id "${cast_repro_id}_outer"
cast_repro_python=${CAST_PYTHON:-$cast_repro_dir/../../.venv/bin/python}
"$cast_repro_python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("Verified scientific goal:", r["goal_passed"]); sys.exit(0 if r["passed"] and r["goal_passed"] else 3)' "$cast_repro_dir/evaluations/${cast_repro_id}_outer/verification.json"
