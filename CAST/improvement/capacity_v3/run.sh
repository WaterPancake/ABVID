#!/bin/sh
set -eu
cast_cap_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_cap_root=$(CDPATH= cd -- "$cast_cap_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_cap_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_cap_dir/../.cache"
export PYTHONPATH="$cast_cap_dir/../src:$cast_cap_root/CAST/generalization/src:$cast_cap_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_cap_action=$1
shift
case "$cast_cap_action" in
  prepare|synthetic|fit) exec "${CAST_PYTHON:-$cast_cap_root/.venv/bin/python}" -m cast_improvement.capacity_cli "$cast_cap_action" "$@" ;;
  audit-fits) exec "${CAST_PYTHON:-$cast_cap_root/.venv/bin/python}" -m cast_improvement.capacity_audit "$@" ;;
  source|audit-source) exec "${CAST_PYTHON:-$cast_cap_root/.venv/bin/python}" -m cast_improvement.capacity_evaluation "$cast_cap_action" "$@" ;;
  diagnostics) exec "${CAST_PYTHON:-$cast_cap_root/.venv/bin/python}" -m cast_improvement.capacity_diagnostics "$@" ;;
  *) echo "Choose prepare, synthetic, fit, audit-fits, source, audit-source or diagnostics" >&2; exit 2 ;;
esac
