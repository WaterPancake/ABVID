#!/bin/sh
set -eu
cast_v1_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_v1_root=$(CDPATH= cd -- "$cast_v1_dir/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_v1_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$cast_v1_dir/.cache"
export PYTHONPATH="$cast_v1_dir/src:$cast_v1_root/CAST/generalization/src:$cast_v1_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
cast_analysis_module=$1
shift
case "$cast_analysis_module" in
  source) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.evaluate source "$@" ;;
  outer) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.outer "$@" ;;
  audit-source) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.audit_source "$@" ;;
  audit-fits) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.audit_fits "$@" ;;
  diagnostics) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.diagnostics "$@" ;;
  gallery) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.gallery "$@" ;;
  block) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.block_evaluation "$@" ;;
  block-figures) exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement.block_figures "$@" ;;
  *) echo "Choose source, outer, audit-source, audit-fits, diagnostics, gallery, block or block-figures" >&2; exit 2 ;;
esac
