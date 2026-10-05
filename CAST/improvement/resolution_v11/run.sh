#!/bin/sh
set -eu
cast_resolution_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_resolution_root=$(CDPATH= cd -- "$cast_resolution_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_resolution_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_resolution_dir/../.cache"
export PYTHONPATH="$cast_resolution_dir/../src:$cast_resolution_root/CAST/generalization/src:$cast_resolution_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_resolution_root/.venv/bin/python}" -m cast_improvement.resolution_evaluation "$@"
