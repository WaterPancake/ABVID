#!/bin/sh
set -eu
cast_temporal_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_temporal_root=$(CDPATH= cd -- "$cast_temporal_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_temporal_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_temporal_dir/../.cache"
export PYTHONPATH="$cast_temporal_dir/../src:$cast_temporal_root/CAST/generalization/src:$cast_temporal_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_temporal_root/.venv/bin/python}" -m cast_improvement.temporal_evaluation "$@"
