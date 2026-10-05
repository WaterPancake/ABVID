#!/bin/sh
set -eu
cast_temporal_outer_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_temporal_outer_root=$(CDPATH= cd -- "$cast_temporal_outer_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_temporal_outer_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_temporal_outer_dir/../.cache"
export PYTHONPATH="$cast_temporal_outer_dir/../src:$cast_temporal_outer_root/CAST/generalization/src:$cast_temporal_outer_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_temporal_outer_root/.venv/bin/python}" -m cast_improvement.temporal_outer "$@"
