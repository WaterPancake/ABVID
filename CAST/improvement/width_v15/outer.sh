#!/bin/sh
set -eu
cast_width_outer_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_width_outer_root=$(CDPATH= cd -- "$cast_width_outer_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_width_outer_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_width_outer_dir/../.cache"
export PYTHONPATH="$cast_width_outer_dir/../src:$cast_width_outer_root/CAST/generalization/src:$cast_width_outer_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_width_outer_root/.venv/bin/python}" -m cast_improvement.width_outer "$@"
