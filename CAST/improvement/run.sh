#!/bin/sh
set -eu
cast_v1_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_v1_root=$(CDPATH= cd -- "$cast_v1_dir/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_v1_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$cast_v1_dir/.cache"
export PYTHONPATH="$cast_v1_dir/src:$cast_v1_root/CAST/generalization/src:$cast_v1_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_v1_root/.venv/bin/python}" -m cast_improvement "$@"
