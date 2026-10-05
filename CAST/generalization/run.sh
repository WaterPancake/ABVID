#!/bin/sh
set -eu
cast_ext_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_root=$(CDPATH= cd -- "$cast_ext_dir/../.." && pwd)
cast_python=${CAST_PYTHON:-"$cast_root/.venv/bin/python"}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_ext_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$cast_ext_dir/.cache"
export PYTHONPATH="$cast_ext_dir/src:$cast_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$cast_python" -m cast_generalization "$@"
