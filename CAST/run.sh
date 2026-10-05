#!/bin/sh
set -eu
cast_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_python=${CAST_PYTHON:-"$cast_dir/../.venv/bin/python"}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_dir/.cache/matplotlib"
export XDG_CACHE_HOME="$cast_dir/.cache"
export PYTHONPATH="$cast_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$cast_python" -m cast.cli "$@"
