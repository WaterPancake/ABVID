#!/bin/sh
set -eu
cast_alt_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_alt_root=$(CDPATH= cd -- "$cast_alt_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_alt_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_alt_dir/../.cache"
export PYTHONPATH="$cast_alt_dir/../src:$cast_alt_root/CAST/generalization/src:$cast_alt_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_alt_root/.venv/bin/python}" -m cast_improvement.alternative_evaluation "$@"
