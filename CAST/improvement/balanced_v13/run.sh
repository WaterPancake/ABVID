#!/bin/sh
set -eu
cast_balanced_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_balanced_root=$(CDPATH= cd -- "$cast_balanced_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_balanced_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_balanced_dir/../.cache"
export PYTHONPATH="$cast_balanced_dir/../src:$cast_balanced_root/CAST/generalization/src:$cast_balanced_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_balanced_root/.venv/bin/python}" -m cast_improvement.balanced_evaluation "$@"
