#!/bin/sh
set -eu
cast_group_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_group_root=$(CDPATH= cd -- "$cast_group_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_group_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_group_dir/../.cache"
export PYTHONPATH="$cast_group_dir/../src:$cast_group_root/CAST/generalization/src:$cast_group_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_group_root/.venv/bin/python}" -m cast_improvement.group_evaluation "$@"
