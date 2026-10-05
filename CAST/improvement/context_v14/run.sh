#!/bin/sh
set -eu
cast_context_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_context_root=$(CDPATH= cd -- "$cast_context_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_context_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_context_dir/../.cache"
export PYTHONPATH="$cast_context_dir/../src:$cast_context_root/CAST/generalization/src:$cast_context_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_context_root/.venv/bin/python}" -m cast_improvement.context_evaluation "$@"
