#!/bin/sh
set -eu
cast_mix_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_mix_root=$(CDPATH= cd -- "$cast_mix_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_mix_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_mix_dir/../.cache"
export PYTHONPATH="$cast_mix_dir/../src:$cast_mix_root/CAST/generalization/src:$cast_mix_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_mix_root/.venv/bin/python}" -m cast_improvement.mixture_evaluation "$@"
