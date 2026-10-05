#!/bin/sh
set -eu
cast_cal_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_cal_root=$(CDPATH= cd -- "$cast_cal_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_cal_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_cal_dir/../.cache"
export PYTHONPATH="$cast_cal_dir/../src:$cast_cal_root/CAST/generalization/src:$cast_cal_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_cal_root/.venv/bin/python}" -m cast_improvement.calibration_evaluation "$@"
