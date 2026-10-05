#!/bin/sh
set -eu
cast_spectrum_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_spectrum_root=$(CDPATH= cd -- "$cast_spectrum_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_spectrum_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_spectrum_dir/../.cache"
export PYTHONPATH="$cast_spectrum_dir/../src:$cast_spectrum_root/CAST/generalization/src:$cast_spectrum_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_spectrum_root/.venv/bin/python}" -m cast_improvement.spectrum_evaluation "$@"
