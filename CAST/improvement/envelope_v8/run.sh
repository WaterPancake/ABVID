#!/bin/sh
set -eu
cast_envelope_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cast_envelope_root=$(CDPATH= cd -- "$cast_envelope_dir/../../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$cast_envelope_dir/../.cache/matplotlib"
export XDG_CACHE_HOME="$cast_envelope_dir/../.cache"
export PYTHONPATH="$cast_envelope_dir/../src:$cast_envelope_root/CAST/generalization/src:$cast_envelope_root/CAST/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CAST_PYTHON:-$cast_envelope_root/.venv/bin/python}" -m cast_improvement.envelope_evaluation "$@"
