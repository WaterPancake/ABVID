"""Numeric components extracted from CAST/improvement/src/cast_improvement/data.py."""
from abvid.paths import archive_root
ROOT = archive_root()


from abvid.provenance.io import read, readl


PILOT = ROOT/"CAST/runs/cast_pilot_v0_20261004"


PREVIOUS = ROOT/"CAST/generalization/runs/cast_generalization_v0_20261004_r1"
