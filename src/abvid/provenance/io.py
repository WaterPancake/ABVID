"""Numeric components extracted from CAST/generalization/src/cast_generalization/pipeline.py."""
import json


from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def readl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]
