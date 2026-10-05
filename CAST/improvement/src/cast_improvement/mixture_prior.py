"""Fixed effective mixture-odds adjustment with unchanged matched controls."""
from copy import deepcopy
from itertools import product
import numpy as np
from cast.renderer import validate_controls
from .block_prior import transform_bank as temper, draw as base_draw
from .evaluate import aggregate

GRID = tuple(product((.65, .8, 1.), (0., -1.)))


def method_id(spectral, offset):
    if (spectral, offset) not in GRID:
        raise ValueError("Undeclared allocation candidate")
    return f"smooth8_s{spectral:g}_mix{offset:g}"


def transform_bank(bank, cfg, spectral, offset):
    method_id(spectral, offset)
    result, stats = temper(bank, cfg, spectral, 1.)
    for s in stats.values():
        s.update(harmonic_logit_offset=offset, allocation_rule="bounded odds multiplication after spectral temperature")
    if offset != 0:
        ratio = np.exp(offset)
        for row in result:
            p = deepcopy(row["parameters"])
            m = p["harmonic_fraction"]
            p["harmonic_fraction"] = float(ratio*m/(1-m+ratio*m))
            row["parameters"] = validate_controls(p, cfg)
    return result, stats


def draw(bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, offset):
    r = base_draw(bank, statistics, cfg, metric_cfg, label, seed, index, arm, fold, spectral, 1.)
    r.pop("envelope_temperature")
    r.update(method_id=method_id(spectral, offset), harmonic_logit_offset=offset,
             calibration_role="all fold-training class-center parents; offset is a fixed source-development candidate")
    return r


def summarize(records, metric_cfg):
    result = aggregate(records, metric_cfg)
    settings = {method_id(s, o):(s,o) for s,o in GRID}
    for r in result.values():
        spectral, offset = settings[r["variant"]]
        r.update(spectral_temperature=spectral, harmonic_logit_offset=offset)
        r["selection_key"] = r["selection_key"][:3]+[abs(spectral-1)+abs(offset), spectral, offset]
    return result
