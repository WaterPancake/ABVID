"""Numeric components extracted from CAST/improvement/src/cast_improvement/resolution_evaluation.py."""
from abvid.cast.population.resolution_bank import MODES


from abvid.cast.population.evaluate import rendered as smooth_rendered


from abvid.cast.population.resolution_model import ResolutionRenderer


from abvid.cast.audio import shape


from abvid.cast.renderer import tensors, seed_for


import torch


def render(record, cfg, variant):
    if variant not in MODES:
        raise ValueError("Undeclared prior mode")
    if variant=="temporal41":return smooth_rendered(record,cfg,"smooth8")
    renderer=ResolutionRenderer(cfg,record["input_id"],sampling=True)
    with torch.no_grad():wave=shape(renderer.render(tensors(record["parameters"]),"sampling"))[0,0].numpy()
    return wave,{"phase":seed_for(cfg,record["input_id"],"phase","sampling"),
                 "noise":[seed_for(cfg,record["input_id"],"noise","sampling",0)],"purpose":"sampling"}
