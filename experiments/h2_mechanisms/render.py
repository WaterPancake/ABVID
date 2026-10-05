"""Expose independent ground/air toggles using unchanged, validated H2 primitives."""
from common import *
from renderer import build_plan,render_sources,exact_trajectory,observation


def render_all(sources,trajectory,mic=None,unit_ground=False):
    plan=build_plan(trajectory,mic,air=True,unit_ground=unit_ground)
    with_air=render_sources(sources,plan,components=True)
    without_air=render_sources(sources,dict(plan,air=None),components=True)
    np.testing.assert_array_equal(with_air['P0'],without_air['P0'])
    return dict(G0A0=with_air['P0'],G0A1=with_air['direct_filtered'],
        G1A0=without_air['P1'],G1A1=with_air['P1'])
