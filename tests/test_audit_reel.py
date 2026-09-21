"""Audit reels must neither replay overlap nor include unselected source gaps."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'audit_reel', Path(__file__).parents[1] / 'scripts/build_audit_reel.py'
)
reel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reel)


def window(identifier, start, end, condition='idle'):
    return {'sample_id': identifier, 'window_start_seconds': start,
            'window_end_seconds': end, 'operating_condition': condition}


def test_union_preserves_gaps_and_condition_boundaries():
    segments = [
        {'start_seconds': 0, 'end_seconds': 10, 'operating_condition': 'idle'},
        {'start_seconds': 10, 'end_seconds': 14, 'operating_condition': 'steady_speed'},
    ]
    rows = [window('a', 1, 3), window('b', 2, 4), window('c', 7, 9),
            window('d', 10, 12, 'steady_speed')]
    result = reel.covered_ranges(rows, segments)
    assert [(r['start_seconds'], r['end_seconds']) for r in result] == [(1, 4), (7, 9), (10, 12)]
    assert result[-1]['operating_condition'] == 'steady_speed'


def test_window_crossing_review_boundary_is_rejected():
    with pytest.raises(ValueError, match='no matching reviewed interval'):
        reel.covered_ranges([window('a', 8, 11)], [
            {'start_seconds': 0, 'end_seconds': 10, 'operating_condition': 'idle'}
        ])


def test_reuse_requires_exact_segment_and_existing_video(tmp_path):
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'local-rendered-chapter')
    prior = {'clips': [{'source_id': 'source', 'source_start': 3, 'source_end': 5,
                        'condition': 'idle', 'video_available': True,
                        'clip_path': 'clip.mp4', 'notes': 'reviewed'}]}
    segment = {'start_seconds': 3, 'end_seconds': 5, 'operating_condition': 'idle', 'notes': 'reviewed'}
    assert reel.reusable_clip(prior, tmp_path, 'source', segment) == clip
    assert reel.reusable_clip(prior, tmp_path, 'source', {**segment, 'end_seconds': 6}) is None
    assert reel.reusable_clip(prior, tmp_path, 'source', {**segment, 'notes': 'changed'}) is None
    prior['clips'][0]['video_available'] = False
    assert reel.reusable_clip(prior, tmp_path, 'source', segment) is None


def test_reuse_rejects_path_outside_prior_reel(tmp_path):
    previous = tmp_path / 'prior'
    previous.mkdir()
    (tmp_path / 'outside.mp4').write_bytes(b'outside')
    prior = {'clips': [{'source_id': 'source', 'source_start': 3, 'source_end': 5,
                        'condition': 'idle', 'video_available': True,
                        'clip_path': '../outside.mp4'}]}
    segment = {'start_seconds': 3, 'end_seconds': 5, 'operating_condition': 'idle'}
    assert reel.reusable_clip(prior, previous, 'source', segment) is None
