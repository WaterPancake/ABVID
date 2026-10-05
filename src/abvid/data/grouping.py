"""Conservative original-source grouping, without inventing independent sessions."""
from collections import defaultdict
from math import gcd
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial import cKDTree

class Union:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.find(self.parent[k])
        return self.parent[k]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def parent_proxy(row):
    if row['dataset_id'] == 'IDMT':
        return row['candidate_group_id']
    if row['dataset_id'] == 'MELAUDIS':
        # Overmerge rather than invent independence for numbered street positions.
        return 'MELAUDIS|day|' + row['calendar_date']
    return row['dataset_id'] + '|entire_released_bank'

def interval(row):
    if row['dataset_id'] == 'IDMT':
        center = row['sample_position_center'] / row['native_sample_rate_hz']
        return row['candidate_recording_id'], center - 1., center + 1.
    if row['dataset_id'] == 'MELAUDIS' and row['metadata_parse_ok']:
        h, m, s = row['timestamp_token'].split('-')[:3]
        center = int(h) * 3600 + int(m) * 60 + float(s)
        return row['candidate_group_id'], center - 1., center + 1.
    return None

def overlapping_components(rows):
    buckets = defaultdict(list)
    for row in rows:
        iv = interval(row)
        if iv:
            buckets[iv[0]].append((iv[1], iv[2], row['file_id']))
    result = []
    for key, spans in sorted(buckets.items()):
        component, end = [], -float('inf')
        for start, stop, fid in sorted(spans):
            if start >= end:
                if len(component) > 1:
                    result.append(dict(parent_time_axis=key, members=component))
                component = []
            component.append(fid)
            end = max(end, stop)
        if len(component) > 1:
            result.append(dict(parent_time_axis=key, members=component))
    return result


PROTOCOL = 'dataset_admission_v1.3.1'


RULES = dict(protocol=PROTOCOL, rate_hz=2000, seconds=2, normalize='demean_then_l2',
             exact_quantization=100000, projection_seed=20261002,
             projection_dimensions=16, nearest_neighbors=4, correlation_threshold=0.9999,
             search='aligned center 2s; random projections; kNN within and across datasets',
             polarity_invariant=True, verification_rate_hz=8000,
             melaudis_groups='whole_calendar_day_then_connected_reuse',
             idmt_groups='site_calendar_day_then_connected_reuse',
             synthetic_groups='one_training_only_group_per_released_generator_bank',
             limitations=['Not exhaustive arbitrary crop/time-stretch/codec reuse detection',
                          'Original media and physical-vehicle IDs remain unknown',
                          'No held-out synthetic source/template inference is supported'])


def normalized_center(x, sr, rate=2000):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    g = gcd(int(sr), rate)
    y = resample_poly(x, rate // g, int(sr) // g)
    frames = 2 * rate
    if len(y) < frames:
        return None
    start = (len(y) - frames) // 2
    y = y[start:start + frames]
    y -= y.mean()
    norm = np.linalg.norm(y)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return np.asarray(y / norm, np.float32)


def verified_pairs(waves, dataset_names, hashes):
    """Conservative waveform-equivalence candidates, independent of class labels."""
    n = len(waves)
    projection = np.random.default_rng(RULES['projection_seed']).normal(
        size=(waves.shape[1], RULES['projection_dimensions'])).astype(np.float32)
    z = waves @ projection
    pairs = set()
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        buckets[h].append(i)
    for members in buckets.values():
        for j in members[1:]:
            pairs.add((members[0], j))
    for ds in sorted(set(dataset_names)):
        idx = np.flatnonzero(np.asarray(dataset_names) == ds)
        tree = cKDTree(z[idx])
        for polarity in (1, -1):
            _, neighbors = tree.query(z * polarity, k=min(5, len(idx)), workers=4)
            for i, ns in enumerate(np.atleast_2d(neighbors)):
                for j in idx[ns]:
                    if i != j:
                        pairs.add((min(i, int(j)), max(i, int(j))))
    accepted = []
    ordered = sorted(pairs)
    for start in range(0, len(ordered), 4096):
        block = ordered[start:start + 4096]
        a, b = np.asarray(block, dtype=int).T
        corrs = np.abs(np.einsum('ij,ij->i', waves[a], waves[b]))
        for (i, j), corr in zip(block, corrs):
            if corr >= RULES['correlation_threshold']:
                accepted.append((i, j, float(min(1., corr))))
    return accepted, len(ordered)
