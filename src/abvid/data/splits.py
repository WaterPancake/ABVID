"""Shared numeric functions from experiments/h1/freeze.py."""
from collections import Counter
import hashlib
import numpy as np

def stream_seed(seed, stream):
    return int(hashlib.sha256(f'h1_common_budget_v1.3|{seed}|{stream}'.encode()).hexdigest()[:16], 16)


def select(rows, n, seed, stream):
    result = []
    for label in [0, 1]:
        pools = {}
        for g in sorted(set(r['provenance_group_id'] for r in rows)):
            pool = sorted(r['file_id'] for r in rows if r['provenance_group_id'] == g and r['class_id'] == label)
            rng = np.random.default_rng(stream_seed(seed, f'{stream}|{g}|{label}'))
            pools[g] = list(rng.permutation(pool))
        # Deterministic round-robin avoids allowing the largest source date to
        # supply the whole car class. Exhausted small pools simply drop out.
        selected = []
        while len(selected) < n:
            before = len(selected)
            for g in sorted(pools):
                if pools[g] and len(selected) < n:
                    selected.append(str(pools[g].pop()))
            if len(selected) == before:
                raise ValueError('Insufficient support')
        result.extend(selected)
    assert len(set(result)) == 2*n
    return result


def assert_separation(train_ids, test_ids, index):
    if set(train_ids) & set(test_ids):
        raise ValueError('File leakage')
    for field in ('provenance_group_id', 'source_file_sha256', 'duplicate_group_id'):
        if {index[i][field] for i in train_ids} & {index[i][field] for i in test_ids}:
            raise ValueError(f'{field} leakage')
