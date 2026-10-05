"""Admission v1.3: conservative grouping and bounded waveform-reuse search.

No classifier, embeddings, label corrections, or writes to source audio. The
near-duplicate search is explicitly incomplete for arbitrary shifted excerpts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from math import gcd
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.spatial import cKDTree

from audit_integrity import ROOT, digest, write_json, write_jsonl

INTEGRITY = ROOT / 'experiments/admission/results/integrity_20261002_v1_2_1'
EXTRACTED = ROOT / 'experiments/admission/work/integrity_20261002_v1_2_1/melaudis'
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


def source_path(row):
    if row['dataset_id'] == 'IDMT':
        return ROOT / 'dataset/IDMT_Traffic' / row['relative_path']
    if row['dataset_id'] == 'MELAUDIS':
        return EXTRACTED / row['relative_path']
    return ROOT / 'dataset/AI4TEN' / row['relative_path']


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


def fingerprint(row):
    path = source_path(row)
    if digest(path) != row['source_file_sha256']:
        raise ValueError(f'Source changed: {path}')
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    y = normalized_center(x, sr)
    h = None if y is None else hashlib.sha256(
        np.rint(y * RULES['exact_quantization']).astype('<i4').tobytes()).hexdigest()
    return row['file_id'], y, h


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(out / 'rules_before_audit.json', RULES)
    shutil.copy2(__file__, out / 'executed_audit.py')
    summary12 = json.loads((INTEGRITY / 'summary.json').read_text())
    assert digest(INTEGRITY / 'all_files.jsonl') == summary12['artifacts_sha256']['all_files.jsonl']
    rows = [json.loads(s) for s in (INTEGRITY / 'all_files.jsonl').open()]
    index = {r['file_id']: r for r in rows}
    group = Union([r['file_id'] for r in rows])
    duplicate = Union([r['file_id'] for r in rows])
    for keyfn in (parent_proxy, lambda r: r.get('paired_event_id'),
                  lambda r: r.get('decoded_audio_sha256')):
        seen = {}
        for r in rows:
            key = keyfn(r)
            if key is not None:
                if key in seen:
                    group.join(r['file_id'], seen[key])
                seen[key] = r['file_id']
    exact = defaultdict(list)
    for r in rows:
        exact[r['decoded_audio_sha256']].append(r['file_id'])
    for members in exact.values():
        for fid in members[1:]:
            duplicate.join(members[0], fid)
    spans = overlapping_components(rows)
    for c in spans:
        for fid in c['members'][1:]:
            group.join(c['members'][0], fid)
    write_json(out / 'overlap_components.json', spans)

    valid_rows = [r for r in rows if r['integrity_pass'] and r['duration_s'] >= 2]
    waves, fids, hashes = [], [], []
    invalid_fingerprints = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k, (fid, wave, h) in enumerate(pool.map(fingerprint, valid_rows), 1):
            if wave is None:
                invalid_fingerprints.append(fid)
            else:
                fids.append(fid); hashes.append(h); waves.append(wave)
            if k % 5000 == 0:
                print(f'fingerprinted {k}/{len(valid_rows)}', flush=True)
    waves = np.stack(waves)
    datasets = [index[fid]['dataset_id'] for fid in fids]
    print('Searching aligned gain/polarity/resampling duplicates', flush=True)
    matches, comparisons = verified_pairs(waves, datasets, hashes)
    del waves
    edges, rejected_candidates, verification_waves = [], [], {}
    for i, j, corr in matches:
        a, b = fids[i], fids[j]
        for fid in (a, b):
            if fid not in verification_waves:
                x, sr = sf.read(source_path(index[fid]), dtype='float64', always_2d=True)
                verification_waves[fid] = normalized_center(x, sr, rate=8000)
        full_corr = float(min(1., abs(verification_waves[a] @ verification_waves[b])))
        if full_corr < RULES['correlation_threshold']:
            rejected_candidates.append(dict(a=a,b=b,abs_correlation_8kHz=full_corr))
            continue
        group.join(a, b); duplicate.join(a, b)
        edges.append(dict(a=a, b=b, abs_correlation=corr,
                          abs_correlation_8kHz=full_corr,
                          method='aligned_2s_2kHz_retrieval_8kHz_waveform_verification',
                          cross_dataset=index[a]['dataset_id'] != index[b]['dataset_id']))
    write_json(out / 'waveform_reuse_edges.json', edges)
    write_json(out / 'rejected_candidates_8kHz.json', rejected_candidates)
    write_jsonl(out / 'fingerprints.jsonl', [dict(file_id=fid, normalized_center_sha256=h)
                                            for fid, h in zip(fids, hashes)])

    components, group_members = defaultdict(list), defaultdict(list)
    for r in rows:
        components[duplicate.find(r['file_id'])].append(r)
        group_members[group.find(r['file_id'])].append(r)
    exclusions, duplicate_report, survivors = {}, [], set()
    for key, members in components.items():
        eligible = [r for r in members if r['integrity_and_class_pass']]
        # Only actual equivalent-waveform edges create label conflicts. Merely
        # overlapping intervals may contain distinct passing vehicles.
        labels = {(r['dataset_id'] == 'MELAUDIS', r.get('raw_label')) for r in members}
        semantic = {r.get('provider_class', r.get('raw_label')) for r in members}
        semantic = {str(s).removeprefix('1V-').lower() for s in semantic}
        ds = {r['dataset_id'] for r in members}
        reason = None
        if len(ds) > 1:
            reason = 'cross_dataset_waveform_reuse'
        elif len(semantic) > 1:
            reason = 'equivalent_waveform_conflicting_annotations'
        if reason:
            for r in eligible:
                exclusions[r['file_id']] = reason
        elif eligible:
            ordered = sorted(eligible, key=lambda r: (r['relative_path'], r['file_id']))
            survivors.add(ordered[0]['file_id'])
            for r in ordered[1:]:
                exclusions[r['file_id']] = 'same_label_waveform_duplicate'
        if len(members) > 1:
            duplicate_report.append(dict(duplicate_group_id=key,
                members=[r['file_id'] for r in members], datasets=sorted(ds),
                annotations=sorted(semantic), disposition=reason or 'retain_one_eligible_copy'))
    for fid in invalid_fingerprints:
        if index[fid]['integrity_and_class_pass']:
            survivors.discard(fid)
            exclusions[fid] = 'zero_or_invalid_common_band_fingerprint'

    admitted = []
    provenance = []
    for key, members in sorted(group_members.items()):
        proxies = sorted(set(parent_proxy(r) for r in members))
        gid = 'connected_' + hashlib.sha256('\n'.join(proxies).encode()).hexdigest()[:16]
        provenance.append(dict(provenance_group_id=gid, conservative_parent_proxies=proxies,
                               all_files=len(members), admitted_files=sum(r['file_id'] in survivors for r in members),
                               original_session_ids_verified=False))
        for r in members:
            if r['file_id'] not in survivors:
                continue
            r = dict(r)
            iv = interval(r)
            real = r['dataset_id'] in {'IDMT', 'MELAUDIS'}
            r.update(provenance_group_id=gid,
                     grouping_basis='connected_conservative_proxy; original sessions unknown',
                     conservative_parent_proxy=parent_proxy(r),
                     source_start_s=iv[1] if iv else None, source_end_s=iv[2] if iv else None,
                     source_time_axis=iv[0] if iv else None,
                     original_media_id=None, recording_session=None,
                     admitted_for_h1=True,
                     admitted_for_training=r['dataset_id'] != 'MELAUDIS',
                     split_role='target_development_only' if r['dataset_id'] == 'MELAUDIS'
                         else 'source_logo_pool' if r['dataset_id'] == 'IDMT' else 'training_only_bank',
                     status='qualified_h1_admission',
                     source_path=str(source_path(r).relative_to(ROOT)),
                     unknown_reason={**r['unknown_reason'],
                         'provenance_group_id':'Resolved conservative group; not a verified original session',
                         'split_role':'H1 scope assigned; exact training selections frozen separately'},
                     independent_physical_source_established=False,
                     duplicate_group_id=duplicate.find(r['file_id']),
                     reusable_synthetic_holdout=False if not real else None)
            if r['dataset_id'] == 'MELAUDIS':
                r['device_annotation'] = {'mono':'iPhone6', 'stereo':'iPhone12'}[r['provider_channel_layout']]
                r['device_annotation_source'] = 'MELAUDIS descriptor, Data Features; not measured hardware verification'
            admitted.append(r)
    admitted.sort(key=lambda r: r['file_id'])
    write_jsonl(out / 'admitted.jsonl', admitted)
    write_json(out / 'groups.json', provenance)
    write_json(out / 'duplicate_components.json', duplicate_report)
    write_jsonl(out / 'additional_exclusions.jsonl', [dict(file_id=fid, reason=reason,
                source_file_sha256=index[fid]['source_file_sha256']) for fid, reason in sorted(exclusions.items())])

    metadata = {}
    for name, url in [('melaudis','https://api.figshare.com/v2/articles/27115870'),
                      ('p02','https://zenodo.org/api/records/21264250')]:
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                doc = json.load(f)
            write_json(out / (name + '_release_metadata.json'), doc)
            files = doc['files']
            metadata[name] = [f.get('name', f.get('key')) for f in files]
        except Exception as exc:
            metadata[name] = {'error': str(exc)}
    lineage_root = ROOT / 'experiments/reproduction/releases/P02/code_and_results'
    lineage = [dict(path=str(p.relative_to(ROOT)), sha256=digest(p))
               for p in sorted(lineage_root.rglob('*')) if p.is_file() and
               p.suffix.lower() in {'.py','.json','.md','.txt','.csv','.yaml','.yml','.log','.xlsx'}]
    write_json(out / 'synthetic_metadata_inventory.json', lineage)
    counts = {}
    for ds in sorted(set(r['dataset_id'] for r in admitted)):
        subset = [r for r in admitted if r['dataset_id'] == ds]
        counts[ds] = dict(classes=dict(Counter(r['canonical_class'] for r in subset)),
                         groups={g: dict(Counter(r['canonical_class'] for r in subset if r['provenance_group_id']==g))
                                 for g in sorted(set(r['provenance_group_id'] for r in subset))})
    summary = dict(protocol=PROTOCOL, completed_utc=datetime.now(timezone.utc).isoformat(),
                   integrity_input_sha256=digest(INTEGRITY / 'all_files.jsonl'),
                   source_files= len(rows), fingerprinted=len(fids), comparison_pairs=comparisons,
                   high_correlation_edges=len(edges), cross_dataset_edges=sum(e['cross_dataset'] for e in edges),
                   overlap_components=len(spans), additional_exclusions=dict(Counter(exclusions.values())),
                   admitted_files=len(admitted), datasets=counts, release_file_lists=metadata,
                   synthetic_lineage='unknown; entire bank training-only; backend not certified',
                   independence='conservative known-link grouping, not independently verified original sessions',
                   runtime_s=time.perf_counter()-started, python=platform.python_version(),
                   git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   artifacts_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    write_json(out / 'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ['admitted_files','datasets','additional_exclusions','runtime_s']},indent=2),flush=True)


if __name__ == '__main__':
    main()
