"""Verify audit artifacts and eligibility decisions without decoding audio again."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    p = args.result_dir
    summary = json.loads((p/'summary.json').read_text())
    for name, digest in summary['artifacts_sha256'].items():
        assert sha(p/name) == digest, name
    assert sha(p/'executed_audit.py') == summary['script_sha256']
    assert sha(p/'protocol_at_run.md') == summary['input_documents']['experiments/DATASET_PROTOCOL.md']
    inventory = rows(p/'inventory.jsonl')
    audited = rows(p/'all_files.jsonl')
    candidates = rows(p/'car_truck_candidates.jsonl')
    excluded = rows(p/'exclusions.jsonl')
    mapping = {r['file_id']:r for r in audited}
    assert len(mapping) == len(audited) == len(inventory) == summary['all_files']
    assert {r['file_id'] for r in inventory} == set(mapping)
    for row in inventory:
        assert all(mapping[row['file_id']][key] == val for key,val in row.items() if key != 'label_conflict_errors'), row['relative_path']
    passed_ids = {r['file_id'] for r in candidates}
    excluded_ids = {r['file_id'] for r in excluded}
    assert not passed_ids & excluded_ids
    assert passed_ids | excluded_ids == set(mapping)
    assert len(candidates) + len(excluded) == len(audited)
    assert len(candidates) == summary['candidates']
    for row in candidates + excluded:
        assert row == mapping[row['file_id']]
    for row in audited:
        assert row['split_role'] is None and row['provenance_group_id'] is None
        assert row['admitted_for_training'] is False and row['raw_audio_modified'] is False
        passed = not (row['metadata_exclusion_reasons'] or row['integrity_errors'] or row['channel_errors'] or row.get('label_conflict_errors'))
        assert passed == row['integrity_and_class_pass'] == (row['file_id'] in passed_ids)
        if not passed:
            continue
        assert row['decode_complete'] and row['nonfinite_samples'] == 0 and row['duration_s'] >= 2
        assert len(row['source_file_sha256']) == len(row['decoded_audio_sha256']) == 64
        assert {'car':0,'truck':1}[row['canonical_class']] == row['class_id']
        if row['dataset_id'] == 'IDMT':
            assert row['device_id'] == 'SE' and row['provider_channel_pair'] == 'CH34'
            assert row['native_channels'] == 2 and row['original_channel_ids'] == [3,4]
            assert row['raw_label'] in {'C','T'}
        elif row['dataset_id'] == 'MELAUDIS':
            assert row['traffic_state'] == 'FF' and row['raw_label'] in {'1V-Car','1V-Truck'}
            assert row['direction'] in {'LR','RL'} and row['multiplicity'] == 1
            if row.get('provider_channel_layout') == 'mono' and row['native_channels'] == 2:
                assert row['channels_exactly_equal']
        else:
            assert row['native_channels'] == 1 and row['source_seed'] is None
            assert row['raw_label'] == row['canonical_class']
    equality = json.loads((p/'exact_equality_groups.json').read_text())
    for key, groups in equality.items():
        counts = Counter(r[key] for r in audited if r.get(key))
        assert len(groups) == sum(n > 1 for n in counts.values())
        for group in groups:
            assert len(group['file_ids']) == counts[group['hash']]
            assert all(mapping[i][key] == group['hash'] for i in group['file_ids'])
    conflict_file = p/'label_conflict_groups.json'
    if conflict_file.exists():
        conflicts = json.loads(conflict_file.read_text())
        conflicted_ids = {i for g in conflicts for i in g['file_ids']}
        assert not conflicted_ids & passed_ids
        assert conflicted_ids == {r['file_id'] for r in audited if r['label_conflict_errors']}
    pair_rows = rows(p/'idmt_pair_links.jsonl')
    paired_ids = []
    for pair in pair_rows:
        members = [mapping[i] for i in pair['file_ids']]
        assert all(r['paired_event_id'] == pair['paired_event_id'] for r in members)
        assert pair['labels_agree'] == (len({r['raw_label'] for r in members}) == 1)
        paired_ids.extend(pair['file_ids'])
    assert len(paired_ids) == len(set(paired_ids)) == summary['datasets']['IDMT']['inventory']
    assert set(paired_ids) == {r['file_id'] for r in audited if r['dataset_id']=='IDMT'}
    counts = {name: dict(Counter(r['canonical_class'] for r in candidates if r['dataset_id']==name)) for name in summary['datasets']}
    assert all(counts[name] == data['class_counts'] for name,data in summary['datasets'].items())
    print(json.dumps(dict(status='verified', all_files=len(audited), candidates=len(candidates),
                         excluded=len(excluded), class_counts=counts,
                         artifact_hashes=len(summary['artifacts_sha256']), no_split_or_training_admission=True), indent=2))


if __name__ == '__main__':
    main()
