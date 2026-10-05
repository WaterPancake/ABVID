"""Read-only civilian audio admission steps 1-2; no preprocessing or split assignment.

Run with the existing reproduction venv. Originals are read only; RAR extraction
goes to a fresh work directory. Output directories must also be fresh.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import struct
import subprocess
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'dataset_admission_v1.2.1'
CLASS_MAP = {'car': 0, 'truck': 1}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def base_row(dataset, release, path):
    return dict(file_id=hashlib.sha256(f'{dataset}|{path}'.encode()).hexdigest(),
                dataset_id=dataset, release_id=release, relative_path=path,
                canonical_class=None, class_id=None, raw_label=None,
                metadata_exclusion_reasons=[], metadata_parse_ok=True, label_conflict_errors=[],
                provenance_group_id=None, split_role=None, admitted_for_training=False,
                candidate_group_id=None, candidate_group_basis=None,
                original_media_id=None, recording_session=None, paired_event_id=None,
                source_template_id=None, simulation_run_id=None, source_seed=None,
                unknown_reason={'provenance_group_id': 'Connected provenance audit not complete',
                                'split_role': 'Future split design deferred',
                                'physical_vehicle_id': 'Not established by category annotation'},
                intended_channel_operation='mean(all_channels)',
                raw_audio_modified=False)


def map_class(row, cls):
    if cls in CLASS_MAP:
        row.update(canonical_class=cls, class_id=CLASS_MAP[cls])
    else:
        row['metadata_exclusion_reasons'].append('class_out_of_scope_v1.1')


def parse_idmt(name):
    row = base_row('IDMT', 'IDMT_V1', 'audio/' + name)
    row.update(licence='CC BY-NC-ND 4.0 (provider README)',
               metadata_reference='IDMT_Traffic/readme.md: File naming convention; annotation/import_idmt_traffic_dataset.py: docstring',
               exposure='historical R0 domain inspected; future roles unassigned',
               expected_channels=2)
    parts = Path(name).stem.split('_')
    try:
        if name.endswith('-BG.wav'):
            stamp, site, speed, position, mic, channel = parts
            daytime = road = direction = None
            vehicle = 'BG'
            channel = channel.removesuffix('-BG')
        else:
            stamp, site, speed, position, daytime, road, vd, mic, channel = parts
            vehicle, direction = vd
            if daytime not in {'M', 'A'} or road not in {'D', 'W'} or direction not in {'L', 'R'}:
                raise ValueError('Unknown vehicle-context token')
        datetime.strptime(stamp, '%Y-%m-%d-%H-%M')
        if vehicle not in {'BG', 'B', 'C', 'M', 'T'} or mic not in {'SE', 'ME'} or channel not in {'CH12', 'CH34'}:
            raise ValueError('Unknown label/sensor/channel token')
        pos = int(position)
        if pos < 0 or not re.fullmatch(r'(?:\d+|unknown)Kmh', speed):
            raise ValueError('Invalid position/speed-limit token')
        label = {'BG': 'background', 'B': 'bus', 'C': 'car', 'M': 'motorcycle', 'T': 'truck'}[vehicle]
        row.update(raw_label=vehicle, provider_class=label, site_id=site, calendar_date=stamp[:10],
                   recording_timestamp=stamp, sample_position_center=pos,
                   speed_limit_token=speed, vehicle_speed=None, road_condition_token=road,
                   daytime_token=daytime, direction=direction, device_id=mic, provider_channel_pair=channel,
                   original_channel_ids=[1, 2] if channel == 'CH12' else [3, 4],
                   candidate_recording_id=f'IDMT|{stamp}|{site}|{speed}',
                   paired_event_id=f'IDMT|{stamp}|{site}|{speed}|{pos}',
                   candidate_group_id=f'IDMT|{site}|{stamp[:10]}',
                   candidate_group_basis='provider site/calendar-date tokens; not independent-session certification')
        map_class(row, label)
        if mic != 'SE':
            row['metadata_exclusion_reasons'].append('sensor_not_selected_ME')
        elif label in CLASS_MAP and channel != 'CH34':
            row['metadata_exclusion_reasons'].append('unexpected_core_sensor_channel_combination')
    except (ValueError, TypeError):
        row['metadata_parse_ok'] = False
        row['metadata_exclusion_reasons'].append('unparsed_or_unknown_idmt_metadata')
    return row


def parse_melaudis(member):
    row = base_row('MELAUDIS', 'MELAUDIS_V1', member)
    row.update(archive_member=member, archive='MELAUDIS_Vehicles.rar',
               licence='CC BY 4.0 (release metadata)',
               metadata_reference='MELAUDIS archive filename tokens; DATASET_PROTOCOL.md sections 4 and 8',
               exposure='development_only; historical P01 feature-domain exposure; exact session resolution pending')
    try:
        date, stamp_site, state, label, *tokens = PurePosixPath(member).stem.split('_')
        datetime.strptime(date, '%Y-%m-%d')
        match = re.fullmatch(r'(\d+)-(\d+)-(\d+(?:\.\d+)?)-(.+)', stamp_site)
        if not match or state not in {'FF', 'TJN', 'TJF'}:
            raise ValueError('Unknown timestamp/state')
        hour, minute, second, site = match.groups()
        valid_time = int(hour) < 24 and int(minute) < 60 and float(second) < 60
        layout = tokens[-1]
        if layout not in {'mono', 'stereo'}:
            raise ValueError('Unknown channel token')
        directions = [t for t in tokens if t in {'LR', 'RL', 'Idling'}]
        if len(directions) != 1:
            raise ValueError('Ambiguous direction')
        row.update(raw_label=label, traffic_state=state, calendar_date=date, site_id=site,
                   timestamp_token=stamp_site, direction=directions[0],
                   lane_tokens=[t for t in tokens if re.fullmatch(r'L[123]+|[123]Lanes?', t)],
                   remaining_tokens=tokens, expected_channels=1 if layout == 'mono' else 2,
                   provider_channel_layout=layout,
                   candidate_group_id=f'MELAUDIS|{site}|{date}',
                   candidate_group_basis='unresolved provider location token/calendar-date; aliases not merged',
                   event_id=f'MELAUDIS|{member}')
        label_match = re.fullmatch(r'(\d+)V-(.+)', label)
        row['multiplicity'] = int(label_match[1]) if label_match else None
        cls = {'1V-Car': 'car', '1V-Truck': 'truck'}.get(label)
        map_class(row, cls)
        if state != 'FF':
            row['metadata_exclusion_reasons'].append('traffic_state_not_FF')
        if row['multiplicity'] != 1:
            row['metadata_exclusion_reasons'].append('not_exact_single_vehicle')
        if cls in CLASS_MAP and directions[0] not in {'LR', 'RL'} and state == 'FF':
            row['metadata_exclusion_reasons'].append('unexpected_free_flow_direction')
        if not valid_time:
            # Preserve independently readable labels/layout; do not guess a repaired time.
            row['metadata_parse_ok'] = False
            row['metadata_exclusion_reasons'].append('invalid_timestamp_token')
    except (ValueError, IndexError):
        row['metadata_parse_ok'] = False
        row['metadata_exclusion_reasons'].append('unparsed_or_unknown_melaudis_metadata')
    return row


def parse_synthetic(path):
    family, folder = Path(path).parts[-3:-1]
    row = base_row('AI4TEN_' + family, 'P02_RELEASE', path)
    row.update(raw_label=folder, generator_family_release_label=family, generator_backend_verified=False,
               expected_channels=1, licence='CC BY 4.0 (release README; source lineage unresolved)',
               metadata_reference='AI4TEN/README.md: data; exact synthetic folder/filename label',
               exposure='supplied synthetic candidate; future roles unassigned')
    row['unknown_reason']['synthetic_lineage'] = 'No complete source/template/run sidecars supplied; suffix is an index, not a seed'
    prefix = {'audioldm': 'ALDM', 'pyroadacoustics': 'PYROAD'}.get(family)
    m = re.fullmatch(r'(ALDM|PYROAD)_(car|truck|motorcycle)_(\d+)\.wav', Path(path).name)
    if not m or m[1] != prefix or m[2] != folder:
        row['metadata_parse_ok'] = False
        row['metadata_exclusion_reasons'].append('synthetic_folder_filename_label_mismatch')
    else:
        row['release_file_index'] = int(m[3])
    map_class(row, folder)
    return row


def riff_check(data):
    """Check declared RIFF boundaries, including truncation libsndfile tolerates."""
    errors, flags = [], []
    if len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        return {'errors': ['unsupported_or_invalid_riff_wave'], 'flags': [], 'data_bytes': None}
    declared = struct.unpack_from('<I', data, 4)[0] + 8
    if declared > len(data):
        errors.append('riff_declared_length_exceeds_file')
    elif declared < len(data):
        flags.append('bytes_after_declared_riff')
    pos, fmt, payloads = 12, None, []
    while pos < min(declared, len(data)):
        if pos + 8 > min(declared, len(data)):
            errors.append('incomplete_riff_chunk_header')
            break
        tag, size = struct.unpack_from('<4sI', data, pos)
        start, end = pos + 8, pos + 8 + size
        if end > min(declared, len(data)):
            errors.append('riff_chunk_exceeds_container')
            break
        if tag == b'fmt ':
            if size < 16:
                errors.append('invalid_fmt_chunk')
            else:
                fmt = struct.unpack_from('<HHIIHH', data, start)
        if tag == b'data':
            payloads.append(size)
        pos = end + (size % 2)
    if fmt is None or len(payloads) != 1:
        errors.append('missing_or_multiple_fmt_data_payload')
    result = {'errors': errors, 'flags': flags, 'riff_declared_bytes': declared,
              'data_bytes': sum(payloads) if payloads else None}
    if fmt:
        format_code, channels, rate, byte_rate, align, bits = fmt
        if format_code not in {1, 3, 65534}:
            errors.append('unsupported_wave_encoding')
        if not channels or not rate or not align or not bits or byte_rate != rate * align or align != channels * ((bits + 7) // 8):
            errors.append('inconsistent_fmt_layout')
        if payloads and align:
            if sum(payloads) % align:
                errors.append('data_size_not_whole_frames')
            result['declared_payload_frames'] = sum(payloads) // align
        result.update(format_code=format_code, bits_per_sample=bits, fmt_channels=channels, fmt_rate=rate)
    return result


def audit_file(job):
    row, path = job
    row = dict(row)
    row.update(integrity_errors=[], channel_errors=[], flags=[], decoded_audio_sha256=None,
               decode_complete=False, integrity_pass=False)
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        row.update(source_file_sha256=hashlib.sha256(data).hexdigest(), source_file_bytes=len(data))
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            row['integrity_errors'].append('source_changed_during_read')
        container = riff_check(data)
        row['container'] = container
        row['integrity_errors'].extend(container['errors'])
        row['flags'].extend(container['flags'])
        with sf.SoundFile(io.BytesIO(data)) as f:
            rate, channels, frames = f.samplerate, f.channels, f.frames
            row.update(native_sample_rate_hz=rate, native_channels=channels,
                       header_frames=frames, format=f.format, subtype=f.subtype)
            x = f.read(dtype='float64', always_2d=True)
            remaining = f.read(1, dtype='float64', always_2d=True)
        row.update(decoded_frames=len(x), duration_s=len(x) / rate,
                   decode_complete=len(x) == frames and len(remaining) == 0)
        if not row['decode_complete'] or len(x) != container.get('declared_payload_frames'):
            row['integrity_errors'].append('incomplete_or_inconsistent_frame_count')
        if rate != container.get('fmt_rate') or channels != container.get('fmt_channels'):
            row['integrity_errors'].append('decoder_container_format_disagreement')
        header = f'abvid-decoded-f64le-v1|{rate}|{channels}|{len(x)}\n'.encode()
        h = hashlib.sha256(header)
        h.update(np.ascontiguousarray(x, dtype='<f8').tobytes())
        row['decoded_audio_sha256'] = h.hexdigest()
        bad = int(np.count_nonzero(~np.isfinite(x)))
        row['nonfinite_samples'] = bad
        if bad:
            row['integrity_errors'].append('nonfinite_samples')
        elif not len(x):
            row['integrity_errors'].append('empty_waveform')
        else:
            zero_channels = np.flatnonzero(~np.any(x != 0, axis=0)).tolist()
            row.update(all_zero=not bool(np.any(x != 0)), zero_channel_indices=zero_channels,
                       peak_abs=float(np.max(np.abs(x))),
                       near_full_scale_samples=int(np.count_nonzero(np.abs(x) >= 0.999)))
            if row['all_zero']:
                row['integrity_errors'].append('all_zero_waveform')
            if not np.any(np.mean(x, axis=1) != 0):
                row['integrity_errors'].append('all_zero_mono_downmix')
            if zero_channels:
                row['flags'].append('one_or_more_zero_channels')
            if row['near_full_scale_samples']:
                row['flags'].append('near_full_scale_not_proof_of_clipping')
        if len(x) < 2 * rate:
            row['integrity_errors'].append('duration_below_two_seconds')
        expected = row.get('expected_channels')
        if row['dataset_id'] == 'MELAUDIS' and row.get('provider_channel_layout') == 'mono' and channels == 2:
            row['channels_exactly_equal'] = bool(np.array_equal(x[:, 0], x[:, 1]))
            if row['channels_exactly_equal']:
                row['flags'].append('mono_label_with_identical_stereo_copies')
            else:
                row['channel_errors'].append('mono_label_with_distinct_channels')
        elif expected is not None and channels != expected:
            row['channel_errors'].append('decoded_channels_disagree_with_provider_layout')
        row['integrity_pass'] = not row['integrity_errors']
    except (OSError, ValueError, RuntimeError, struct.error) as exc:
        row['integrity_errors'].append('read_or_decode_failure')
        row['decode_error'] = f'{type(exc).__name__}: {exc}'
    finish_status(row)
    return row


def finish_status(row):
    row['integrity_and_class_pass'] = not (row['metadata_exclusion_reasons'] or row['integrity_errors'] or row['channel_errors'] or row.get('label_conflict_errors'))
    row['status'] = 'integrity_and_class_pass' if row['integrity_and_class_pass'] else 'excluded_or_quarantined_steps_1_2'


def quarantine_label_conflicts(rows):
    """Exact identical audio with incompatible provider annotations is ambiguous."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get('decoded_audio_sha256'):
            buckets[row['decoded_audio_sha256']].append(row)
    conflicts = []
    mel_labels = {'1V-Car':'car', '1V-Truck':'truck', '1V-MC':'motorcycle',
                  '1V-Bus':'bus', '1V-Tram':'tram', '1V-Bic':'bicycle', 'NoV':'no_vehicle'}
    for key, members in sorted(buckets.items()):
        labels = set()
        for row in members:
            label = row.get('provider_class', row['raw_label'])
            if row['dataset_id'] == 'MELAUDIS':
                label = mel_labels.get(label, label)
            if label is not None:
                labels.add(label)
        if len(labels) > 1:
            conflicts.append(dict(decoded_audio_sha256=key, annotations=sorted(labels), file_ids=[r['file_id'] for r in members]))
            for row in members:
                row['label_conflict_errors'].append('exact_audio_conflicting_provider_annotations')
                finish_status(row)
    return conflicts


def equal_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row['file_id'])
    return [{'hash': key_hash, 'file_ids': members} for key_hash, members in sorted(groups.items()) if len(members) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out, work = args.output.resolve(), args.work_dir.resolve()
    if out.exists() or work.exists():
        parser.error('Use fresh output and work directories; existing artifacts are never overwritten.')
    if args.workers < 1 or args.workers > 8:
        parser.error('workers must be 1..8')
    # All destinations must remain outside the immutable source tree.
    for destination in [out, work]:
        if destination.is_relative_to(ROOT / 'dataset'):
            parser.error('Output/work must not be inside dataset/')
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    out.mkdir(parents=True); work.mkdir(parents=True)
    shutil.copyfile(__file__, out / 'executed_audit.py')
    shutil.copyfile(ROOT / 'experiments/DATASET_PROTOCOL.md', out / 'protocol_at_run.md')
    releases = []
    archive = ROOT / 'dataset/MELAUDIS/MELAUDIS_Vehicles.rar'
    print('Checking vehicle archive and inventory', flush=True)
    archive_md5 = digest(archive, 'md5')
    if archive_md5 != '0bf2cf3ad13436ec8bf060b378508252':
        raise RuntimeError('MELAUDIS vehicle archive fails provider MD5')
    archive_sha = digest(archive)
    releases.append(dict(path=relative(archive), md5=archive_md5, sha256=archive_sha, provider_md5_match=True))
    listing = subprocess.check_output(['/usr/bin/tar', '-tf', str(archive)], text=True).splitlines()
    members = sorted(n for n in listing if n.lower().endswith('.wav'))
    if len(members) != len(set(members)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in listing):
        raise RuntimeError('Duplicate or unsafe archive members')
    (out / 'melaudis_vehicle_members.txt').write_text('\n'.join(members) + '\n')
    idmt = ROOT / 'dataset/IDMT_Traffic'
    listed = (idmt / 'annotation/idmt_traffic_all.txt').read_text().splitlines()
    actual = sorted(p.name for p in (idmt / 'audio').glob('*.wav'))
    if len(listed) != len(set(listed)) or sorted(listed) != actual:
        raise RuntimeError('IDMT supplied inventory differs from actual WAVs')
    provider_train = set((idmt / 'annotation/eusipco_2021_train.txt').read_text().splitlines())
    provider_test = set((idmt / 'annotation/eusipco_2021_test.txt').read_text().splitlines())
    jobs = []
    for name in actual:
        row = parse_idmt(name)
        row['provider_eusipco_roles'] = [role for role, names in [('train', provider_train), ('test', provider_test)] if name in names]
        jobs.append((row, idmt / 'audio' / name))
    for name in members:
        row = parse_melaudis(name)
        row['archive_sha256'] = archive_sha
        jobs.append((row, work / 'melaudis' / name))
    ai4ten = ROOT / 'dataset/AI4TEN'
    for path in sorted((ai4ten / 'data/synthetic').rglob('*.wav')):
        jobs.append((parse_synthetic(str(path.relative_to(ai4ten))), path))
    write_jsonl(out / 'inventory.jsonl', (row for row, _ in jobs))
    config = dict(protocol=PROTOCOL, steps=[1, 2], workers=args.workers,
                  min_duration_seconds=2, decoded_hash='abvid-decoded-f64le-v1|rate|channels|frames + newline + interleaved little-endian float64',
                  waveform_modifications=False, model_execution=False, split_assignment=False,
                  silence_rule='exactly zero, including arithmetic mono downmix', clipping_rule='flag only at abs(sample)>=0.999',
                  background_archive_audited=False, candidate_classes=CLASS_MAP,
                  duplicate_policy='quarantine exact-audio annotation conflicts; same-label deduplication and near-duplicate audit deferred',
                  mono_label_rule='two identical decoded channels accepted as redundant mono; distinct channels quarantined',
                  invalid_timestamp_rule='retain independently parsed labels/layout and original token; quarantine, never normalize guessed time',
                  random_seed=None, randomness='none')
    write_json(out / 'config.json', config)
    print(f'Inventory frozen: {len(jobs)} WAVs; extracting vehicle archive into separate work directory', flush=True)
    raw = work / 'melaudis'; raw.mkdir()
    with (out / 'extraction.log').open('w') as log:
        subprocess.run(['/usr/bin/tar', '-xf', str(archive), '-C', str(raw), '-T', str(out / 'melaudis_vehicle_members.txt'), '--no-same-owner', '--no-same-permissions'], check=True, stdout=log, stderr=subprocess.STDOUT)
    for member in members:
        path = raw / member
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(raw):
            raise RuntimeError(f'Invalid extracted file: {member}')
    results = []
    with (out / 'decoded_files.jsonl').open('w') as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(audit_file, jobs), 1):
            results.append(row)
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            if n % 2000 == 0:
                f.flush()
                print(f'{n}/{len(jobs)} fully decoded; elapsed {time.perf_counter()-clock:.1f}s', flush=True)
    conflicts = quarantine_label_conflicts(results)
    write_json(out / 'label_conflict_groups.json', conflicts)
    write_jsonl(out / 'all_files.jsonl', results)
    candidates = [r for r in results if r['integrity_and_class_pass']]
    write_jsonl(out / 'car_truck_candidates.jsonl', candidates)
    write_jsonl(out / 'exclusions.jsonl', (r for r in results if not r['integrity_and_class_pass']))
    equality = {key: equal_groups(results, key) for key in ['source_file_sha256', 'decoded_audio_sha256']}
    write_json(out / 'exact_equality_groups.json', equality)
    pairs = defaultdict(list)
    for row in results:
        if row.get('paired_event_id'):
            pairs[row['paired_event_id']].append(row)
    pair_rows = [dict(paired_event_id=key, file_ids=[r['file_id'] for r in rows],
                      sensors=[r['device_id'] for r in rows], labels=sorted({r['raw_label'] for r in rows}),
                      labels_agree=len({r['raw_label'] for r in rows}) == 1)
                 for key, rows in sorted(pairs.items())]
    write_jsonl(out / 'idmt_pair_links.jsonl', pair_rows)
    extra = next(r for r in results if r['dataset_id'] == 'AI4TEN_audioldm' and Path(r['relative_path']).name == 'ALDM_car_0201.wav')
    finding = dict(file=extra, byte_identical_other_files=[r['file_id'] for r in results if r['file_id'] != extra['file_id'] and r.get('source_file_sha256') == extra['source_file_sha256']],
                   decoded_identical_other_files=[r['file_id'] for r in results if r['file_id'] != extra['file_id'] and r.get('decoded_audio_sha256') == extra['decoded_audio_sha256']],
                   generation_reason='unknown', action='retain original; no README-count-driven deletion',
                   limitation='Exact equality cannot establish source independence or near-duplicate absence')
    write_json(out / 'audioldm_extra_car.json', finding)
    datasets = {}
    for name in sorted({r['dataset_id'] for r in results}):
        rows = [r for r in results if r['dataset_id'] == name]
        passed = [r for r in rows if r['integrity_and_class_pass']]
        datasets[name] = dict(inventory=len(rows), decoded_complete=sum(r['decode_complete'] for r in rows),
             integrity_errors=sum(bool(r['integrity_errors']) for r in rows), channel_errors=sum(bool(r['channel_errors']) for r in rows),
             metadata_parse_errors=sum(not r['metadata_parse_ok'] for r in rows),
             metadata_candidates=sum(not r['metadata_exclusion_reasons'] for r in rows),
             integrity_and_class_pass=len(passed), class_counts=dict(Counter(r['canonical_class'] for r in passed)),
             label_conflict_files=sum(bool(r['label_conflict_errors']) for r in rows),
             exclusion_reasons=dict(Counter(reason for r in rows for reason in r['metadata_exclusion_reasons'] + r['integrity_errors'] + r['channel_errors'] + r['label_conflict_errors'])),
             flags=dict(Counter(flag for r in rows for flag in r['flags'])),
             formats=[dict(sample_rate_hz=k[0], channels=k[1], frames=k[2], subtype=k[3], count=v) for k,v in sorted(Counter((r.get('native_sample_rate_hz',0),r.get('native_channels',0),r.get('decoded_frames',0),r.get('subtype','unknown')) for r in rows).items())],
             candidate_group_class_counts={key:dict(Counter(r['canonical_class'] for r in passed if r['candidate_group_id']==key)) for key in sorted({r['candidate_group_id'] for r in passed if r['candidate_group_id']})})
    docs = ['experiments/DATASET_PROTOCOL.md','experiments/AMENDMENT_v1.1.md',
            'reports/proposed_experiments.md','reports/published_reproduction_status.md',
            'dataset/IDMT_Traffic/readme.md','dataset/IDMT_Traffic/annotation/import_idmt_traffic_dataset.py',
            'dataset/IDMT_Traffic/annotation/idmt_traffic_all.txt','dataset/AI4TEN/README.md']
    summary = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
         elapsed_seconds=time.perf_counter()-clock, scope='Admission steps 1-2 only', protocol=PROTOCOL,
         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         script_sha256=digest(__file__), config_sha256=digest(out/'config.json'),
         input_documents={p:digest(ROOT/p) for p in docs}, archive_checks=releases,
         versions=dict(python=platform.python_version(),numpy=np.__version__,soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__,platform=platform.platform(),tar=subprocess.check_output(['/usr/bin/tar','--version'],text=True).strip()),
         datasets=datasets, all_files=len(results), candidates=len(candidates),
         all_candidates_audited=True, full_admission_complete=False, model_execution=False, split_assignment=False,
         exact_equality_group_counts={k:len(v) for k,v in equality.items()},
         conflicting_annotation_groups=len(conflicts),
         idmt_pair_key_count=len(pair_rows), idmt_multifile_pair_keys=sum(len(p['file_ids'])>1 for p in pair_rows),
         idmt_pair_label_conflicts=sum(not p['labels_agree'] for p in pair_rows),
         pending=['connected original-session/location/overlap audit','synthetic source/run lineage','future split design and final admission lock'],
         limitations=['Provider labels are not independent human content validation','No near-duplicate fingerprinting','Background archive not decoded','Expanded IDMT/AI4TEN original ZIP identity not verified','Exact hash equality does not prove independence','No preprocessing/model inference/training or assigned splits'])
    summary['artifacts_sha256'] = {p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ['all_files','candidates','elapsed_seconds','exact_equality_group_counts','idmt_pair_label_conflicts']},indent=2),flush=True)


if __name__ == '__main__':
    main()
