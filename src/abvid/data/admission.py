"""Original admission metadata rules; unknown session identity stays unknown."""
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
import io
import soundfile as sf
import hashlib
import re
import struct
import numpy as np
CLASS_MAP = {"car": 0, "truck": 1}
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
