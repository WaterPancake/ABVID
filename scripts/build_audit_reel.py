#!/usr/bin/env python3
"""Build a local audiovisual review reel from a frozen native-real manifest.

Restore approved originals through the collection downloader, verifying recorded
raw hashes. Select the union of benchmark windows within each reviewed interval.
Preserve source channel 0 without gain normalization. Never run model inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import textwrap
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont
import yaml

from vehicle_audio.collection import _download


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(command):
    subprocess.run(command, check=True, capture_output=True)


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)
    ]))


def timestamp(seconds):
    return f'{int(seconds // 60):02d}:{seconds % 60:05.2f}'


def covered_ranges(rows, segments):
    """Union overlapping windows, retaining reviewed condition boundaries."""
    output = []
    assigned = set()
    for segment in segments:
        windows = sorted(
            (r['window_start_seconds'], r['window_end_seconds'], r['sample_id'])
            for r in rows
            if r['window_start_seconds'] >= segment['start_seconds'] - 1e-6
            and r['window_end_seconds'] <= segment['end_seconds'] + 1e-6
            and r['operating_condition'] == segment['operating_condition']
        )
        merged = []
        for start, end, sample_id in windows:
            if sample_id in assigned:
                raise ValueError('Window belongs to overlapping review intervals')
            assigned.add(sample_id)
            if merged and start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        for start, end in merged:
            output.append({**segment, 'start_seconds': start, 'end_seconds': end})
    if assigned != {r['sample_id'] for r in rows}:
        raise ValueError('Some benchmark windows have no matching reviewed interval')
    return output


def panel(path, source, segment, number, video_available):
    image = Image.new('RGBA', (960, 540), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 66), fill='#101a29')
    draw.rectangle((0, 466, 960, 540), fill='#101a29')
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    large = ImageFont.truetype(font_path, 24)
    small = ImageFont.truetype(font_path, 17)
    draw.text((16, 7), f'{number:03d} | {source["vehicle_class"].upper()} | {source["vehicle_model"]}', font=large, fill='white')
    draw.text((16, 38), source['recording_session'], font=small, fill='#b1cbe0')
    start, end = segment['start_seconds'], segment['end_seconds']
    draw.text((16, 475), f'ORIGINAL {timestamp(start)} - {timestamp(end)}  |  {segment["operating_condition"]}  |  channel 0, original gain', font=small, fill='white')
    for i, line in enumerate(textwrap.wrap(segment.get('notes') or 'Human-reviewed benchmark interval', width=110)[:2]):
        draw.text((16, 501 + i * 17), line, font=small, fill='#b1cbe0')
    if not video_available:
        draw.rectangle((0, 66, 960, 466), fill='#182638')
        draw.text((45, 180), 'AUDIO REVIEW', font=large, fill='white')
        draw.text((45, 226), 'No source video available in this reel.', font=large, fill='white')
        draw.text((45, 272), 'Listen to the recorded vehicle audio; see index for provenance.', font=small, fill='#b1cbe0')
    image.save(path)


def restore_original(source, metadata, directory, local_hashes):
    expected = metadata['raw_sha256']
    if expected in local_hashes:
        return local_hashes[expected]
    url = metadata['download_url']
    suffix = Path(urlparse(url).path).suffix
    if suffix.lower() in {'.ogg', '.wav', '.flac'}:
        return None
    path = directory / (source['id'] + suffix)
    if path.exists() and digest(path) == expected:
        return path
    print(f'Restoring original video: {source["id"]}', flush=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    actual = _download(url, temporary, 1_000_000_000)
    if actual != expected:
        raise ValueError(f'Original download hash changed: {actual} != {expected}')
    temporary.replace(path)
    return path


def reusable_clip(previous, previous_dir, source_id, segment):
    """Reuse only exact same reviewed interval from a manifest-matched prior reel."""
    if previous is None:
        return None
    matches = [r for r in previous['clips']
               if r['source_id'] == source_id and r['video_available']
               and r['source_start'] == segment['start_seconds']
               and r['source_end'] == segment['end_seconds']
               and r['condition'] == segment['operating_condition']
               and r.get('notes') == segment.get('notes')]
    if len(matches) != 1:
        return None
    path = (previous_dir / matches[0]['clip_path']).resolve()
    if not path.is_relative_to(previous_dir.resolve()) or not path.is_file():
        return None
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.artifacts/audit_reel_7t5w'))
    parser.add_argument('--download-originals', action='store_true')
    parser.add_argument('--skip-remote-provider', action='append', default=[],
                        help='Use only local originals for a rate-limited provider')
    parser.add_argument('--reuse-reel', type=Path,
                        help='Reuse cached originals and exact video chapters from a previous reel')
    parser.add_argument('--audio-only-source-id', action='append', default=[],
                        help='Keep source audio but intentionally omit its video')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / 'audit_reel.mp4').exists():
        raise FileExistsError('Use a new output directory; completed reels are immutable')
    for directory in ('originals', 'clips', 'panels'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    snapshot = Path('benchmarks/v0.1/native_real_7t5w')
    lock = json.loads((snapshot / 'dataset_lock.json').read_text())
    manifest = snapshot / 'real_manifest.jsonl'
    if digest(manifest) != lock['manifest_sha256']:
        raise ValueError('Frozen manifest hash mismatch')
    previous = None
    if args.reuse_reel:
        previous = json.loads((args.reuse_reel / 'index.json').read_text())
        if previous['dataset_version'] != lock['dataset_version']:
            raise ValueError('Prior reel uses a different manifest')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_id']].append(row)
    catalog = yaml.safe_load((snapshot / 'audio_sources.yaml').read_text())
    sources = {s['id']: s for s in catalog['sources']}
    local_files = list(Path('.').iterdir())
    if args.reuse_reel:
        local_files += list((args.reuse_reel / 'originals').glob('*'))
    local_hashes = {digest(p): p.resolve() for p in local_files
                    if p.is_file() and p.suffix.lower() in {'.webm', '.mp4', '.ogv'}}
    index = []
    availability = []
    reel_time = 0.0
    for source_id in sorted(grouped, key=lambda key: (sources[key]['vehicle_class'], sources[key]['recording_session'])):
        source = sources[source_id]
        if source.get('admitted_to_corpus') is not True or source['status'] != 'approved':
            raise ValueError('Only admitted approved development sources are allowed')
        wav = Path('data') / source['output_path']
        if digest(wav) != grouped[source_id][0]['normalized_source_sha256']:
            raise ValueError(f'Audio changed: {source_id}')
        metadata = json.loads(wav.with_suffix('.json').read_text())
        original = None
        error = None
        segments = covered_ranges(grouped[source_id], source['condition_segments'])
        reused = {} if source_id in args.audio_only_source_id else {
            (segment['start_seconds'], segment['end_seconds']): reusable_clip(previous, args.reuse_reel, source_id, segment)
            for segment in segments}
        try:
            audio_only = Path(urlparse(metadata['download_url']).path).suffix.lower() in {'.ogg', '.wav', '.flac'}
            if source_id in args.audio_only_source_id:
                error = 'Video intentionally omitted at user request; audio retained.'
            elif audio_only:
                pass
            elif reused and all(reused.values()) and metadata['raw_sha256'] not in local_hashes:
                pass
            elif source['provider'] in args.skip_remote_provider and metadata['raw_sha256'] not in local_hashes:
                error = 'Remote video deferred after provider rate limiting; local audio retained.'
            elif args.download_originals or metadata['raw_sha256'] in local_hashes:
                original = restore_original(source, metadata, output / 'originals', local_hashes)
            if original and not any(s['codec_type'] == 'video' for s in probe(original)['streams']):
                original = None
        except Exception as exc:
            error = str(exc)
            print(f'Video unavailable; keeping source audio: {source_id}: {error}', flush=True)
        video_available = original is not None or (bool(reused) and all(reused.values()))
        availability.append({'source_id': source_id, 'video_path': str(original) if original else None,
                             'video_available': video_available,
                             'video_reused_from': str(args.reuse_reel) if not original and video_available else None,
                             'audio_only_original': audio_only,
                             'video_error': error, 'source_page': source['source_page'],
                             'attribution': source.get('attribution'), 'license': source['expected_license']})
        for segment in segments:
            number = len(index) + 1
            start, end = segment['start_seconds'], segment['end_seconds']
            duration = end - start
            overlay = output / 'panels' / f'{number:03d}.png'
            reuse = reused.get((start, end)) if not original else None
            panel(overlay, source, segment, number, original is not None or reuse is not None)
            clip = output / 'clips' / f'{number:03d}.mp4'
            if reuse and not clip.exists():
                shutil.copy2(reuse, clip)
            # Explicit source-channel selection; no normalization, crossfade or effects.
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-y']
            if original:
                command += ['-ss', str(start), '-i', str(original)]
            else:
                command += ['-f', 'lavfi', '-i', 'color=c=0x182638:s=960x540:r=30']
            command += ['-ss', str(start), '-i', str(wav), '-loop', '1', '-i', str(overlay)]
            command += ['-filter_complex',
                        '[0:v]setpts=PTS-STARTPTS,scale=960:400:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:66,setsar=1,fps=30[v];[v][2:v]overlay=0:0:shortest=1[out];[1:a]pan=mono|c0=c0,asetpts=PTS-STARTPTS[a]',
                        '-map', '[out]', '-map', '[a]', '-t', str(duration),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-ar', '48000',
                        '-movflags', '+faststart', str(clip)]
            print(f'Clip {number}: {source["vehicle_model"]} {timestamp(start)}-{timestamp(end)}', flush=True)
            if not clip.exists():
                try:
                    run(command)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(exc.stderr.decode()) from exc
            actual_duration = float(probe(clip)['format']['duration'])
            index.append({'number': number, 'source_id': source_id, 'model': source['vehicle_model'],
                          'vehicle_class': source['vehicle_class'], 'condition': segment['operating_condition'],
                          'source_start': start, 'source_end': end, 'reel_start': reel_time,
                          'reel_end': reel_time + actual_duration, 'video_available': original is not None or reuse is not None,
                          'reused_from': str(reuse) if reuse else None,
                          'reused_sha256': digest(reuse) if reuse else None,
                          'clip_path': f'clips/{number:03d}.mp4', 'notes': segment.get('notes')})
            reel_time += actual_duration
    concat = output / 'concat.txt'
    concat.write_text(''.join(f"file 'clips/{r['number']:03d}.mp4'\n" for r in index))
    run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat),
         '-c', 'copy', '-movflags', '+faststart', str(output / 'audit_reel.mp4')])
    info = {'dataset_version': lock['dataset_version'], 'sources': availability, 'clips': index,
            'selection': 'union of exact benchmark windows within each reviewed interval',
            'audio': 'normalized source WAV channel 0, original gain; AAC listening copy',
            'excluded': 'consumed Sherman/PDSounds and reserved T90M/JLTV',
            'duration_seconds': float(probe(output / 'audit_reel.mp4')['format']['duration']),
            'audio_only_source_ids': args.audio_only_source_id,
            'prior_reel_index_sha256': digest(args.reuse_reel / 'index.json') if args.reuse_reel else None}
    (output / 'index.json').write_text(json.dumps(info, indent=2) + '\n')
    buttons = '\n'.join(f'<button onclick="seek({r["reel_start"]})">{r["number"]:03d} · {html.escape(r["model"])} · {timestamp(r["source_start"])}–{timestamp(r["source_end"])} · {r["condition"]}</button>' for r in index)
    credits = ''.join(f'<li><a href="{html.escape(s["source_page"])}">{html.escape(s["source_id"])}</a>: {html.escape(s["attribution"] or "")} ({html.escape(s["license"])})'+ (' — '+html.escape(s['video_error']) if s['video_error'] else ' — audio-only original' if s['audio_only_original'] else ' — video retained from prior reel' if s['video_reused_from'] else '')+'</li>' for s in availability)
    page = '''<!doctype html><html><meta charset="utf-8"><title>ABVID recording audit</title>
<style>body{background:#101a29;color:#eee;font:17px system-ui;max-width:1100px;margin:24px auto;padding:0 20px}video{width:100%;max-height:62vh;background:black}button{display:block;background:#20334b;color:white;border:0;padding:12px;margin:5px 0;text-align:left;width:100%;cursor:pointer}a{color:#91d9ff}#now{position:sticky;top:0;background:#101a29;padding:12px}small{color:#b1cbe0}</style>
<h1>ABVID · Recording audit</h1><p>12 development sessions · only time covered by the 593 benchmark windows. Overlapping windows play once; gaps and rejected intervals are skipped. Train/eval roles vary by fold.</p>
<video id="player" controls src="audit_reel.mp4"></video><div id="now">Select a chapter below.</div>
<p>Review the original source time shown below the player when reporting corrections. Audio uses channel 0 at original gain; no loudness normalization. This is a compressed listening copy. Audio-only sources use a labeled panel. T90M/JLTV and Sherman/PDSounds are excluded.</p>
CHAPTERS<h2>Sources and attribution</h2><ul>CREDITS</ul>
<script>const clips=CLIPDATA;const player=document.getElementById('player');function seek(t){player.currentTime=t;player.play()}function ts(s){return Math.floor(s/60).toString().padStart(2,'0')+':'+(s%60).toFixed(2).padStart(5,'0')}player.ontimeupdate=()=>{const c=clips.find(c=>player.currentTime>=c.reel_start&&player.currentTime<c.reel_end);if(c){document.getElementById('now').textContent=c.number+' · '+c.model+' · original '+ts(Math.min(c.source_end,c.source_start+player.currentTime-c.reel_start))+' · '+c.condition}};</script></html>'''
    (output / 'index.html').write_text(page.replace('CHAPTERS', buttons).replace('CREDITS', credits).replace('CLIPDATA', json.dumps(index).replace('</', '<\\/')))
    print(json.dumps({'output': str(output), 'clips': len(index), 'duration_seconds': info['duration_seconds'],
                      'video_sources': sum(s['video_available'] for s in availability)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
