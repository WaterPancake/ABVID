# Recording audit reel

**Historical snapshot:** this reel covers the frozen, capped 593-window benchmark.
Startup segments were subsequently excluded from the active catalog on 2026-09-20.
The old reel is retained for review history and can still contain those intervals;
it is not the manifest for the new uncapped preprocessing grid. See
[the new protocol](preprocessing_grid_protocol.md) and
`configs/archived_startup_intervals.yaml`.

Build a local viewing copy of the frozen 7/5 benchmark, reusing cached
video and intentionally keeping AMX-30/Ford Model T audio-only:

```bash
uv run python scripts/build_audit_reel.py \
  --output .artifacts/audit_reel_7t5w_v2 \
  --reuse-reel .artifacts/audit_reel_7t5w \
  --audio-only-source-id candidate-target-tracked-retromobile-amx30-2015 \
  --audio-only-source-id target-wheeled-ford-model-t-start
```

Open `.artifacts/audit_reel_7t5w_v2/index.html` in a browser, or play
`.artifacts/audit_reel_7t5w_v2/audit_reel.mp4` directly. The browser page provides
clickable chapters and a readout of the corresponding original source time.
Individual chapters are also available in the `clips/` directory.

The reel contains 76 chapters and 829 seconds of selected source audio
(the MP4 container is 829.021 seconds, approximately 74 MiB). The full MP4 decoded
without errors. All 76 chapter selections and timings match the initial reel exactly;
decoded PCM audio also has the identical SHA-256
`c81a15781738b11d21fd9843abab328ca0a52197956b48230ed36af27f2ceb24`.
The full test suite passes 109 tests, including exact-chapter reuse and path checks.

The updated v2 reel adds hash-matched local originals for StuG, T-72/BMP-3 and
Mobile Protected Firepower. It reuses four cached originals and the exact prior
T-80/MT-LB rendered chapters, since that original is no longer present locally.
The index records the reused clip paths and hashes. Eight sessions now have video.
AMX-30 and Ford Model T remain audio-only at the user's request; Abarth and Maserati
were audio-only originals. No training/evaluation audio, labels, or selections changed.
No network downloads are needed. Choose a new output directory for another build.

## Initial build history

In the initial reel, pictures were present for T-80/MT-LB, Abrams Bright Star, Bradley/Abrams Poland,
HMMWV and Stryker. Wikimedia rate limiting prevented restoration of StuG,
T-72/BMP-3 and AMX-30, after which remaining remote Commons requests for MPF and
Ford Model T were deferred. Those five sources retained audio with labeled panels.
Abarth and Maserati were audio-only originals. Source-page links are in the player.
The subsequent manual downloads and audio-only preference are incorporated into v2;
no further downloads are required for the requested reel.

### Original video links

The user supplied the first, second and fourth originals below. AMX-30 and Ford
Model T downloads are no longer requested. These links are retained for provenance:

- [StuG III, Lappeenranta](https://commons.wikimedia.org/wiki/File:Taistelun%C3%A4yt%C3%B6s_Lippujuhlan_p%C3%A4iv%C3%A4_2014_16_Stug_IIIG.webm)
- [T-72B3/BMP-3, training-ground march](https://commons.wikimedia.org/wiki/File:102nd_Motorized_Rifle_Regiment_marching_to_the_training_ground_%282022-01-26%29.webm)
- [AMX-30, Retromobile 2015](https://commons.wikimedia.org/wiki/File:R%C3%A9tromobile_2015_-_Char_AMX_30_-_001.ogv)
- [Mobile Protected Firepower](https://commons.wikimedia.org/wiki/File:U.S._Army_Armor_%26_Cavalry_Collection_Mobile_Protected_Firepower.webm)
- [Ford Model T, starting the engine](https://commons.wikimedia.org/wiki/File:Ford_Model_T_%22Tin_Lizzy%22_-_Starting_the_Engine.webm)

These are the URLs stored in the reel provenance, not newly downloaded copies.
Abarth and Maserati have audio-only originals and need no video download.

## Selection and provenance

The reel uses the frozen manifest and catalog in
`benchmarks/v0.1/native_real_7t5w/`. It includes all 12 development sessions that
rotate between training and evaluation in the nested benchmark. Selected two-second
windows are joined as a union within each reviewed condition interval, so overlap
plays once and unselected gaps stay excluded. It does not include every second of
the full recordings or every possible reviewed window.

Audio comes from the hash-verified normalized source WAV, selecting channel 0 at
its original gain, with no normalization or crossfades. The MP4 uses AAC for
convenient listening; it is not an exact copy of the 16 kHz model input tensors.
Pictures come from originals matched to recorded raw hashes. A labeled panel
replaces the picture when the source is audio-only or its video cannot be retrieved.
No classifier predictions appear in this review artifact.

`index.json` records source URLs, attribution, license, video availability/errors,
condition labels, and the mapping from reel time to original source time. Titles
may list multiple vehicle models when a source contains them; the review notes
under the picture retain the segment-specific annotation.

Use original timestamps when reporting corrections, for example:

```text
HMMWV, original 00:37–00:41: reject, foreground speech
Stryker, original 00:05–00:16: steady_speed
```

The consumed Sherman/PDSounds pair and reserved T90M/JLTV are excluded. Reviewing
their audiovisual content is a separate activity from this development reel.

The optional original downloads restore only previously collected, approved sources
through the existing collection downloader and validate their raw hashes. Media,
individual clips and the reel remain local under `.artifacts/`. A completed reel
is never overwritten; use a new `--output` directory after changing labels.
