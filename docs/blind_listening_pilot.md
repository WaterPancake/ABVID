# Take the blind listening pilot

The generated test is `.artifacts/blind_listening_v1/participant/index.html`.
Open it directly in a browser, or serve **only** its participant directory:

```bash
uv run python -m http.server 8767 --bind 127.0.0.1 --directory .artifacts/blind_listening_v1/participant
```

Then open <http://127.0.0.1:8767>. Do not serve the repository or the parent
directory: the answer key is in the sibling `private` directory. No server uploads,
accounts, or dependencies beyond a browser are needed. The page saves progress in
browser local storage when available. Export the JSON at the end and return it for
scoring. Use headphones, a comfortable fixed volume, and answer without video.

28 trials: 14 short clips, then their longer counterparts. All development sessions
represented; seven unique anchors per class. Startup, T90M, JLTV and consumed locked
sources excluded. Native audio, no inferred SNR or synthetic sound. Source credits
are in the administrator key; this is a local research artifact, not a redistribution
package. Familiarity and repeated exposure limit the conclusions.

Rebuild a fresh artifact (refuses existing output):

```bash
uv run python scripts/build_blind_listening.py --output .artifacts/blind_listening_repeat
```

Score a completed export, without overwriting existing results:

```bash
uv run python scripts/score_blind_listening.py path/to/abvid-listening-responses.json --output runs/listening_participant_01.json
```

Selection/scoring protocol: [blind_listening_protocol.md](blind_listening_protocol.md).
No participant results have been collected. Synthetic test responses used in software
tests are not human-performance evidence. The short-block result is primary; longer
clips have already been heard in abbreviated form, so improvement cannot be attributed
solely to context duration. This pilot does not advance the M7 gate.

Software verification: selection and strict scoring tests pass; a simulated 28-trial
DOM test exercises playback gating, local save/resume, and JSON export without
creating human results. All 28 excerpt hashes and review bounds were checked. The
participant-only HTTP server returns 404 for the private answer key. Browser UI
render inspection was unavailable in this tool session, so visual/audio playback
compatibility still needs a real-browser check by the participant.
