# Blind listening pilot v1

Preregistered 2026-09-20 before selection or human responses. This is a development
diagnostic, not independent confirmation or an M7 gate. Use only the non-startup
785-window manifest SHA256
`c705cc95ef506cbd0ad4d504e53a5cc8fcae71baca63355d9b6b54c24ba81359`.

Select one anchor per each of the 12 development sessions using seed 42, uniformly
from two-second windows whose centered six-second context fits wholly within one
currently reviewed interval. Add two wheeled anchors, in different sessions, with
nonoverlapping six-second contexts. This yields 14 anchors, seven per class, and
28 trials. Selection never reads model predictions. If geometry cannot satisfy this,
fail instead of changing the sampling rule. All sessions contribute; session-level
scoring prevents the two extra wheeled clips receiving disproportionate weight.

Present all two-second clips first in shuffled order, then the matched six-second
clips in a second shuffled block. Random opaque IDs, no video, provenance, class,
source filename, model predictions, or answer feedback in the participant page.
The answer key lives outside the served participant directory. Raw mono channel 0,
16 kHz; no denoising, gain normalization, padding, or concatenation across gaps.
Use the exact benchmark WAV for short trials. Record choice (tracked/wheeled/unsure),
confidence, replay count, self-reported familiarity, and response time. Browser-local
autosave and JSON export; no data sent to an external service.

Primary descriptive result: short-block class/session-macro recall, counting unsure
as incorrect; show accuracy, coverage, conditional accuracy among committed choices,
confusion counts, per-session results, and descriptive session-bootstrap intervals.
Long-block results are secondary. Longer clips repeat an event already heard:
duration and repeat exposure are confounded, so differences are NOT a causal context
benefit. Familiarity with these previously reviewed recordings also limits blindness.
Do not inspect the key or browse source videos before taking the test; ideally recruit
an unfamiliar listener. No population-level human-performance claim from one person.

Do not use human answers to change source labels automatically or select benchmark
hyperparameters. No human results exist until an actual response export is scored.
