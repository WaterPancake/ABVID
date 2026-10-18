# CAST-5A: classification utility within IDMT

Frozen protocol: [PROTOCOL.md](PROTOCOL.md) (`cast5a_utility_v1`). The SHA256 of this
protocol is pinned in the design bundle, so do not edit it. Outputs live under
`artifact:cast/utility/cast5a_run_20261011_v1/`, and every stage is create-only.

| Stage | Environment | Command |
|---|---|---|
| Design freeze (metadata only) | CAST readiness | `python -m abvid.cast.utility_design freeze` |
| Plan / progress | CAST readiness | `python -m abvid.cast.utility_design plan` |
| Fit the 1,496 new parents | CAST readiness | `python -m abvid.cast.utility_design fit --workers 4` |
| RU banks (5 seeds) | CAST readiness | `python -m abvid.cast.utility_arms ru` |
| RC/C banks (30 roles; needs all parents) | CAST readiness | `python -m abvid.cast.utility_arms rc` |
| Real, RA and RU features | baseline | `python experiments/cast_utility/learn.py features-base` |
| RC features | baseline | `python experiments/cast_utility/learn.py features-rc` |
| 420 heads (saved and hashed before scoring) | baseline | `python experiments/cast_utility/learn.py heads` |
| Held scores, bootstrap, decision flags | baseline | `python experiments/cast_utility/learn.py score` |

CAST readiness Python:
`/Users/magi/workspace/abvid-artifacts/environments/cast-readiness-20261007/bin/python`.
Baseline Python: `.venv-baseline/bin/python`.

Parents already prepared by the CAST-4A run are read through the 4A context's
validated loader, so the 4A run must have published them. No CAST-4A pinned module is
modified. The reusable code is in:
- `src/abvid/cast/utility_design.py`
- `src/abvid/cast/utility_arms.py`
- `src/abvid/data/augmentation.py`
- `src/abvid/learning/utility.py`
