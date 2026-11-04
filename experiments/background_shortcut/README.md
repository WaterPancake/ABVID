# Background-shortcut audit

Status: scaffold. The data side is implemented and tested; the probe is not written and
the [protocol](PROTOCOL.md) is a draft. No head has been fitted.

The audit asks whether the IDMT car/truck head uses site acoustics. Vehicle-free
background clips from one site are scored by heads trained on the other sites. If
background from truck-heavy sites scores as more truck-like, part of the head's evidence
is the site, and a CAST re-render, which has no site background, loses it.

## Layout

| Path | Role |
|---|---|
| `config.json` | Feature pin, selection, folds, budget, seeds, classifier reference |
| `PROTOCOL.md` | Question, inherited H1 settings, data counts, open decisions |
| `run.py` | Thin runner |
| `src/abvid/audit/background_shortcut.py` | Feature pin check, clip roles, site folds, separation |
| `src/abvid/audit/run_record.py` | Create-only run directory with provenance written first |
| `src/abvid/audit/background_probe.py` | Probe interface, not yet implemented |

## Run

Use the baseline environment:

```bash
.venv/bin/python experiments/background_shortcut/run.py table
```

`table` verifies the feature hash, assigns roles, builds the 15 folds (3 sites × 5 seeds)
and checks file, event, group and site separation in every fold. It writes
`provenance.json`, `site_summary.json` and `folds.json` to a new
`artifact:audits/background_shortcut/<UTC stamp>-<commit>/` directory and fits nothing.
It refuses to run from uncommitted code unless `--allow-dirty` is passed, which records
the dirty paths.

## Registry

Not added to `configs/experiments/`: the registry schema has no diagnostic phase and
restricts execution modes. Whether to add one is a maintainer decision.
