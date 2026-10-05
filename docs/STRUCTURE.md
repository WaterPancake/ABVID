# Repository structure and ownership

`src/abvid/` contains reusable numerical code. It does not import an experiment
folder. `experiments/` holds scientific instructions and canonical registry
entries. `reports/` holds interpretations and small evidence; full results live
outside Git. Tests use synthetic fixtures unless explicitly marked as requiring
local frozen artifacts.

| Package | Responsibility |
|---|---|
| `data` | Fixed observations, grouped sampling and leakage assertions |
| `simulation` | Procedural source controls, deterministic filters, propagation |
| `cast` | Effective observation renderer, fitting, descriptors and coverage |
| `cast/population` | Population calibration/sampling and coverage selection |
| `representations` | Frozen BEATs interface and MFCC features |
| `learning` | Training-only scaler and fixed linear head |
| `evaluation` | Metrics, paired bootstrap and source/path contrasts |
| `provenance` | File hashes and metadata IO |

CAST belongs in the source-model branch. Its numerical controls describe observed
recordings with unknown scene/device contributions. The v15 coverage finding is
source-domain development evidence, not proof of clean source identification or
vehicle classification transfer. The larger source-versus-propagation hypothesis
remains open.

Storage defaults are sibling directories:

```text
workspace/
  ABVID/                                    code checkout
  abvid-data/civilian/                       view of archived original civilian data
  abvid-data/MELAUDIS_extracted/              extraction used by admitted manifests
  abvid-artifacts/models/                    checkpoints and upstream encoder code
  abvid-artifacts/archive/pre-restructure-20261005/
    experiments/                            completed studies, generated data, features
    CAST/                                   original studies and fitted banks
    data/, runs/, benchmarks/, openclaw/     retired project material
```

The archive retains original relative paths. The original `dataset/` directory remains a real directory inside the external
archive because CAST rejects symlinked source roots. `abvid-data/civilian` is a
symlink to that preserved data. Compatibility links for the admitted MELAUDIS
extraction path and `.artifacts/models` point to the external extraction/model
stores. Moving on
the same filesystem reduces checkout clutter; it does not reclaim those bytes.
No raw recordings or frozen results were deleted. Removing those later is a
separate retention decision.

Earlier published benchmark commits remain in Git history. The pre-refactor
research source snapshot is also preserved in history; full locally generated
artifacts and environments are not distributed in the code repository.

The archive has separate Git metadata referencing its original commit `0d242c6d` and retaining the exact original tracked diff;
its object store is shared with this checkout. Keep the checkout with the archive
or make the archive repository self-contained before moving it to another disk.
