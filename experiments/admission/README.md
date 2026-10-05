# Admission

Integrity and provenance before grouping/splits.

- Canonical wrapper: [configuration](../../configs/experiments/admission.json).
- Original contract: `archive:experiments/DATASET_PROTOCOL.md` (SHA256 pinned by the wrapper).
- Original runner directory: `archive:experiments/admission`.
- [Original command sequences](REPLAY.md) retain their historical paths. Run them
  from `ABVID_ARCHIVE_ROOT`, using the environment described there. New destinations
  must be distinct from completed runs; never rewrite frozen manifests or results.
- Shared numerical code now lives in `src/abvid/`.

```bash
abvid validate configs/experiments/admission.json --check-files
abvid replay configs/experiments/admission.json
```

`replay` prints the original command; `--execute` explicitly runs it. The bundled
wrapper is a regression/help entry point, not a new training protocol. The complete
historical workflow remains in the original command sequences above.

- [Report: dataset_admission_integrity.md](../../reports/dataset_admission_integrity.md)
- [Report: dataset_admission_provenance.md](../../reports/dataset_admission_provenance.md)
