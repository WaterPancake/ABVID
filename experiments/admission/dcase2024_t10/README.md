# DCASE 2024 Task 10 metadata admission

Metadata-only admission and seal for the DCASE 2024 Task 10 development release.
Read the [protocol](PROTOCOL.md) and the [report](../../../reports/dataset_admission_dcase2024_t10.md).
`PROTOCOL.md` and `seal.json` are hash-bound: editing either breaks `verify_seal`.

```bash
# Re-run admission into a NEW directory (create-only; transfers ~3.4 MB, no audio).
uv run --extra cast python -m abvid.data.dcase_t10 data:DCASE2024_T10/metadata_v3

# Check the seal against the admitted metadata.
uv run --extra cast python -c "from abvid.data.sealing import verify_seal; \
print(verify_seal('experiments/admission/dcase2024_t10/seal.json')['status'])"
```

Admitted record: `data:DCASE2024_T10/metadata_v2/`, which holds the archive
directory inventories, the 18 metadata members, `segments.jsonl`, `summary.json`,
`receipt.json` and `hashes.json`.

To open the audio, a human commits `authorizations/<study>.json` with the fields
checked by `abvid.data.sealing.authorize`. It names the seal SHA256, the frozen
comparison protocol, the overlap-audit result and exactly one scoring pass. The
audio download belongs to that frozen study and is not part of this admission.

**Planned use (decided 2026-10-11):** one joint confirmation batch for the CAST
transfer comparison and a corrected H2 replication, frozen together and scored in
a single pass. See the [joint confirmation plan](../../joint_confirmation/PLAN.md).
The set is not split between studies.
