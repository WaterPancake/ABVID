# Source-only equivalent-fit mixture

This uses the complete audited smooth8 bank and its original 5% ambiguity
threshold. It introduces no new fits, recordings or outer data access.
Every matched arm receives the same eligible alternatives, with equal original
parent and group weighting. See [PROTOCOL.md](PROTOCOL.md).

Reproduce from the ABVID root with the existing local parent artifacts and
pinned Python environment:

```sh
bash CAST/improvement/alternatives_v4/run.sh source --eval-id NEW_SOURCE_ID
bash CAST/improvement/alternatives_v4/run.sh audit-source --eval-id NEW_SOURCE_ID
```

The source command runs the full numerical/provenance test gate, freezes code,
bank, descriptors and configuration, then preserves every generated donor,
alternative start, waveform sample hash, descriptor and score. The audit
replays the complete fixed schedule, recomputes scores and confirms that all
150 winner-only scores reproduce the previous full-bank source comparison.
A successful audit establishes reproducibility, not scientific success. Inspect
the saved per-class criteria and `scientific_status.json`.
