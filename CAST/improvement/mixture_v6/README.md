# Fixed effective mixture adjustment

This six-case source-only experiment is motivated by the audited training
component diagnostic. Every arm receives the same transformed donor bank.
No original fitted artifact is changed. See [PROTOCOL.md](PROTOCOL.md).

Reproduce with the existing local parent artifacts and pinned environment:

```sh
bash CAST/improvement/mixture_v6/run.sh source --run-id cast_smooth8_v1_20261004_r1 --eval-id NEW_MIXTURE_SOURCE_ID
bash CAST/improvement/mixture_v6/run.sh audit --eval-id NEW_MIXTURE_SOURCE_ID
```

The complete v5 failure and its successful replay audit are prerequisites.
Generation runs the full test suite and freezes all inputs before sampling.
Audit replays every sample, ancestry and score, and requires exact reproduction
of all 450 zero-offset reference score records. The same per-class coverage,
margin and spread thresholds apply. A failed source candidate does not permit
outer access, even if it ranks first. Source scores remain exposed development.

The conditional outer runner has separately tested access guards. Use it only
after a passing source candidate and matching successful audit:

```sh
bash CAST/improvement/mixture_v6/outer.sh run --source-eval-id NEW_MIXTURE_SOURCE_ID --eval-id NEW_OUTER_ID
bash CAST/improvement/mixture_v6/outer.sh verify --eval-id NEW_OUTER_ID
```

It uses the original outer scales and IDs, completes all 1,500 generated audio
samples before any held descriptor read, and preserves full calibration
ancestry. Verification replays every generated sample and all 570 raw held
descriptors. It is development evidence, not fresh confirmation.
