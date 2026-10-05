> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H2 execution clarifications and repairs

2026-10-04, before any H2 fitting or target scoring. The immutable design remains
`source_path_20261004_v1`. Its four factors, parameter draws, 190/class training
budget, five banks, target population, fixed heads and statistical contrasts are
unchanged. This record distinguishes numerical implementation changes from
physical realism or empirical calibration. Failed runs are retained.

1. **Indexing and propagation corrections.** Retain every complex angle/frequency
   row; select matching rows. Apply geometric spreading once over the full image
   path, initialize the second delay from microphone height, use environment `c`
   throughout, and detect source motion in all coordinates. Fix Sinc table weights
   to use their normalized fractional position, and air-filter angular frequencies
   to `2*pi*f/fs`.
2. **Sinc accuracy.** The corrected 11-tap Hann interpolator failed four frozen
   3 kHz checks (maximum 1.112 dB error). Use 31 taps for every path and factor.
   The 0.5 dB tolerance is unchanged; the repeated static probe's maximum tone
   error is 0.00882 dB. No source/path prior or factor receives a special filter.
3. **Deterministic arithmetic.** The first complete source and renderer checks
   found last-bit replay differences from vector complex arithmetic. Use fixed
   scalar ordering for Butterworth prototype/prewarp/bilinear/polynomial algebra,
   and fixed scalar/Numba complex ordering for ground coefficients. No waveform
   rounding, integer quantization or tolerance-based hash comparison is allowed.
   Source validation against the unchanged released functions is numerical, since
   their implementation itself exhibits these differences; local adapter replay
   remains byte-exact. The full 480-case source check agrees with the released
   source within `2.73e-12` maximum absolute waveform error.
4. **Execution speed.** Batch the same samplewise delay and FIR operations across
   source columns sharing geometry. Evaluate each sample's ground equation;
   introduce no new angle/distance grid, approximation or source-specific physics.
   A scalar reference remains independent of the batched implementation. Both
   paths agree over 52,000 samples, including a circular-buffer wrap, within
   `3e-17` absolute output error. Exact replay is checked separately, including
   fresh-process corpus rerenders.
   Four spawned rendering workers may process independent complete geometry
   groups, with at most four groups in flight and one BLAS thread per worker.
   All eight members of a group are validated/written together in fixed ID order.
   The sequential and four-worker 30-slot banks have identical source and
   observation manifests and waveform hashes. Measured four-worker time is
   about 36 seconds including startup for the 240-observation check; the
   conservative full-bank extrapolation is about 21 minutes. These worker
   settings affect execution resources, not the factor definitions or budget.
5. **Passivity criterion clarification.** Check the local plane-wave coefficient
   `Rp` and air-absorption FIRs against the original 0.5 dB gain allowance. Retain
   and report the spherical field coefficient `Q` and its FIR gains separately;
   do not clamp them to unity. `Q` combines the plane-wave and curved-wave/ground
   contribution and is a field multiplier in the image-path expression, rather
   than the local incident-to-reflected energy ratio. See the distinct definitions
   in [Acoustic-Toolbox's primary implementation documentation, reflection module](https://acoustic-toolbox.readthedocs.io/en/doc-fix/reflection/)
   and [Damiano et al., §3.2.1](https://doi.org/10.1186/s13636-024-00372-4).
   This is an explicit clarification of the design's ambiguous phrase “passive
   gains”; it is not a proof that the complete inherited ground model accurately
   describes a real road. Some retained spherical FIR gains exceed unity by more
   than 0.5 dB, and the validation report exposes them. The independently checked
   unit-reflection fixture verifies the spreading law without this ambiguity.
6. **Normalization location.** As specified by the design's operation order, the
   -26 dBFS RMS and float32 rounding check apply to the 8 kHz receiver crop before
   resampling. The final 16 kHz RMS is also recorded; resampling can change it.
   Do not add a second normalization after resampling. The .98 peak gate applies
   to the actual 16 kHz classifier input.
7. **Storage estimate correction.** Raw full renders alone require 6.144 GB, plus
   0.307 GB of dry sources and 1.229 GB of observations, before component masters,
   headers, features and reports. Budget roughly 9–11 GB of additional disk space,
   not the earlier approximate 5 GiB. Check actual disk space and measure the 30
   prescribed validation slots before bulk generation; do not reduce the bank.

The inherited 40-tap spherical ground filter, 11-tap air filters, one-degree
angle selection, current-position delay approximation, raw ground parameter and
unknown real-road calibration remain explicit limitations. The moving-frequency
gate compares against independently solved retarded time; passing a 1% numerical
tolerance does not make the renderer an exact moving-source wave equation.
H2 estimates the effect of these specified source and path interventions, not an
intrinsic decomposition of all real-world domain shift.

Evidence: `results/physics_probe_20261004_v1` (failed),
`results/physics_probe_20261004_v2` (static pass),
`results/source_validation_20261004_v1` (replay failure),
`results/source_validation_20261004_v2` (all source checks pass),
`results/renderer_validation_20261004_v1` (replay failure), and
`results/renderer_validation_20261004_v2` (analytic/equivalence/replay checks pass).
The complete execution lock and paired-corpus check remain separate required gates.
