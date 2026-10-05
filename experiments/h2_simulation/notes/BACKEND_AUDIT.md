> Historical protocol/command reference. Commands retain their original paths
> and run from `ABVID_ARCHIVE_ROOT`, not from the reorganized code checkout.
> See the [experiment registry](../../README.md).

# H2 source and renderer audit

2026-10-04. This is a source-code/provenance audit, not an acoustic validation run. Snapshot bytes are checked against SHA256 and, for upstream files, Git blob IDs from the pinned recursive Git tree. No upstream module or P02 generator was imported/executed. Evidence paths are retained under `evidence/` and copied into the freeze. Findings below distinguish inspected code from our physical interpretation and required future work.

## Dry source reference

Released P02 [generator snapshot](../../../reports/ARTIFACT_INDEX.md#artifact-c797053c90b4): SHA256 `01e177684b9204d36b0410ad6bd42a26aeda3f73fadd498aa812c6e7457b7c0e`. Source: `experiments/reproduction/releases/P02/code_and_results/scripts/_2_genSyntheticData.py`.

| Evidence locator | Observed fact | H2 disposition |
|---|---|---|
| Lines84–113 | Harmonic engine, per-harmonic phases, firing/2 AM, peak normalization | Preserve in S0; explicit parameters/RNG in future adapter |
| Lines116–138 | Filtered broadband tire component | Preserve band/filter/normalization |
| Lines141–180 | Lowpass broadband + harmonic exhaust mixture | Preserve; only truck uses it |
| Lines188–244 | Car/truck RPM priors and mixture weights | Freeze as source-code priors, not empirical vehicle measurements |
| Lines295–329 | Wrapper attempts `pyroadacoustics.simulatorScene.road_scene`; exception returns None | Exclude wrapper; this API is absent in our pinned upstream tree |
| Lines332 onward; per-file generation around416 onward | Alternate rendering path and Python hash-based seeding | Exclude; the released archive cannot be a verified factor cell |

The script header states MIT; the [release README](../../../reports/ARTIFACT_INDEX.md#artifact-100e0c205402) distinguishes MIT code from CC BY4.0 data/models. Preserve notices. A comment referring to CNOSSOS does not validate/calibrate these source functions against that standard. Actual backend/run/source lineage of the distributed P02 audio remains unknown: this audit does not establish which branch generated those files.

The existing local `src/vehicle_audio/source_simulation.py` is also not a drop-in civilian source: its VehicleProfile requires tracked/wheeled labels and its synthesis includes geometry attenuation. Reusing it unchanged would mix the source/path factors. Military profiles/results are left unchanged.

## Reference renderer and findings

Reference: [official pyroadacoustics revision642f95e](https://github.com/steDamiano/pyroadacoustics/tree/642f95ea83417b662906712c5902d7b7dd665092). The snapshot [pyproject](../../../reports/ARTIFACT_INDEX.md#artifact-5d8322a832a8) identifies version1.1.0 and numpy/scipy/matplotlib dependencies; the retained [LICENSE](../../../reports/ARTIFACT_INDEX.md#artifact-52782c08903a) is GPLv3. This records package notices, not a new assessment of dataset redistribution rights. The package is not installed or integrated here.

| Inspected locator at that revision | Code observation | Consequence / interpretation |
|---|---|---|
| [environment.py](../../../reports/ARTIFACT_INDEX.md#artifact-6d832a984137):109–159,190–225,562–578 | Constructor temperature is Celsius, pressure atm; explicit `set_simulation_params` flags; speed of sound formula | Use20°C,1atm,RH50 and explicitly request Sinc. Do not inherit differing method defaults or pass Kelvin. |
| environment.py:230–313,426–483 | Too-short signals are looped, absent signals become a siren; one sample update per supplied trajectory point | Require full explicit source; forbid hidden loop/default signal and verify exact dimensions. |
| [soundSource.py](../../../reports/ARTIFACT_INDEX.md#artifact-2d555a903495):149–169 | Segment sample count uses round(T*fs−1) and then appends endpoint | Use/check exact analytic sample grid through an adapter; otherwise nominal speed differs slightly from requested speed. |
| [simulatorManager.py](../../../reports/ARTIFACT_INDEX.md#artifact-981030b85bf3):142–157,670–710 | Ground-angle loop overwrites `w_tmp`/`Rp` each iteration; returns one frequency vector. Later `w_tmp[idx]` indexes this vector as if it were an angle table; `Rp` is not selected by angle. | **Blocking indexing defect for the configured integer-ground branch.** Direct code inspection establishes the mismatched dimensions; acoustic size/direction of its effect has not been measured. |
| simulatorManager.py:373–385,421–425,501–517 | Reflected signal is multiplied by1/a then1/b; helper is explicitly1/d. Reflection filter does not introduce a compensating a*b/(a+b) spreading factor. | **Blocking physical-contract discrepancy:** this produces1/(a*b), while the intended point-source image-path amplitude is1/(a+b), before ground reflection. This latter expected law is our H2 model specification; a validation must check it independently. Receiver RMS matching cannot generally remove changed direct/reflected balance. |
| simulatorManager.py:695 | Ground calculation hardcodes343m/s while environment computes c | Require common c; magnitude of resulting discrepancy has not been tested. |
| simulatorManager.py:297–321,330–333,371–401 | Uses current-position variable delays; short FIRs add latency; reflected filters/delays are cascaded | Doppler/latency accuracy requires independent moving/static checks; no claim of exact retarded-time propagation from inspection. |
| simulatorManager.py:152–157; environment.py:119–126 | Integer20000 called flow-resistance; formula includes1000*f/sigma | Audited API does not establish physical units/calibration. Preserve raw value, mark units unknown; do not silently reinterpret as a measured asphalt surface. |

Links to primary code: [renderer implementation](https://github.com/steDamiano/pyroadacoustics/blob/642f95ea83417b662906712c5902d7b7dd665092/pyroadacoustics/simulatorManager.py), [environment API](https://github.com/steDamiano/pyroadacoustics/blob/642f95ea83417b662906712c5902d7b7dd665092/pyroadacoustics/environment.py). The source is authoritative for these implementation observations. They are not a claim that every upstream version or the unverified P02 audio has these defects.

## H2-R1 repair contract — future work, not implemented

The factor definitions stay fixed. The backend named in the design is a future reviewed derivative of this reference, with an explicit patch and execution hash; upstream unchanged is **not an eligible execution backend**.

1. Store complex ground `w_tmp` and `Rp` for every angle/frequency with shape(179,512); select both by the same angle index. Check several angles against direct evaluation of the retained equations, including the grazing boundary. Do not change material priors in the repair.
2. Apply spherical spreading once per total path:1/d direct and1/(a+b) reflected. Remove the pair of segment attenuation factors; retain path-dependent air and ground filtering. Test unity reflection/air-disabled against the image-source formula. Do not make separate source-specific/path-specific backend patches.
3. Use the environment's c throughout the ground computation. Supply/check the exact frozen sampled trajectory and length-matched source. Record every code alteration; no silent substitution/fallback.
4. Run all physical/manipulation gates in [PROTOCOL.md](PROTOCOL.md). Any extra issue discovered is a pre-target disclosed revision, with repaired code/dependency hashes and all matched scenes rechecked. A correction that changes factor definitions/priors/budget requires a new design version.

`source_adapter_sha256`, `renderer_patch_sha256`, `environment_lock_sha256`, `waveform_manifest_sha256` and `validation_report_sha256` remain null. No acoustic accuracy, runtime, compatible dependency installation, generated data or transfer effect is asserted. In particular, retaining a reference snapshot plus metadata is **not** successful simulator admission. This keeps the next implementation task concrete without pretending it was completed during the design freeze.
