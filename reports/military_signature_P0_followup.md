# Military acoustic signature P0 evidence follow-up

2026-10-11 local date; protocol `military_signature_p0_v1`.
This is an additive supplement to the [frozen P0 report](military_signature_P0.md).
The original report, source ledger and 55-file freeze remain the dated P0 snapshot.
This follow-up covers public literature and the artificial browser round trip.

The supplied annotation passed import, connected-group validation and native
export. The user has now confirmed that Chrome audio playback, waveform,
spectrogram and interval controls displayed correctly. The [round-trip record](../experiments/military_signature/ANNOTATION_ROUNDTRIP.md)
links the explicit confirmation receipt and sample-exact export evidence. This
closes the human browser handoff listed in the original report. It is run-level
user evidence; there is no instrumented or per-recording playback trace.

## Additional powerplant evidence

Six additional original public PDFs were retrieved. Their extracted Markdown
was read before the selected original pages were rendered and visually checked.
The [supplementary source ledger](../experiments/military_signature/source_evidence_supplement.json)
records PDF/Markdown hashes, page and section locators, author statements,
interpretations, rendered pages and failed access attempts. Documents and
renders are outside the checkout at
`artifact:military_signature/p0/literature-followup-20261011/`.

| Verified scope | Author evidence and locator | Interpretation and remaining unknowns |
|---|---|---|
| Bradley production beginning in 1981 | Cummins associates its start with VTA903-500; `CUMMINS_DEFENSE`, PDF p13 / printed p11. `CUMMINS_V903`, PDF p2, specifies V903-family V8 four-stroke diesel architecture. | Historical association plus a later family specification. The proposed Bradley recording's variant and engine remain unknown. |
| M109A6 in a 1992 Army report | `ARMY_M109_1992`, PDF p57 / printed p55, describes an eight-cylinder diesel modification designated 8V71T-LHR. | Supports this reported variant. Stroke count and engines in other M109 variants or recordings remain unknown. |
| Strykers in a documented 2020 reset program | `ARMY_STRYKER_2020`, PDF p1 and p4, reports arrivals with C7 or 3126 engines and serviced C7 departures. `CAT_DIESEL_FLUIDS_2025`, PDF/printed p49, names both families as diesel. | Establishes engine heterogeneity in this population. A Stryker caption cannot identify the engine of a recording. Cylinder/stroke counts are not established here. |
| M1165A1 Special Ops brochure configuration | `AMGENERAL_M1165`, PDF p1–2, identifies the HMMWV variant and specifies a GEP 6.5-litre turbocharged V8 diesel. | Applies to the brochure configuration. It does not establish other HMMWV variants, JLTV, recording identity or stroke count. |

The source PDFs are [Cummins defense](https://mart.cummins.com/imagelibrary/data/assetfiles/0075965.pdf),
[Cummins V903](https://mart.cummins.com/imagelibrary/data/assetfiles/0063416.pdf),
[Army RD&A September–October 1992](https://asc.army.mil/docs/pubs/alt/archives/1992/Sep-Oct_1992.PDF),
[Anniston Tracks January 23, 2020](https://anad.army.mil/docs/librariesprovider2/default-document-library/tr012320_v2.pdf?sfvrsn=7a536430_0),
[Caterpillar SEBU6251-32](https://caterpillar.scene7.com/is/content/Caterpillar/CM20250210-d820f-61dca)
and [AM General M1165A1](https://www.amgeneral.com/wp-content/uploads/2019/12/M1165-Special-Ops.pdf).

Initial Python certificate failures are retained in the retrieval record.
System curl retrieved the affected documents using its default TLS validation;
certificate verification was not disabled. Three ARMOR originals still timed
out or lacked usable original-page verification. T-72/T-80/T-80UD/T-90 entries
therefore remain unverified under this project's evidence contract. Search
snippets do not supply verified powerplant labels.

## Consequences for a future pilot

Engine family and vehicle model need separate evidence fields. The documented
Stryker reset population supplies a concrete reason to retain within-model
engine heterogeneity. Manufacturer architecture cannot identify a recording's
firing rate, engine order or dominant acoustic mechanism without additional
ground truth. An engine specification provides no independent-session evidence.

The six-model roster remains a future pilot decision. “HMMWV/JLTV” cannot serve
as one model identity. An M1-only turbine population still cannot separate a
generic turbine effect from the M1 model effect. The existing distinction
between eight development groups and the proposal's ten-group minimum remains;
neither is a power analysis or a sufficient model-generalization guarantee.

The new documents add no admitted media, usable-audio observations or verified
independent recording groups. They support protocol preparation only. The
remaining handoffs are a human harness/test review before merge and separately
scoped P1 authorization with the exact population, assets, representation,
budgets and evaluation allocation frozen before execution.
