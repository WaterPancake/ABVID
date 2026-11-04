# Background-shortcut audit: protocol (draft, not frozen)

Freeze this file before the first probe run: record its SHA256 in the run's provenance
and do not edit it afterwards. A changed statistic, control, split or head needs a new
protocol version.

## Question

Does the frozen BEATs car/truck head respond to site acoustics as well as to the vehicle?
Under the active question (which simulation components drive synthetic-to-real
classification), background is a candidate component: a CAST re-render carries no site
background, so a decision that leans on it can flip between an original and its
re-render.

## Fixed, inherited from H1 unchanged

- Data: the pinned IDMT_V1 BEATs768 features in `config.json`. H1 preprocessing: channel
  mean, native rate → 8 kHz → 16 kHz, centre 2 s, no gain normalisation. Only 0–4 kHz
  reaches the encoder.
- Classes: car = 0, truck = 1. Motorcycle and bus excluded. SE sensor, CH34, as admitted.
- Head: the `classifier` block of `configs/reference/h1.json`, fitted on training rows
  only.
- Budget: 190 clips per class per head, the H1 budget. Seeds 42, 123, 456, 789, 1024.
- Folds: leave one provider site out. Sites are coarser than any candidate session; no
  session is inferred. Hohenwarte has no car/truck clips and no background, so it is
  not a fold.

## Data (counted from the pinned features, 4 Nov 2026)

| Site | Car | Truck | Truck share | Background |
|---|---:|---:|---:|---:|
| Fraunhofer-IDMT | 556 | 24 | 4.1% | 615 |
| Langewiesener-Strasse | 2,189 | 298 | 12.0% | 2,044 |
| Schleusinger-Allee | 1,157 | 189 | 14.0% | 1,412 |

The smallest training pool is 213 trucks (Langewiesener-Strasse held out), so the
190-per-class budget fits every fold.

- The selected clips carry 18 `candidate_recording_id` tokens (5, 6 and 7 per site),
  and each token has both background and car/truck clips. Truck share varies within
  sites: 0.8–10.8% (Fraunhofer-IDMT), 10.9–16.0% (Schleusinger-Allee), 8.1–21.4%
  (Langewiesener-Strasse). The grouped-coverage protocol notes that these tokens do
  not certify independent sessions.
- Background clips were "randomly sampled in between annotated vehicle passings" of
  the same recordings (Abeßer et al., EUSIPCO 2021), so faint vehicle sound may remain
  in them.
- Under the 8 kHz route, 30 of the encoder's 128 mel bins lie entirely above 4 kHz.

## Open decisions (Alex, before freezing)

1. What is the smallest run whose result would tell you the head reads the site?
2. What single number answers the question, and what is its unit of analysis? Clips
   within a site are not independent. With three sites, an exact test of a site
   ordering cannot reach p below 1/6.
3. What would make a positive result wrong, and which control would show it?

## Not claimed

No transfer result, no session certification, no physical mechanism. IDMT is exposed
development data, so any result is diagnostic evidence.
