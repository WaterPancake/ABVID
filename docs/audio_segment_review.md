# Vehicle Audio Segment Review Worksheet

Generated on 2026-08-28 from `configs/audio_sources.yaml` and the collected audio under `data/targets/`.

Edit the segment blocks in this file. After review, Codex can transfer the accepted changes into the catalog and collected sidecars. Until that happens, `configs/audio_sources.yaml` remains the implementation source of truth.

## Editing conventions

Active operating-condition tags are:

- `idle`
- `accelerating`
- `steady_speed`
- `decelerating`
- `mixed`
- `unknown`

On 2026-09-20 the operator excluded startup intervals from active training and
evaluation. Original recordings and historical annotations remain intact in
`configs/archived_startup_intervals.yaml` and frozen benchmark snapshots. The
historical `startup` tag is not relabeled as idle. Protected-source review history
below remains archival and does not authorize those sources for development.

Review decisions used only in this worksheet are:

- `keep`: currently admitted by the implementation.
- `reject`: explicitly rejected because of speech, wind, silence, weapons, unrelated vehicles, or other contamination.
- `exclude`: outside the current allowlist, but not necessarily conclusively rejected.
- `unreviewed`: requires audiovisual review before admission.

Use `MM:SS.s` timestamps. Segment ends are treated as exclusive boundaries. Small gaps are intentional unless you edit them. A pass-by at approximately constant speed is `steady_speed`; note `pass-by` separately. When motion is visible but acceleration state is unclear, use `mixed`.

Example:

```yaml
segments:
  - start: "00:00"
    end: "00:08"
    decision: keep
    tag: accelerating
    vehicle: Example vehicle
    notes: Clean approach.
  - start: "00:08"
    end: "00:12"
    decision: reject
    tag: null
    vehicle: null
    notes: Foreground speech.
```

## Tracked recordings

### M4 Sherman

- Source ID: `candidate-target-tracked-sherman-passby-gvn-43670951`
- Local audio: `data/targets/tracked/beeldengeluid_gvn_sherman_passby_43670951/sherman_passby.wav`
- Duration: `01:42.008`
- Original media: [audio-only Wikimedia page](https://commons.wikimedia.org/wiki/File:Sherman_tank,_rijden_-_SoundCloud_-_Beeld_en_Geluid.ogg)
- Corpus status: consumed by the completed locked evaluation; exclude from future model development

```yaml
segments:
  - start: "00:00"
    end: "00:25"
    decision: exclude
    tag: null
    vehicle: M4 Sherman
    notes: Outside the current allowlist.
  - start: "00:25"
    end: "01:30"
    decision: keep
    tag: mixed
    vehicle: M4 Sherman
    notes: Tank-dominant approach and pass-by; exact motion state is unavailable.
  - start: "01:30"
    end: "01:42.008"
    decision: exclude
    tag: null
    vehicle: M4 Sherman
    notes: Vehicle fades out.
```

### StuG III Ausf. G

- Source ID: `target-tracked-stug-iiig-lappeenranta`
- Local audio: `data/targets/tracked/commons_lappeenranta_flag_day_reenactment_2014/stug_iiig_arrival.wav`
- Duration: `00:25.920`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:Taistelun%C3%A4yt%C3%B6s_Lippujuhlan_p%C3%A4iv%C3%A4_2014_16_Stug_IIIG.webm)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "00:13"
    decision: reject
    tag: null
    vehicle: StuG III Ausf. G
    notes: Foreground speech.
  - start: "00:13"
    end: "00:14"
    decision: exclude
    tag: null
    vehicle: StuG III Ausf. G
    notes: Conservative boundary gap.
  - start: "00:14"
    end: "00:21"
    decision: keep
    tag: steady_speed
    vehicle: StuG III Ausf. G
    notes: Reviewed steady-speed movement.
  - start: "00:21"
    end: "00:22"
    decision: exclude
    tag: null
    vehicle: StuG III Ausf. G
    notes: Conservative boundary gap.
  - start: "00:22"
    end: "00:25"
    decision: keep
    tag: decelerating
    vehicle: StuG III Ausf. G
    notes: Reviewed deceleration.
  - start: "00:25"
    end: "00:25.920"
    decision: exclude
    tag: null
    vehicle: StuG III Ausf. G
    notes: Outside the current allowlist.
```

### T-72B3 and BMP-3 road march

- Source ID: `candidate-target-tracked-t72-bmp3-102nd-march-2022`
- Local audio: `data/targets/tracked/commons_milru_102nd_march_training_ground_2022_01_26/t72_bmp3_road_march.wav`
- Duration: `02:28.575`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:102nd_Motorized_Rifle_Regiment_marching_to_the_training_ground_(2022-01-26).webm)
- Corpus status: admitted using the reviewed segments below

```yaml
segments:
  - start: "00:00"
    end: "00:36"
    decision: reject
    tag: null
    vehicle: null
    notes: Unrelated material.
  - start: "00:36"
    end: "00:45"
    decision: keep
    tag: idle
    vehicle: T-72B3
    notes: Reviewed T-72B3 idle.
  - start: "00:45"
    end: "00:46"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "00:46"
    end: "00:54"
    decision: keep
    tag: steady_speed
    vehicle: T-72B3
    notes: Reviewed steady-speed T-72B3 movement.
  - start: "00:54"
    end: "00:55"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "00:55"
    end: "01:07"
    decision: keep
    tag: steady_speed
    vehicle: BMP-3
    notes: Reviewed steady-speed BMP-3 movement.
  - start: "01:07"
    end: "01:21"
    decision: reject
    tag: null
    vehicle: null
    notes: Other vehicles.
  - start: "01:21"
    end: "01:37"
    decision: keep
    tag: steady_speed
    vehicle: BMP-3
    notes: Reviewed steady-speed BMP-3 movement.
  - start: "01:37"
    end: "01:38"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "01:38"
    end: "02:01"
    decision: keep
    tag: steady_speed
    vehicle: T-72B3
    notes: Reviewed steady-speed T-72B3 movement.
  - start: "02:01"
    end: "02:02"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "02:02"
    end: "02:09"
    decision: keep
    tag: steady_speed
    vehicle: BMP-3
    notes: Reviewed steady-speed BMP-3 movement.
  - start: "02:09"
    end: "02:28.575"
    decision: reject
    tag: null
    vehicle: null
    notes: Other vehicles.
```

### T-80BVM and MT-LBVMK

- Source ID: `candidate-target-tracked-t80bvm-driving-readiness-2021`
- Local audio: `data/targets/tracked/commons_milru_200th_t80bvm_mtlb_driving_2021_02_02/t80bvm_mtlb_driving.wav`
- Duration: `01:10.822`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:A_week_of_combat_readiness_passed_in_the_200th_Motorized_Rifle_Brigade_(02-02-2021).webm)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "00:24"
    decision: keep
    tag: accelerating
    vehicle: T-80BVM
    notes: Reviewed accelerating movement.
  - start: "00:24"
    end: "00:41"
    decision: keep
    tag: mixed
    vehicle: MT-LBVMK
    notes: Vehicle identified; exact operating state remains mixed.
  - start: "00:41"
    end: "01:02"
    decision: keep
    tag: steady_speed
    vehicle: MT-LBVMK
    notes: Reviewed steady-speed movement.
  - start: "01:02"
    end: "01:03"
    decision: exclude
    tag: null
    vehicle: MT-LBVMK
    notes: Conservative boundary gap.
  - start: "01:03"
    end: "01:05"
    decision: keep
    tag: decelerating
    vehicle: MT-LBVMK
    notes: Reviewed deceleration.
  - start: "01:05"
    end: "01:10.822"
    decision: reject
    tag: null
    vehicle: null
    notes: Silence.
```

### T-90M Proryv

- Source ID: `candidate-target-tracked-t90m-27guards-2021`
- Local audio: `data/targets/tracked/commons_milru_t90m_27guards_2021_03_10/t90m_delivery.wav`
- Duration: `00:30.093`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:T-90M_tanks_of_the_27th_Guards_Motorized_Rifle_Brigade_(2021-03-10).webm)
- Corpus status: reviewed and reserved outside training for a future locked evaluation

```yaml
segments:
  - start: "00:00"
    end: "00:24"
    decision: keep
    tag: steady_speed
    vehicle: T-90M Proryv
    notes: Conservative resolution of the prior overlapping 00:00-00:25 keep and 00:24-end reject annotations.
  - start: "00:24"
    end: "00:30.093"
    decision: reject
    tag: null
    vehicle: null
    notes: Silence.
```

### AMX-30

- Source ID: `candidate-target-tracked-retromobile-amx30-2015`
- Local audio: `data/targets/tracked/commons_retromobile_amx30_demo_2015/amx30_demo.wav`
- Duration: `02:54.165`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:R%C3%A9tromobile_2015_-_Char_AMX_30_-_001.ogv)
- Corpus status: admitted using the reviewed operating-state segments below

```yaml
segments:
  - start: "00:00"
    end: "00:08"
    decision: reject
    tag: null
    vehicle: AMX-30
    notes: Rejected in the latest review.
  - start: "00:09"
    end: "00:40"
    decision: exclude
    tag: startup
    vehicle: AMX-30
    notes: Reviewed AMX-30 startup; parked by operator on 2026-09-20, original review preserved.
  - start: "00:41"
    end: "00:49"
    decision: keep
    tag: idle
    vehicle: AMX-30
    notes: Reviewed AMX-30 idle.
  - start: "00:50"
    end: "01:05"
    decision: keep
    tag: accelerating
    vehicle: AMX-30
    notes: Reviewed AMX-30 acceleration.
  - start: "01:06"
    end: "01:35"
    decision: keep
    tag: idle
    vehicle: AMX-30
    notes: Reviewed AMX-30 idle.
  - start: "01:35"
    end: "01:55"
    decision: keep
    tag: steady_speed
    vehicle: AMX-30
    notes: Reviewed steady-speed AMX-30 movement.
  - start: "01:56"
    end: "02:00"
    decision: keep
    tag: decelerating
    vehicle: AMX-30
    notes: Reviewed AMX-30 deceleration.
  - start: "02:00"
    end: "02:11"
    decision: keep
    tag: steady_speed
    vehicle: AMX-30
    notes: Reviewed steady-speed AMX-30 movement.
  - start: "02:11"
    end: "02:47"
    decision: keep
    tag: steady_speed
    vehicle: AMX-30
    notes: Reviewed steady-speed AMX-30 movement.
  - start: "02:47"
    end: "02:54.165"
    decision: reject
    tag: null
    vehicle: null
    notes: Final silent tail excluded by the current catalog.
```

### M1 Abrams — Bright Star 2017

- Source ID: `candidate-target-tracked-abrams-bright-star-2017`
- Local audio: `data/targets/tracked/dvids_bright_star_abrams_mohamed_naguib_2017/m1_abrams_maneuver.wav`
- Duration: `04:35.989`
- Video: [DVIDS source video](https://www.dvidshub.net/video/550523/us-army-abrams-tanks-maneuver-exercise-bright-star-2017)
- Corpus status: admitted using the keep-segments below

```yaml
segments:
  - start: "00:00"
    end: "00:12"
    decision: exclude
    tag: null
    vehicle: M1 Abrams
    notes: Outside the latest allowlist.
  - start: "00:12"
    end: "00:29"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed movement.
  - start: "00:29"
    end: "02:17"
    decision: reject
    tag: null
    vehicle: null
    notes: Speech- and wind-dominated material, plus excluded boundaries.
  - start: "02:17"
    end: "02:24"
    decision: keep
    tag: decelerating
    vehicle: M1 Abrams
    notes: Wind is audible; retained by the latest annotation.
  - start: "02:24"
    end: "02:35"
    decision: reject
    tag: null
    vehicle: null
    notes: Outside the reviewed vehicle-dominant interval.
  - start: "02:35"
    end: "02:40"
    decision: keep
    tag: accelerating
    vehicle: M1 Abrams
    notes: Reviewed acceleration.
  - start: "02:40"
    end: "02:41"
    decision: exclude
    tag: null
    vehicle: M1 Abrams
    notes: Conservative boundary gap.
  - start: "02:41"
    end: "02:46"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed movement.
  - start: "02:46"
    end: "03:45"
    decision: reject
    tag: null
    vehicle: null
    notes: Wind-dominated material and excluded edit boundaries.
  - start: "03:45"
    end: "04:02"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed movement.
  - start: "04:02"
    end: "04:35.989"
    decision: reject
    tag: null
    vehicle: null
    notes: Foreground speech or outside the latest reviewed allowlist.
```

### M2 Bradley — Poland 2025

- Source ID: `candidate-target-tracked-bradley-nato-muddy-maneuver-2025`
- Local audio: `data/targets/tracked/dvids_nato_bradley_bemowo_livefire_2025/bradley_muddy_maneuver_review.wav`
- Duration: `07:11.220`
- Video: [DVIDS source video](https://www.dvidshub.net/video/972401/nato-allies-demonstrate-their-readiness-through-live-fire-training-poland-b-roll)
- Corpus status: admitted as one tracked session containing reviewed M2 Bradley and M1 Abrams intervals

```yaml
segments:
  - start: "00:00"
    end: "00:41"
    decision: keep
    tag: steady_speed
    vehicle: M2 Bradley
    notes: Reviewed steady-speed M2 Bradley movement.
  - start: "00:41"
    end: "00:42"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "00:42"
    end: "00:48"
    decision: reject
    tag: null
    vehicle: unidentified wheeled vehicle
    notes: Car-like vehicle, possibly a HMMWV; rejected rather than mislabeled as tracked.
  - start: "00:48"
    end: "01:52"
    decision: reject
    tag: null
    vehicle: null
    notes: Interview or non-target material.
  - start: "01:52"
    end: "02:32"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed M1 Abrams movement.
  - start: "02:32"
    end: "02:33"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "02:33"
    end: "02:37"
    decision: keep
    tag: accelerating
    vehicle: M1 Abrams
    notes: Reviewed M1 Abrams acceleration.
  - start: "02:37"
    end: "02:38"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "02:38"
    end: "02:45"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed M1 Abrams movement.
  - start: "02:45"
    end: "03:38"
    decision: reject
    tag: null
    vehicle: null
    notes: Interview or non-target material.
  - start: "03:38"
    end: "03:44"
    decision: keep
    tag: steady_speed
    vehicle: M2 Bradley
    notes: Reviewed steady-speed M2 Bradley movement.
  - start: "03:44"
    end: "03:45"
    decision: reject
    tag: null
    vehicle: null
    notes: Boundary gap.
  - start: "03:45"
    end: "03:53"
    decision: keep
    tag: steady_speed
    vehicle: M1 Abrams
    notes: Reviewed steady-speed M1 Abrams movement.
  - start: "03:53"
    end: "07:11.220"
    decision: reject
    tag: null
    vehicle: null
    notes: Interview or non-target material.
```

### BMP-1 — Trident Juncture 2018

- Source ID: `candidate-target-tracked-bmp1-trident-juncture-moveout-2018`
- Local audio: `data/targets/tracked/dvids_nato_trident_juncture_bmp1_elval_2018/bmp1_moveout_review.wav`
- Duration: `03:45.792`
- Video: [DVIDS source video](https://www.dvidshub.net/video/637281/trident-juncture-2018-polish-tank-and-fighting-vehicle-and-british-engineering-company-during-training-maneuvers-near-elval-norway)
- Corpus status: rejected; no approved interval

```yaml
segments:
  - start: "00:00"
    end: "03:45.792"
    decision: reject
    tag: null
    vehicle: BMP-1
    notes: Entire recording rejected after audiovisual review because the audio quality is poor.
```

### TR-85M1 Romanian tank-range recording

- Source ID: `target-tracked-tr85m1-tank-range`
- Local audio: `data/targets/tracked/dvids_romanian_tank_range_2014/tr85m1_tank_range.wav`
- Duration: `02:43.627`
- Video: [DVIDS source video](https://www.dvidshub.net/video/343966/romanian-tanks)
- Corpus status: blocked; no approved interval

```yaml
segments:
  - start: "00:00"
    end: "02:43.627"
    decision: reject
    tag: null
    vehicle: TR-85M1
    notes: Near-silence at the beginning, background or foreground speech, small-arms and main-gun-like reports around 01:10-01:59, and foreground speech after 02:00. No interval is currently approved.
```

### Mobile Protected Firepower testbed

- Source ID: `target-tracked-mpf-firepower`
- Local audio: `data/targets/tracked/us_army_armor_cavalry_collection_tankodrome_2023/mpf_testbed.wav`
- Duration: `01:38.048`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:U.S._Army_Armor_%26_Cavalry_Collection_Mobile_Protected_Firepower.webm)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "01:28"
    decision: keep
    tag: steady_speed
    vehicle: Mobile Protected Firepower testbed
    notes: Latest audiovisual annotation supersedes the older idle/mixed subdivisions.
  - start: "01:28"
    end: "01:38.048"
    decision: exclude
    tag: null
    vehicle: null
    notes: Outside the latest allowlist.
```

## Wheeled recordings

### M1151 Up-Armored HMMWV — driver training

- Source ID: `candidate-target-wheeled-hmmwv-m1151-training-2014`
- Local audio: `data/targets/wheeled/dvids_m1151_driver_training_jbmdl_2014_04_03/hmmwv_m1151_training.wav`
- Duration: `04:23.147`
- Video: [DVIDS source video](https://www.dvidshub.net/video/329602/humvee-training-b-roll)
- Corpus status: admitted using the reviewed segments below

Every interval not explicitly marked `keep` is rejected from corpus use.

```yaml
segments:
  - start: "00:00"
    end: "00:15"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "00:15"
    end: "00:36"
    decision: keep
    tag: idle
    vehicle: M1151 HMMWV
  - start: "00:36"
    end: "00:37"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "00:37"
    end: "00:47"
    decision: keep
    tag: steady_speed
    vehicle: M1151 HMMWV
  - start: "00:47"
    end: "00:50"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "00:50"
    end: "01:21"
    decision: keep
    tag: steady_speed
    vehicle: M1151 HMMWV
  - start: "01:21"
    end: "02:10"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "02:10"
    end: "02:17"
    decision: keep
    tag: accelerating
    vehicle: M1151 HMMWV
  - start: "02:17"
    end: "02:27"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "02:27"
    end: "02:30"
    decision: keep
    tag: accelerating
    vehicle: M1151 HMMWV
  - start: "02:30"
    end: "02:40"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "02:40"
    end: "02:47"
    decision: keep
    tag: steady_speed
    vehicle: M1151 HMMWV
  - start: "02:47"
    end: "02:50"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "02:50"
    end: "02:52"
    decision: keep
    tag: accelerating
    vehicle: M1151 HMMWV
  - start: "02:52"
    end: "04:10"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
  - start: "04:10"
    end: "04:15"
    decision: keep
    tag: idle
    vehicle: M1151 HMMWV
  - start: "04:15"
    end: "04:23.147"
    decision: reject
    tag: null
    vehicle: M1151 HMMWV
    notes: Outside the reviewed allowlist.
```

### M1126 Stryker — Pinon Canyon convoy

- Source ID: `candidate-target-wheeled-m1126-stryker-convoy-pinon-2024`
- Local audio: `data/targets/wheeled/dvids_m1126_pinon_canyon_convoy_2024_09_04/m1126_stryker_convoy.wav`
- Duration: `04:23.817`
- Video: [DVIDS source video](https://www.dvidshub.net/video/937004/pinon-canyon-maneuver-site-b-roll-package)
- Corpus status: admitted using the reviewed segments below

Multiple M1126 Strykers are audible in this convoy recording. They collectively
remain one recording session and are not independent datapoints. Every interval not
explicitly marked `keep` is rejected from corpus use.

```yaml
segments:
  - start: "00:00"
    end: "00:05"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "00:05"
    end: "00:16"
    decision: keep
    tag: unknown
    vehicle: M1126 Stryker convoy
    notes: Retained by audiovisual review; no operating-state label was supplied.
  - start: "00:16"
    end: "00:17"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "00:17"
    end: "00:35"
    decision: keep
    tag: idle
    vehicle: M1126 Stryker convoy
  - start: "00:35"
    end: "00:36"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "00:36"
    end: "00:58"
    decision: keep
    tag: steady_speed
    vehicle: M1126 Stryker convoy
  - start: "00:58"
    end: "01:02"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "01:02"
    end: "01:10"
    decision: keep
    tag: accelerating
    vehicle: M1126 Stryker convoy
  - start: "01:10"
    end: "01:12"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "01:12"
    end: "01:33"
    decision: keep
    tag: steady_speed
    vehicle: M1126 Stryker convoy
  - start: "01:33"
    end: "01:45"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "01:45"
    end: "01:50"
    decision: keep
    tag: steady_speed
    vehicle: M1126 Stryker convoy
  - start: "01:50"
    end: "02:33"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "02:33"
    end: "02:41"
    decision: keep
    tag: accelerating
    vehicle: M1126 Stryker convoy
  - start: "02:41"
    end: "02:57"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "02:57"
    end: "03:09"
    decision: keep
    tag: idle
    vehicle: M1126 Stryker convoy
  - start: "03:09"
    end: "03:15"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
  - start: "03:15"
    end: "03:53"
    decision: keep
    tag: steady_speed
    vehicle: M1126 Stryker convoy
  - start: "03:53"
    end: "04:23.817"
    decision: reject
    tag: null
    vehicle: M1126 Stryker convoy
    notes: Outside the reviewed allowlist.
```

### Abarth 205 Monza

- Source ID: `target-wheeled-abarth-205-goodwood`
- Local audio: `data/targets/wheeled/commons_goodwood_festival_of_speed_2009/abarth_205_monza.wav`
- Duration: `00:15.337`
- Original media: [audio-only Wikimedia page](https://commons.wikimedia.org/wiki/File:Abarth_205_Monza_(1950).ogg)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "00:08.800"
    decision: keep
    tag: accelerating
    vehicle: Abarth 205 Monza
    notes: Clean coming-and-going audio; source description supports acceleration.
  - start: "00:08.800"
    end: "00:15.337"
    decision: exclude
    tag: null
    vehicle: null
    notes: Quieter trailing ambience.
```

### Ford Model T

- Source ID: `target-wheeled-ford-model-t-start`
- Local audio: `data/targets/wheeled/commons_soundsofchanges_private_collection/ford_model_t_starting.wav`
- Duration: `00:38.267`
- Video: [Wikimedia source video](https://commons.wikimedia.org/wiki/File:Ford_Model_T_%22Tin_Lizzy%22_-_Starting_the_Engine.webm)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "00:06"
    decision: exclude
    tag: null
    vehicle: Ford Model T
    notes: Outside the latest allowlist.
  - start: "00:06"
    end: "00:14"
    decision: exclude
    tag: startup
    vehicle: Ford Model T
    notes: Reviewed engine-start interval; parked by operator on 2026-09-20, original review preserved.
  - start: "00:14"
    end: "00:15"
    decision: exclude
    tag: null
    vehicle: Ford Model T
    notes: Conservative boundary gap.
  - start: "00:15"
    end: "00:37.800"
    decision: keep
    tag: idle
    vehicle: Ford Model T
    notes: Reviewed post-start idle.
  - start: "00:37.800"
    end: "00:38.267"
    decision: exclude
    tag: null
    vehicle: null
    notes: End boundary.
```

### JLTV — Fort McCoy driver training

- Source ID: `candidate-target-wheeled-jltv-fort-mccoy-2019`
- Local audio: `data/targets/wheeled/dvids_fort_mccoy_jltv_precision_course_2019/jltv_driver_training.wav`
- Duration: `06:50.411`
- Video: [DVIDS source video](https://www.dvidshub.net/video/770767/jltv-drivers-training)
- Corpus status: reserved outside training; blocked pending review

```yaml
segments:
  - start: "00:00"
    end: "06:50.411"
    decision: unreviewed
    tag: null
    vehicle: JLTV
    notes: Identify vehicle-dominant intervals and reject instructors, crew speech, wind, other vehicles, repeated takes, edits, and camera-position changes.
```

### Maserati GranTurismo S

- Source ID: `target-wheeled-maserati-granturismo-exhaust`
- Local audio: `data/targets/wheeled/freesound_lmartins_maserati_granturismo_2019/maserati_start_rev_idle.wav`
- Duration: `01:06.175`
- Original media: [audio-only Wikimedia page](https://commons.wikimedia.org/wiki/File:Maserati_GranTurismo_S_Exhaust.ogg)
- Corpus status: admitted

```yaml
segments:
  - start: "00:00"
    end: "00:02.500"
    decision: exclude
    tag: startup
    vehicle: Maserati GranTurismo S
    notes: Reviewed engine-start transient; parked by operator on 2026-09-20, original review preserved.
  - start: "00:02.500"
    end: "00:14.500"
    decision: keep
    tag: idle
    vehicle: Maserati GranTurismo S
    notes: Low-amplitude steady engine interval.
  - start: "00:14.500"
    end: "00:15"
    decision: exclude
    tag: null
    vehicle: Maserati GranTurismo S
    notes: Conservative boundary gap.
  - start: "00:15"
    end: "01:01.500"
    decision: keep
    tag: mixed
    vehicle: Maserati GranTurismo S
    notes: Frequent high-rev cycles. Speed and acceleration are unavailable because this source is audio-only.
  - start: "01:01.500"
    end: "01:06"
    decision: keep
    tag: idle
    vehicle: Maserati GranTurismo S
    notes: Final low-amplitude engine-running interval.
  - start: "01:06"
    end: "01:06.175"
    decision: exclude
    tag: null
    vehicle: null
    notes: End boundary.
```

### Unspecified car — starting and driving

- Source ID: `candidate-target-wheeled-car-start-drive-pdsounds-194`
- Local audio: `data/targets/wheeled/pdsounds_194_stephan_car_start_drive_2007_04_26/starting_car_and_driving.wav`
- Duration: `00:44.852`
- Original media: [audio-only Wikimedia page](https://commons.wikimedia.org/wiki/File:Starting_a_car_and_driving.ogg)
- Corpus status: consumed by the completed locked evaluation; exclude from future model development

```yaml
segments:
  - start: "00:00"
    end: "00:14.500"
    decision: exclude
    tag: null
    vehicle: null
    notes: Outside the current allowlist.
  - start: "00:14.500"
    end: "00:17.200"
    decision: keep
    tag: startup
    vehicle: unspecified car
    notes: Reviewed engine-start interval.
  - start: "00:17.200"
    end: "00:20.500"
    decision: keep
    tag: idle
    vehicle: unspecified car
    notes: Reviewed post-start idle.
  - start: "00:20.500"
    end: "00:41.500"
    decision: keep
    tag: mixed
    vehicle: unspecified car
    notes: Vehicle-dominant driving interval. Exact state cannot be inferred from this audio-only source.
  - start: "00:41.500"
    end: "00:44.852"
    decision: exclude
    tag: null
    vehicle: null
    notes: Outside the current allowlist.
```

## Handoff checklist

Before asking Codex to apply the worksheet:

1. Replace every relevant `unreviewed` decision with `keep`, `reject`, or `exclude`.
2. Give every `keep` segment one allowed operating-condition tag.
3. Ensure segments do not overlap.
4. Identify the visible vehicle for mixed-source videos whenever possible.
5. Note speech, wind, music, weapons, unrelated vehicles, silence, or edit boundaries explicitly.
6. Leave uncertain but usable vehicle audio as `tag: unknown`; do not guess RPM, speed, or acceleration.
