---
title: 6.551J Microphones and Transducers
tags:
  - abvid
  - mit
  - acoustics
status: active
---

# 05 — Microphones & Transducers (L10, L11)

## The transducer chain

A microphone is an **electro-mechano-acoustic** transducer: sound pressure → diaphragm
force/motion → electrical signal.

```mermaid
flowchart LR
    A[Sound pressure] -->|acoustic stage| B[Diaphragm motion]
    B -->|mechanical stage| C[Force / velocity]
    C -->|electrical stage| D[Voltage / current]
```

## Impedance & transformers

- Each stage is connected by an **ideal transformer** that trades one variable for
  another while conserving power:
  - **Electrical**: turns ratio $T$ maps voltage↔current.
  - **Mechanical**: lever arm maps force↔velocity.
  - **Acoustic**: coupled pistons map pressure↔volume velocity; ratio = area ratio.
  - **Impedance transforms as $T^2$**: a big area ratio (e.g. middle ear 20:1) gives a
    400:1 impedance transformation.

> Why this matters: every stage loads the previous one. The total frequency response of
> a mic = how acoustic, mechanical, and electrical impedances combine.

## The capacitive (electrostatic) microphone

- A charged capacitor with a fixed backplate and a moving diaphragm.
- Sound moves the diaphragm → changes capacitance → changes voltage/current.
- **Low-frequency model**: the diaphragm's mechanical compliance $C_M$ in series with
  the acoustic source; the static electrical capacitance $C_E$ in parallel.
- Stiff (low-compliance) membranes + small (high-impedance) capacitance = the "standard
  microphone" design (expensive reference mics).

## Thevenin view

- The microphone seen from its electrical port = a voltage source + output impedance.
- The output impedance depends on the acoustic and mechanical stages through the
  transformer — so the *load* (preamp) interacts with the mic's response.
- **Reciprocity** (Rayleigh): in a fixed environment, a source at position 1 producing
  pressure at 2 is equivalent to the same source at 2 producing the same pressure at 1.
  This is why you can calibrate a mic by treating it as a source.

## Frequency response & coloration

- A mic's **frequency response** is not flat — its mechanical/acoustic resonances and
  impedance matching shape the output vs frequency.
- **Directionality** (omni vs cardioid etc.) comes from the geometry / phase combination
  of the diaphragm's front and back — the same array-directivity idea as
  [[03 - Sources and Arrays]].
- This is why the augmentation pipeline includes a **microphone frequency response**
  stage and why each mic in M4 can have a different response.

> Q: Why do different microphones "color" the same sound differently?
> A: Each is a different LTI system — different diaphragm compliance, capacitance, and
> resonance — so each has a different frequency response $H(\omega)$. It's the filter
> idea from 6.003 applied to a physical device.

## The loudspeaker (reverse transducer)

- **Electrostatic**: voltage → force → diaphragm → sound (the mirror of the mic).
- **Electrodynamic**: voice coil in a magnetic field; current → force → cone.
- The speaker's **enclosure** makes the cone act as a monopole source (and adds the
  box's compliance); bass-reflex ports modify it.

## Where in ABVID

- M1's microphone-response stage and M4's per-mic frequency responses are the direct
  implementation of "each transducer has an $H(\omega)$".
- The augmentation engine's `configs/default.yaml` mic-response ranges (-6 to +3 dB)
  randomize coloration across mics.
- See [[01 - Signal Fundamentals]] and
  `src/vehicle_audio/augment.py`.
