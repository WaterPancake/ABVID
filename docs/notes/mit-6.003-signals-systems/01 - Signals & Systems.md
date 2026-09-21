---
title: 6.003 Signals and Systems
tags:
  - abvid
  - mit
  - signals-systems
status: active
---

# 01 — Signals & Systems

## Signals

- **Signal** = a function of one or more variables carrying information (audio = pressure varying in time).
- **Continuous-time (CT)** $x(t)$ vs **Discrete-time (DT)** $x[n]$ (indexed by integer sample).
- Audio in code is always **DT**: `x[n]`, with sample rate $f_s$ connecting $n$ to time $t = n/f_s$.

## Systems

- **System** = a mapping from input signal to output signal: $y = T\{x\}$.
- Important system properties:
  - **Linearity** — $T\{a x_1 + b x_2\} = a T\{x_1\} + b T\{x_2\}$
  - **Time invariance** — shifting input shifts output the same way
  - **Causality** — output depends only on present/past inputs
  - **Stability** — bounded input → bounded output (BIBO)

## Difference equations (DT)

A DT system is often written as a **difference equation**:

$$y[n] + a_1 y[n-1] = b_0 x[n] + b_1 x[n-1]$$

- The output depends on past outputs (feedback/recursive) and current/past inputs.
- Solvable by iterating forward in time — exactly how a digital filter runs.

## Block diagrams & modularity

```mermaid
flowchart LR
    x --> op[+ / − / × / delay]
    op --> y
    y -.feedback.-> op
```

- Systems compose: cascade, parallel, feedback.
- **Modularity**: build big systems from small LTI blocks — the design principle behind
  the augmentation pipeline's composable stages.

## The perfect sine wave

The building block of everything: $x(t) = A\cos(\omega t + \phi)$

- $A$ = amplitude, $\omega = 2\pi f$ = radian frequency, $\phi$ = phase.
- Period $T = 1/f$.
- Euler's identity ties sinusoids to complex exponentials:
  $$e^{j\theta} = \cos\theta + j\sin\theta$$
  so $A\cos(\omega t + \phi) = \mathrm{Re}\{A e^{j\phi} e^{j\omega t}\}$.

> Key idea: complex exponentials are **eigenfunctions of LTI systems** — an LTI system
> just scales and phase-shifts a complex exponential. That's why Fourier analysis is
> the natural tool for LTI systems (see [[03 - Fourier Analysis]]).

## Where in ABVID

- Waveforms in `src/vehicle_audio/audio.py` are DT tensors `[channels, samples]`.
- The pipeline is a **cascade of systems** — each augmentation stage is a block
  transforming the signal (see `src/vehicle_audio/augment.py`).
- The sine-wave/eigenfunction idea underlies harmonic order analysis for engine RPM
  (see [[MIT 6.551J Acoustics of Speech and Hearing]] and [[03 - Vehicle Acoustics & Synthetic Data]]).
