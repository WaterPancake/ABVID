---
title: Equation Reference
tags:
  - abvid
  - reference
status: active
---

# Equation Reference

## Decibels & SNR

**Amplitude ratio (dB):**
$$G_{dB} = 20 \log_{10}\frac{A_2}{A_1}$$

**Power SNR (dB):**
$$SNR_{dB} = 10 \log_{10}\frac{P_{signal}}{P_{noise}}$$

**Controlled-SNR noise scaling** (to hit requested $R$ dB):
$$\alpha = \sqrt{\frac{P_s}{P_n \cdot 10^{R/10}}} \qquad y = x + \alpha n$$

## Sampling

**Nyquist:**
$$f_{Nyquist} = \frac{f_s}{2}$$

## Convolution

$$y[n] = \sum_k x[k]\, h[n-k]$$

## Acoustics

**Spherical free-field:** pressure $p \propto 1/r$, intensity $I \propto 1/r^2$.

## Order analysis

**Rotational / engine order:**
$$f_{order} = \frac{RPM}{60} \times order$$

## Array / localization

**Azimuth unit vector** (broadside: 0° = +y, toward +x):
$$u = (\sin\theta,\ \cos\theta,\ 0)$$

**Far-field arrival delay** at mic $i$ relative to mic 0:
$$\tau_i = -\frac{p_i \cdot u}{c}, \qquad c = 343\ \text{m/s}$$

**GCC-PHAT:**
$$\hat{R}(\tau) = \int \frac{X_1(\omega) X_2^*(\omega)}{|X_1(\omega) X_2^*(\omega)|} e^{j\omega\tau}\,d\omega, \qquad \hat\tau = \arg\max_\tau \hat R(\tau)$$

## Metrics

**Precision / recall / F1:**
$$P = \frac{TP}{TP+FP} \qquad R = \frac{TP}{TP+FN} \qquad F_1 = \frac{2PR}{P+R}$$

**Balanced accuracy (2 classes):**
$$BA = \frac{1}{2}\left(\text{recall}_{tracked} + \text{recall}_{wheeled}\right)$$

**Cross-entropy:**
$$L = -\log p(y_{correct} \mid x)$$

## Invariance

**Triplet margin (cosine distance):**
$$L_{triplet} = \max\left(0,\ d(a,p) - d(a,n) + m\right)$$

**Total invariance objective:**
$$L = L_{CE} + \lambda_{cons} \cdot \text{cosine}(z_{clean}, z_{corrupted}) + \lambda_{hp} \cdot L_{hard\_pos} + \lambda_{hn} \cdot L_{triplet}$$

with (in the checked run) $\lambda_{cons}=0.5$, $\lambda_{hp}=0.1$, $\lambda_{hn}=0.1$, $m=0.2$.

## Generative model

$$x = g(v,\ s,\ e,\ m,\ n)$$

$v$ vehicle · $s$ state · $e$ environment · $m$ microphone · $n$ noise
