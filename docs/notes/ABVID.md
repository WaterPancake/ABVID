> Maybe rename AVID (Acoustic Vehicle IDentification)
# Learning Goals
- Terms: sample rate, filtering, convolution, controlled-SNR
- Log-MEL spectrogram and their use in vehicle classification
- Metrics: MFCC, spectral centroid, bandwidth, rolloff 
- CNN usage for time-frequency structure
- recording-session leakage invalidation in evaluation
- Accuracy against SNR and test domain
- How Engine RPM produces harmonic orders
- Unseen test is different from random test split
- The possibilities and limitations of domain randomization
- Using real and synthetic data.

# Study notes
- [[README]] — condensed ABVID study guide (cheat-sheet of all milestones)
- [[Think DSP - Digital Signal Processing in Python]] — waveforms and spectrum (Ch 1-4)
- [[MIT 6.003 Signals and Systems]] — signals, LTI, Fourier, sampling, filters
- [[MIT 6.551J Acoustics of Speech and Hearing]] — wave physics, dB, arrays, mics

# Section 1 
[[Think DSP - Digital Signal Processing in Python]] (Waveforms and Spectrum)

## Equations of note
- Amplitude Gain
$$
G_{dB} = 20\cdot\log_{10} \left(\frac{A_{out}}{A_{in}}\right)
$$
- Power Signal-to-Noise  (SNR) 
$$
\operatorname{SNR}_{dB}=10\cdot\log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)
$$
	- Q: What does a 0 dB SNR mean and why does it not mean that the waveform is silent?) 
		A: 
	- We log log decibel-scale as many signals have a very wide dynamic range
	- SNR Compares the level of a desired signal and background noise. Higher ratio, beyond 1:1 corespond to 

- Nyquist frequency
- 
$$
f_{\text{Nyquist}}=\frac{f_s}{2}
$$