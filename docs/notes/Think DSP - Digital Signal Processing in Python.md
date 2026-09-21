# Chpt. 1: Sounds and Signals
- **Signal** a quantity that varies over time (e.g. "sound" is just the varying of air pressure).
- **Transducers** are devices that transduce (convert) signals from one form to another (e.g. microphones or speaker).
- **Periodic signals** are a specific class of signals that repeat themselves over a period of time e.g. a bell's vibration which creates the ring repeats itself)
	- **Sinusoid** are periodic signals that represent the trigonometric sine function.
	- The period which the signal repeats itself is called it **cycle** and occur of a set duration of time called its **period**.
	- The shape of a periodic signal is called it **waveform**. The shape determine **timbre** i.e. our perception of the quality of the sound
- Frequency is measure in "per second" unit called **Herz** or **Hz** (the standard turning *A* is tuned to 440 Hz)
- **Spectral decomposition** is the idea that any signal can be expressed as the sum of sinusoids with different frequencies.
	- The **discrete Fourier transform (DFT)** is an important mathematical idea that transform a signal into a **spectrum**, the set of sinusoids that add up to produce the signal.
		- The **Fast Fourier transform (FFT)** is an efficient way of competing the DFT.
	- A spectrum can be visualized by plotting the range of sinusoids frequencies in Hz on the x-axis and their **amplitude** or strength of the signal on the y-axis.
	- The lowest frequency or **fundamental frequency** (often) has the largest amplitude and so is (often) also called the **dominant frequency** 
		- We often perceive pitch by the fundamental frequency even if its not the dominate frequency 
	- **Harmonics** signals who's frequency are integer multiple of of the fundamental.
		- An **octave** is a doubling in frequency.
- **Phase offset** determines when in a period of time the signal starts. Often using radians as the unit of measurement (because of sinusoids).
- Every discrete(???) point in time is referred too as a **Frame** (like in movies), sometimes may be called a **Sample** (though this often refers to the measurement at the point in time in question). The more samples or frames over a given interval, the "higher quality" the sound.
	- **Time step** is the time between frames.
# Chpt. 2: Harmonics 
- The set of sinusoids that make up a spectrum is understood as its **harmonic structure** or the relationships between the amplitude and frequencies in the spectrum.
	- e.g. say the first two harmonics 200 and 600 Hz, have a ratio of 3, and their respective amplitude ration of 9 (e.g 100 and 900),
- Types of waves:
	- Triangle waves are look like a series of straight-lines between a peak and valley (like a v or w). Also harmonics of a triangle wave are all odd multiples of the fundamental.
	- Square waves, as the name suggest, are 
- **Aliasing** is an important phenomena in digital signal processing where high frequency signals can appear as low(er) frequency signals.
	Consider a triangle wave with a foundation of 1100 Hz sampled at 10,000 frames. The waves harmonics occur at integer multiple of the foundation (3300, 5500, 7700, etc.), but the spectrum reveals peaks, some at the expected frequencies, but also at strange multiples like 4500 and 2300 Hz.
	This occurs because in sampling at discrete time points, we can lose information between the samples. With a frequency of  5000 Hz, sampling at 10,000 frames per second, you only have 2 samples per period. This causes the signal at higher frequency, say 5500 Hz sampled at same sample rate to be indistinguishable from a 4500 Hz signal.
	- We "fold back" (modulate) sample above 5000 Hz, we also fold frequencies below zero to their positive 
	- **Nyquist frequency** or the **folding frequency** is the frequency that if the frequency goes below. e.g. the 5th harmonic of a 1100 Hz triangle would be 12,100 Hz, folded it becomes -2100 Hz, folded up too 2100 Hz (the next would be 14300 Hz $\to$ -4300 Hz $\to$ 4300 Hz)
- **Fast Fourier Transform** (FFT) algorithm calculates the **Discrete Fourier Transform** (DFT),
	- Aside: consider two ways to represent complex numbers:
		1. As the sum of the real and imaginary, written $x + iy$ where $i$ is the imaginary unit $\sqrt{-1}$. This can be visualized as the $x,y$ coordinate system 
		2. As the product of a magnitude and a complex exponential $Ae^{i\phi}$, where $A$ is the *magnitude* and $\phi$ in *angle* (in radians).
# Chpt. 3: Non-periodic signals
- A **Chirp** is a signal with a variable frequency (that may change over time) that causes the sinusoid to sweep (e.g. linearly or exponential) through a range of frequency.
	- Any function (e.g. exponential) can be used to determine the relationship between the variable and the frequency.
	- The change in the phase ($\phi$) of a chirp can be expressed the period of a typical sinusoid ($2\pi$). With a constant frequency ($f$) the phase increase linearly over time ($t$) as 
		$\phi =2\pi f t$
	- When frequency changes over time, we instead look for the *change in phase* ($\Delta \phi$) over a short time interval ($\Delta t$) as  
		$\Delta \phi = 2 \pi f(t) \Delta t$  
	 Note that the limit as $\Delta t$ approaches zero, the equation becomes:
		$d|phi =2\pi f(t)dt \quad$, which can yield, $\quad \frac{d\phi}{dt} = 2\pi f(t)$
	This means that *Frequency is the derivative of phase* and inversely, *phase is the integral of frequency*.
- **Interval** refers to the perceived difference between two pitches as a ratio (e.g. an octave has a ratio of 2, a doubling of frequency, e.g. 220 and 440 are both the "A" note). 
	- Our perception of pitch depends on the logarithm of the frequency (220 $\to$ 440 and 440 $\to$ 880 have a ratio of 2, but difference is greater).
	- If frequency increase linearly, the perceived pitch increase is logarithmically. 
		- This is why we may use an exponential Chirp instead of a linear one 
	
- **Short-time Fourier Transform** (STFT) is a way to visualize the **spectrogram** of Chirp.
	- The Spectrogram of a Chirp is harder to understand visually. It may give hints to the structure of the signal, but the relationship between frequency and time is obscure.
	- **Spectrogram** differs from a spectrum by plotting time (x-axis) against frequency (y-axis) using a color to represent amplitude. 
		- **Time Resolution** is the temporal duration of the segments corresponding to a cell in the spectrogram (e.g. if there are 11,025 frames per second, each segment might be 512, thus have a resolution of 0.046 seconds, ) 
			- Generally, given $n$ segment lengths, the spectrum contains $n/2$ components. If the frame rate is $r$, the maximum frequency of the spectrum is $r/2$. Thus the *time resolution* is $n/r$ with a *frequency resolution i*s $\frac{r/2}{n/2} = r/n$
- **Gabor limit** is the trade off between increasing segment length (which cuts frequency resolution, good), but doubles time resolutions (bad). Note that the two resolutions are the inverse of each other.
- **Windowing** involves creating a function that transforms a non-periodic segment into something about periodic. 
	- Common one is the *Hamming window* 

# Chpt. 4: Noise
- **Power** is the square of amplitude. This chapter primarily uses this quantity. It is not interchangeable with amplitude
- **Noise** in DSP refers too:
	1. Unwanted signal of any kind. If two signals interface with each other, each signal would consider the other to be the noise.
	2. A signal that contains components at many frequencies do, but lack harmonic structure of a periodic signals. *we care about this kind in this section.*
- Types of noise
	- **Uncorrelated Uniform (UU) Noise**. *Uniform* meaning the signal contains random values form a uniform distribution. *Uncorrelated* meaning that the values are independent across time.
	- **Gaussian Noise** is a signal woes value are that of a Gaussian distribution $\mathcal{N}(0,1)$
	- **Brownian Noise** is generated by summing the previous two values in a signal and adding a random "step." 
		- **Random Walk** a mathematical model of a path where the distance between steps is charactered by a random distribution.
		- **Red Noise** is another name for Brownian noise, for when combined with 
	- **Pink Noise** the power is inversely related to frequency. The power at frequency $f$ is drawn from a distribution with mean $1/f$
	- **White Noise** noise with equal power at all frequencies. 
		- Called white noise as it analogous with light, an equal mixture across all frequencies (visual and noise frequency).
- Aspects of noise
	- *Distribution* or the range of possible random signal values and their respective probability. (e.g. Guassian)
	- *Correlation* the values may be correlated or in some cases dependent on previous values. (e.g. Brownian)
	- *Relationship between power and frequency*  (e.g. UU or pink)
- **Integrated spectrum** is a function of frequency and the cumulative amplitude in the spectrum up too $f$. This makes the relationship between power and frequency easy to visualize. Sometimes the plot yses log-log scale.
	- e,g, If the Integrated spectrum is a straight line, then the power at all the frequencies is, on average, constant,
	- Sometimes a visual is hard to understand without changing the scale of the axies, for instance, if most of the power of the signal is in lower frequencies. (it the peak and fall is almost exactly along the y-axis).
- 

# Chpt. 7: Discrete Fourier Transform
- The **Gamma Function** is a generalization of the factorial operator to non-integers defined as a power series: $e^{\phi} = 1 + \phi + \frac{\phi^2}{2!} + \frac{\phi^3}{3!} + \cdots$  
	- The function also works for (pure) imaginary numbers $i\phi$ as $e^{i\phi} = 1 + i\phi + \frac{i\phi^2}{2!} + \frac{i\phi^3}{3!} + \cdots$ which can be shown to be equivalent to $e^{i\phi} = \cos(\phi) + i \sin(\phi)$.
		- One way to visualize this is to imagine a point in the complex lain that is alwasy on the unit circle. 
		- Another way is a a vector where with respect to the x-axis is the argument $\phi$
- The **Complex Plane** as an extension of the Cartesian plane  (the real field),  ???? FILL OUT LATER ???

* **Complex signals**
 
# Chpt 8: Filtering and Convolution 
# Chpt 10: LTI system
# Chpt 11: Modulation and sampling
