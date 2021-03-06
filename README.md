---
head: https://hackmd.io/r-CG8R7xTZ2S3pRa6JD9FA
---

$$
\newcommand{aa}{\alpha}
\newcommand{bb}{\beta}
\newcommand{cc}{\gamma}
\newcommand{dd}{\delta}
\newcommand{DD}{\Delta}
% can extend to overview
\newcommand{\inv}[1]{{#1}^{-1}}
% this note
\newcommand{Dt}{\Delta T}
\newcommand{uu}{\mu}
\newcommand{zz}{\zeta}
\newcommand{FT}{\mathfrak{F}}
\newcommand{st}{s_{\Delta T}}
\DeclareMathOperator{sinc}{sinc}
$$

# Digital Image Processing

[TOC]

## Policies

## Textbook
- [DIP 4e](https://www.codecool.ir/extra/2020816204611411Digital.Image.Processing.4th.Edition.www.EBooksWorld.ir.pdf)

---

## HW
- [Digital Image Processing, HW1](/YzFUi19yQIW5hRx4RxS4Iw)

---

## Chap 3. Intensity Transformation, Spatial Filtering

## Chap 3-1. Histogram Processing

### Histogram Equalization
- Intensity Domain Transformation 
    - map $r_k$ to $s_k$ both in $[0, L)$
- $T : [0, L) \to [0, L)$
    - origial
        - (a) $T$ monotone increasing
            - to avoid making **artifact**s
        - (b) $\text{range}(T) \subset [0.L)$
            - to make sure the intensity range is the same
    - to make $T$ invertible
        - (a') $T$ strictly monotone increasing
        - discrete case can see example 3.7 
            - It can be shown (see Problem 3.9) that this inverse transformation satisfies conditions  (a') and (b) defined earlier only if all intensity levels are present in the input image
- Formula
    - $s = T(r) = (L-1)\int\limits_{0}^{r}p_r(w) dw$
    - this results in $p_s(s) = p_r(r) |\frac{dr}{ds}| = (L-1)p_r(r)$

### Histogram Matching (TBA)

---

## Chap 3-2. Spatial Convolution(linear spatial filtering)
- discrete calculation and padding (1D)
    - $(m-1)/2$ where m is the size of kernel
        - standard
    - full version len is $n+m-1$
    - standard version len is $n$
- motivation of convolution is associativity, commutativity, and identity (discrete unit impulse)

### 2D Convolution
- convolution
    - $w {\star}f(x,y) = \sum_{s,t}w(s,t)f(x-s,y-t)$
        - think of w as filter
- computational advantage of performing convolution with a separable, as opposed to a nonseparable, kernel is defined as
- convolution in the spatial domain ~= multiplication in the frequency domain

----

### Lowpass Filters (Smoothing)
- lowpass means smoothing
    - reduce irrelevant detail in an image

#### Box Filter

#### Lowpass Gaussian Filter

----

### Highpass Filters (Sharpening)

#### how
- second order diff
- isotropic

#### Laplacian filter
- $\nabla^2 f= \frac {\partial^2 f} {\partial x ^2} + \frac {\partial^2 f} {\partial y ^2}$
- $\nabla^2 f(x,y) = \sum_{dx,dy}f(x+dx, y+dy)-4f(x,y)$
- $g(x,y)=f(x,y) - \nabla^2 f(x,y)$

#### UNSHARP MASKING
- substract an unsharped version of the original 
    - Blurred = blur(Origin)
    - Mask = Origin - Blurred
    - New = Origin + Mask
- in math
    - $g_{mask}=f-\bar{f}$
    - $g=f+kg_{mask}$
        - When k = 1 we have unsharp masking
        - When k > 1, the process is referred to as highboost filtering.

#### Gradient and Sharpenning filter
- $M=\|\nabla f\|= f_x^2+f_y^2$
    - approx with $\|f_x\|+\|f_y\|$
        - $f_x=z_9-z_5, f_y=z_8-z_6$
        - $f_x=z_9-z_5, f_y=z_8-z_6$
- Sobel and Gradient
http://fourier.eng.hmc.edu/e161/lectures/gradient/node1.html

---

## Chap 4. Frequency Domain 
- define $\sinc(x)=\frac{\sin (\pi x)}{x}$

----

### Fourier Transform
- denote as $\FT$ in this note (latex use `$\FT$`)
- $f(t) \overset{\FT}{\to} F(\uu)$
	- $F(\uu)=\int_{-\infty}^{\infty}f(t)e^{-2\pi i\uu t}dt$
- inverse
	- $f(t)=\int_{-\infty}^{\infty}F(\uu)e^{2\pi i\uu t}d\uu$
		- Fourier expansion!
- periodoic with cycle $T$
	- $F(\uu)=\frac 1 T\int_{-\frac T 2}^{\frac T 2}f(t)e^{-2\pi i\uu t}dt$

#### Example: Box Filter
![](https://i.imgur.com/LPQKYmz.png =50%x)


----

### Sampling

#### the Delta Funciton
- $\delta(t)$
- properties
	- $\int\delta(t)f(t)dt=f(0)$
- $\FT \dd(t-t_0) = F(\uu) = \int \dd(t-t_0)e^{-2\pi i \uu t}dt=e^{-2\pi i \uu t_0}$

#### Impulse Train
* $s_{\DD T}=\sum\limits_{n=-\infty}^{\infty}\delta(t-n\DD T)$
* Fourier
	* $S(\uu)=\FT(s_{\DD T}(t))\overset{\text{Fourier Series}}{=}\FT(\frac 1 {\DD T} \sum_n e^{2\pi i t\frac{n}{\Delta T}})=\frac{1}{\DD T}\sum\dd(t-\frac{n}{\DD T})$

#### Sampling as Multiplication with Impulse Train
* as $f\cdot p$

#### The Convolution Theorem
- $\FT(f\star g)=\FT(f) \cdot \FT(g)$
- $\FT(f\cdot g)=\FT(f) \star \FT(g)$

#### The Sampling Theorem and Nyquist Frequency
![](https://i.imgur.com/bbgpR08.png)
- statement
	- $\frac{1}{\DD T} \gt 2\uu_{max} = \text{the nyquist frequency}$
		- exact can cause problem (e.g. $\sin$)
- proof by show the equations and figures
	- calculate $\FT(f(t)\cdot s_{\DD T}(t))=F(\uu)\star S_{\DD T}(\uu)$

#### Aliasing
- loss of the ability to recover the original signal
- https://www.youtube.com/watch?v=OFIQwsodJrg&list=PLdciPPorsHukc4z-Jr1-D8d4fQNHiHZZf&index=6&ab_channel=AdamPanagos
- under sampling
- Thm: **no function of finite duration can be band-limited**
	- proof by
		- box fiter is of infinite bandwidth
		- convolution theorem ($\FT(f\cdot g)=\FT(f) \star \FT(g)$)
- anti-aliasing
	- try to "reduce influence of high-frequency signal" "before" sampling
	- $\sinc$
	- blurring can achieve similar result but is not anti-aliasing by definition

#### Signal Recovery
\begin{equation}
f(t)=\inv\FT [F(\uu)]\inv\FT [H_{box}(\uu)\tilde{F}(\uu)] =h(t) \star \tilde{f}(t) \\
f(t) =\sum_n f(n\DD T) \sinc(\frac{t-n\DD T}{n\DD T})
\end{equation}
- note
	- $\sinc(0)=1, \sinc(n)=0~\forall n \in \mathbb{N}$
	- for sampled point, true
	- others are interpolated results of infinity number of $\sinc$s


(p.227 still have to go to p.310 ...)
(3:22 finish 310, fill in DFT motivation )
(must test: Histogram Eq + DFT)

----

### DFT
\begin{equation}
F(\uu)=\sum_{x=0}^{M-1}f(x) e^{-i2\pi\frac{\uu x}{M}} \\
f(x)=\sum_{\uu=0}^{M-1}F(\uu) e^{i2\pi\frac{\uu x}{M}}
\end{equation}

#### Motivation
- The sampled version of transform
\begin{equation}
\tilde{F}(\uu)=\int_{-\infty}^{\infty}\tilde f(t)e^{-2\pi i\uu t}dt 
\end{equation}
- then we get
\begin{equation}
\tilde{F}(\uu)=\int_{-\infty}^{\infty}\tilde f(t)e^{-2\pi i\uu t}dt \\
= \int \sum_n f(t)\dd(t-n\DD T)e^{-2\pi i\uu t}dt \\
=\sum_n f(n\Dt)e^{-2\pi i\uu \Dt}
\end{equation} 
- want sample $M$ points in $[0,\frac{1}{\Dt}]$
	- $\DD\uu = \frac{1}{M \Dt}$
- $F_m=\sum_{n=0}^{M-1}f_n e^{-i2\pi\frac{mn}{M}}$

#### Relation of spacing between domains
- $M$ samples with time spacing $\Dt$
	- frequency spacing = $\DD\uu=\frac{1}{M\Dt}$
	- frequency range = $R=\frac{1}{\Dt}$

----

### 2D FT

### 2D aliasing

#### Image Resampling and Interpolation
- zoom and shrink ~= undersampling and oversampling
- shrink can worsen aliasing
	- The digital “equivalent” of the defocusing of continuous images mentioned earlier for reducing aliasing, is to attenuate the high frequencies of a digital image by smoothing it with a lowpass filter before resampling. 

#### Moiré effect
- artifactural frequency by two wave?

----

### 2D DFT

----

### DFT properties
- tranlation
- rotation
- perioadic
- special case of transform 
	- $f(x)-1^{x} \iff F(\uu - \frac M2)$
- symmetric
	- key: real/img, even/odd
	- decomposition into even and odd
	- real function
		- conjudate symmetric

----

### Spectrum and Phase
- real f so
	- $|F(u,v)|=|F(-u,-v)|$
	- $\phi(u,v)=-\phi(-u,-v)$
- $F(0,0)=MN\bar{f}$ so is bright
	- index = $[0,M)$
- to do centering
	- multiply $f$ by $(-1)^{x+y}$
- to show detail
	- log transform to make bright the details

#### Relation to image
- phase ~= shape
- spectrum ~= intensity 

----

### Summary of 2D properties
- Table 4.4

----

### Frequency Domain Filtering
- $g(x,y)=Real(\FT^{-1}(H(u,v)\cdot F(u,v)))$

#### Filters
- Low/High Pass Filters (with offsets)
	- Low Pass (Band Pass)
		- $D_0$s are cut-off frequency
		- ideal
			- $H(u,v)=\mathbb{I}_{D(u,v)\le D_0}$
		- Gaussian
			- $e^{-D(u,v)^2/2D_0^2}$
		- Butterworth of order $n$
			- $H(u,v) = \frac{1}{1+(D(u,v)/D_0)^{2n}}$
	- High Pass (Band Reject)
		- Hight From Low
			- $H_{HP}=1-H_{LP}$
			- and $\FT^{-1}H_{LP}=\dd(x,y)-h_{LP}(x,y)$
		- Laplacian
			- use differential
			- $H(u,v)=-4\pi^2D^2(u,v)$
			- $g(x,y)=\FT^{-1}((1+H(u,v))F(u,v)))$
	- Unsharp Masking, High-Boost Filtering, High-Frequency Emphasizing
		- linearity
		- $g(x,y)=\FT^{-1}((1+kH_{HP}(u,v))F(u,v)))$
- HOMOMORPHIC FILTERING and illumination model
	- $f=i \cdot r$
	- define $z= \ln f = \ln i + \ln r$
		- can use linearity
-  notch filtering
	-  $H_{NR}=\prod_{k=1}^Q H_{-k} H_{k}$
		-  for $k$ notchs, $H_{k}=H_{HPF}(u_k,v_k)$

#### Padding
- zero-, mirror-, or replicate padding

----

### FFT (2D)
- product + 1D FFT

https://dsp.stackexchange.com/questions/20061/removing-periodic-noise-from-image-using-fourier-transform

---

## Chapter 5. Image Restoration and Reconstruction

### Degradation and Restoration
![](https://i.imgur.com/6wrsgce.png)

- $f$ + degradation $h$ and noise $\eta$
    - $g(x,y) = (h\star f)(x,y)+\eta(x,y)$
    - Fourier basis
        - $G = HF+N$

### PDF of Noises
- Gaussian
- Rayleigh
- Gamma
- Exponential
- Uniform
- Salt-and-Pepper

### Removal of Noise Only

#### General => Spatial
- Mean
    - Arithmetic
    - Geometric
    - Harmonic
    - Contraharmonic
- Ordered Statistics
    - Mediam
    - MinMax
    - Midpoint
    - Alpha-Trimmed Mean
- Adaptive

#### Periodic => Frequency
- Notch
    - 3 types 
- Optinum notch

### 5.5 LINEAR, POSITION-INVARIANT DEGRADATIONS
- ESTIMATION BY IMAGE OBSERVATION
    - use high contrast part to guess H by dividing out f
- ESTIMATION BY EXPERIMENTATION
- ESTIMATION BY MODELING

### Restoration Methods (Technical)
- Inverse Filtering
    - $\hat F = G/H$
    - may need to cut close to zero value
        - or $\hat F = F + N/H$ and $N/H$ dominates $F$
- MMSE(Wiener) Filtering
    - Min MSE
        - $\hat{f} = \arg \min _{\hat f} E[(f-\hat{f})^2]$
    - $\hat{F} = \frac{1}{H}\frac{|H|^2}{|H|^2+K}G$
        - $K$ is a constant function for approximating $SNR = \frac {S_\eta}{S_f} = \frac{|N|^2}{|F|^2}$
- Constrained Wiener
- Geometric Mean
- From Projection

![Uploading file..._1iyx0dic6]()


---

## Chapter 6. Color Image Processing