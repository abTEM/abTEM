#!/usr/bin/env python
# coding: utf-8

# # Deriving the Multislice Algorithm

# This is a short description on the theory of the multislice method. For a more complete, formal discussion including detailed theory see Advanced Computing in Electron Microscopy by EJ Kirkland.
# 
# The energy of the incident electron waves (100 - 1000 keV) are much higher than the specimen potential, which provides only minor perturbations on the forward motion of the electrons. Hence, it is useful to write the wave function, $\psi$, of the propagating electrons as a slowly varying plane wave along the optical axis, $z$, with an amplitude modulation
# 
# $$
#     \psi(\vec{r}) = \phi(\vec{r})\exp(2\pi iz/\lambda) \quad ,
# $$
# 
# where $\vec{r} = (x,y,z)$ and $\lambda$ is the de Broglie wavelength of the electrons. Substituting this into the Schrödinger equation we obtain
# 
# $$
#     -\frac{\hbar^2}{2m} \left[\nabla_{xy}^2 + \frac{\partial^2}{\partial z^2} + \frac{4\pi i}{\lambda}\frac{\partial}{\partial z} + \frac{2meV(\vec{r})}{\hbar^2}  \right] \phi(\vec{r}) = 0 \qquad \nabla_{xy}^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} \quad .
# $$
# 
# In the high energy approximation, we assume that the wavefunction varies slowly in the $z$-direction compared to the potential and that the wavelength is small, thus
# 
# $$
#     \left| \frac{\partial^2 \phi}{\partial z^2} \right| \ll \left| \frac{1}{\lambda} \frac{\partial \phi}{\partial z} \right| \quad .
# $$
# 
# Hence, the Schrödinger equation simplifies to a first order differential equation in $z$
# 
# $$
#     \frac{\partial \phi(\vec{r})}{\partial z} = \left[\frac{i\lambda}{4\pi} \nabla_{xy}^2 + i \sigma V(\vec{r}) \right] \phi(\vec{r}) \quad ,
# $$
# 
# where $\sigma=2\pi me\lambda/h^2$ is the interaction parameter. This equation is integrated numerically by slicing the potential into thin slices, such that the influence of each slice can be approximated as a simple phase shift of the wave function. The wave function is propagated between slices as a small angle outgoing wave (Fresnel diffraction). The transmission and propagation across a single slice can be written
# 
# $$
#     \phi(x, y, z + \Delta z) = p(x,y,\Delta z) * [t(r) \psi(\vec{r})] + \mathcal{O}(\Delta z^2) \quad ,
# $$
# 
# where $*$ represents a convolution. The transmission function, $t(r)$, for the portion of the potential between $z$ and $z+\Delta z$ is
# 
# $$
#     t(\vec{r}) = \exp\left[i\sigma \int_z^{z+\Delta z} V(\vec{r}) dz'\right] \quad ,
# $$
# 
# and the Fresnel propagator $p(x, y, \Delta z)$ is
# 
# $$
#     p(x,y,\Delta z) = \frac{1}{i \lambda \Delta z}\exp\left[\frac{i\pi}{\lambda \Delta z}(x^2+y^2)\right] \, .
# $$
# 
# The wave at the exit plane of the specimen is obtained by sequentially propagating and transmitting the wave function starting with an assumed input wave.
# 
# Convolutions can be performed efficiently by utilizing the Fast Fourier Transform (FFT). The implemented form of Eq. the single slice propagation is
# 
# $$
#     \phi_{z + \Delta z}(x,y) = \mathcal{F}^{-1}\{P(k_x, k_y, \Delta z) \mathcal{F}[t(\vec{r}) \phi(\vec{r})] \} \quad ,
# $$
# 
# where $P$ is the Fourier transform of the fresnel propagator. The computational cost for the FFT scales as $N \log(N)$ with the number of samples $N$.
