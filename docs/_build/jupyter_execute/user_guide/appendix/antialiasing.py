#!/usr/bin/env python
# coding: utf-8

# # Antialiasing and Sampling

# ## Real-space sampling
# 
# The real-space sampling is extremely important because it controls the accuracy of the simulation at high scattering angles. The sampling defines the maximum spatial frequency $k_{max}$ via the formula:
# 
# $$ k_{max} = \frac{1}{2p} \quad , $$
# 
# where $p$ is the real-space sampling distance. To counteract aliasing artifacts due to the periodicity assumption of a discrete Fourier transform, abTEM supresses spatial frequencies above 2 / 3 of the maximum scattering angle, further reducing the maximum effective scattering angle by a factor of 2 / 3. Hence the maximum scattering angle $\alpha_{max}$ is given by:
# 
# $$ \alpha_{max} = \frac{2}{3}\frac{\lambda}{2p} \quad , $$
# 
# where $\lambda$ is the relativistic electron wavelength. As an example, consider a case where we want to simulate 80 keV electron scattering up to angles of 200 mrads. Plugging these values into the above equation gives a sampling of $\sim0.052$ Å, i.e. we require at least 0.05 Å pixel size in order to reach a maximum scattering angle of 200 mrads. In practice, you should ensure that the simulation is converged with respect to pixel size.
# 
# The maximum scattering angles in the $x$- and $y$-direction of the wave functions object can obtained
# 
# print(f'Maximal simulated scattering angles = {waves.cutoff_angles[0]:.3f} mrad')
