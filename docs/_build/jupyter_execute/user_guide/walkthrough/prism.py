#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read

import abtem
abtem.config.set({'fft':'fftw'})
abtem.config.set({'local_diagnostics.progress_bar': False});


# # PRISM
# 
# Multslice simulations of STEM images is computationally expensive because the selectron cattering have to be calculated from scratch at each probe position. This can be mitigated by using parallel hardware or GPUs, see our introduction to distributed parallelization and GPUs. 
# 
# An alternative, that does not require more hardware, is to use the PRISM algorithm {cite}`prism`. At zero cost to accuracy, PRISM almost always speeds up simulations of images with many probe positions. Introducing approximations in the form of Fourier space interpolation, PRISM can reach speedups of several 100 times at a minor cost to the accuracy.
# 
# The downside to PRISM is an increased memory requirement, which can be prohibitive, especially if a requirement of good Fourier space sampling prohibits interpolation. A solution to this issue is using the partitioned PRISM algorithm introduced at the end of this document.
# 
# ```{seealso}
# Our PRISM quickstart example provides a basic template for doing STEM simulations with PRISM in *abTEM*.
# ```
# ## The PRISM scattering matrix
# 
# We derive the PRISM algorithm from the perspective of writing the (discretized) probe wave function as an expansion of plane waves.
# 
# $$
#     \phi(\vec{r}) = \sum_{nm} C_{nm} S_{nm}
# $$
# 
# where
# 
# $$
#     S_{nm} = \exp(-2\pi i \vec{q}_{nm} \cdot \vec{r})
# $$
# 
# is a plane wave with a wave vector $\vec{q}_{nm}$ that (due to discretization) must fulfill
# 
# $$
#     \vec{q}_{nm} = (n \Delta_q, m \Delta_q) \quad ,
# $$ 
# 
# where $\Delta_{q}$ is the Fourier space sampling rate. The complex expansion coefficients $C_{nm}$ may be chosen to select a specified probe position or phase error. Since the Fourier space probe wave function is zero beyond the convergence semi-angle, we may limit the number of terms in the expansion by a cutoff $\alpha_{max}$. Thus only the terms fulfilling 
# 
# $$
#     \sqrt{n ^ 2 + m ^ 2} \lambda \Delta_q \leq \alpha_{max}
# $$ (eq:prism_cutoff)
# 
# needs to be included in the expansion.
# 
# We use the symbol $\mathcal{M}$ to represent the application of the multislice algorithm (see appendix Eq. ), thus the exit wave is given by 
# 
# $$
#     \phi_{exit}(\vec{r}) = \mathcal{M} \sum_{nm} C_{nm} S_{nm} = \sum_{nm} C_{nm} \mathcal{M} S_{nm} \quad ,
# $$ (eq:prism)
# 
# where the second equality uses that the multislice algorithm is coherent.
# 
# The PRISM algorithm for STEM simulations may be summarised as two consecutive stages: 
# 
# **Multislice stage:** Build the PRISM scattering matrix by applying the multislice algorithm to each plane wave of the expansion, i.e. calculate the set of waves $\mathcal{M} S_{nm}$.
# 
# **Reduction stage:** For each probe position, perform the reduction in Eq. {eq}`eq:prism` with a set of expansion coefficients determined by the probe position.
# 
# ## The `SMatrix`
# 
# We import the atomic model of the SrTiO<sub>3</sub>/LaTiO<sub>3</sub> interface created [here](user_guide:walkthrough:atomic_models) and make a `Potential`.

# In[2]:


atoms = read('./data/STO_LTO.cif') 

potential = abtem.Potential(atoms, gpts = (400, 800))

abtem.show_atoms(atoms, legend=True);


# In *abTEM*, the PRISM scattering matrix (i.e. the set of waves $\mathcal{M} S_{nm}$) is represented by the `SMatrix`. The expansion cutoff angle, $\alpha_{max}$, is given as `planewave_cutoff`. In addition to $\alpha_{max}$, the number of plane wave in the expansion increases with the energy and the extent of the potential, because they, respectively, decrease $\lambda$ and $\Delta_q$ in Eq. {eq}`eq:prism_cutoff`.
# 
# We create an `SMatrix` representing the SrTiO<sub>3</sub>/LaTiO<sub>3</sub> model for an energy of $150 \ \mathrm{keV}$ and an expansion cutoff angle of $20 \ \mathrm{mrad}$.

# In[8]:


s_matrix = abtem.SMatrix(
    potential = potential,
    energy = 150e3,
    planewave_cutoff = 20,
)


# We can `build` the `SMatrix` to produce an `SMatrixArray`. The `SMatrixArray` wraps a 3d `numpy` array where the first dimension indexes `S_{nm}` (running over both indices $n$ and $m$), the last two dimensions represents the $x$ and $y$ axis of the waves.
# 
# We see that the expansion of our probe requires 371 plane waves.

# In[9]:


s_matrix_array = s_matrix.build()
s_matrix_array


# ```{note}
# The number of grid points of the waves is smaller than the `Potential` by a factor of $2 / 3$. This is because the array representing the PRISM S-Matrix is downsampled to the angle of the anti-aliasing aperture after the multislice stage. This is done to save memory, but can be disabled by setting `downsample=False` when creating the `SMatrix`.
# ```

# The number of plane waves should be compared to the number of probe positions required to scan over the extent of the potential at Nyquist sampling.

# In[10]:


sampling = abtem.waves.transfer.nyquist_sampling(s_matrix.planewave_cutoff, 
                                                 s_matrix.wavelength)
scan = abtem.GridScan.from_fractional_coordinates(potential, sampling=sampling)

print(f'Number of probe positions: {len(scan)}')
print(f'Number of plane waves: {len(s_matrix)}')
print(f'Ratio: {len(scan) / len(s_matrix):.1f}')


# Thus, PRISM (without approximations) requires 5.4 times fewer multislice steps than the conventional multislice algorithm. 
# 
# However, this factor is an upper bound on possible speedup, as PRISM also requires the reduction stage. To perform the reduction we call `scan` with the `GridScan` defined above and an `AnnularDetector` to detect the exit waves as the expansion is reduced with varying `C_{nm}`. 
# 
# We also call `compute` which in this case will compute the task graph of both the multislice and reduction stage. Finally, the resulting HAADF image is shown.

# In[15]:


detector = abtem.AnnularDetector(inner=50, outer=200)

with abtem.config()
measurement = s_matrix.scan(scan=scan, detectors=detector).compute()

measurement.interpolate(.1).show();


# The reduction stage is typically significantly faster than the multislice stage, however, this depends completely on the number of slices in the potential. Using PRISM the present example took about 1 min 30s on a mid-range laptop, while the equivalent conventional multislice algorithm took 6 min, hence the actual speedup from using PRISM was about a factor of 4.
# 
# ## PRISM interpolation
# 
# The real speedup with PRISM comes when introducing interpolation. Choosing the interpolation factors $(f_n, f_m)$ the wave vectors should fulfill 
# 
# $$
#     \vec{q}_{nm} = (n f_n \Delta_q, m f_m \Delta_q) \quad ,
# $$
# 
# hence we keep only every $f_n$'th and $f_m$'th wave vector. For example, for $(f_n, f_m) = (2, 1)$ we keep only every second wave vector, skipping every second of the $n$-indices. The total number of wave vectors is thus reduced by a factor $f_nf_m$.
# 
# We create the same `SMatrix` as we did in the preceding section, however, we create it with three different interpolation factors $(2, 1)$, $(4, 2)$ and $(8, 4)$.

# In[51]:


s_matrix_interpolated = abtem.SMatrix(
    potential = potential,
    energy = 150e3,
    planewave_cutoff = 20,
    interpolation = (1, 2),
)

s_matrix_more_interpolated = abtem.SMatrix(
    potential = potential,
    energy = 150e3,
    planewave_cutoff = 20,
    interpolation = (2, 4),
)

s_matrix_very_interpolated = abtem.SMatrix(
    potential = potential,
    energy = 150e3,
    planewave_cutoff = 20,
    interpolation = (4, 8),
)


# We run the PRISM algorithm with each of the interpolated scattering matrices.

# In[52]:


with abtem.config.set({'local_diagnostics.progress_bar' : True}):

measurement_interpolated = s_matrix_interpolated.scan(scan=scan, detectors=detector).compute()
measurement_more_interpolated = s_matrix_more_interpolated.scan(scan=scan, detectors=detector).compute()
measurement_very_interpolated = s_matrix_very_interpolated.scan(scan=scan, detectors=detector).compute()


# We show a comparison of the results below. At a glance, we see that the results look identical up to and including an interpolation factor of $(2,4)$, however, at an interpolation factor of $(4, 8)$ visible errors are introduced.

# In[53]:


abtem.stack([
    measurement,
    measurement_interpolated,
    measurement_more_interpolated,
    measurement_very_interpolated
],
    ('(fn, fm) = (1, 1)', '(fn, fm) = (1, 2)', '(fn, fm) = (2, 4)', '(fn, fm) = (4, 8)')
).interpolate(.1).show(explode=True, figsize=(8,6), common_color_scale=True, cbar=True, image_grid_kwargs={'axes_pad': .1});


# A quantitative comparison is shown below as the difference between the exact and interpolated as a percent of the maximum value. We see that the error is a couple of percent for the interpolation factors $(1,2)$ and $(2,4)$, which is probably well within the range of error source due to unknowns and implicit approximantions. The error of up to $15\ \%$ for the interpolation factor of $(4,8)$ is clearly too large for creating production quality simulations, but it  might be tolerated while dialing in other parameters. 

# In[65]:


abtem.stack([
    (measurement - measurement) / measurement.max() * 100,
    (measurement_interpolated - measurement) / measurement.max() * 100,
    (measurement_more_interpolated - measurement) / measurement.max() * 100,
    (measurement_very_interpolated - measurement) / measurement.max() * 100
],
    ('(fn, fm) = (1, 1)', '(fn, fm) = (2, 1)', '(fn, fm) = (4, 2)', '(fn, fm) = (8, 4)')
).interpolate(.1).show(explode = True, 
                       figsize = (8,6),
                       common_color_scale = True, 
                       cbar = True, 
                       image_grid_kwargs = {'axes_pad': .1}, 
                       cmap = 'bwr',
                       cbar_labels = 'Error [%]',
                       vmin = -15,
                       vmax = 15);


# In the above we interpolated by twice the factor in $y$ compared to $x$. Interpolation effectively repeats the probe, hence the "window" that the repeated probes shrinks as interpolation increases. Hence when the potential extent is larger, we can interpolate by a larger factor before the probes starts to get squeezed.
# 
# ### Choosing the interpolation
# 
# Choosing an appropriate interpolation factor can 

# In[71]:



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
s_matrix.dummy_probes().show(ax=ax1)
s_matrix_interpolated.dummy_probes().show(ax=ax2)
s_matrix_more_interpolated.dummy_probes().show(ax=ax3)
s_matrix_very_interpolated.dummy_probes().show(ax=ax4)


# ## PRISM with phase aberrations

# ## Parallel reduction in *abTEM*

# In[ ]:




