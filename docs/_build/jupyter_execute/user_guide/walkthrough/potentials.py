#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from ase.io import read

import abtem
from abtem.potentials.parametrizations import LobatoParametrization, KirklandParametrization, PengParametrization

abtem.config.set({'local_diagnostics.progress_bar': False});


# # Potentials
# An electron beam interacts with a specimen through the Coulomb potential of its electrons and nuclei. Thus the total electrostatic potential of the sample is required for quantitative image simulations. Typically, the so-called indepedent atom model (IAM) is used, which neglects any bonding effects and treats the sample as an array of atomic potentials.

# ## Atomic potential parametrization
# 
# The electron charge distribution of an atom can be calculated from a first-principles electronic structure calculation, while the atomic nuclei are point charges at any realistic spatial resolution. Given a charge distribution, the potential can be obtained via Poisson's equation. Most multislice simulation codes include a parametrization of the atomic potentials, with a table of parameters for each element fitted to Hartree-Fock calculations. 
# 
# The radial dependence of the electrostatic potential and (electron) scattering factor of five selected elements is shown below.

# In[2]:


symbols = ['C', 'N', 'Si', 'Au', 'U']

scattering_factors = abtem.concatenate([
    LobatoParametrization().line_profiles(symbol, cutoff=3, name='scattering_factor')
    for symbol in symbols
])

potentials = abtem.concatenate([
    LobatoParametrization().line_profiles(symbol, cutoff=2, name='potential', )
    for symbol in symbols
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
scattering_factors.show(ax=ax1)
potentials.show(ax=ax2)

ax2.set_ylim([0, 800]);


# The default parametrization in abTEM is created by Ivan Lobato{cite}`lobato-parameters`. We also implement the parametrization by Earl J. Kirkland{cite}`kirkland-parameters` and the parametrization by Peng{cite}`peng-parameters`. The differences between the parametrizations are generally negligible at low scattering angles, however, the parametrization by Lobato has the most accurate behavior at high scattering angles.

# In[3]:


lobato_scattering_factors = abtem.concatenate([
    LobatoParametrization().line_profiles(symbol, cutoff=3, name='scattering_factor')
    for symbol in symbols
])
kirkland_scattering_factors = abtem.concatenate([
    KirklandParametrization().line_profiles(symbol, cutoff=3, name='scattering_factor')
    for symbol in symbols
])

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
lobato_scattering_factors.show(ax=ax1)
kirkland_scattering_factors.show(ax=ax2);


# ## Independent atom model
# The full specimen potential, $V(r)$, is usually obtained as a linear superposition of atomic potentials
# 
# $$
#     V(r) = \sum_i V_i(r-r_i) \quad,
# $$
# 
# where $V_i(r)$ is the atomic potential of the $i$'th atom. This is known as the Independent Atom Model (IAM). The IAM neglects bonding effects, which is usually a good approximation. In some special cases it may be useful to include bonding effects, see section ???.
# 
# ## The `Potential`
# 
# Here, we create a `Potential` representing SrTiO<sub>3</sub> which represents a sliced IAM potential using a given parametrization. The parameter `sampling` denotes the spacing of the $xy$-samples of the potential, `parametrization` specifies the potential parametrization as described above, `slice_thickness` denotes the spacing

# In[4]:


sto_lto_interface = read('./data/STO_LTO.cif')

potential = abtem.Potential(sto_lto_interface,
                            sampling = .05,
                            parametrization = 'lobato',
                            slice_thickness = 1,
                            projection = 'finite')


# The potential has 23 slices as may be determined from getting its length.

# In[5]:


len(potential)


# The projected potential, i.e. the sum of all the slices, may be shown using the `.show` method.

# In[6]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

abtem.show_atoms(sto_lto_interface, ax=ax1, legend=True)
potential.show(ax=ax2);


# The `Potential` may be indexed to return a subset of slices. Below we select the first 5 slices and show them all by setting `explode=True`.

# In[7]:


potential[:5].show(explode=True, figsize=(18, 5), image_grid_kwargs = {'axes_pad': .05});


# ````{seealso}
# 
# ````

# ## Building and saving the potential
# The `Potential` object does not store the calculated potential slices. Hence, if a simulation, such as STEM requires multiple propagations, each slice have to be calculated multiple times. For this reason, abTEM often precalculates the potential, whenever it has to be used more than once. 
# 
# The potential can be precalculated manually using the `.build` method. however, it should typically be preferred to let abTEM precalculate the potential.

# In[8]:


potential_array = potential.build(lazy=True)


# This returns an `PotentialArray` object, which stores each 2D potential slice in a 3D array. The first dimension is the slice index and the last two are the spatial dimensions.

# In[9]:


potential_array.array


# The calculated potential can be stored in the `zarr` file format and read back.

# In[10]:


potential_array.to_zarr('data/srtio3_110_potential.zarr', overwrite=True)

abtem.from_zarr('data/srtio3_110_potential.zarr')


# ## Slicing of the potential
# 
# The multislice method requires a mathematical slicing of the potential and is only correct in the limit of thin slices, however, more slices increases the computational cost. A reasonable value for slice thickness is generally between $0.5 \ Å$ and $2 \ Å$. The default is $0.5 \ Å$. abTEM provides multiple options for evaluating the integrals required for slicing the potential.
# 
# ### Finite projection integrals
# 
# abTEM implements an accurate finite potential projection method. Numerical integration is used to calculate the integrals of the form
# 
# $$
# V_{proj}^{(i)}(x, y) = \int_{z_i}^{z_i+\Delta z} V(x,y,z) dz \quad ,
# $$
# 
# where $z_i$ is the $z$-position at the entrance of the $i$'th slice and $\Delta z$ is the slice thickness.
# 
# We used `abTEM`s default method of handling finite projection integrals above, additional options and a description of the methods is given in appendix ???.
# 
# ### Infinite projection integrals
# 
# Since the finite projection method can be computationally demanding abTEM also implements a potential using infinite projection integrals. The finite integrals are replaced by infinite integrals, which may be evaluated analytically
# 
# $$ \int_{-\infty}^{\infty} V(x,y,z) dz \approx \int_{z_i}^{z_i+\Delta z} V(x,y,z) dz $$
# 
# The infinite projection of the atomic potential for each atom is assigned to a single slice. The implementation uses the hybrid real-space/Fourier-space approach by W. van den Broek et. al. (https://doi.org/10.1016/j.ultramic.2015.07.005). Using infinite projections can be *much* faster, especially for potentials with a large numbers of atoms. The error introduced by using infinite integrals is negligible compared to other sources of error in most cases.
# 
# Below we create the same SrTiO3 potential as above with infinite projections.

# In[11]:


potential_infinite = abtem.Potential(sto_lto_interface, 
                                     sampling = .05, 
                                     parametrization = 'lobato', 
                                     slice_thickness = 1,
                                     projection = 'infinite')

potential_infinite[:5].show(explode=True, figsize=(18, 5), image_grid_kwargs = {'axes_pad': .05});


# ## Crystal potentials
# 
# Calculating the potential is generally not a significant cost for simulations where the same potential is used in many runs of the multislice algorithm. However, for simulations that require just one or a few wave functions, such as HRTEM and CBED, it is an advantage to use a `CrystalPotential`.
# 
# The `CrystalPotential` allows fast calculation for potentials of crystals by tiling a repeating unit.

# In[12]:


atoms = read('./data/SrTiO3.cif')

potential_unit = abtem.Potential(atoms, sampling=.05, projection='finite')

crystal_potential = abtem.CrystalPotential(potential_unit=potential_unit, repetitions=(10, 10, 20))


# In[13]:


crystal_potential.build().array


# We see that computing this *significantly* larger `Potential` is much faster than the previous potentials. 

# In[14]:


get_ipython().run_cell_magic('time', '', 'crystal_potential_array = crystal_potential.build().compute()')


# ```{note}
# Frozen phonons also works for the `CrystalPotential`, however, the implementation is different. See our user guide to frozen phonons.
# ```
