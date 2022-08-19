#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from abtem.ionization import SubshellTransitions, TransitionPotential, EELSDetector
from abtem import SMatrix, Potential, GridScan

from ase import units
from ase.io import read


# # Core energy-loss with SrTiO3 (early experimental version)

# We calculate an "energy-filtered" STEM image of SrTiO$_3$ targeting oxygen, specifically we target its *K*-edge (i.e. the 1*s* subshell; quantum numbers $(n, \ell) = (1, 0)$)). In a hypothetical experiment, this would be roughly equivalent to setting your energy filter to 456 eV.
# 
# The following code calculates projected overlap integrals following to Dwyer *et al.* (see https://doi.org/10.1016/j.ultramic.2005.03.005 or https://doi.org/10.1103/PhysRevB.57.3273), dynamical scattering following Brown *et al.* (https://doi.org/10.1103/PhysRevResearch.1.033186) and uses the density functional theory code [GPAW](https://wiki.fysik.dtu.dk/gpaw/index.html) for calculating wave functions. Please see our citation guide if you use this code in a publication.

# In[ ]:


Z = 8 # atomic number
n = 1 # principal quantum number
l = 0 # azimuthal quantum number
xc = 'PBE' # exchange-correlation functional

O_transitions = SubshellTransitions(Z = Z, n = n, l = l, xc = 'PBE')

print('bound electron configuration:', O_transitions.bound_configuration)
print('ionic electron configuration:', O_transitions.excited_configuration)


# Applying the selection rules
# 
# $$
#  \Delta \ell = \pm 1 \quad\mathrm{and}\quad \Delta m = 0, \pm 1
# $$
# 
# we obtain the following dipole transitions.

# In[4]:


for bound_state, continuum_state in O_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')


# For a fast electron with an energy of 100 keV and a specified grid, we obtain the following transition potentials; the code will run an all-electron density function theory calculation using GPAW.

# In[5]:


atomic_transition_potentials = O_transitions.get_transition_potentials(extent = 5, 
                                                                       gpts = 256, 
                                                                       energy = 100e3)


# In[6]:


fig, axes = plt.subplots(1,3, figsize = (10,5))

for ax, atomic_transition_potential in zip(axes, atomic_transition_potentials):    
    atomic_transition_potential.show(ax = ax, title = str(atomic_transition_potential))


# We have created transition potentials for single atoms, now we need to put them together in a multislice simulation.
# 
# We import our `Atoms` as usual and make a 2$\times$2 supercell.

# In[7]:


atoms = read('data/srtio3_100.cif') * (2,2,1)
atoms.center(axis = 2)

from abtem import show_atoms
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10,2))

show_atoms(atoms, ax = ax1, title = 'Beam view')
show_atoms(atoms, ax = ax2, plane = 'yz', title = 'Side view')
show_atoms(atoms[atoms.numbers == 8], ax = ax3, plane = 'xy', title = 'Beam view (Oxygen)');


# Next, we create a `TransitionPotential`. The overlap integrals depend on the incoming energy, hence we have to provide the acceleration voltage. We also provide a sampling in order to show the potential below.

# In[8]:


transition_potential = TransitionPotential(O_transitions, 
                                           atoms = atoms, 
                                           sampling = .05, 
                                           energy = 100e3, 
                                           slice_thickness = 2)


# We can show the projected intensity of the transition potential.

# In[9]:


transition_potential.show()


# Finally, we can do a full "energy-filtered" STEM simulation targeting the oxygen *K* edge.
# 
# We create a scattering matrix `SMatrix` as usual (note: interpolation is not yet implemented!), and an `EELSDetector` (interpolation is implemented), as well as a standard electrotrostatic potential.
# 
# We also create a new `TransitionPotential`, which will automatically adopt the appropriate atoms and energy from the other simulation objects.

# In[10]:


S = SMatrix(energy = 100e3, semiangle_cutoff = 25) # interpolation not implemented!


detector = EELSDetector(collection_angle = 100, interpolation = 4)


potential = Potential(atoms, sampling = .05, slice_thickness = .5, 
                      projection = 'infinite', parametrization = 'kirkland')


transition_potential = TransitionPotential(O_transitions)


scan = GridScan((0,0), potential.extent, sampling = .9*S.ctf.nyquist_sampling)


measurement = S.coreloss_scan(scan, detector, potential, transition_potential)


# We show the final (tiled, interpolated) energy-filtered image below.

# In[11]:


measurement.tile((2,2)).interpolate(.02).show(figsize = (6,6));


# We further target the *K*-edge of oxygen as above and the *L*$_{23}$-edge of titanium and strontium. We use the PBE functional to calculate the transitions.

# In[12]:


O_transitions = SubshellTransitions(Z = 8, n = 1, l = 0, xc = 'PBE')
Ti_transitions = SubshellTransitions(Z = 22, n = 2, l = 1, xc = 'PBE')
Sr_transitions = SubshellTransitions(Z = 38, n = 2, l = 1, xc = 'PBE')

transitions = [O_transitions, Ti_transitions, Sr_transitions]

transition_potential = TransitionPotential(transitions)


# In[13]:


print('Oxygen:')
for bound_state, continuum_state in O_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')
print('Titanium:')
for bound_state, continuum_state in Ti_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')
print('Strontium:')
for bound_state, continuum_state in Sr_transitions.get_transition_quantum_numbers():
    print(f'(l, ml) = {bound_state} -> {continuum_state}')


# In[14]:


measurements = S.coreloss_scan(scan, detector, potential, transition_potential)


# By abTEM convention, the first dimensions always represent the scan or navigation dimensions. Hence, in our case the third dimension represents the subshell of an electron (or, experimentally, a specific energy loss).

# In[15]:


fig, (ax1, ax2, ax3)= plt.subplots(1, 3, figsize = (10,2.7))

measurements[0].tile((2, 2)).interpolate(.1).show(ax = ax1)
measurements[1].tile((2, 2)).interpolate(.1).show(ax = ax2)
measurements[2].tile((2, 2)).interpolate(.1).show(ax = ax3);


# In[16]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4))

measurements[0].interpolate_line((0,0), (0, potential.extent[1]), 
                                 sampling = .1).show(ax = ax1, label = 'O')
measurements[1].interpolate_line((0,0), (0, potential.extent[1]), 
                                 sampling = .1).show(ax = ax1, label = 'Ti')
measurements[2].interpolate_line((0,0), (0, potential.extent[1]), 
                                 sampling = .1).show(ax = ax1, label = 'Sr')
ax1.legend()

measurements[0].interpolate_line((atoms[3].x, 0), (atoms[3].x,potential.extent[1]), 
                                 sampling = .1).show(ax = ax2, label = 'O')
measurements[1].interpolate_line((atoms[3].x, 0),(atoms[3].x, potential.extent[1]), 
                                 sampling = .1).show(ax = ax2, label = 'Ti')
measurements[2].interpolate_line((atoms[3].x, 0),(atoms[3].x,potential.extent[1]), 
                                 sampling = .1).show(ax = ax2, label = 'Sr')
ax2.legend();


# In[ ]:




