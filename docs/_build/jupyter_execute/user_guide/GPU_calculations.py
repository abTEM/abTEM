#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from abtem import *
from ase.io import read


# # Calculations on (single) GPU's

# Performing calculations on a (single) GPU is no different than CPU calculations. We just have to set `device = gpu` on the object we wish to use for GPU calculations. The objects currently supporting GPU calculations are:
# 
# * `Potential`
# * `PlaneWave`
# * `Probe`
# * `SMatrix`
# 
# For example; to create the potential on the GPU and propagate a scattering matrix.

# In[2]:


atoms = read('data/orthogonal_graphene.cif') 

potential = Potential(atoms, sampling=.03, device='gpu').build()

S = SMatrix(expansion_cutoff=32, energy=80e3, device='gpu')

S_array = S.multislice(potential, pbar=False)


# The potential and the scattering matrix currently exists on the GPU as CuPy arrays.

# In[3]:


type(potential.array), type(S_array.array)


# ## Memory intensive calculations

# The preffered way of doing GPU calculations is to keep everyting in the GPU's memory. However, since GPU memory can be limited abTEM also supports storing the large objects on the host and loading it on the GPU as needed. This is controlled using the `storage` argument.

# In[4]:


S2 = SMatrix(expansion_cutoff=32, energy=80e3, device='gpu', storage='cpu')

S_on_cpu = S2.multislice(potential, pbar=False)


# The scattering matrix is in CPU memory, however, the calculation device is the GPU.

# In[5]:


type(S_on_cpu.array), S_on_cpu.calculation_device

