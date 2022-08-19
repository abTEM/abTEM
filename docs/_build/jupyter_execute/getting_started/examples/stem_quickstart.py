#!/usr/bin/env python
# coding: utf-8

# In[8]:


from ase.build import graphene
from ase.io import write
import abtem


# In[12]:


atoms = abtem.orthogonalize_cell(graphene(vacuum=2))

atoms *= (3,2,1)

atoms.numbers[10] = 14

abtem.show_atoms(atoms)

write('../../user_guide/data/graphene_with_si.cif', atoms)

