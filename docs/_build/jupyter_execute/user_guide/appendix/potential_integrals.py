#!/usr/bin/env python
# coding: utf-8

# ### Real space integrators (advanced options)
# 
# #### Quadrature rule
# 
# The numerical integrals by creating a table of integrals by using the Gaussian quadrature method , which is designed for accurate results using a minimum number of evaluations for functions with singularities.
# 
# The potential of a single atom is localized, but in principle infinite in extent, hence we need to set a reasonable cutoff. The cutoff is calculated by solving the equation
# 
# $$
# V(r) = V_{tol} \quad ,
# $$
# 
# where $V_{tol}$ is the error at the cut-off. The equation is solved for each species. The use of the cut-off radius creates a discontinuity; hence, abTEM uses a tapering near the cut-off. $V_{cut}$ can be modified using the `cutoff_tolerance` argument of the `Potential` or `AtomicPotential` objects. abTEM uses a tapering cutoff starting at $85 \ \%$ of the full cutoff.

# In[ ]:


from abtem.potentials.integrals import ProjectionQuadratureRule

integrator = ProjectionQuadratureRule()

integrator

