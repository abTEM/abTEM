# from abtem.potentials.parametrizations import EwaldParametrization
# from abtem.potentials.integrals import ProjectionQuadratureRule
# from abtem.potentials.poisson import solve_point_charges

# parametrization = EwaldParametrization(width=.3)
#
# integrator = ProjectionQuadratureRule(parametrization=parametrization, quad_order=90, cutoff_tolerance=1e-9)
#
# potential = abtem.Potential(atoms, parametrization=parametrization, integral_space='real', gpts=(128,128))
#
# from ase import Atoms
#
# atoms = Atoms('C', [(0,0,0)], cell=(4,4,4))
#
# v1 = solve_point_charges(atoms, shape=(128,128,128), width=.3)
# v0 = solve_point_charges(atoms, shape=(128,128,128), width=0)
#
# v = v0[0].sum(0) #- v1[0].sum(0)
# v -= v.min()
#
# p = potential.build().compute().project().array[0]
# p -= p.min()
# #plt.plot(v[0].sum(0))
#
# plt.plot(v)
# plt.plot(p*32)
