from ase.io import read
from gpaw import GPAW, PW
from abtem.dft import GPAWPotential
from abtem.potentials import import_potential
import matplotlib.pyplot as plt

#atoms = read('../data/graphene.traj')
#atoms.center(vacuum=4, axis=2)
#atoms *= (2, 1, 1)

#calc = GPAW(mode=PW(600), eigensolver='cg', h=.2, txt=None)
#atoms.set_calculator(calc)
#atoms.get_potential_energy()
#calc.write('graphene.gpw')

calc = GPAW('graphene.gpw', txt=None)

potential = GPAWPotential(calc, sampling=.015, slice_thickness=.4).precalculate()
potential.export('potential.npz', overwrite=True)

potential = import_potential('potential.npz')

plt.imshow(potential.array.sum(axis=0))
plt.show()