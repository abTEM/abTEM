# atoms = bulk('Au') #graphene()
# #atoms.center(vacuum=2, axis=2)
# atoms.pbc = True
# #atoms.center()
#
# ortho_atoms = orthogonalize_cell(atoms)
#
# gpaw = GPAW(h=.2, txt=None, kpts=(3, 3, 3))
# atoms.calc = gpaw
#
# ortho_gpaw = GPAW(h=.2, txt=None, kpts=(3, 3, 3))
# ortho_atoms.calc = ortho_gpaw
#
# atoms.get_potential_energy()
# ortho_atoms.get_potential_energy()
