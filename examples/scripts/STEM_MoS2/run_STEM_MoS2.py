from abtem import __version__

print('current version:', __version__)

from ase.build import mx2

from abtem import *
from abtem.structures import orthogonalize_cell

device = 'cpu'

##### Set up atomic structure #####
atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None)

repetitions = (3, 2, 1)

atoms = orthogonalize_cell(atoms)

atoms *= repetitions

atoms.center(vacuum=2, axis=2)

##### Define potential #####
potential = Potential(atoms,
                      gpts=256,
                      projection='finite',
                      slice_thickness=1,
                      parametrization='kirkland')

##### Define probe #####
probe = SMatrix(energy=80e3,
                semiangle_cutoff=20,
                expansion_cutoff=20,  # Should be bigger than or equal to the semiangle_cutoff
                rolloff=0.1,
                defocus=40,
                Cs=3e5,
                focal_spread=20,
                device=device)

##### Define detector #####
detector = AnnularDetector(inner=70, outer=200, save_file='STEM_MoS2.hdf5')

##### Define scan region #####
end = (potential.extent[0] / repetitions[0], potential.extent[1] / repetitions[1])

gridscan = GridScan(start=[0, 0], end=end, sampling=probe.ctf.nyquist_sampling * .9)

##### Run simulation #####
measurements = probe.scan(gridscan, [detector], potential)
