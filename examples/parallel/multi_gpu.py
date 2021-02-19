import cupy as cp
import matplotlib.pyplot as plt
from ase.build import bulk, surface

from abtem import GridScan, Potential, Probe, AnnularDetector

"""
In this example, we parallelize over scan positions by partitioning the probe positions of the grid scan and calculating
each partition on a different GPU.

WARNING: The example is currently untested.
"""
NUM_GPUS = 2

atoms = bulk('Si', crystalstructure='diamond', cubic=True)
atoms = surface(atoms, (1, 1, 0), 1)
atoms.center(axis=2, vacuum=5)
atoms *= (1, 1, 1)
atoms.wrap()

potential = Potential(atoms,
                      gpts=512,
                      slice_thickness=1,
                      projection='infinite',
                      precalculate=False,
                      parametrization='kirkland')

probe = Probe(semiangle_cutoff=15, energy=300e3, device='gpu')
probe.match_grid(potential)

gridscan = GridScan(start=[0, 0], end=potential.extent, sampling=.9 * probe.ctf.nyquist_sampling)
scans = gridscan.partition_scan((1, NUM_GPUS))

detector = AnnularDetector(inner=70, outer=100)
measurements = detector.allocate_measurement(probe, gridscan)

for i, scan in enumerate(scans):
    with cp.cuda.Device(0):
        probe.scan(scans[i], detector, potential, measurements=measurements, pbar=False)

measurements.show()
plt.show()