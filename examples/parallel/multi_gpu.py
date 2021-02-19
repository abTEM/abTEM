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
atoms = surface(atoms, (1, 1, 0), 3)
atoms.center(axis=2, vacuum=5)
reps = (3, 4, 1)
atoms *= reps
atoms.wrap()

potential = Potential(atoms,
                      gpts=768,
                      slice_thickness=1,
                      projection='infinite',
                      precalculate=False,
                      device='gpu',
                      parametrization='kirkland')

probe = Probe(semiangle_cutoff=15, energy=300e3, device='gpu')
probe.match_grid(potential)

scan_end = (potential.extent[0] / reps[0], potential.extent[1] / reps[1])
gridscan = GridScan(start=(0, 0), end=scan_end, sampling=.9 * probe.ctf.nyquist_sampling)
scans = gridscan.partition_scan((1, NUM_GPUS))

detector = AnnularDetector(inner=70, outer=100)
measurements = detector.allocate_measurement(probe, gridscan)

for i, scan in enumerate(scans):
    with cp.cuda.Device(i):
        probe.scan(scan, detector, potential, measurements=measurements, pbar=False)

measurements.interpolate(.05).gaussian_filter(.25).show()
plt.show()
