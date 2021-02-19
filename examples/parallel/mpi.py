import matplotlib.pyplot as plt
from ase.build import bulk, surface
from mpi4py import MPI

from abtem import GridScan, Potential, Probe, AnnularDetector, Measurement

"""
In this example, we parallelize over scan positions by partitioning the probe positions of the grid scan and calculating
each partition on a different CPU core. The results from each are saved to disk, 
"""

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

atoms = bulk('Si', crystalstructure='diamond', cubic=True)
atoms = surface(atoms, (1, 1, 0), 10)
atoms.center(axis=2, vacuum=5)
atoms *= (1, 1, 1)
atoms.wrap()

potential = Potential(atoms,
                      sampling=.05,
                      projection='infinite',
                      parametrization='kirkland',
                      precalculate=True,
                      slice_thickness=2)

probe = Probe(semiangle_cutoff=30, energy=160e3)
probe.match_grid(potential)

temp_fname = f'temp_{str(rank).zfill(len(str(size - 1)))}.hdf5'
detector = AnnularDetector(60, 240, save_file=temp_fname)

scan = GridScan((0, 0), (potential.extent[0] / 3, potential.extent[1] / 4), sampling=.9 * probe.ctf.nyquist_sampling)

measurement = detector.allocate_measurement(probe, scan)
scans = scan.partition_scan((1, size))

measurement = probe.scan(scans[rank], detector, potential, measurements={detector: measurement}, pbar=False)

comm.Barrier()
if rank == 0:
    measurement = Measurement.read(temp_fname)

    for i in range(1, size):
        measurement += Measurement.read(f'temp_{str(i).zfill(len(str(size - 1)))}.hdf5')

    measurement.write('silicon_110.hdf5')
    measurement.show()
    plt.show()
