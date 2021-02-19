import matplotlib.pyplot as plt
from ase.build import bulk, surface
from mpi4py import MPI

from abtem import GridScan, Potential, Probe, AnnularDetector, Measurement

"""
In this example, we parallelize over scan positions by partitioning the probe positions of the grid scan and calculating
each partition on a different CPU core. The results from each process are saved to disk, after all processes are done 
the results are retrived and combined by the master.  
"""

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

atoms = bulk('Si', crystalstructure='diamond', cubic=True)
atoms = surface(atoms, (1, 1, 0), 3)
atoms.center(axis=2, vacuum=5)
reps = (3, 4, 1)
atoms *= reps
atoms.wrap()

potential = Potential(atoms,
                      gpts=768,
                      projection='infinite',
                      parametrization='kirkland',
                      precalculate=True,
                      slice_thickness=2)

probe = Probe(semiangle_cutoff=30, energy=160e3)
probe.match_grid(potential)

temp_fname = f'temp_{str(rank).zfill(len(str(size - 1)))}.hdf5'
detector = AnnularDetector(60, 240, save_file=temp_fname)

scan_end = (potential.extent[0] / reps[0], potential.extent[1] / reps[1])
scan = GridScan((0, 0), scan_end, sampling=.9 * probe.ctf.nyquist_sampling)

measurement = detector.allocate_measurement(probe, scan)
scans = scan.partition_scan((1, size))

measurement = probe.scan(scans[rank], detector, potential, measurements={detector: measurement}, pbar=False)

comm.Barrier()
if rank == 0:
    measurement = Measurement.read(temp_fname)

    for i in range(1, size):
        measurement += Measurement.read(f'temp_{str(i).zfill(len(str(size - 1)))}.hdf5')

    measurement.write('silicon_110.hdf5')
    measurement.interpolate(.05).gaussian_filter(.25).show()
    plt.show()
