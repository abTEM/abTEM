import numpy as np
from mpi4py import MPI

from abtem.detect import RingDetector, PtychographyDetector
from abtem.potentials import import_potential
from abtem.scan import GridScan
from abtem.utils import ind2sub
from abtem.waves import ProbeWaves

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(rank)

partitions = (2, 2)

assigned_partition = ind2sub(partitions, rank)

potential = import_potential('potential.npz')
extent = potential.extent

probe = ProbeWaves(energy=80e3, cutoff=.02)

scan = GridScan((0., 0), (extent[0] / 2, extent[1]), sampling=.1, endpoint=False)
scans = scan.partition(partitions)

detectors = [PtychographyDetector(), RingDetector(inner=.04, outer=.2)]
measurements = probe.scan(scan=scans[assigned_partition], potential=potential, detectors=detectors, max_batch=50, )

np.save('export/ptycho-partition-{}-{}.npy'.format(*assigned_partition), measurements[detectors[0]])
np.save('export/stem-partition-{}-{}.npy'.format(*assigned_partition), measurements[detectors[1]])

