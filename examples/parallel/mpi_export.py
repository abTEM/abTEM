import numpy as np
from ase.io import read

from abtem.detect import RingDetector
from abtem.scan import GridScan
from abtem.utils import ind2sub
from abtem.waves import ProbeWaves
from abtem.potentials import import_potential

size = 4
partitions = (2, 2)

for rank in range(size):
    assigned_partition = ind2sub(partitions, rank)

    potential = import_potential('potential.npz')
    extent = potential.extent

    probe = ProbeWaves(energy=80e3, cutoff=.02)

    scan = GridScan((0., 0), (extent[0] / 2, extent[1]), sampling=.1, endpoint=False)
    scans = scan.partition(partitions)

    detector = RingDetector(inner=.04, outer=.2)
    measurements = probe.scan(scan=scans[assigned_partition], potential=potential, detectors=detector, max_batch=50, )

    np.save('export/mpi-partition-{}-{}.npy'.format(*assigned_partition), measurements[detector])
