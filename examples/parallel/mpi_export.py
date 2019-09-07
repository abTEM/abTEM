import numpy as np
from ase.io import read

from abtem.detect import RingDetector
from abtem.scan import GridScan
from abtem.utils import ind2sub
from abtem.waves import ProbeWaves

size = 4
partitions = (2, 2)

for rank in range(size):
    assigned_partition = ind2sub(partitions, rank)

    atoms = read('../data/mos2.traj')
    cell = np.diag(atoms.get_cell())

    probe = ProbeWaves(energy=80e3, cutoff=.02, sampling=.025)

    scan = GridScan((0., 0), (cell[0] / 2, cell[1] / 2), sampling=.2, endpoint=False)
    scans = scan.partition(partitions)

    detector = RingDetector(inner=.04, outer=.2)
    measurements = probe.scan(scan=scans[assigned_partition], potential=atoms, detectors=detector, max_batch=50, )

    np.save('export/mpi-partition-{}-{}.npy'.format(*assigned_partition), measurements[detector])
