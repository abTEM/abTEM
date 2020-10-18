import numpy as np
from ase.io import read

from abtem.detect import AnnularDetector, FlexibleAnnularDetector
from abtem.potentials import Potential
from abtem.scan import GridScan
from abtem.waves import Probe


def test_flexible_annular_detector():
    atoms = read('data/srtio3_100.cif')
    atoms *= (4, 4, 1)

    potential = Potential(atoms, gpts=512, device='gpu', projection='infinite', slice_thickness=.5,
                          parametrization='kirkland', storage='gpu').build(pbar=False)

    probe = Probe(energy=300e3, semiangle_cutoff=9.4, device='gpu', rolloff=0.05)

    flexible_detector = FlexibleAnnularDetector()
    annular_detector = AnnularDetector(inner=30, outer=80)

    end = (potential.extent[0] / 4, potential.extent[1] / 4)

    gridscan = GridScan(start=[0, 0], end=end, sampling=.2)

    measurements = probe.scan(gridscan, [flexible_detector, annular_detector], potential, pbar=False)

    assert np.allclose(measurements[flexible_detector].integrate(30, 80).array, measurements[annular_detector].array)
