import numpy as np
from ase import Atoms

from abtem.detect import AnnularDetector
from abtem.potentials import Potential
from abtem.scan import LineScan
from abtem.waves import PlaneWave, Probe


def test_fig_5_12():
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland', cutoff_tolerance=1e-4)

    waves = PlaneWave(energy=200e3)

    waves = waves.multislice(potential, pbar=False)
    waves = waves.apply_ctf(defocus=700, Cs=1.3e7, semiangle_cutoff=10.37, rolloff=0.)

    intensity = np.abs(waves.array) ** 2

    assert np.round(intensity.min(), 2) == np.float32(.72)
    assert np.round(intensity.max(), 2) == np.float32(1.03)


def test_fig():
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 4) for x in np.linspace(5, 45, 5)], cell=(50, 50, 8))

    gpts = 2048

    potential = Potential(atoms=atoms, gpts=gpts, parametrization='kirkland', slice_thickness=8)

    probe = Probe(energy=200e3, defocus=700, Cs=1.3e7, semiangle_cutoff=10.37)

    probe.grid.match(potential)

    scan = LineScan(start=[5, 25], end=[45, 25], gpts=5)

    detector = AnnularDetector(inner=40, outer=200)

    measurements = probe.scan(scan, [detector], potential, pbar=False)

    #assert np.allclose(measurements[detector].array, [0.00010976, 0.00054356, 0.00198158, 0.00997221, 0.01098883])
    assert np.allclose(measurements[detector].array, [0.00010933, 0.00054426, 0.00205304, 0.00986171, 0.01197426],
                      atol=1e-5)
    # assert np.allclose(measurements[detector].array, [0.00012474, 0.00060268, 0.0022271 , 0.00963283, 0.01076559],
    #                    atol=1e-5)