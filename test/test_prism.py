import os

import numpy as np
import pytest
from abtem.detect import AnnularDetector, FlexibleAnnularDetector
from abtem.device import asnumpy, cp
from abtem.potentials import Potential
from abtem.scan import LineScan, GridScan
from abtem.waves import Probe, SMatrix
from ase import Atoms
from ase.io import read


def test_prism_raises():
    with pytest.raises(ValueError) as e:
        SMatrix(.01, 80e3, .5)

    assert str(e.value) == 'Interpolation factor must be int'

    with pytest.raises(RuntimeError) as e:
        prism = SMatrix(10, 80e3, 1)
        prism.build()

    assert str(e.value) == 'Grid extent is not defined'


def test_prism_match_probe():
    S = SMatrix(semiangle_cutoff=30., energy=60e3, interpolation=1, extent=5, gpts=50, rolloff=0.)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=30., rolloff=0.)
    assert np.allclose(probe.build([(0., 0.)]).array, S.build().collapse([(0., 0.)]).array, atol=2e-5)


def test_prism_translate():
    S = SMatrix(semiangle_cutoff=30., energy=60e3, interpolation=1, extent=5, gpts=50, rolloff=0.)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=30, rolloff=0)
    assert np.allclose(probe.build(np.array([(2.5, 2.5)])).array,
                       S.build().collapse([(2.5, 2.5)]).array, atol=1e-5)


def test_prism_tilt():
    S = SMatrix(semiangle_cutoff=30., energy=60e3, interpolation=1,  extent=5, gpts=50, tilt=(1, 2))
    S_array = S.build()
    wave = S_array.collapse()
    assert S_array.tilt == S.tilt == wave.tilt


def test_prism_interpolation():
    S = SMatrix(semiangle_cutoff=30., energy=60e3, interpolation=2, extent=10, gpts=100, rolloff=0.)
    probe = Probe(extent=5, gpts=50, energy=60e3, semiangle_cutoff=30, rolloff=0)

    probe_array = probe.build(np.array([(2.5, 2.5)])).array
    S_array = S.build().collapse([(2.5, 2.5)]).array

    assert np.allclose(probe_array, S_array, atol=1e-5)


def test_prism_multislice():
    potential = Potential(Atoms('C', positions=[(0, 0, 2)], cell=(5, 5, 4)))
    S = SMatrix(semiangle_cutoff=30., energy=60e3, interpolation=1, extent=5, gpts=500, rolloff=0.)
    probe = Probe(extent=5, gpts=500, energy=60e3, semiangle_cutoff=30, rolloff=0.)
    assert np.allclose(probe.build(np.array([[2.5, 2.5]])).multislice(potential, pbar=False).array,
                       S.multislice(potential, pbar=False).collapse([(2.5, 2.5)]).array, atol=2e-5)


def test_probe_waves_line_scan():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))
    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=80, outer=200)

    S = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500).multislice(potential, pbar=False)
    probe = Probe(semiangle_cutoff=30, energy=80e3, gpts=500)

    prism_measurement = S.scan(linescan, detector, max_batch_probes=10, pbar=False)
    measurement = probe.scan(linescan, detector, potential, max_batch=50, pbar=False)

    assert np.allclose(measurement.array, prism_measurement.array, atol=1e-6)


def test_interpolation_scan():
    atoms = Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4))
    potential = Potential(atoms)
    linescan = LineScan(start=[0, 0], end=[2.5, 2.5], gpts=10)
    detector = AnnularDetector(inner=80, outer=200)

    probe = Probe(semiangle_cutoff=30, energy=80e3, gpts=250)
    measurements = probe.scan(linescan, [detector], potential, max_batch=50, pbar=False)

    S_builder = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=2, gpts=500)
    atoms = Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4))
    atoms *= (2, 2, 1)
    potential = Potential(atoms)
    S = S_builder.multislice(potential, pbar=False)
    prism_measurements = S.scan(linescan, detector, max_batch_probes=10, pbar=False)

    assert np.allclose(measurements.array, prism_measurements.array, atol=1e-6)


def test_prism_batch():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500)
    S1 = S_builder.multislice(potential, pbar=False)
    S2 = S_builder.multislice(potential, max_batch=5, pbar=False)

    assert np.allclose(S1.array, S2.array)


def test_downsample_smatrix():
    S = SMatrix(expansion_cutoff=10, interpolation=2, energy=300e3, extent=10, sampling=.05)
    S = S.build().downsample()

    S2 = SMatrix(expansion_cutoff=10, interpolation=2, energy=300e3, extent=10, gpts=S.gpts)
    S2 = S2.build()

    assert np.allclose(S.array - S2.array, 0., atol=5e-6)


def test_downsample_max_angle():
    S = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500)
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))
    S = S.multislice(potential, pbar=False)
    pattern1 = S.collapse((0, 0)).diffraction_pattern(max_angle=64)
    S = S.downsample(max_angle=64)
    pattern2 = S.collapse((0, 0)).diffraction_pattern(None)
    pattern3 = S.collapse((0, 0)).diffraction_pattern(max_angle=64)

    assert np.allclose(pattern1.array, pattern2.array, atol=1e-6 * pattern1.array.max(), rtol=1e-6)
    assert np.allclose(pattern3.array, pattern2.array, atol=1e-6 * pattern1.array.max(), rtol=1e-6)


def test_downsample_detect():
    atoms = read('data/srtio3_100.cif')
    atoms *= (4, 4, 1)

    potential = Potential(atoms, gpts=256, projection='infinite', slice_thickness=.5,
                          parametrization='kirkland').build(pbar=False)

    detector = FlexibleAnnularDetector()
    end = (potential.extent[0] / 4, potential.extent[1] / 4)
    gridscan = GridScan(start=[0, 0], end=end, sampling=.2)
    S = SMatrix(energy=300e3, semiangle_cutoff=9.4, rolloff=0.05, expansion_cutoff=10)
    S_exit = S.multislice(potential, pbar=False)
    measurement = S_exit.scan(gridscan, detector, pbar=False)
    S_downsampled = S_exit.downsample()

    downsampled_measurement = S_downsampled.scan(gridscan, detector, pbar=False)

    assert S_downsampled.array.shape != S_exit.array.shape
    assert not np.all(measurement.array == downsampled_measurement.array)
    assert np.allclose(measurement.array, downsampled_measurement.array)


def test_crop():
    S = SMatrix(expansion_cutoff=30, interpolation=3, energy=300e3, extent=10, sampling=.02).build()
    gridscan = GridScan(start=[0, 0], end=S.extent, gpts=64)

    scans = gridscan.partition_scan((2, 2))
    cropped = S.crop_to_scan(scans[0])

    assert cropped.gpts != S.gpts

    position = (4.9, 0.)
    assert np.allclose(S.collapse(position).array - cropped.collapse(position).array, 0.)


@pytest.mark.gpu
def test_prism_gpu():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500, device='cpu')
    S_cpu = S_builder.multislice(potential, pbar=False)

    assert type(S_cpu.array) is np.ndarray

    S_builder = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500, device='gpu')
    S_gpu = S_builder.multislice(potential, pbar=False)

    assert type(S_gpu.array) is cp.ndarray
    assert np.allclose(S_cpu.array, asnumpy(S_gpu.array))


@pytest.mark.gpu
def test_prism_storage():
    potential = Potential(Atoms('C', positions=[(2.5, 2.5, 2)], cell=(5, 5, 4)))

    S_builder = SMatrix(semiangle_cutoff=30., energy=80e3, interpolation=1, gpts=500, device='gpu', storage='cpu')
    S_gpu = S_builder.multislice(potential, pbar=False)

    assert type(S_gpu.array) is np.ndarray


@pytest.mark.gpu
def test_cropped_scan():
    atoms = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/amorphous_carbon.cif'))
    potential = Potential(atoms, gpts=512, slice_thickness=1, device='gpu', projection='infinite',
                          parametrization='kirkland', storage='gpu').build(pbar=True)
    detector = AnnularDetector(inner=40, outer=60)
    gridscan = GridScan(start=[0, 0], end=potential.extent, gpts=16)

    S = SMatrix(expansion_cutoff=20, interpolation=4, energy=300e3, device='gpu', storage='cpu')  # .build()

    S = S.multislice(potential, pbar=True)
    S = S.downsample('limit')

    measurements = S.scan(gridscan, [detector], max_batch_probes=64)

    scans = gridscan.partition_scan((2, 2))
    cropped_measurements = detector.allocate_measurement(S, gridscan)

    for scan in scans:
        cropped = S.crop_to_scan(scan)
        cropped = cropped.transfer('gpu')
        cropped.scan(scan, detector, measurements=cropped_measurements, pbar=False)

    assert np.allclose(cropped_measurements.array, measurements.array)

