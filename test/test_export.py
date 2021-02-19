import numpy as np

from abtem.detect import PixelatedDetector
from abtem.measure import Measurement, calibrations_from_grid
from abtem.potentials import Potential, PotentialArray
from abtem.scan import LineScan, GridScan
from abtem.waves import Probe, Waves


def test_export_import_potential(tmp_path, graphene_atoms):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'potential.hdf5'

    potential = Potential(graphene_atoms, sampling=.05)
    precalculated_potential = potential.build(pbar=False)
    precalculated_potential.write(path)
    imported_potential = PotentialArray.read(path)
    assert np.allclose(imported_potential.array, precalculated_potential.array)
    assert np.allclose(imported_potential.extent, precalculated_potential.extent)
    assert np.allclose(imported_potential._slice_thicknesses, precalculated_potential._slice_thicknesses)


def test_export_import_waves(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'waves.hdf5'

    waves = Probe(semiangle_cutoff=30, sampling=.05, extent=10, energy=80e3).build()
    waves.write(path)
    imported_waves = Waves.read(path)
    assert np.allclose(waves.array, imported_waves.array)
    assert np.allclose(waves.extent, imported_waves.extent)
    assert np.allclose(waves.energy, imported_waves.energy)


def test_export_import_measurement(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'measurement.hdf5'

    calibrations = calibrations_from_grid((512, 256), (.1, .3), ['x', 'y'], 'Å')

    measurement = Measurement(np.random.rand(512, 256), calibrations)
    measurement.write(path)
    imported_measurement = Measurement.read(path)
    assert np.allclose(measurement.array, imported_measurement.array)
    assert measurement.calibrations[0] == imported_measurement.calibrations[0]
    assert measurement.calibrations[1] == imported_measurement.calibrations[1]


def test_linescan_to_file(tmp_path, graphene_atoms):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'measurement2.hdf5'

    potential = Potential(atoms=graphene_atoms, sampling=.05)

    probe = Probe(energy=200e3, semiangle_cutoff=30)

    probe.grid.match(potential)

    scan = LineScan(start=[0, 0], end=[0, potential.extent[1]], gpts=20)

    detector = PixelatedDetector()
    export_detector = PixelatedDetector(save_file=path)

    measurements = probe.scan(scan, [detector, export_detector], potential, pbar=False)

    measurement = measurements[0]
    imported_measurement = Measurement.read(measurements[1])

    assert np.allclose(measurement.array, imported_measurement.array)
    assert measurement.calibrations[0] == imported_measurement.calibrations[0]
    assert measurement.calibrations[1] == imported_measurement.calibrations[1]


def test_gridscan_to_file(tmp_path, graphene_atoms):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'measurement2.hdf5'

    potential = Potential(atoms=graphene_atoms, sampling=.05)

    probe = Probe(energy=200e3, semiangle_cutoff=30)

    probe.grid.match(potential)

    scan = GridScan(start=[0, 0], end=[0, potential.extent[1]], gpts=(10, 9))

    detector = PixelatedDetector()
    export_detector = PixelatedDetector(save_file=path)

    measurements = probe.scan(scan, [detector, export_detector], potential, pbar=False)

    measurement = measurements[0]
    imported_measurement = Measurement.read(measurements[1])

    assert np.allclose(measurement.array, imported_measurement.array)
    assert measurement.calibrations[0] == imported_measurement.calibrations[0]
    assert measurement.calibrations[1] == imported_measurement.calibrations[1]
