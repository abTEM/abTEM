import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from abtem import Probe, AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from strategies import atoms as atoms_st
from strategies import core as core_st
from utils import assume_valid_probe_and_detectors, gpu


@given(atoms=atoms_st.random_atoms(min_side_length=5, max_side_length=10),
       gpts=core_st.gpts(min_value=128, max_value=128),
       semiangle_cutoff=st.floats(5, 10),
       inner=st.floats(0, 50),
       outer=st.floats(50, 100),
       energy=st.floats(100e3, 200e3))
@pytest.mark.parametrize('lazy', [False, True])
def test_integrate_consistent(atoms, gpts, semiangle_cutoff, energy, lazy, inner, outer):
    probe = Probe(gpts=gpts,
                  semiangle_cutoff=semiangle_cutoff,
                  energy=energy,
                  extent=np.diag(atoms.cell)[:2])

    inner, outer = np.round(inner), np.round(outer)

    detectors = [AnnularDetector(inner=inner, outer=outer),
                 FlexibleAnnularDetector(),
                 PixelatedDetector(max_angle='cutoff')]

    assume_valid_probe_and_detectors(probe, detectors)

    measurements = probe.scan(potential=atoms, detectors=detectors, lazy=lazy)
    measurements.compute()
    annular_measurement, flexible_measurement, pixelated_measurement = measurements

    assert np.allclose(annular_measurement.array, flexible_measurement.integrate_radial(inner, outer).array)
    assert np.allclose(annular_measurement.array, pixelated_measurement.integrate_radial(inner, outer).array)


@given(gpts=st.integers(min_value=64, max_value=128),
       extent=st.floats(min_value=5, max_value=10))
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_interpolate_diffraction_patterns(gpts, extent, device):
    probe1 = Probe(energy=100e3, semiangle_cutoff=30, extent=(extent * 2, extent), gpts=(gpts * 2, gpts), device=device)
    probe2 = Probe(energy=100e3, semiangle_cutoff=30, extent=extent, gpts=gpts, device=device)

    measurement1 = probe1.build(lazy=False).diffraction_patterns(max_angle=None).interpolate('uniform')
    measurement2 = probe2.build(lazy=False).diffraction_patterns(max_angle=None)

    assert np.allclose(measurement1.array, measurement2.array)
