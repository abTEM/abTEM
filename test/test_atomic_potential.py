import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from abtem.integrals import GaussianProjectionIntegrals
from abtem.integrals import ProjectionQuadratureRule

from abtem.parametrizations import LobatoParametrization, KirklandParametrization, PengParametrization


@given(atomic_number=st.integers(min_value=1, max_value=98))
@pytest.mark.parametrize('parametrization_a',
                         [LobatoParametrization(),
                          KirklandParametrization(),
                          PengParametrization(),
                          ], ids=['lobato', 'kirkland', 'peng'])
@pytest.mark.parametrize('parametrization_b',
                         [LobatoParametrization(),
                          KirklandParametrization(),
                          PengParametrization()],
                         ids=['lobato', 'kirkland', 'peng'])
def test_parametrizations(atomic_number, parametrization_a, parametrization_b):
    k = np.linspace(0, 5, 100)
    assert np.allclose(parametrization_a.projected_scattering_factor(atomic_number)(k),
                       parametrization_b.projected_scattering_factor(atomic_number)(k),
                       atol=10, rtol=0.1)
    r = np.linspace(.2, 5, 100)
    assert np.allclose(parametrization_a.projected_potential(atomic_number)(r),
                       parametrization_b.projected_potential(atomic_number)(r),
                       atol=20, rtol=0.1)


@pytest.mark.parametrize('parameters',
                         [{'gaussian_projection_integrals':
                               GaussianProjectionIntegrals(correction_parametrization=None),
                           'parametrization':
                               PengParametrization()
                           },
                          {'gaussian_projection_integrals':
                               GaussianProjectionIntegrals(correction_parametrization='lobato'),
                           'parametrization': LobatoParametrization()
                           }
                          ], ids=['uncorrected', 'corrected'])
def test_gaussian_projection_integrals(parameters):
    parametrization = parameters['parametrization']
    gaussian_projection_integrals = parameters['gaussian_projection_integrals']

    symbol = 'C'
    gpts = (256, 256)
    sampling = (0.1, 0.1)
    positions = np.array([[0, 0, 0]], dtype=np.float32)
    a = -np.inf
    b = np.inf

    gaussian_scattering_factors = gaussian_projection_integrals.gaussian_scattering_factors('C', gpts, sampling)

    projections = gaussian_scattering_factors.integrate_on_grid(
        positions,
        a,
        b,
        gpts,
        sampling,
        fourier_space=True)

    k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
    analytical = parametrization.projected_scattering_factor(symbol)(k ** 2)
    assert np.allclose(analytical, projections[0].real, rtol=1e-6)


@pytest.mark.parametrize('fourier_space', [True, False])
def test_finite_gaussian_projection_integrals(fourier_space):
    gaussian_projection_integrals = GaussianProjectionIntegrals(correction_parametrization=None)

    symbol = 'C'
    gpts = (256, 256)
    sampling = (0.02, 0.02)
    positions = np.array([[0, 0, 0]], dtype=np.float32)
    a = -.2
    b = .2

    gaussian_scattering_factors = gaussian_projection_integrals.gaussian_scattering_factors('C', gpts, sampling)

    projections = gaussian_scattering_factors.integrate_on_grid(
        positions,
        a,
        b,
        gpts,
        sampling,
        fourier_space=fourier_space)

    if fourier_space:
        k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
        analytical = PengParametrization().finite_projected_scattering_factor(symbol)(k, a, b)
        assert np.allclose(analytical, projections[0].real, rtol=1e-6)

    else:
        r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False)
        analytical = PengParametrization().finite_projected_potential(symbol)(r, a, b)
        assert np.allclose(analytical[:len(r) // 2], projections[0][:len(r) // 2], rtol=1e-6, atol=5)


@pytest.mark.parametrize('parametrization',
                         [LobatoParametrization(),
                          KirklandParametrization()])
def test_quadrature(parametrization):
    quadrature = ProjectionQuadratureRule(parametrization, quad_order=20, cutoff_tolerance=1e-9)

    symbol = 'Au'
    gpts = (256, 256)
    sampling = (0.05, 0.05)
    xp = np
    a = -20
    b = 20

    positions = xp.array([[0, 0, 0]], dtype=xp.float32)

    table = quadrature.build_integral_table(symbol, min(sampling) / 2)

    integrated = table.integrate_on_grid(positions, a, b, gpts, sampling)

    r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False).astype(np.float32)
    analytical = parametrization.projected_potential(symbol)(r[1:])

    assert np.allclose(analytical, integrated[0, 1:], atol=2)


def test_finite_projections():
    quadrature = ProjectionQuadratureRule('lobato', quad_order=8, cutoff_tolerance=1e-4)
    gaussian_projection_integrals = GaussianProjectionIntegrals()

    symbol = 'C'
    gpts = (256, 256)
    sampling = (0.05, 0.05)
    a = .1
    b = 1

    positions = np.array([[0, 0, 0]], dtype=np.float32)

    table = quadrature.build_integral_table(symbol, min(sampling) / 2)

    quadrature_potential = table.integrate_on_grid(positions, a, b, gpts, sampling)

    gaussian_scattering_factors = gaussian_projection_integrals.gaussian_scattering_factors(symbol, gpts, sampling)

    gaussian_potential = gaussian_scattering_factors.integrate_on_grid(positions, a, b, gpts, sampling)

    assert np.allclose(quadrature_potential[0, :gpts[1] // 2], gaussian_potential[0, :gpts[1] // 2], atol=2)
