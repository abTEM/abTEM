import ase.build
import numpy as np
import pytest

import abtem
from abtem.core.axes import EnergyLossAxis, PlasmonOrderAxis
from abtem.inelastic.plasmons import drude_loss_function
from abtem.measurements import MeasurementsEnsemble


@pytest.fixture(scope="module")
def si_potential():
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 3)
    return abtem.Potential(atoms, gpts=64, slice_thickness=2.0)


@pytest.fixture(scope="module")
def probe(si_potential):
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
    probe.grid.match(si_potential)
    return probe


def quadrature(**kwargs):
    defaults = dict(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        max_loss_order=3,
        num_angles=2,
        num_azimuthal=3,
        num_depths=2,
    )
    return abtem.QuadraturePlasmons(**{**defaults, **kwargs})


def channels(model, values=None):
    """A measurement with unit channels on the loss-order axis of a model."""
    num_orders = len(model.order_axis.values)
    array = np.ones(num_orders) if values is None else np.asarray(values, float)
    return MeasurementsEnsemble(array, [model.order_axis], {"energy": 200e3})


def test_orders_integrate_to_poisson_weights():
    model = quadrature()
    energies = np.arange(-5.0, 400.0, 0.05)
    spectrum = abtem.plasmon_loss_spectrum(
        channels(model, [1.0, 0.0, 0.0, 0.0]), energies, thickness=800.0
    )
    weights = model.excitation_weights(800.0)
    assert np.isclose(spectrum.array.sum() * 0.05, weights[0], rtol=1e-6)

    for n in range(1, 4):
        values = np.zeros(4)
        values[n] = 1.0
        spectrum = abtem.plasmon_loss_spectrum(
            channels(model, values), energies, thickness=800.0
        )
        assert np.isclose(spectrum.array.sum() * 0.05, weights[n], rtol=2e-3)


def test_peaks_at_multiples_of_plasmon_energy():
    model = quadrature()
    energies = np.arange(-5.0, 80.0, 0.05)

    for n in range(1, 4):
        values = np.zeros(4)
        values[n] = 1.0
        spectrum = abtem.plasmon_loss_spectrum(
            channels(model, values),
            energies,
            thickness=800.0,
            plasmon_width=1.0,
            energy_resolution=0.2,
        )
        assert abs(energies[np.argmax(spectrum.array)] - n * 17.0) < 0.3


def test_custom_loss_function():
    model = quadrature(max_loss_order=2)
    energies = np.arange(-5.0, 60.0, 0.1)

    def narrow(e):
        return np.exp(-((e - 20.0) ** 2) / (2 * 0.3**2))

    spectrum = abtem.plasmon_loss_spectrum(
        channels(model, [0.0, 1.0, 1.0]),
        energies,
        thickness=300.0,
        loss_function=narrow,
        energy_resolution=0.0,
    )
    peaks = [
        energies[np.argmax(np.where(mask, spectrum.array, 0))]
        for mask in (energies < 30, energies > 30)
    ]
    assert np.allclose(peaks, [20.0, 40.0], atol=0.2)


def test_zero_loss_without_resolution_is_one_bin():
    model = quadrature(max_loss_order=1)
    energies = np.arange(-2.0, 2.0 + 1e-9, 0.5)
    spectrum = abtem.plasmon_loss_spectrum(
        channels(model, [1.0, 0.0]),
        energies,
        thickness=100.0,
        energy_resolution=0.0,
    )
    expected = np.zeros(len(energies))
    expected[np.argmin(np.abs(energies))] = np.exp(-100.0 / 1050.0) / 0.5
    assert np.allclose(spectrum.array, expected)


def test_phase_scramble_channels_carry_the_weights():
    model = abtem.PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        max_loss_order=2,
    )
    energies = np.arange(-5.0, 400.0, 0.05)
    values = [0.5, 0.3, 0.2]
    spectrum = abtem.plasmon_loss_spectrum(channels(model, values), energies)
    assert np.isclose(spectrum.array.sum() * 0.05, 1.0, rtol=2e-3)


def test_axis_and_measurement_types(si_potential, probe):
    model = quadrature(max_loss_order=2)
    energies = np.arange(-5.0, 60.0, 0.5)
    detector = abtem.PixelatedDetector(max_angle="valid")
    measurement = probe.multislice(si_potential, detectors=detector, plasmons=model)
    eager = abtem.plasmon_loss_spectrum(
        measurement.compute(), energies, thickness=si_potential.thickness
    )
    lazy = abtem.plasmon_loss_spectrum(
        measurement, energies, thickness=si_potential.thickness
    ).compute()

    assert isinstance(eager, abtem.DiffractionPatterns)
    assert eager.shape == (len(energies),) + measurement.shape[1:]
    axis = eager.axes_metadata[0]
    assert isinstance(axis, EnergyLossAxis)
    assert axis.format_title() == "energy loss = -5 eV"
    assert np.allclose(eager.array, lazy.array, rtol=1e-5, atol=1e-9)


def test_monte_carlo_order_axis_carries_parameters(si_potential, probe):
    model = abtem.MonteCarloPlasmons(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        num_excitations=1,
        num_samples=2,
        seed=1,
        lab_frame=False,
    )
    measurement = probe.multislice(
        si_potential, detectors=abtem.AnnularDetector(0, 10), plasmons=model
    ).compute()
    axis = measurement.axes_metadata[0]
    assert isinstance(axis, PlasmonOrderAxis)
    assert axis.parameters["excitation_energy"] == 17.0

    spectrum = abtem.plasmon_loss_spectrum(
        measurement, np.arange(-5.0, 400.0, 0.05), thickness=si_potential.thickness
    )
    weights = model.excitation_weights(si_potential.thickness)
    assert np.isclose(
        spectrum.array.sum() * 0.05, np.dot(weights, measurement.array), rtol=2e-3
    )


def test_drude_loss_function_peak_and_width():
    energies = np.linspace(1.0, 40.0, 39001)
    loss = drude_loss_function(energies, excitation_energy=17.0, width=2.0)
    assert abs(energies[np.argmax(loss)] - 17.0) < 0.05
    above = energies[loss > loss.max() / 2]
    assert np.isclose(above[-1] - above[0], 2.0, rtol=2e-2)


def test_errors_and_warnings():
    model = quadrature(max_loss_order=1)
    energies = np.arange(-5.0, 60.0, 0.5)
    with pytest.raises(ValueError, match="thickness"):
        abtem.plasmon_loss_spectrum(channels(model), energies)
    with pytest.raises(ValueError, match="uniformly"):
        abtem.plasmon_loss_spectrum(
            channels(model), np.array([0.0, 1.0, 3.0]), thickness=100.0
        )
    with pytest.warns(UserWarning, match="Poisson"):
        abtem.plasmon_loss_spectrum(channels(model), energies, thickness=3000.0)

    bare = MeasurementsEnsemble(
        np.ones(2),
        [PlasmonOrderAxis(values=("Zero loss", "Single plasmon"))],
        {"energy": 200e3},
    )
    with pytest.raises(ValueError, match="parameters"):
        abtem.plasmon_loss_spectrum(bare, energies, thickness=100.0)
    spectrum = abtem.plasmon_loss_spectrum(
        bare, energies, thickness=100.0, plasmons=model
    )
    assert spectrum.shape == (len(energies),)
