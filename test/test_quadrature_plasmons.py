import ase.build
import numpy as np
import pytest

import abtem
from abtem.core.axes import PlasmonAxis, PlasmonOrderAxis
from abtem.inelastic.plasmons import (
    MonteCarloPlasmons,
    QuadraturePlasmons,
    _lorentzian_kernels,
    reduce_plasmon_axes,
)


@pytest.fixture(scope="module")
def si_potential():
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 3)
    return abtem.Potential(atoms, gpts=64, slice_thickness=2.0)


@pytest.fixture(scope="module")
def probe(si_potential):
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
    probe.grid.match(si_potential)
    return probe


def plasmons(**kwargs):
    defaults = dict(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        max_loss_order=2,
        num_angles=2,
        num_azimuthal=3,
        num_depths=2,
    )
    return QuadraturePlasmons(**{**defaults, **kwargs})


def test_excitation_weights_sum_to_poisson():
    p = plasmons(max_loss_order=8)
    weights = p.excitation_weights(thickness=1050.0)
    assert np.isclose(sum(weights), 1.0, atol=1e-6)
    assert np.isclose(weights[0], np.exp(-1.0))


def test_angular_nodes_probabilities():
    p = plasmons(num_angles=4, num_azimuthal=6)
    nodes = p._angular_nodes(200e3, min_angle=0.5)
    assert np.isclose(nodes["p_small"] + nodes["p_large"], 1.0)
    tilts = nodes["single"]["tilts"]
    assert tilts.shape == (24, 2)
    radii = np.linalg.norm(tilts, axis=1)
    assert radii.min() > 0.5
    assert radii.max() < 27.6
    # rings are uniform in cumulative Lorentzian probability
    theta_e = p.characteristic_angle(200e3)
    u = np.log(1 + radii**2 / theta_e**2)
    ring_u = np.unique(np.round(u, 6))
    assert len(ring_u) == 4
    assert np.allclose(np.diff(ring_u), np.diff(ring_u)[0])


def test_kernels_are_normalized():
    p = plasmons(num_angles=3, num_azimuthal=4)
    nodes = p._angular_nodes(200e3, min_angle=1.0)
    kernels = _lorentzian_kernels((64, 64), (1.0, 1.0), nodes)
    for key in ("small", "large"):
        assert np.isclose(kernels[key][0, 0].real, 1.0, atol=1e-5)
    assert kernels["single"].shape == (12, 64, 64)
    assert np.allclose(kernels["single"][:, 0, 0].real, 1.0, atol=1e-5)
    assert np.allclose(kernels["extra"][:, 0, 0].real, 1.0, atol=1e-5)


def test_order_axis():
    axis = plasmons().order_axis
    assert isinstance(axis, PlasmonOrderAxis)
    assert axis.values == ("Zero loss", "Single plasmon", "Double plasmon")
    assert axis.model == "quadrature"
    assert axis.parameters["max_tilt_events"] == 1


@pytest.mark.parametrize("lazy", [False, True])
def test_zero_loss_matches_multislice(si_potential, probe, lazy):
    detector = abtem.PixelatedDetector(max_angle="valid")
    waves = probe.build(lazy=lazy)
    reference = waves.multislice(si_potential, detectors=detector).compute()
    result = waves.multislice(si_potential, detectors=detector, plasmons=plasmons())
    result = result.compute()
    assert result.shape == (3,) + reference.shape
    assert isinstance(result.axes_metadata[0], PlasmonOrderAxis)
    assert np.allclose(result.array[0], reference.array, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("max_tilt_events", [0, 1, 2])
@pytest.mark.parametrize("lab_frame", [False, True])
def test_loss_orders_conserve_intensity(
    si_potential, probe, max_tilt_events, lab_frame
):
    detector = abtem.PixelatedDetector(max_angle="full")
    p = plasmons(max_tilt_events=max_tilt_events, lab_frame=lab_frame)
    result = probe.multislice(si_potential, detectors=detector, plasmons=p).compute()
    totals = result.array.sum((-2, -1))
    assert np.allclose(totals, totals[0], rtol=2e-3)


def test_pure_momentum_transfer_is_convolution(si_potential, probe):
    detector = abtem.PixelatedDetector(max_angle="full")
    p = plasmons(max_tilt_events=0, max_loss_order=1, lab_frame=True)
    result = probe.multislice(si_potential, detectors=detector, plasmons=p).compute()
    nodes = p._angular_nodes(200e3, min_angle=max(result.angular_sampling))
    kernels = _lorentzian_kernels(
        tuple(result.shape[-2:]), tuple(result.angular_sampling), nodes
    )
    elastic = np.fft.fft2(np.fft.ifftshift(result.array[0]))
    kernel = nodes["p_small"] * kernels["small"] + nodes["p_large"] * kernels["large"]
    expected = np.fft.fftshift(np.fft.ifft2(elastic * kernel).real)
    assert np.allclose(result.array[1], expected, rtol=1e-3, atol=1e-6 * expected.max())


def test_tilted_frame_without_tilts_equals_elastic(si_potential, probe):
    detector = abtem.PixelatedDetector(max_angle="full")
    p = plasmons(max_tilt_events=0, lab_frame=False)
    result = probe.multislice(si_potential, detectors=detector, plasmons=p).compute()
    assert np.allclose(result.array[1], result.array[0])


def test_scan_positions_and_polar_detector(si_potential, probe):
    scan = abtem.LineScan(start=(0, 0), end=(3, 0), gpts=2)
    detector = abtem.FlexibleAnnularDetector()
    result = probe.scan(
        si_potential, scan=scan, detectors=detector, plasmons=plasmons(max_loss_order=1)
    ).compute()
    assert result.shape[:2] == (2, 2)
    assert np.all(result.array >= 0)


def test_three_events_conserve_intensity_and_multipass_equivalence(si_potential, probe):
    detector = abtem.PixelatedDetector(max_angle="full")
    kwargs = dict(
        max_loss_order=3,
        num_angles=2,
        num_azimuthal=3,
        num_depths=3,
        max_tilt_events=3,
        event_num_angles=1,
        event_num_azimuthal=2,
        lab_frame=True,
    )
    single_pass = probe.multislice(
        si_potential, detectors=detector, plasmons=plasmons(**kwargs)
    ).compute()
    totals = single_pass.array.sum((-2, -1))
    assert np.allclose(totals, totals[0], rtol=2e-3)
    p = plasmons(**kwargs, max_copies=20)
    assert p.num_copies() == 6 * (3 + 6 * 2 + 10 * 4)
    multi_pass = probe.multislice(
        si_potential, detectors=detector, plasmons=p
    ).compute()
    assert np.allclose(multi_pass.array, single_pass.array, rtol=1e-4, atol=1e-8)


def test_order_axis_round_trips_through_zarr(si_potential, probe, tmp_path):
    detector = abtem.PixelatedDetector(max_angle="valid")
    result = probe.multislice(si_potential, detectors=detector, plasmons=plasmons())
    result = result.compute()
    result.to_zarr(str(tmp_path / "plasmons.zarr"))
    loaded = abtem.from_zarr(str(tmp_path / "plasmons.zarr")).compute()
    assert np.allclose(loaded.array, result.array)
    assert loaded.axes_metadata[0].values == result.axes_metadata[0].values
    assert loaded.axes_metadata[0].parameters["num_depths"] == 2


def monte_carlo(**kwargs):
    defaults = dict(
        mean_free_path=1050.0,
        excitation_energy=17.0,
        critical_angle=27.6,
        num_samples=2,
        num_excitations=1,
        seed=0,
    )
    return MonteCarloPlasmons(**{**defaults, **kwargs})


def test_monte_carlo_events_and_reduction(si_potential, probe):
    detector = abtem.PixelatedDetector(max_angle="valid")
    mc = monte_carlo()
    events = mc.draw_events(probe, si_potential)
    assert events.num_excitations == (0, 1, 1)
    axis = events.ensemble_axes_metadata[0]
    assert isinstance(axis, PlasmonAxis)
    assert len(axis.tilt) == 3
    result = events.apply(probe.build()).multislice(si_potential, detectors=detector)
    reduced = reduce_plasmon_axes(result).compute()
    assert reduced.shape[0] == 2
    assert isinstance(reduced.axes_metadata[0], PlasmonOrderAxis)
    assert reduced.axes_metadata[0].values == ("Zero loss", "Single plasmon")


@pytest.mark.parametrize("lazy", [False, True])
def test_monte_carlo_plasmons_keyword_and_lab_frame(si_potential, probe, lazy):
    detector = abtem.PixelatedDetector(max_angle="valid")
    waves = probe.build(lazy=lazy)
    elastic = waves.multislice(si_potential, detectors=detector).compute()
    tilted = waves.multislice(
        si_potential, detectors=detector, plasmons=monte_carlo(lab_frame=False)
    ).compute()
    lab = waves.multislice(
        si_potential, detectors=detector, plasmons=monte_carlo(lab_frame=True)
    ).compute()
    assert tilted.shape == (2,) + elastic.shape
    assert np.allclose(tilted.array[0], elastic.array)
    assert np.allclose(lab.array[0], elastic.array)
    # the laboratory frame shifts the single-plasmon channel, the tilted frame not
    assert np.isclose(lab.array[1].sum(), tilted.array[1].sum(), rtol=1e-5)
    assert not np.allclose(lab.array[1], tilted.array[1])
    assert lab.axes_metadata[0].model == "monte_carlo"


def test_monte_carlo_axis_round_trips_through_zarr(si_potential, probe, tmp_path):
    detector = abtem.PixelatedDetector(max_angle="valid")
    events = monte_carlo().draw_events(probe, si_potential)
    result = events.apply(probe.build()).multislice(si_potential, detectors=detector)
    result = result.compute()
    result.to_zarr(str(tmp_path / "events.zarr"))
    loaded = abtem.from_zarr(str(tmp_path / "events.zarr")).compute()
    assert isinstance(loaded.axes_metadata[0], PlasmonAxis)
    assert np.allclose(loaded.array, result.array)
