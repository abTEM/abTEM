"""Tests for the plasmon scattering models."""

import math

import ase.build
import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem import (
    CustomScan,
    FrozenPhonons,
    PhaseScramblePlasmons,
    PlaneWave,
    Potential,
    Probe,
    SMatrix,
)
from abtem.core.backend import asnumpy
from abtem.inelastic.plasmons import (
    MonteCarloPlasmons,
    PlasmonAxis,
    PlasmonScatteringEvents,
    estimate_plasmon_parameters,
    excitations_weights,
    scale_critical_angle,
)
from abtem.multislice import FourierMultislice, RealSpaceMultislice


def _to_numpy(array):
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def _setup(device, num_configs=4, sigmas=0.078, ensemble_mean=True, nz=20, seed=1):
    atoms = ase.build.bulk("Si", cubic=True) * (3, 3, nz)
    frozen_phonons = FrozenPhonons(
        atoms,
        num_configs=num_configs,
        sigmas=sigmas,
        seed=seed,
        ensemble_mean=ensemble_mean,
    )
    potential = Potential(frozen_phonons, gpts=96, slice_thickness=2.0, device=device)
    wave = PlaneWave(energy=200e3, device=device)
    return wave, potential, atoms


def _plasmons(seed=7):
    return PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=seed,
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_none_is_noop(device):
    """Passing plasmons=None must reproduce the plain multislice result exactly."""
    wave, potential, _ = _setup(device)
    detector = abtem.PixelatedDetector(max_angle=40)

    ref = wave.multislice(potential, detector).compute()
    out = wave.multislice(potential, detector, plasmons=None).compute()

    assert np.allclose(_to_numpy(ref.array), _to_numpy(out.array))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_zero_loss_is_scaled_elastic(device):
    """The zero-loss channel is the elastic wave scaled by the survival probability
    of every slice, exactly; the electron count is conserved in expectation over the
    repetitions (the scrambled wave of a single configuration is not normalized)."""
    wave, potential, _ = _setup(device, num_configs=8, nz=10, ensemble_mean=False)
    detector = abtem.PixelatedDetector(max_angle="valid")

    d_elastic = _to_numpy(wave.multislice(potential, detector).compute().array)
    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=2,
    )
    result = _to_numpy(
        wave.multislice(potential, detector, plasmons=plasmons).compute().array
    )

    thicknesses = np.array(potential.slice_thickness)
    survival = np.prod(1.0 - thicknesses / 1050.0)
    assert np.allclose(
        result[0], survival * d_elastic, rtol=1e-4, atol=1e-8 * d_elastic.max()
    )

    totals = result.sum((-2, -1))  # (order, config)
    assert np.all(totals > 0)
    total_per_config = totals.sum(0)
    elastic_per_config = d_elastic.sum((-2, -1))
    # expectation value equals the elastic count; 8 repetitions give a loose check
    assert total_per_config.mean() == pytest.approx(elastic_per_config.mean(), rel=0.25)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_redistribute_to_higher_angles(device):
    """Plasmon scattering tilts the beam, reducing the central (zero) beam and
    broadening the diffraction pattern."""
    wave, potential, _ = _setup(device)
    detector = abtem.PixelatedDetector(max_angle=40)

    d_ref = _to_numpy(wave.multislice(potential, detector).compute().array)
    d_pl = _to_numpy(
        wave.multislice(potential, detector, plasmons=_plasmons()).compute().array
    )

    cy, cx = np.unravel_index(np.argmax(d_ref), d_ref.shape)
    # central beam attenuated
    assert d_pl[cy, cx] < d_ref[cy, cx]

    yy, xx = np.indices(d_ref.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mean_r_ref = (r * d_ref).sum() / d_ref.sum()
    mean_r_pl = (r * d_pl).sum() / d_pl.sum()
    # pattern broadened
    assert mean_r_pl > mean_r_ref


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_reproducible_with_seed(device):
    """A fixed seed gives reproducible results across runs (eager and lazy)."""
    wave, potential, _ = _setup(device)
    detector = abtem.PixelatedDetector(max_angle=40)

    a = _to_numpy(
        wave.multislice(potential, detector, plasmons=_plasmons(seed=3)).compute().array
    )
    b = _to_numpy(
        wave.multislice(potential, detector, plasmons=_plasmons(seed=3)).compute().array
    )
    assert np.allclose(a, b)

    c = _to_numpy(
        wave.multislice(potential, detector, plasmons=_plasmons(seed=99))
        .compute()
        .array
    )
    assert not np.allclose(a, c)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_decorrelate_per_configuration(device):
    """Each frozen-phonon configuration must get an independent phase scramble, even
    when the potential is identical across configurations (sigmas=0)."""
    wave, potential, _ = _setup(device, num_configs=3, sigmas=0.0, ensemble_mean=False)
    detector = abtem.PixelatedDetector(max_angle=40)

    a = _to_numpy(
        wave.multislice(potential, detector, plasmons=_plasmons()).compute().array
    )
    assert a.shape[0] == 3
    assert not np.allclose(a[0], a[1])
    assert not np.allclose(a[0], a[2])


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_realspace_order_resolved_matches_fourier(device):
    """Order-resolved plasmon scattering is algorithm-agnostic: the per-order
    diffraction patterns from the real-space and Fourier multislice agree, since
    both apply the identical real-space scatter operator at each slice boundary."""
    wave, potential, _ = _setup(device, num_configs=4, nz=10)
    detector = abtem.PixelatedDetector(max_angle=40)

    def _orders(algorithm):
        plasmons = PhaseScramblePlasmons(
            mean_free_path=1050.0,
            excitation_energy=16.7,
            critical_angle=19.1,
            seed=7,
            max_loss_order=2,
        )
        result = wave.multislice(
            potential, detector, plasmons=plasmons, algorithm=algorithm
        ).compute()
        return _to_numpy(result.array)

    arr_realspace = _orders(RealSpaceMultislice())
    arr_fourier = _orders(FourierMultislice())

    assert arr_realspace.shape == arr_fourier.shape
    assert arr_realspace.shape[0] == 3
    # Per-order total intensities agree (same scatter draw via shared seed).
    for n in range(3):
        assert arr_realspace[n].sum() == pytest.approx(arr_fourier[n].sum(), rel=1e-3)


@pytest.mark.filterwarnings("ignore:Grid size")  # small grid, GPU occupancy hint
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_order_resolved_backscatter(device):
    """Order-resolved plasmon scattering composes with real-space backscattering:
    each loss-order channel yields a forward diffraction pattern and an
    independently reverse-propagated backscattered wave."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 4)
    potential = Potential(
        atoms, gpts=48, slice_thickness=2.0, exit_planes=1, device=device
    )
    wave = PlaneWave(energy=200e3, device=device)
    detector = abtem.PixelatedDetector(max_angle="valid")

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=2,
    )

    forward, backward = wave.multislice(
        potential,
        detector,
        plasmons=plasmons,
        algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
        return_backscattered=True,
    ).compute()

    n_orders = 3
    n_exit = potential.num_exit_planes
    # Both forward and backscatter carry a leading plasmon-order axis.
    assert _to_numpy(forward.array).shape[:2] == (n_orders, n_exit)
    assert _to_numpy(backward.array).shape[:2] == (n_orders, n_exit)
    # Reverse propagation produced finite backscattered waves.
    assert np.all(np.isfinite(_to_numpy(backward.array)))


def test_estimate_plasmon_parameters_silicon():
    """The free-electron plasmon energy is accurate for Si (~16.7 eV); the
    critical angle and mean free path are positive, physically-sized estimates."""
    si = ase.build.bulk("Si", cubic=True)
    E_p, theta_c, lambda_p = estimate_plasmon_parameters(si, 200e3, valence_electrons=4)
    # Plasmon energy is the reliable estimate — within a few percent of 16.7 eV.
    assert E_p == pytest.approx(16.7, abs=0.5)
    # theta_c and lambda_p are order-of-magnitude free-electron estimates.
    assert 1.0 < theta_c < 50.0  # mrad
    assert 200.0 < lambda_p < 5000.0  # Å


def test_estimate_plasmon_parameters_density_intensive():
    """E_p depends on valence-electron density, not on supercell tiling."""
    unit = ase.build.bulk("Si", cubic=True)
    super_cell = unit * (2, 2, 2)
    e_unit = estimate_plasmon_parameters(unit, 200e3, valence_electrons=4)
    e_super = estimate_plasmon_parameters(super_cell, 200e3, valence_electrons=4)
    assert e_unit == pytest.approx(e_super, rel=1e-6)


def test_from_atoms_overrides_and_kwargs():
    """from_atoms estimates E_p but honours explicit overrides and forwards
    constructor kwargs (e.g. max_loss_order)."""
    si = ase.build.bulk("Si", cubic=True)
    plasmons = PhaseScramblePlasmons.from_atoms(
        si,
        200e3,
        valence_electrons=4,
        critical_angle=19.1,  # calibrated override
        mean_free_path=1050.0,  # calibrated override
        max_loss_order=3,
    )
    assert plasmons.excitation_energy == pytest.approx(16.7, abs=0.5)  # estimated
    assert plasmons.critical_angle == 19.1  # override respected
    assert plasmons.mean_free_path == 1050.0  # override respected
    assert plasmons.max_loss_order == 3  # kwarg forwarded


def test_estimate_plasmon_parameters_compound_valence():
    """A {symbol: count} valence mapping works for compounds."""
    gaas = ase.build.bulk("GaAs", crystalstructure="zincblende", a=5.65)
    E_p, _, _ = estimate_plasmon_parameters(
        gaas, 200e3, valence_electrons={"Ga": 3, "As": 5}
    )
    assert E_p == pytest.approx(15.7, abs=1.5)


def test_scale_critical_angle_voltage():
    """theta_c scales with electron wavelength (q_c = K theta_c = const), per
    Mendis (2024) / Barthel (2019): the Si 19.1 mrad at 200 kV corresponds to
    ~15 mrad at 300 kV (Barthel's fitted value)."""
    # Higher energy -> shorter wavelength -> smaller angle.
    at_300 = scale_critical_angle(19.1, 200e3, 300e3)
    assert at_300 == pytest.approx(15.0, abs=0.2)
    # Round-trip is exact.
    assert scale_critical_angle(at_300, 300e3, 200e3) == pytest.approx(19.1, abs=1e-6)
    # Identity at the same energy.
    assert scale_critical_angle(19.1, 200e3, 200e3) == pytest.approx(19.1)


def _static_setup(device, nz=10):
    """A static (no frozen-phonon) potential + plane wave."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, nz)
    potential = Potential(atoms, gpts=64, slice_thickness=2.0, device=device)
    wave = PlaneWave(energy=200e3, device=device)
    return wave, potential


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("max_loss_order", [None, 2])
def test_plasmons_static_num_repetitions(device, max_loss_order):
    """num_repetitions runs plasmons over a static structure (no frozen phonons)
    by averaging independent phase scrambles: a single averaged pattern, or one per
    loss order, with the incident intensity conserved over the orders."""
    wave, potential = _static_setup(device)
    assert potential.num_configurations == 1
    detector = abtem.PixelatedDetector(max_angle="valid")

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        num_repetitions=6,
        seed=7,
        max_loss_order=max_loss_order,
    )
    result = wave.multislice(potential, detector, plasmons=plasmons).compute()
    arr = _to_numpy(result.array)
    # Repetitions are averaged away -> (ny, nx), or (orders, ny, nx).
    assert arr.ndim == (2 if max_loss_order is None else 3)
    if max_loss_order is not None:
        assert arr.shape[0] == max_loss_order + 1
    assert float(arr.sum()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_static_num_repetitions_reproducible(device):
    """A fixed seed makes the static-structure average reproducible; a different
    seed changes it."""
    wave, potential = _static_setup(device)
    detector = abtem.PixelatedDetector(max_angle="valid")

    def run(seed):
        pl = PhaseScramblePlasmons(
            mean_free_path=1050.0,
            excitation_energy=16.7,
            critical_angle=19.1,
            num_repetitions=5,
            seed=seed,
        )
        return _to_numpy(
            wave.multislice(potential, detector, plasmons=pl).compute().array
        )

    first = run(7)
    assert np.allclose(first, run(7))
    assert not np.allclose(first, run(99))


def test_plasmons_num_repetitions_ignored_with_frozen_phonons():
    """When the potential already has frozen phonons, num_repetitions is ignored
    (the configs are the repetitions) and a warning is emitted."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 6)
    fp = FrozenPhonons(atoms, num_configs=3, sigmas=0.05, seed=1)
    potential = Potential(fp, gpts=64, slice_thickness=2.0)
    wave = PlaneWave(energy=200e3)
    detector = abtem.PixelatedDetector(max_angle="valid")

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        num_repetitions=10,
        seed=7,
    )
    with pytest.warns(UserWarning, match="num_repetitions"):
        wave.multislice(potential, detector, plasmons=plasmons).compute()


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_prism_matches_single_probe(device):
    """PRISM with plasmons reproduces the single-probe multislice with plasmons.

    The plasmon operator is linear, so reducing the (un-renormalized) plasmon
    S-matrix is identical to a per-probe multislice; the non-unitary
    renormalization is applied to the recombined probe (issue #296). With
    interpolation=1 and downsampling disabled the two share a grid and agree to
    float precision for a single static configuration and seed."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 6)
    potential = Potential(atoms, gpts=64, slice_thickness=2.0, device=device)
    energy, semiangle = 200e3, 20.0
    pos = (atoms.cell.lengths()[0] / 2, atoms.cell.lengths()[1] / 2)
    detector = abtem.PixelatedDetector(max_angle="valid")

    def plasmons():
        return PhaseScramblePlasmons(
            mean_free_path=1050.0,
            excitation_energy=16.7,
            critical_angle=19.1,
            seed=42,
        )

    s_matrix = SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle,
        interpolation=1,
        downsample=False,
        plasmons=plasmons(),
        device=device,
    ).build(lazy=False)
    prism = _to_numpy(
        s_matrix.reduce(scan=CustomScan([pos]), detectors=detector).compute().array
    )[0]

    probe = Probe(energy=energy, semiangle_cutoff=semiangle, device=device)
    probe.grid.match(potential)
    single = _to_numpy(
        probe.multislice(
            potential, scan=CustomScan([pos]), detectors=detector, plasmons=plasmons()
        )
        .compute()
        .array
    )[0]

    np.testing.assert_allclose(prism, single, rtol=0, atol=1e-5 * single.max())


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_prism_order_resolved_matches_single_probe(device):
    """Order-resolved PRISM reproduces the single-probe order-resolved result:
    each loss-order probe is the linear recombination of that order's S-matrix
    channel, and the orders share a single renormalization scale. Both share a
    grid (interpolation=1, no downsampling), so they agree to float precision."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 6)
    potential = Potential(atoms, gpts=64, slice_thickness=2.0, device=device)
    energy, semiangle = 200e3, 20.0
    pos = (atoms.cell.lengths()[0] / 2, atoms.cell.lengths()[1] / 2)
    detector = abtem.PixelatedDetector(max_angle="valid")

    def plasmons():
        return PhaseScramblePlasmons(
            mean_free_path=1050.0,
            excitation_energy=16.7,
            critical_angle=19.1,
            seed=42,
            max_loss_order=2,
        )

    s_matrix = SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle,
        interpolation=1,
        downsample=False,
        plasmons=plasmons(),
        device=device,
    ).build(lazy=False)
    prism = _to_numpy(
        s_matrix.reduce(scan=CustomScan([pos]), detectors=detector).compute().array
    )  # (n_orders, 1, ny, nx)

    probe = Probe(energy=energy, semiangle_cutoff=semiangle, device=device)
    probe.grid.match(potential)
    single = _to_numpy(
        probe.multislice(
            potential, scan=CustomScan([pos]), detectors=detector, plasmons=plasmons()
        )
        .compute()
        .array
    )  # (n_orders, 1, ny, nx)

    assert prism.shape == single.shape
    assert prism.shape[0] == 3  # leading loss-order axis
    np.testing.assert_allclose(prism, single, rtol=0, atol=1e-5 * single.max())


def test_plasmons_prism_order_resolved_multiconfig_averages():
    """Order-resolved PRISM over a frozen-phonon ensemble averages the
    configurations (the phase-scramble repetitions) exactly once, keeping the
    loss-order axis: the average is the mean of the per-configuration results, and
    eager and lazy builds agree."""
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 5)
    pos = (atoms.cell.lengths()[0] / 2, atoms.cell.lengths()[1] / 2)
    detector = abtem.PixelatedDetector(max_angle="valid")

    def build(lazy, ensemble_mean=True):
        potential = Potential(
            FrozenPhonons(
                atoms, num_configs=2, sigmas=0.0, seed=1, ensemble_mean=ensemble_mean
            ),
            gpts=48,
            slice_thickness=2.0,
        )
        plasmons = PhaseScramblePlasmons(
            mean_free_path=1050.0,
            excitation_energy=16.7,
            critical_angle=19.1,
            seed=42,
            max_loss_order=2,
        )
        sma = SMatrix(
            potential=potential,
            energy=200e3,
            semiangle_cutoff=20.0,
            interpolation=1,
            downsample=False,
            plasmons=plasmons,
        ).build(lazy=lazy)
        return _to_numpy(
            sma.reduce(scan=CustomScan([pos]), detectors=detector).compute().array
        )

    eager, lazy = build(False), build(True)
    per_configuration = build(False, ensemble_mean=False)
    assert eager.shape[0] == 3  # loss orders kept, configurations averaged
    assert per_configuration.shape == (3, 2) + eager.shape[1:]
    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-6 * eager.max())
    np.testing.assert_allclose(
        per_configuration.mean(1), eager, rtol=0, atol=1e-6 * eager.max()
    )


def _plasmon_axis(radial_angles, azimuthal_angles, excitations):
    depths = tuple(float(i + 1) for i in range(len(radial_angles)))
    return PlasmonAxis(
        values=((depths, tuple(radial_angles), tuple(azimuthal_angles), excitations),)
    )


@pytest.mark.parametrize(
    "radial_angles, azimuthal_angles, expected",
    [
        # Equal and opposite deflections cancel.
        ((5.0, 5.0), (0.3, 0.3 + math.pi), (0.0, 0.0)),
        # Perpendicular deflections add in quadrature.
        ((3.0, 4.0), (0.0, math.pi / 2), (3.0, 4.0)),
        # Parallel deflections add linearly along the shared direction.
        ((2.0, 1.5), (math.pi / 2, math.pi / 2), (0.0, 3.5)),
    ],
)
def test_tilt_is_vector_sum_of_deflections(radial_angles, azimuthal_angles, expected):
    axis = _plasmon_axis(radial_angles, azimuthal_angles, excitations=2)
    np.testing.assert_allclose(axis.tilt, (expected,), atol=1e-12)


def test_tilt_counts_only_excitations_above_current_depth():
    axis = _plasmon_axis((3.0, 4.0), (0.0, math.pi / 2), excitations=0)
    assert axis.tilt == ((0.0, 0.0),)

    # Between the first (depth 1) and second (depth 2) scattering event.
    axis.update(1.5)
    np.testing.assert_allclose(axis.tilt, ((3.0, 0.0),), atol=1e-12)

    axis.update(10.0)
    np.testing.assert_allclose(axis.tilt, ((3.0, 4.0),), atol=1e-12)


def test_draw_events_weights_are_poisson():
    potential = abtem.Potential(bulk("Si", cubic=True), gpts=16, slice_thickness=2)
    waves = abtem.PlaneWave(energy=100e3, gpts=16, extent=potential.extent)
    plasmons = MonteCarloPlasmons(
        mean_free_path=10.0,
        excitation_energy=15.0,
        critical_angle=10.0,
        num_excitations=2,
        num_samples=3,
        seed=0,
    )
    events = plasmons.draw_events(waves, potential)

    x = potential.thickness / plasmons.mean_free_path
    expected = [math.exp(-x)] + [x * math.exp(-x)] * 3 + [x**2 * math.exp(-x) / 2] * 3
    np.testing.assert_allclose(events.weights, expected)
    assert [len(depths) for depths in events.depths] == [0, 1, 1, 1, 2, 2, 2]
    assert events.ensemble_shape == (7,)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmon_deflection_in_multislice(device, lazy):
    potential = abtem.Potential(
        bulk("Si", cubic=True) * (1, 1, 2),
        gpts=24,
        slice_thickness=2,
        device=device,
    ).build(lazy=False)
    waves = abtem.PlaneWave(
        energy=100e3, gpts=24, extent=potential.extent, device=device
    ).build(lazy=lazy)
    thickness = potential.thickness
    theta, phi = 5.0, 0.3
    # The deflection applies from the first slice boundary below the event.
    k = potential.num_slices // 2
    depth = sum(potential.slice_thickness[:k]) - 1e-3

    events = PlasmonScatteringEvents(
        depths=((), (thickness + 1.0,), (depth,), (depth, depth)),
        radial_angles=((), (theta,), (theta,), (theta, theta)),
        azimuthal_angles=((), (phi,), (phi,), (phi, phi + math.pi)),
        weights=(1.0,) * 4,
        ensemble_mean=False,
    )
    exit_waves = events.apply(waves).multislice(potential).compute()
    assert isinstance(exit_waves.ensemble_axes_metadata[0], PlasmonAxis)
    zero_loss, beyond_exit, single, opposite_pair = asnumpy(exit_waves.array)

    upper = waves.multislice(potential[:k]).compute()
    upper.metadata.update(
        base_tilt_x=theta * math.cos(phi), base_tilt_y=theta * math.sin(phi)
    )
    expected_single = asnumpy(upper.multislice(potential[k:]).compute().array)

    # An event below the exit surface never happens.
    np.testing.assert_array_equal(beyond_exit, zero_loss)
    # Equal and opposite deflections cancel.
    np.testing.assert_allclose(opposite_pair, zero_loss, atol=1e-5)
    assert np.abs(single - zero_loss).max() > 1e-2
    np.testing.assert_allclose(single, expected_single, atol=1e-5)


@pytest.mark.parametrize("thickness, mean_free_path", [(10.0, 1000.0), (500.0, 800.0)])
def test_excitations_weights_are_poisson_probabilities(thickness, mean_free_path):
    # np.math was removed in NumPy 2, which made excitations_weights raise
    # AttributeError for every input.
    from scipy.stats import poisson

    n = np.arange(0, 20)
    weights = np.array([excitations_weights(i, thickness, mean_free_path) for i in n])

    assert np.allclose(weights, poisson.pmf(n, thickness / mean_free_path))
    assert excitations_weights(0, thickness, mean_free_path) == pytest.approx(
        np.exp(-thickness / mean_free_path)
    )
