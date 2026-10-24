import warnings

import ase
import ase.build
import numpy as np
import pytest
from utils import _ACCELERATOR, to_host_array

import abtem
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.multislice import FourierMultislice
from abtem.twoway import (
    MeanFreePathAbsorber,
    TwoWaySolver,
    structure_glides,
    structure_symmetry,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# The solver runs on the CPU and on CUDA devices (not on Metal, which has no double
# precision and no complex linear algebra).
cuda = pytest.param(
    "gpu",
    marks=(
        pytest.mark.gpu,
        pytest.mark.skipif(_ACCELERATOR != "gpu", reason="no CUDA device"),
    ),
)


def graphene_atoms(vacuum=6.0):
    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=vacuum, axis=2)
    return atoms


def bernal_graphite(layers, spacing=3.35, a=2.46):
    """AB-stacked graphene layers in the orthogonal cell, with 6 Å of vacuum."""
    cell = np.array(
        [
            [a, 0, 0],
            [-a / 2, a * np.sqrt(3) / 2, 0],
            [0, 0, 12.0 + (layers - 1) * spacing],
        ]
    )
    sites = {0: [(1 / 3, 2 / 3), (2 / 3, 1 / 3)], 1: [(2 / 3, 1 / 3), (0.0, 0.0)]}
    positions = [
        np.array([fx, fy, 0.0]) @ cell + [0.0, 0.0, i * spacing]
        for i in range(layers)
        for fx, fy in sites[i % 2]
    ]
    atoms = ase.Atoms(
        f"C{2 * layers}", positions=positions, cell=cell, pbc=[True, True, False]
    )
    atoms = abtem.orthogonalize_cell(atoms)
    atoms.center(vacuum=6.0, axis=2)
    atoms.pbc = [True, True, False]
    return atoms


def strontium_titanate(cells=2, displaced_titanium=(0.0, 0.0, 0.0)):
    a = 3.905
    atoms = ase.Atoms(
        "SrTiO3",
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ],
        cell=[a, a, a],
        pbc=True,
    ) * (1, 1, cells)
    atoms.positions[atoms.numbers == 22] += np.array(displaced_titanium)
    atoms.center(vacuum=4.0, axis=2)
    atoms.pbc = [True, True, False]
    return atoms


def flux_weights(shape, extent, energy):
    """kz / k of every beam of an array with this shape and extent, 0 if evanescent."""
    k = 2 * np.pi / energy2wavelength(energy)
    nx, ny = shape
    g2 = (np.fft.fftfreq(nx, extent[0] / nx)[:, None]) ** 2 + (
        np.fft.fftfreq(ny, extent[1] / ny)[None, :]
    ) ** 2
    return np.sqrt(np.maximum(k**2 - 4 * np.pi**2 * g2, 0.0)) / k


def beam_amplitudes(waves):
    return np.fft.fft2(waves.array) / waves.array.size


# The tests that compare two ways to the same solution of the same discretised
# problem (the reduced and the full basis, the ensemble and its members, the device
# and the host) do not need a converged discretisation, so they use this coarse one:
# 300 eV, slices of 0.04 Å and 3 Å of vacuum, on a grid that divides the hexagonal
# lattice (so the sampled potential keeps its symmetry exactly). Each solve takes
# about a second, a converged one tens of seconds.
COARSE_ENERGY = 300.0


def coarse_graphene(**kwargs):
    kwargs = {
        "gpts": (24, 42),
        "slice_thickness": 0.04,
        "projection": "finite",
        "periodic": True,
        **kwargs,
    }
    return abtem.Potential(graphene_atoms(vacuum=3.0), **kwargs)


def test_structure_symmetry_of_graphene():
    # The orthogonal cell holds two primitive cells: one translation, and the twelve
    # operations of the 6mm point group about a hexagon centre.
    origin, operations, translations = structure_symmetry(graphene_atoms())
    assert {name for name, _ in operations} == {
        "rot0",
        "rot60",
        "rot120",
        "rot180",
        "rot240",
        "rot300",
        "mir0",
        "mir30",
        "mir60",
        "mir90",
        "mir120",
        "mir150",
    }
    assert len(translations) == 1
    np.testing.assert_allclose(translations[0], (0.5, 0.5), atol=1e-6)


@pytest.mark.parametrize(
    "displacement, expected",
    [
        ((0.0, 0.0, 0.02), 8),  # along z: the four-fold axis and mirrors stay
        ((0.005, 0.0, 0.0), 2),  # along x: only the mirror normal to x
        ((0.01, 0.007, 0.0), 1),  # in a general direction: only the identity
    ],
)
def test_structure_symmetry_follows_small_displacements(displacement, expected):
    # Oracle: displacing every Ti by a vector lowers the point group to the operations
    # that
    # leave that vector invariant; the point group of a square lattice has 8.
    _, operations, translations = structure_symmetry(
        strontium_titanate(1, displacement), tolerance=1e-4
    )
    assert len(operations) == expected
    assert len(translations) == 0


@pytest.mark.float64
def test_empty_slab_transmits_the_wave_unchanged():
    # Oracle: no potential, no scattering: the transmitted wave has unit amplitude,
    # nothing
    # is reflected, and the phase is the vacuum phase, which the solver removes. The
    # Numerov dispersion over the slab is corrected with the exact discrete relation, so
    # nothing but rounding remains, at any step.
    def phase_error(slice_thickness):
        potential = abtem.Potential(
            ase.Atoms(cell=[2.46, 4.2608, 4.0]),
            gpts=(18, 30),
            slice_thickness=slice_thickness,
        )
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(cutoff=1.5).scatter(
                abtem.PlaneWave(energy=1000.0), potential
            )
        np.testing.assert_allclose(np.abs(transmitted.array), 1.0, atol=1e-8)
        np.testing.assert_allclose(reflected.array, 0.0, atol=1e-8)
        return np.abs(np.angle(transmitted.array)).max()

    assert phase_error(0.016) < 1e-9
    assert phase_error(0.008) < 1e-9


@pytest.mark.parametrize("energy, step", [(200.0, 0.02), (1000.0, 0.015)])
def test_flux_is_conserved(energy, step):
    # Oracle: without absorption the transmitted and reflected fluxes add up to the
    # incident
    # one; the cutoff is below the aliasing limit of the potential's sampling.
    potential = coarse_graphene(slice_thickness=step)
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=energy), potential
        )
    extent = potential.extent
    weights = flux_weights(transmitted.array.shape, extent, energy)
    flux = (np.abs(beam_amplitudes(transmitted)) ** 2 * weights).sum() + (
        np.abs(beam_amplitudes(reflected)) ** 2 * weights
    ).sum()
    assert abs(flux - 1.0) < 1e-4
    # and the reflection is real: evanescent coupling gives several percent at 200 eV
    if energy == 200.0:
        assert (np.abs(beam_amplitudes(reflected)) ** 2 * weights).sum() > 0.02


@pytest.mark.slow
@pytest.mark.float64
def test_agrees_with_multislice_where_multislice_is_accurate():
    # Oracle: at 20 keV, with slices much thinner than the 0.086 Å wavelength, split-
    # step
    # multislice is accurate and independent of the solver, so the two must agree in
    # amplitude and phase on every strong beam (the phase checks the convention that the
    # vacuum phase is removed).
    energy, cutoff, gpts = 20e3, 1.5, (16, 28)
    with abtem.config.set({"precision": "float64"}):
        potential = abtem.Potential(
            graphene_atoms(vacuum=1.5),
            gpts=gpts,
            slice_thickness=0.005,
            projection="finite",
            periodic=True,
        ).build(lazy=False)
        transmitted, reflected = TwoWaySolver(cutoff=cutoff).scatter(
            abtem.PlaneWave(energy=energy), potential
        )
        reference = abtem.PlaneWave(energy=energy).multislice(
            potential, algorithm=FourierMultislice(), lazy=False
        )
    a, b = beam_amplitudes(transmitted), beam_amplitudes(reference)
    g = np.hypot(
        np.fft.fftfreq(gpts[0], 2.46 / gpts[0])[:, None],
        np.fft.fftfreq(gpts[1], 4.2608 / gpts[1])[None],
    )
    strong = (np.abs(b) > 1e-3) & (g < cutoff)
    assert strong.sum() > 20
    assert np.abs(a - b)[strong].max() < 5e-4
    assert np.abs(reflected.array).max() < 1e-4


def tilted_beam_geometry(shape, extent, energy, tilt):
    """kz of every beam of a wave stored without the carrier of a beam of this tilt
    [mrad], and the kz of that beam: the stored beam g is the plane wave g + k_t."""
    tangent = np.tan(np.asarray(tilt) / 1e3)
    norm = np.sqrt(1 + np.sum(tangent**2))
    k = 2 * np.pi / energy2wavelength(energy)
    nx, ny = shape
    qx = np.fft.fftfreq(nx, extent[0] / nx)[:, None] + k * tangent[0] / norm / (
        2 * np.pi
    )
    qy = np.fft.fftfreq(ny, extent[1] / ny)[None, :] + k * tangent[1] / norm / (
        2 * np.pi
    )
    kz = np.sqrt(np.maximum(k**2 - 4 * np.pi**2 * (qx**2 + qy**2), 0.0))
    return kz, k / norm


@pytest.mark.float64
def test_a_tilted_empty_slab_transmits_the_wave_unchanged():
    # Oracle: no potential, no scattering, at any direction: the wave, stored without
    # the carrier of its direction, has unit amplitude and zero phase after the slab,
    # and nothing is reflected.
    potential = abtem.Potential(
        ase.Atoms(cell=[2.46, 4.2608, 4.0]), gpts=(18, 30), slice_thickness=0.016
    )
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=1.5).scatter(
            abtem.PlaneWave(energy=1000.0, tilt=(150.0, -60.0)), potential
        )
    np.testing.assert_allclose(transmitted.array, 1.0, atol=1e-8)
    np.testing.assert_allclose(reflected.array, 0.0, atol=1e-8)
    assert transmitted.base_tilt == (150.0, -60.0)


@pytest.mark.parametrize("tilt", [(80.0, 0.0), (40.0, -60.0)])
def test_flux_is_conserved_for_a_tilted_wave(tilt):
    # Oracle: without absorption the transmitted and reflected fluxes add up to the
    # incident one, at any direction of the incident wave: the flux of a beam is its
    # kz, over that of the incident beam.
    energy = COARSE_ENERGY
    potential = coarse_graphene()
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=energy, tilt=tilt), potential
        )
    kz, incident_kz = tilted_beam_geometry(
        transmitted.array.shape, potential.extent, energy, tilt
    )
    flux = (
        (
            np.abs(beam_amplitudes(transmitted)) ** 2
            + np.abs(beam_amplitudes(reflected)) ** 2
        )
        * kz
    ).sum() / incident_kz
    assert abs(flux - 1.0) < 1e-4


@pytest.mark.slow
@pytest.mark.float64
def test_agrees_with_multislice_for_a_tilted_wave():
    # Oracle: as without tilt, at 20 keV with slices much thinner than the wavelength
    # the exact-order multislice is accurate: in the frame of the tilted beam (the
    # wave is stored without the carrier of its direction) both give the same
    # amplitudes and phases on the strong beams, which checks the frame and the
    # vacuum phase of the oblique wave.
    energy, cutoff, gpts, tilt = 20e3, 1.4, (16, 28), (30.0, 10.0)
    with abtem.config.set({"precision": "float64"}):
        potential = abtem.Potential(
            graphene_atoms(vacuum=1.5),
            gpts=gpts,
            slice_thickness=0.005,
            projection="finite",
            periodic=True,
        ).build(lazy=False)
        transmitted, reflected = TwoWaySolver(cutoff=cutoff).scatter(
            abtem.PlaneWave(energy=energy, tilt=tilt), potential
        )
        reference = abtem.PlaneWave(energy=energy, tilt=tilt).multislice(
            potential, algorithm=FourierMultislice(), lazy=False
        )
    a, b = beam_amplitudes(transmitted), beam_amplitudes(reference)
    wavelength = energy2wavelength(energy)
    tangent = np.tan(np.asarray(tilt) / 1e3)
    k_t = tangent / (wavelength * np.sqrt(1 + np.sum(tangent**2)))
    q = np.hypot(
        np.fft.fftfreq(gpts[0], 2.46 / gpts[0])[:, None] + k_t[0],
        np.fft.fftfreq(gpts[1], 4.2608 / gpts[1])[None] + k_t[1],
    )
    strong = (np.abs(b) > 1e-3) & (q < cutoff)
    assert strong.sum() > 15
    assert np.abs(a - b)[strong].max() < 1e-3
    assert np.abs(reflected.array).max() < 1e-4


def single_precision_pair(energy, potential, cutoff):
    results = {}
    for precision in ("float64", "float32"):
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(
                cutoff=cutoff, precision=precision
            ).scatter(abtem.PlaneWave(energy=energy), potential)
        results[precision] = (beam_amplitudes(transmitted), beam_amplitudes(reflected))
    return results


def assert_single_precision_is_accurate(results):
    t64, r64 = results["float64"]
    t32, r32 = results["float32"]
    strong = np.abs(t64) > 1e-3
    relative = np.abs((t32 - t64)[strong] / t64[strong]).max()
    assert 1e-9 < relative < 1e-3
    assert np.abs(r32 - r64).max() < 1e-4


def test_single_precision_is_accurate_to_1e_3_on_the_strong_beams():
    # Oracle: the double precision solution. The evanescent beams make the blocks
    # ill-conditioned, but the error stays about 1e-4 relative on the strong beams
    # and 1e-5 absolute on the reflected wave, and it is not zero (a single precision
    # solve was done).
    assert_single_precision_is_accurate(
        single_precision_pair(COARSE_ENERGY, coarse_graphene(), cutoff=2.0)
    )


@pytest.mark.slow
@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_single_precision_is_accurate_on_a_converged_problem(energy):
    # As above for the converged problem, with 1200 slices of 0.01 Å to accumulate the
    # rounding over and the evanescent beams up to 3 / Å.
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    assert_single_precision_is_accurate(single_precision_pair(energy, potential, 3.0))


def test_the_precision_follows_the_configuration_by_default():
    # Oracle: the explicit precisions. The default follows abtem.config["precision"]
    # (single precision gives the single precision result, not the double one), and
    # the dtype of the waves follows it too.
    potential = coarse_graphene(slice_thickness=0.05)
    arrays = {}
    for configured in ("float32", "float64"):
        with abtem.config.set({"precision": configured}):
            transmitted, _ = TwoWaySolver(cutoff=2.0).scatter(
                abtem.PlaneWave(energy=COARSE_ENERGY), potential
            )
        assert transmitted.array.dtype == (
            np.complex64 if configured == "float32" else np.complex128
        )
        arrays[configured] = beam_amplitudes(transmitted)
    explicit = single_precision_pair(COARSE_ENERGY, potential, cutoff=2.0)
    np.testing.assert_allclose(arrays["float64"], explicit["float64"][0], atol=1e-12)
    # (the potential of the single precision configuration is itself single precision)
    np.testing.assert_allclose(arrays["float32"], explicit["float32"][0], atol=5e-5)
    assert np.abs(arrays["float32"] - arrays["float64"]).max() > 1e-9


@pytest.mark.parametrize("device", [cuda, "cpu"])
@pytest.mark.parametrize("tilt", [(0.0, 0.0), (60.0, -20.0)])
def test_the_device_gives_the_waves_of_the_cpu(device, tilt):
    # Oracle: the CPU result in double precision, to rounding (the CUDA case is skipped
    # without a CUDA device).
    with abtem.config.set({"precision": "float64"}):
        reference = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY, tilt=tilt), coarse_graphene()
        )
        result = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY, tilt=tilt, device=device),
            coarse_graphene(device=device),
        )
    for a, b in zip(result, reference):
        host = np.asarray(to_host_array(a))
        np.testing.assert_allclose(
            np.fft.fft2(host) / host.size, beam_amplitudes(b), atol=1e-9
        )


@pytest.mark.parametrize("tilt", [(60.0, 0.0), (0.0, -45.0), (50.0, 30.0)])
def test_the_symmetry_of_a_tilted_wave_gives_the_waves_of_the_full_basis(tilt):
    # Oracle: the full beam basis. A wave tilted along a mirror axis of the lattice
    # keeps the mirror, a wave in a general direction only the translations; either
    # way the reduced basis must reproduce the amplitudes and phases of the full one.
    potential = coarse_graphene()
    amplitudes = {}
    for symmetry in (True, False):
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(
                cutoff=2.0, symmetry=symmetry
            ).scatter(abtem.PlaneWave(energy=COARSE_ENERGY, tilt=tilt), potential)
        amplitudes[symmetry] = (
            beam_amplitudes(transmitted),
            beam_amplitudes(reflected),
        )
    for reduced, full in zip(amplitudes[True], amplitudes[False]):
        np.testing.assert_allclose(reduced, full, atol=2e-6)


@pytest.mark.float64
def test_an_ensemble_of_tilts_is_the_tilts_one_by_one():
    # Oracle: each member of the ensemble is the solution of that tilt alone; the
    # axes of the ensemble keep their order and their metadata.
    tilts_x, tilts_y = [0.0, 80.0], [0.0, 25.0]
    potential = coarse_graphene(slice_thickness=0.05)
    solver = TwoWaySolver(cutoff=1.0)
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = solver.scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY, tilt=(tilts_x, tilts_y)), potential
        )
        assert transmitted.shape == (2, 2, 24, 42)
        assert reflected.shape == (2, 2, 24, 42)
        assert len(transmitted.ensemble_axes_metadata) == 2
        for i, tx in enumerate(tilts_x):
            for j, ty in enumerate(tilts_y):
                single_t, single_r = solver.scatter(
                    abtem.PlaneWave(energy=COARSE_ENERGY, tilt=(tx, ty)), potential
                )
                np.testing.assert_allclose(
                    transmitted.array[i, j], single_t.array, atol=1e-9
                )
                np.testing.assert_allclose(
                    reflected.array[i, j], single_r.array, atol=1e-9
                )


def glide_slab():
    """A slab of the plane group pmg: two atoms, related by a mirror along y with a
    translation of half the cell along y (a glide), placed on the grid of the potential
    so that the sampled potential has the symmetry exactly."""
    a, b, nx, ny = 5.0, 6.0, 24, 32
    x0, y0 = 9 * a / nx, 13 * b / ny
    atoms = ase.Atoms(
        "Si2",
        positions=[[x0, y0, 4.0], [(-x0) % a, y0 + b / 2, 4.0]],
        cell=[a, b, 8.0],
        pbc=[True, True, False],
    )
    return atoms, (nx, ny)


def test_structure_glides_finds_the_glide_that_the_point_operations_miss():
    atoms, _ = glide_slab()
    origin, operations, translations = structure_symmetry(atoms)
    glides = structure_glides(atoms, origin, operations, translations)
    assert {name for name, _, _ in glides} == {"mir90", "rot180"}
    for _, matrix, translation in glides:
        # a half period along y, whatever the origin: the translation, taken in the
        # frame of the glide, moves the atoms by (0, b / 2) modulo the lattice
        moved = (atoms.positions[:, :2] - origin) @ matrix.T + origin + translation
        difference = moved[:, None, :] - atoms.positions[None, :, :2]
        difference -= np.round(difference / [5.0, 6.0]) * [5.0, 6.0]
        assert (np.linalg.norm(difference, axis=-1).min(axis=1) < 1e-6).all()


def test_symmorphic_structures_have_no_glides():
    # Oracle: graphene and SrTiO3 (symmorphic): every operation of the point group is
    # found about the origin.
    for atoms in (graphene_atoms(), strontium_titanate(2)):
        origin, operations, translations = structure_symmetry(atoms)
        assert structure_glides(atoms, origin, operations, translations) == []


@pytest.mark.parametrize("tilt", [(0.0, 0.0), (0.0, 60.0), (45.0, 0.0)])
def test_the_glide_gives_the_waves_of_the_full_basis_and_the_systematic_absences(tilt):
    # Oracle: the full beam basis. The reduced basis (a glide gives orbits with fixed
    # phases, and none at all where the beams are absent) reproduces the amplitudes
    # and phases of the full one, with and without a tilt along the mirror, and the
    # beams (0, odd) that the glide extinguishes are zero in the full solution.
    atoms, gpts = glide_slab()
    potential = abtem.Potential(
        atoms, gpts=gpts, slice_thickness=0.05, projection="finite", periodic=True
    )
    amplitudes = {}
    for symmetry in (True, False):
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(
                cutoff=1.0, symmetry=symmetry
            ).scatter(abtem.PlaneWave(energy=COARSE_ENERGY, tilt=tilt), potential)
        amplitudes[symmetry] = (
            beam_amplitudes(transmitted),
            beam_amplitudes(reflected),
        )
    for reduced, full in zip(amplitudes[True], amplitudes[False]):
        np.testing.assert_allclose(reduced, full, atol=1e-10)
    if tilt[0] == 0.0:
        absent = amplitudes[False][0][0, 1::2]
        assert np.abs(absent[: gpts[1] // 2 // 2]).max() < 1e-10


@pytest.mark.float64
def test_an_ensemble_of_energies_and_tilts_is_the_members_one_by_one():
    # Oracle: each member is the solution of its energy and tilt alone, in the order of
    # the axes of the ensemble (here the tilts first, then the energies).
    tilts, energies = [0.0, 40.0], [250.0, COARSE_ENERGY]
    potential = coarse_graphene(slice_thickness=0.05)
    solver = TwoWaySolver(cutoff=1.0)
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = solver.scatter(
            abtem.PlaneWave(energy=energies, tilt=(tilts, 0.0)), potential
        )
        assert transmitted.shape == (2, 2, 24, 42)
        for i, tilt in enumerate(tilts):
            for j, energy in enumerate(energies):
                single_t, single_r = solver.scatter(
                    abtem.PlaneWave(energy=energy, tilt=(tilt, 0.0)), potential
                )
                np.testing.assert_allclose(
                    transmitted.array[i, j], single_t.array, atol=1e-9
                )
                np.testing.assert_allclose(
                    reflected.array[i, j], single_r.array, atol=1e-9
                )
        # the energy matters: the members differ
        assert np.abs(transmitted.array[0, 0] - transmitted.array[0, 1]).max() > 1e-3


def test_an_ensemble_of_energies_follows_the_wavelength():
    # Oracle: the reflection of graphene falls with the energy, 7 % at 200 eV and less
    # than 1 % at 1 keV, and the flux of every member is conserved.
    potential = coarse_graphene(slice_thickness=0.015)
    energies = [200.0, 1000.0]
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=energies), potential
        )
    reflections = []
    for index, energy in enumerate(energies):
        weights = flux_weights(transmitted.array.shape[-2:], potential.extent, energy)
        r = (np.abs(beam_amplitudes(reflected[index])) ** 2 * weights).sum()
        t = (np.abs(beam_amplitudes(transmitted[index])) ** 2 * weights).sum()
        assert abs(r + t - 1.0) < 1e-4
        reflections.append(r)
    assert reflections[0] > 0.02 and reflections[1] < 0.01


@pytest.mark.float64
@pytest.mark.parametrize("absorption", [0.0, 0.1, 0.5])
def test_a_flat_absorbing_slab_follows_the_airy_formulas(absorption):
    # Oracle: a laterally uniform potential is a slab of a medium with the complex wave
    # number k2 = sqrt(k^2 + 2 k sigma V (1 + i absorption)), and the amplitudes of a
    # plane wave on it are the Fresnel coefficients with the multiple reflection of its
    # two faces; the vacuum phase is removed. The step of the Numerov method costs 2e-4.
    energy, thickness, step, volts = 1000.0, 2.0, 0.01, 15.0
    wavelength = energy2wavelength(energy)
    k, sigma = 2 * np.pi / wavelength, energy2sigma(energy)
    count = int(round(thickness / step))
    projected = np.full((count, 24, 24), volts * step)
    potential = abtem.PotentialArray(projected, slice_thickness=step, extent=(3.0, 3.0))
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(
            cutoff=1.0, symmetry=False, absorption=absorption
        ).scatter(abtem.PlaneWave(energy=energy), potential)

    k2 = np.sqrt(k**2 + 2 * k * sigma * volts * (1 + 1j * absorption) + 0j)
    r12, t12, t21 = (k - k2) / (k + k2), 2 * k / (k + k2), 2 * k2 / (k + k2)
    r21 = -r12
    round_trip = np.exp(2j * k2 * thickness)
    expected_t = (
        t12 * t21 * np.exp(1j * (k2 - k) * thickness) / (1 - r21**2 * round_trip)
    )
    expected_r = r12 + t12 * t21 * r21 * round_trip / (1 - r21**2 * round_trip)
    t, r = transmitted.array[0, 0], reflected.array[0, 0]
    assert abs(t - expected_t) < 5e-4
    assert abs(r - expected_r) < 1e-4
    flux = abs(t) ** 2 + abs(r) ** 2
    assert (flux < 1 - 1e-3) == (absorption > 0)
    # the absorbed part is what the analytic amplitudes lack
    assert flux == pytest.approx(abs(expected_t) ** 2 + abs(expected_r) ** 2, abs=1e-3)


@pytest.mark.float64
def test_absorption_is_the_imaginary_potential_that_scatter_can_also_be_given():
    # Oracle: absorption mu is an imaginary potential mu V, so the same result follows
    # from an absorptive potential of mu V (built as an array), and the two add up.
    potential = coarse_graphene()
    wave = abtem.PlaneWave(energy=COARSE_ENERGY)
    with abtem.config.set({"precision": "float64"}):
        built = potential.build(lazy=False)
        scaled = abtem.PotentialArray(
            0.1 * np.asarray(built.array),
            slice_thickness=built.slice_thickness,
            extent=built.extent,
        )
        proportional = TwoWaySolver(cutoff=1.5, absorption=0.1).scatter(wave, potential)
        separate = TwoWaySolver(cutoff=1.5).scatter(wave, potential, scaled)
        both = TwoWaySolver(cutoff=1.5, absorption=0.05).scatter(
            wave,
            potential,
            abtem.PotentialArray(
                0.05 * np.asarray(built.array),
                slice_thickness=built.slice_thickness,
                extent=built.extent,
            ),
        )
        lossless = TwoWaySolver(cutoff=1.5).scatter(wave, potential)
    for a, b, c, d in zip(proportional, separate, both, lossless):
        np.testing.assert_allclose(a.array, b.array, atol=1e-9)
        np.testing.assert_allclose(a.array, c.array, atol=1e-9)
        assert np.abs(a.array - d.array).max() > 1e-4


@pytest.mark.float64
def test_symmetry_reduction_with_absorption_gives_the_waves_of_the_full_basis():
    # Oracle: the full beam basis. The symmetry does not need the coupling matrix to be
    # Hermitian; the loss of flux is the same in both.
    potential = coarse_graphene()
    waves = {}
    with abtem.config.set({"precision": "float64"}):
        for symmetry in (True, False):
            waves[symmetry] = TwoWaySolver(
                cutoff=2.0, symmetry=symmetry, absorption=0.2
            ).scatter(abtem.PlaneWave(energy=COARSE_ENERGY), potential)
    # (the rectangular grid does not hold the sixfold axis exactly: 2e-5 remains)
    for reduced, full in zip(waves[True], waves[False]):
        np.testing.assert_allclose(
            beam_amplitudes(reduced), beam_amplitudes(full), atol=5e-5
        )
    weights = flux_weights((24, 42), potential.extent, COARSE_ENERGY)
    flux = sum((np.abs(beam_amplitudes(w)) ** 2 * weights).sum() for w in waves[False])
    assert flux < 0.95


def test_absorption_is_validated():
    with pytest.raises(ValueError, match="absorption"):
        TwoWaySolver(absorption=-0.1)

    potential = coarse_graphene()
    built = potential.build(lazy=False)
    negative = abtem.PotentialArray(
        -np.asarray(built.array),
        slice_thickness=built.slice_thickness,
        extent=built.extent,
    )
    with pytest.raises(ValueError, match="must not be negative"):
        TwoWaySolver(cutoff=1.5).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY), potential, negative
        )
    other = abtem.Potential(
        graphene_atoms(vacuum=5.0),
        gpts=(24, 42),
        slice_thickness=0.04,
        projection="finite",
    )
    with pytest.raises(ValueError, match="same atoms|grid"):
        TwoWaySolver(cutoff=1.5).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY), potential, other
        )


def test_symmetry_reduction_gives_the_same_waves_as_the_full_basis():
    # Oracle: the full beam basis. The reduced basis must reproduce its amplitudes,
    # including phase, on every beam.
    potential = abtem.Potential(
        strontium_titanate(1),
        gpts=(24, 24),
        slice_thickness=0.05,
        projection="finite",
        periodic=True,
    )
    with abtem.config.set({"precision": "float64"}):
        reduced = TwoWaySolver(cutoff=1.5, symmetry=True).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY), potential
        )
        full = TwoWaySolver(cutoff=1.5, symmetry=False).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY), potential
        )
    for waves_reduced, waves_full in zip(reduced, full):
        np.testing.assert_allclose(waves_reduced.array, waves_full.array, atol=1e-8)
    assert np.abs(full[0].array - 1).max() > 0.05  # a nontrivial wave


def test_a_built_potential_gives_the_same_waves_as_the_potential():
    # A built PotentialArray carries no atoms, so it is solved in the full beam basis;
    # the symmetry-reduced solution of the potential differs by the asymmetry of the
    # sampled potential, about 1e-4 for graphene on this rectangular grid.
    potential = coarse_graphene()
    solver = TwoWaySolver(cutoff=1.5)
    from_potential = solver.scatter(abtem.PlaneWave(energy=COARSE_ENERGY), potential)
    from_built = solver.scatter(
        abtem.PlaneWave(energy=COARSE_ENERGY), potential.build(lazy=False)
    )
    for a, b in zip(from_potential, from_built):
        np.testing.assert_allclose(a.array, b.array, atol=3e-4)


def test_the_solver_refuses_unsupported_input(monkeypatch):
    # The step and the sampling are checked before the potential is built, which for
    # slices this thin would take minutes.
    def refuse_to_build(*args, **kwargs):
        raise AssertionError("the potential was built before the input was refused")

    monkeypatch.setattr(abtem.Potential, "build", refuse_to_build)
    coarse = coarse_graphene(slice_thickness=0.0002)
    with pytest.raises(ValueError, match="sampled too coarsely"):
        TwoWaySolver(cutoff=4.5).scatter(abtem.PlaneWave(energy=1000.0), coarse)

    with pytest.raises(ValueError, match="too thick for the Numerov step"):
        TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=200e3),
            coarse_graphene(),
        )

    uneven = coarse_graphene(slice_thickness=[2.0, 4.0])
    with pytest.raises(ValueError, match="equal thickness"):
        TwoWaySolver(cutoff=1.0).scatter(abtem.PlaneWave(energy=1000.0), uneven)

    with pytest.raises(ValueError, match="cutoff"):
        TwoWaySolver(cutoff=0.0)

    with pytest.raises(ValueError, match="precision"):
        TwoWaySolver(precision="float16")


def spglib_inplane_symmetry(atoms, symprec=1e-4):
    """(in-plane point operations, translations) that spglib finds for the slab.

    spglib sees the lattice it is given: for a supercell it only finds the operations
    that map that lattice onto itself, and a space group lists operations about any
    origin only through their translations. So it is the reference for lattices aligned
    with the cell, and a lower bound on the point operations otherwise.
    """
    spglib = pytest.importorskip("spglib")
    cell = (atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers)
    with warnings.catch_warnings():
        # spglib warns about its own internal error handling
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry(cell, symprec=symprec)
    lattice = atoms.cell.array
    in_plane = []
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
        cartesian = lattice.T @ rotation @ np.linalg.inv(lattice.T)
        keeps_z = (
            abs(cartesian[2, 2] - 1) < 1e-8
            and np.abs(cartesian[2, :2]).max() < 1e-8
            and np.abs(cartesian[:2, 2]).max() < 1e-8
            and abs(translation[2] - np.round(translation[2])) < 1e-6
        )
        if keeps_z:
            in_plane.append((cartesian[:2, :2], translation[:2]))
    pure = [
        t
        for matrix, t in in_plane
        if np.allclose(matrix, np.eye(2)) and np.abs(t - np.round(t)).max() > 1e-6
    ]
    return len(in_plane) // (1 + len(pure)), len(pure)


@pytest.mark.parametrize(
    "displacement",
    [(0.0, 0.0, 0.0), (0.0, 0.0, 0.02), (0.005, 0.0, 0.0), (0.01, 0.007, 0.0)],
)
def test_structure_symmetry_agrees_with_spglib_on_a_square_lattice(displacement):
    # Oracle: spglib, an independent implementation, on a lattice the cell aligns with.
    atoms = strontium_titanate(2, displacement)
    _, operations, translations = structure_symmetry(atoms)
    order, number_of_translations = spglib_inplane_symmetry(atoms)
    assert len(operations) == order
    assert len(translations) == number_of_translations


@pytest.mark.parametrize("layers", [1, 4])
def test_structure_symmetry_finds_what_spglib_finds_in_a_hexagonal_supercell(layers):
    # spglib is bound to the rectangular supercell lattice, so it finds a subgroup of
    # the hexagonal point group; the translation of the centred cell is the same.
    atoms = graphene_atoms() if layers == 1 else bernal_graphite(layers)
    _, operations, translations = structure_symmetry(atoms)
    order, number_of_translations = spglib_inplane_symmetry(atoms)
    assert len(operations) % order == 0 and len(operations) > order
    assert len(translations) == number_of_translations


# --- the absorber from an inelastic mean free path ---


def flat_slab(volts, thickness=2.0, step=0.01):
    count = int(round(thickness / step))
    return abtem.PotentialArray(
        np.full((count, 24, 24), volts * step), slice_thickness=step, extent=(3.0, 3.0)
    )


@pytest.mark.float64
@pytest.mark.parametrize("volts", [0.0, 15.0])
def test_a_mean_free_path_absorber_follows_the_airy_formulas(volts):
    # Oracle: a laterally uniform slab of a medium of complex wave number
    # k2 = sqrt(k^2 + 2 k sigma V + i k / mean_free_path), Fresnel coefficients with the
    # multiple reflection of the two faces, as for the proportional absorption. For a
    # clean slab the intensity decays as exp(-thickness / mean_free_path) (to the
    # reflection at the faces).
    energy, thickness, mean_free_path = COARSE_ENERGY, 2.0, 2.0
    k, sigma = 2 * np.pi / energy2wavelength(energy), energy2sigma(energy)
    potential = flat_slab(volts, thickness)
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(
            cutoff=1.0, symmetry=False, absorber=MeanFreePathAbsorber(mean_free_path)
        ).scatter(abtem.PlaneWave(energy=energy), potential)
    k2 = np.sqrt(k**2 + 2 * k * sigma * volts + 1j * k / mean_free_path)
    r12, t12, t21 = (k - k2) / (k + k2), 2 * k / (k + k2), 2 * k2 / (k + k2)
    round_trip = np.exp(2j * k2 * thickness)
    expected_t = (
        t12 * t21 * np.exp(1j * (k2 - k) * thickness) / (1 - r12**2 * round_trip)
    )
    expected_r = r12 + t12 * t21 * (-r12) * round_trip / (1 - r12**2 * round_trip)
    t, r = transmitted.array[0, 0], reflected.array[0, 0]
    assert abs(t - expected_t) < 5e-4
    assert abs(r - expected_r) < 5e-4
    if volts == 0.0:
        assert abs(t) ** 2 == pytest.approx(
            np.exp(-thickness / mean_free_path), rel=5e-3
        )


@pytest.mark.float64
def test_the_absorber_acts_only_in_its_range():
    # Oracle: the intensity that crosses the slab falls by exp(-length / mean free
    # path) over the length of the slices whose centres are in the range; a long mean
    # free path keeps the reflection of the interfaces negligible.
    energy, mean_free_path = COARSE_ENERGY, 10.0
    absorber = MeanFreePathAbsorber(mean_free_path, z_range=(0.5, 1.5))
    potential = flat_slab(0.0)
    length = absorber.slices(potential.num_slices, 0.01).sum() * 0.01
    assert length == pytest.approx(1.0, abs=0.02)
    with abtem.config.set({"precision": "float64"}):
        transmitted, _ = TwoWaySolver(
            cutoff=1.0, symmetry=False, absorber=absorber
        ).scatter(abtem.PlaneWave(energy=energy), potential)
    assert abs(transmitted.array[0, 0]) ** 2 == pytest.approx(
        np.exp(-length / mean_free_path), rel=1e-3
    )


@pytest.mark.float64
def test_the_absorber_is_the_imaginary_potential_of_one_over_two_sigma_lambda():
    # Oracle: the absorptive potential of the absorber, V_i = 1 / (2 sigma lambda) built
    # as an array with the potential's grid, and given to scatter, in the symmetry
    # reduced basis; the interaction constant cancels in the wave equation, so this is
    # an independent route to the same operator.
    potential = coarse_graphene()
    absorber = MeanFreePathAbsorber(3.0, z_range=(0.5, 5.0))
    wave = abtem.PlaneWave(energy=COARSE_ENERGY)
    assert absorber.imaginary_potential(COARSE_ENERGY) == pytest.approx(
        1.0 / (2.0 * energy2sigma(COARSE_ENERGY) * 3.0)
    )
    with abtem.config.set({"precision": "float64"}):
        by_array = TwoWaySolver(cutoff=1.5).scatter(
            wave, potential, absorber.absorptive_potential(potential, COARSE_ENERGY)
        )
        by_absorber = TwoWaySolver(cutoff=1.5, absorber=absorber).scatter(
            wave, potential
        )
        lossless = TwoWaySolver(cutoff=1.5).scatter(wave, potential)
    for a, b, c in zip(by_array, by_absorber, lossless):
        np.testing.assert_allclose(a.array, b.array, atol=1e-9)
        assert np.abs(a.array - c.array).max() > 1e-3


@pytest.mark.float64
def test_the_absorber_follows_the_energy_of_every_member():
    # Oracle: each member of an ensemble of energies is the solution of its energy
    # alone, with the mean free path of that energy (here a function of it), in the
    # symmetry-reduced and the full basis.
    def mean_free_path(energy):
        return 1.0 + energy / 100.0

    potential = coarse_graphene(slice_thickness=0.05)
    energies = [250.0, COARSE_ENERGY]
    absorber = MeanFreePathAbsorber(mean_free_path)
    with abtem.config.set({"precision": "float64"}):
        solver = TwoWaySolver(cutoff=1.0, absorber=absorber)
        transmitted, reflected = solver.scatter(
            abtem.PlaneWave(energy=energies), potential
        )
        for i, energy in enumerate(energies):
            single_t, single_r = solver.scatter(
                abtem.PlaneWave(energy=energy), potential
            )
            np.testing.assert_allclose(transmitted.array[i], single_t.array, atol=1e-9)
            np.testing.assert_allclose(reflected.array[i], single_r.array, atol=1e-9)
        full = TwoWaySolver(cutoff=1.0, absorber=absorber, symmetry=False).scatter(
            abtem.PlaneWave(energy=COARSE_ENERGY), potential
        )
    np.testing.assert_allclose(transmitted.array[1], full[0].array, atol=5e-5)


def test_the_absorber_is_validated():
    with pytest.raises(ValueError, match="mean_free_path"):
        MeanFreePathAbsorber(-1.0)
    with pytest.raises(ValueError, match="z_range"):
        MeanFreePathAbsorber(1.0, z_range=(2.0, 1.0))
    with pytest.raises(TypeError, match="absorber"):
        TwoWaySolver(absorber=3.0)
    with pytest.raises(ValueError, match="not positive"):
        MeanFreePathAbsorber(lambda energy: -1.0).imaginary_potential(100.0)
