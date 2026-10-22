import warnings

import ase
import ase.build
import numpy as np
import pytest
from utils import _ACCELERATOR, to_host_array

import abtem
from abtem.core.energy import energy2wavelength
from abtem.multislice import FourierMultislice
from abtem.twoway import TwoWaySolver, structure_symmetry

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
            ase.Atoms(cell=[2.46, 4.2608, 8.0]),
            gpts=(48, 84),
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


@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_flux_is_conserved(energy):
    # Oracle: without absorption the transmitted and reflected fluxes add up to the
    # incident
    # one; the cutoff is below the aliasing limit of the potential's sampling.
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=3.0).scatter(
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
            graphene_atoms(vacuum=3.0),
            gpts=gpts,
            slice_thickness=0.0035,
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
        ase.Atoms(cell=[2.46, 4.2608, 8.0]), gpts=(48, 84), slice_thickness=0.016
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
    energy = 1000.0
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    with abtem.config.set({"precision": "float64"}):
        transmitted, reflected = TwoWaySolver(cutoff=2.5).scatter(
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
            graphene_atoms(vacuum=3.0),
            gpts=gpts,
            slice_thickness=0.0035,
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


def single_precision_pair(energy, solver_kwargs=None):
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    results = {}
    for precision in ("float64", "float32"):
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(
                cutoff=3.0, precision=precision, **(solver_kwargs or {})
            ).scatter(abtem.PlaneWave(energy=energy), potential)
        results[precision] = (beam_amplitudes(transmitted), beam_amplitudes(reflected))
    return results


@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_single_precision_is_accurate_to_1e_3_on_the_strong_beams(energy):
    # Oracle: the double precision solution. The evanescent beams make the blocks
    # ill-conditioned, but the error stays about 1e-4 relative on the strong beams
    # and 1e-5 absolute on the reflected wave, and it is not zero (a single precision
    # solve was done).
    results = single_precision_pair(energy)
    t64, r64 = results["float64"]
    t32, r32 = results["float32"]
    strong = np.abs(t64) > 1e-3
    relative = np.abs((t32 - t64)[strong] / t64[strong]).max()
    assert 1e-9 < relative < 1e-3
    assert np.abs(r32 - r64).max() < 1e-4


def test_the_precision_follows_the_configuration_by_default():
    # Oracle: the explicit precisions. The default follows abtem.config["precision"]
    # (single precision gives the single precision result, not the double one), and
    # the dtype of the waves follows it too.
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    arrays = {}
    for configured in ("float32", "float64"):
        with abtem.config.set({"precision": configured}):
            transmitted, _ = TwoWaySolver(cutoff=3.0).scatter(
                abtem.PlaneWave(energy=1000.0), potential
            )
        assert transmitted.array.dtype == (
            np.complex64 if configured == "float32" else np.complex128
        )
        arrays[configured] = beam_amplitudes(transmitted)
    explicit = single_precision_pair(1000.0)
    np.testing.assert_allclose(arrays["float64"], explicit["float64"][0], atol=1e-12)
    # (the potential of the single precision configuration is itself single precision)
    np.testing.assert_allclose(arrays["float32"], explicit["float32"][0], atol=5e-5)
    assert np.abs(arrays["float32"] - arrays["float64"]).max() > 1e-9


@pytest.mark.parametrize("device", [cuda, "cpu"])
@pytest.mark.parametrize("tilt", [(0.0, 0.0), (60.0, -20.0)])
def test_the_device_gives_the_waves_of_the_cpu(device, tilt):
    # Oracle: the CPU result in double precision, to rounding (the CUDA case is skipped
    # without a CUDA device).
    kwargs = dict(
        gpts=(48, 84), slice_thickness=0.01, projection="finite", periodic=True
    )
    with abtem.config.set({"precision": "float64"}):
        reference = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=1000.0, tilt=tilt),
            abtem.Potential(graphene_atoms(), **kwargs),
        )
        result = TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=1000.0, tilt=tilt, device=device),
            abtem.Potential(graphene_atoms(), device=device, **kwargs),
        )
    for a, b in zip(result, reference):
        host = np.asarray(to_host_array(a))
        np.testing.assert_allclose(
            np.fft.fft2(host) / host.size, beam_amplitudes(b), atol=1e-9
        )


@pytest.mark.parametrize("tilt", [(60.0, 0.0), (0.0, -45.0), (50.0, 30.0)])
def test_the_symmetry_of_a_tilted_wave_gives_the_waves_of_the_full_basis(tilt):
    # Oracle: the full beam basis. A wave tilted along a mirror axis of the lattice
    # keeps the mirror, a wave in a general direction only the translations; either way the
    # reduced basis must reproduce the amplitudes and phases of the full one.
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.01,
        projection="finite",
        periodic=True,
    )
    amplitudes = {}
    for symmetry in (True, False):
        with abtem.config.set({"precision": "float64"}):
            transmitted, reflected = TwoWaySolver(
                cutoff=2.5, symmetry=symmetry
            ).scatter(abtem.PlaneWave(energy=1000.0, tilt=tilt), potential)
        amplitudes[symmetry] = (
            beam_amplitudes(transmitted),
            beam_amplitudes(reflected),
        )
    for reduced, full in zip(amplitudes[True], amplitudes[False]):
        np.testing.assert_allclose(reduced, full, atol=2e-6)


def test_symmetry_reduction_gives_the_same_waves_as_the_full_basis():
    # Oracle: the full beam basis. The reduced basis must reproduce its amplitudes,
    # including phase, on every beam.
    potential = abtem.Potential(
        strontium_titanate(2),
        gpts=(30, 30),
        slice_thickness=0.015,
        projection="finite",
        periodic=True,
    )
    with abtem.config.set({"precision": "float64"}):
        reduced = TwoWaySolver(cutoff=1.5, symmetry=True).scatter(
            abtem.PlaneWave(energy=1000.0), potential
        )
        full = TwoWaySolver(cutoff=1.5, symmetry=False).scatter(
            abtem.PlaneWave(energy=1000.0), potential
        )
    for waves_reduced, waves_full in zip(reduced, full):
        np.testing.assert_allclose(waves_reduced.array, waves_full.array, atol=1e-8)
    assert np.abs(full[0].array - 1).max() > 0.05  # a nontrivial wave


def test_a_built_potential_gives_the_same_waves_as_the_potential():
    # A built PotentialArray carries no atoms, so it is solved in the full beam basis;
    # the
    # symmetry-reduced solution of the potential differs by the asymmetry of the sampled
    # potential, about 1e-4 for graphene on a rectangular grid.
    potential = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=0.015,
        projection="finite",
        periodic=True,
    )
    solver = TwoWaySolver(cutoff=1.5)
    from_potential = solver.scatter(abtem.PlaneWave(energy=1000.0), potential)
    from_built = solver.scatter(
        abtem.PlaneWave(energy=1000.0), potential.build(lazy=False)
    )
    for a, b in zip(from_potential, from_built):
        np.testing.assert_allclose(a.array, b.array, atol=1e-4)


def test_the_solver_refuses_unsupported_input():
    potential = abtem.Potential(
        graphene_atoms(), gpts=(48, 84), slice_thickness=0.01, projection="finite"
    )
    with pytest.raises(NotImplementedError, match="tilt ensembles"):
        TwoWaySolver().scatter(
            abtem.PlaneWave(energy=1000.0, tilt=([5.0, 10.0], 0.0)), potential
        )

    coarse = abtem.Potential(
        graphene_atoms(), gpts=(24, 42), slice_thickness=0.01, projection="finite"
    )
    with pytest.raises(ValueError, match="sampled too coarsely"):
        TwoWaySolver(cutoff=4.5).scatter(abtem.PlaneWave(energy=1000.0), coarse)

    with pytest.raises(ValueError, match="too thick for the Numerov step"):
        TwoWaySolver(cutoff=2.0).scatter(
            abtem.PlaneWave(energy=200e3),
            abtem.Potential(
                graphene_atoms(),
                gpts=(24, 42),
                slice_thickness=0.02,
                projection="finite",
            ),
        )

    uneven = abtem.Potential(
        graphene_atoms(),
        gpts=(48, 84),
        slice_thickness=[4.0, 8.0],
        projection="finite",
    )
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
