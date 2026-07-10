"""Tests for the paraxial Pauli multislice solver (abtem.magnetism.multislice).

The analytic references follow Edström, Lubk & Rusz, PRB 94, 174414 (2016):
the spin of the beam precesses about a constant magnetic field at the rate
e*B/(hbar*k) per unit length [Eq. (21)], and a vortex beam with OAM l in the
vector potential A = B x r / 2 acquires phase at half that rate per quantum
of l — the g = 2 ratio of spin and orbital moments [Eq. (19)].
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem.core.axes import SpinAxis
from abtem.core.energy import energy2wavelength
from abtem.magnetism.iam import (
    MagneticField,
    MagneticFieldArray,
    VectorPotential,
    VectorPotentialArray,
)
from abtem.magnetism.multislice import e_over_hbar, pauli_multislice
from abtem.magnetism.utils import set_magnetic_moments
from abtem.multislice import RealSpaceMultislice, multislice_and_detect

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)

ENERGY = 100e3


@pytest.fixture(autouse=True)
def float64_precision():
    previous = abtem.config.get("precision")
    abtem.config.set({"precision": "float64"})
    yield
    abtem.config.set({"precision": previous})


def to_numpy(array):
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)


def zero_fields(n_slices, gpts, extent, slice_thickness):
    zeros = np.zeros((n_slices, 3, gpts, gpts))
    A = VectorPotentialArray(
        zeros.copy(), extent=(extent, extent), slice_thickness=slice_thickness
    )
    B = MagneticFieldArray(
        zeros.copy(), extent=(extent, extent), slice_thickness=slice_thickness
    )
    return A, B


def vacuum_potential(n_slices, gpts, extent, slice_thickness):
    return abtem.PotentialArray(
        np.zeros((n_slices, gpts, gpts)),
        slice_thickness=slice_thickness,
        extent=(extent, extent),
    )


def spin_expectation(array):
    up, down = array[0].ravel(), array[1].ravel()
    n = (np.abs(up) ** 2 + np.abs(down) ** 2).sum()
    sx = 2 * np.real(np.conj(up) @ down) / n
    sy = 2 * np.imag(np.conj(up) @ down) / n
    sz = ((np.abs(up) ** 2 - np.abs(down) ** 2).sum()) / n
    return np.array([sx, sy, sz])


def test_to_spinor():
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, gpts=64, extent=10)
    waves = probe.build(lazy=False)

    spinor = waves.to_spinor((1, 1j))

    assert spinor.shape == (2,) + waves.shape
    assert isinstance(spinor.ensemble_axes_metadata[0], SpinAxis)

    intensity = np.abs(to_numpy(spinor.array)) ** 2
    total = np.abs(to_numpy(waves.array)) ** 2
    assert np.allclose(intensity.sum(0), total, atol=1e-12)
    assert np.allclose(intensity[0], intensity[1])

    lazy = probe.build(lazy=True).to_spinor()
    assert lazy.array.chunks[0] == (2,)


def test_requires_spin_axis():
    gpts, extent, n, dz = 16, 10.0, 2, 1.0
    potential = vacuum_potential(n, gpts, extent, dz)
    A, B = zero_fields(n, gpts, extent, dz)
    waves = abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent).build(lazy=False)

    with pytest.raises(ValueError, match="to_spinor"):
        pauli_multislice(waves, potential, vector_potential=A, magnetic_field=B)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_scalar_limit(device):
    """With A = B = 0 each spinor channel must match the scalar real-space
    multislice."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    potential = abtem.Potential(
        atoms, gpts=64, slice_thickness=1.35, device=device
    ).build(lazy=False)

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25, device=device)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    reference = multislice_and_detect(
        waves.copy(), potential, detectors=None, algorithm=RealSpaceMultislice()
    )[0]

    n = potential.num_slices
    A, B = zero_fields(n, potential.gpts[0], potential.extent[0], 1.35)

    out = pauli_multislice(
        waves.copy().to_spinor((0.6, 0.8j)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    )

    reference_array = to_numpy(reference.array)
    out_array = to_numpy(out.array)
    scale = np.abs(reference_array).max()

    assert np.abs(out_array[0] - 0.6 * reference_array).max() / scale < 1e-12
    assert np.abs(out_array[1] - 0.8j * reference_array).max() / scale < 1e-12


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_constant_field_spin_precession(device):
    """Spin initially along z precesses about a constant B along x at the
    analytic rate e*B/(hbar*k)."""
    gpts, extent = 32, 20.0
    n_slices, dz = 50, 10.0
    B0 = 2e4  # T; exaggerated to give an O(1 rad) rotation

    potential = vacuum_potential(n_slices, gpts, extent, dz)
    A, _ = zero_fields(n_slices, gpts, extent, dz)

    B_array = np.zeros((n_slices, 3, gpts, gpts))
    B_array[:, 0] = B0 * dz  # projected (slice-integrated) field
    B = MagneticFieldArray(B_array, slice_thickness=dz, extent=(extent, extent))

    waves = abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent, device=device)
    spinor = waves.build(lazy=False).to_spinor((1, 0))

    out = pauli_multislice(spinor, potential, vector_potential=A, magnetic_field=B)

    wavelength = energy2wavelength(ENERGY)
    theta = e_over_hbar * B0 * wavelength / (2 * np.pi) * (n_slices * dz)
    expected = np.array([0.0, -np.sin(theta), np.cos(theta)])

    s = spin_expectation(to_numpy(out.array))
    assert np.allclose(s, expected, atol=1e-6)

    # a spin parallel to the field is stationary
    spinor = waves.build(lazy=False).to_spinor((1, 1))
    out = pauli_multislice(spinor, potential, vector_potential=A, magnetic_field=B)
    s = spin_expectation(to_numpy(out.array))
    assert np.allclose(s, [1, 0, 0], atol=1e-6)


def test_average_field_precession():
    """The uniform-field path (average_field -> A_np + constant Zeeman)
    reproduces the analytic precession for a realistic field strength."""
    gpts, extent = 32, 20.0
    n_slices, dz = 50, 10.0
    B0 = 2.0  # T, like the saturation field of Fe

    potential = vacuum_potential(n_slices, gpts, extent, dz)
    A, B = zero_fields(n_slices, gpts, extent, dz)

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent)
        .build(lazy=False)
        .to_spinor((1, 0))
    )

    out = pauli_multislice(
        spinor,
        potential,
        vector_potential=A,
        magnetic_field=B,
        average_field=(B0, 0, 0),
    )

    wavelength = energy2wavelength(ENERGY)
    theta = e_over_hbar * B0 * wavelength / (2 * np.pi) * (n_slices * dz)

    s = spin_expectation(to_numpy(out.array))
    assert abs(s[1] + theta) < 1e-7
    assert abs(s[2] - 1) < 1e-6

    # norm conservation (small loss from bandlimiting the non-periodic A_np
    # phase at the supercell boundary is expected)
    norm = (np.abs(to_numpy(out.array)) ** 2).sum() / (
        np.abs(to_numpy(spinor.array)) ** 2
    ).sum()
    assert abs(norm - 1) < 1e-4


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("l", [1, -1])
def test_vortex_orbital_phase(device, l):
    """A vortex beam with OAM l in A = B x r / 2 acquires the orbital Zeeman
    phase -e*B*l/(2*hbar*k) per unit length."""
    gpts, extent = 128, 40.0
    n_slices, dz = 10, 1.0
    B0 = 5e4  # T

    x = (np.arange(gpts) - gpts / 2) * (extent / gpts)
    X, Y = np.meshgrid(x, x, indexing="ij")

    potential = vacuum_potential(n_slices, gpts, extent, dz)

    A_array = np.zeros((n_slices, 3, gpts, gpts))
    A_array[:, 0] = -0.5 * B0 * Y * dz
    A_array[:, 1] = +0.5 * B0 * X * dz
    A = VectorPotentialArray(A_array, slice_thickness=dz, extent=(extent, extent))
    A_zero, B_zero = zero_fields(n_slices, gpts, extent, dz)

    vortex = (X + 1j * np.sign(l) * Y) ** abs(l) * np.exp(
        -(X**2 + Y**2) / (2 * 6.0**2)
    )
    vortex /= np.sqrt((np.abs(vortex) ** 2).sum())

    from abtem.core.backend import get_array_module

    waves = abtem.PlaneWave(
        energy=ENERGY, gpts=gpts, extent=extent, device=device
    ).build(lazy=False)
    xp = get_array_module(device)
    waves._array = xp.asarray(vortex.astype(to_numpy(waves.array).dtype))

    spinor = waves.to_spinor((1, 0))

    out_B = pauli_multislice(
        spinor.copy(), potential, vector_potential=A, magnetic_field=B_zero
    )
    out_0 = pauli_multislice(
        spinor.copy(), potential, vector_potential=A_zero, magnetic_field=B_zero
    )

    overlap = np.vdot(to_numpy(out_0.array[0]), to_numpy(out_B.array[0]))
    phase = np.angle(overlap)

    wavelength = energy2wavelength(ENERGY)
    expected = -e_over_hbar * wavelength / (2 * np.pi) * B0 * l / 2 * (n_slices * dz)

    assert abs(phase - expected) < 0.02 * abs(expected)


def test_collinear_matches_adjusted_potential():
    """For a collinear sample (A along z only, no Zeeman) the Pauli solver
    must match the scalar path through adjust_coulomb_potential."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    dz = 1.35
    potential = abtem.Potential(atoms, gpts=64, slice_thickness=dz).build(lazy=False)
    n, gpts, extent = potential.num_slices, potential.gpts[0], potential.extent[0]

    rng = np.random.default_rng(7)
    A_array = np.zeros((n, 3, gpts, gpts))
    smooth = rng.standard_normal((n, 4, 4))
    from abtem.core.fft import fft_interpolate

    A_array[:, 2] = fft_interpolate(smooth, (n, gpts, gpts)) * 5.0 * dz
    A = VectorPotentialArray(
        A_array, extent=(extent, extent), slice_thickness=dz
    )
    _, B = zero_fields(n, gpts, extent, dz)

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    adjusted = A.adjust_coulomb_potential(potential, energy=ENERGY)
    reference = multislice_and_detect(
        waves.copy(), adjusted, detectors=None, algorithm=RealSpaceMultislice()
    )[0]

    out = pauli_multislice(
        waves.copy().to_spinor((1, 0)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    )

    reference_array = to_numpy(reference.array)
    out_array = to_numpy(out.array)
    scale = np.abs(reference_array).max()

    assert np.abs(out_array[0] - reference_array).max() / scale < 1e-10
    assert np.abs(out_array[1]).max() / scale < 1e-14


def test_lazy_matches_eager():
    """The lazy dask path (spin axis pinned to one chunk) is identical to the
    eager path, including genuine spin mixing."""
    gpts, extent, n, dz = 32, 20.0, 10, 2.0

    rng = np.random.RandomState(0)
    potential = abtem.PotentialArray(
        rng.rand(n, gpts, gpts) * 10, slice_thickness=dz, extent=(extent, extent)
    )
    A = VectorPotentialArray(
        np.random.RandomState(1).randn(n, 3, gpts, gpts) * 0.5,
        slice_thickness=dz,
        extent=(extent, extent),
    )
    B = MagneticFieldArray(
        np.random.RandomState(2).randn(n, 3, gpts, gpts) * 5000,
        slice_thickness=dz,
        extent=(extent, extent),
    )

    pw = abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent)

    eager = pauli_multislice(
        pw.build(lazy=False).to_spinor((1, 0)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    )
    lazy = pauli_multislice(
        pw.build(lazy=True).to_spinor((1, 0)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    ).compute()

    assert np.abs(to_numpy(eager.array) - to_numpy(lazy.array)).max() < 1e-12
    # the off-diagonal Zeeman terms populated the spin-down channel
    assert np.abs(to_numpy(eager.array[1])).max() > 1e-4


def test_iam_noncollinear_smoke():
    """End-to-end run with quasi-dipole IAM fields from canted atomic
    moments: finite result, near-unit norm, and a magnetic signal that
    vanishes when the moments are removed."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    moments = np.zeros((len(atoms), 3))
    moments[:, 0] = 2.2 * np.sign(np.cos(np.arange(len(atoms))))  # canted/alternating
    moments[:, 2] = 2.2
    set_magnetic_moments(atoms, moments)

    gpts, dz = 64, 0.5
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=dz).build(
        lazy=False
    )

    A = VectorPotential(atoms, gpts=gpts, slice_thickness=dz).build(lazy=False)
    B = MagneticField(atoms, gpts=gpts, slice_thickness=dz).build(lazy=False)

    assert A.num_slices == potential.num_slices

    # genuinely non-collinear fields: all three components populated
    assert all(np.abs(np.asarray(B.array)[:, i]).max() > 0 for i in range(3))

    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25)
    probe.grid.match(potential)
    spinor = probe.build(lazy=False).to_spinor((1, 0))

    out = pauli_multislice(
        spinor.copy(), potential, vector_potential=A, magnetic_field=B
    )
    out_array = to_numpy(out.array)

    assert np.all(np.isfinite(out_array))
    # a percent-level intensity loss is expected: the antialias bandlimit
    # clips high-angle scattering from the strongly scattering Fe crystal
    norm = (np.abs(out_array) ** 2).sum() / (np.abs(to_numpy(spinor.array)) ** 2).sum()
    assert abs(norm - 1) < 0.05

    # removing the moments removes the magnetic signal
    A_zero, B_zero = zero_fields(
        potential.num_slices, gpts, potential.extent[0], dz
    )
    out_zero = pauli_multislice(
        spinor.copy(), potential, vector_potential=A_zero, magnetic_field=B_zero
    )
    difference = np.abs(out_array - to_numpy(out_zero.array)).max()
    assert difference > 1e-8

    assert np.abs(to_numpy(out_zero.array)[1]).max() == 0
    assert np.abs(out_array[1]).max() > 0
