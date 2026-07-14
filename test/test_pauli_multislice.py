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
from abtem.multislice import (
    FourierMultislice,
    RealSpaceMultislice,
    multislice_and_detect,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)

ENERGY = 100e3

# the two Pauli evolution schemes, keyed by short test-ID names
ALGORITHMS = {"series": RealSpaceMultislice(), "split": FourierMultislice()}


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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_to_spinor(device):
    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=20, gpts=64, extent=10, device=device
    )
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
    A, B = zero_fields(
        n, potential.gpts[0], potential.extent[0], potential.slice_thickness
    )

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
@pytest.mark.parametrize("method", ["series", "split"])
def test_constant_field_spin_precession(device, method):
    """Spin initially along z precesses about a constant B along x at the
    analytic rate e*B/(hbar*k). The split method's Zeeman factor is an exact
    per-pixel rotation, so it must reproduce this too."""
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

    out = pauli_multislice(
        spinor, potential, vector_potential=A, magnetic_field=B,
        algorithm=ALGORITHMS[method],
    )

    wavelength = energy2wavelength(ENERGY)
    theta = e_over_hbar * B0 * wavelength / (2 * np.pi) * (n_slices * dz)
    expected = np.array([0.0, -np.sin(theta), np.cos(theta)])

    s = spin_expectation(to_numpy(out.array))
    assert np.allclose(s, expected, atol=1e-6)

    # a spin parallel to the field is stationary
    spinor = waves.build(lazy=False).to_spinor((1, 1))
    out = pauli_multislice(
        spinor, potential, vector_potential=A, magnetic_field=B,
        algorithm=ALGORITHMS[method],
    )
    s = spin_expectation(to_numpy(out.array))
    assert np.allclose(s, [1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_average_field_precession(device):
    """The uniform-field path (average_field -> A_np + constant Zeeman)
    reproduces the analytic precession for a realistic field strength."""
    gpts, extent = 32, 20.0
    n_slices, dz = 50, 10.0
    B0 = 2.0  # T, like the saturation field of Fe

    potential = vacuum_potential(n_slices, gpts, extent, dz)
    A, B = zero_fields(n_slices, gpts, extent, dz)

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent, device=device)
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
@pytest.mark.parametrize("method", ["series", "split"])
@pytest.mark.parametrize("l", [1, -1])
def test_vortex_orbital_phase(device, method, l):
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
        spinor.copy(),
        potential,
        vector_potential=A,
        magnetic_field=B_zero,
        algorithm=ALGORITHMS[method],
    )
    out_0 = pauli_multislice(
        spinor.copy(),
        potential,
        vector_potential=A_zero,
        magnetic_field=B_zero,
        algorithm=ALGORITHMS[method],
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

    # use the potential's actual (depth-adjusted) slice thicknesses
    dzs = np.array(potential.slice_thickness)
    A_array[:, 2] = fft_interpolate(smooth, (n, gpts, gpts)) * 5.0 * dzs[:, None, None]
    A = VectorPotentialArray(
        A_array, extent=(extent, extent), slice_thickness=potential.slice_thickness
    )
    _, B = zero_fields(n, gpts, extent, potential.slice_thickness)

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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lazy_matches_eager(device):
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

    pw = abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent, device=device)

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


def test_vortex_probe_spinor():
    """A Probe with the built-in Vortex aperture composes with to_spinor and
    the Pauli solver: scanned batches work, the lazy path matches eager
    bit-exactly, and the beam equals the hand-built vortex construction
    (hard aperture disk times exp(il*phi) in reciprocal space)."""
    from abtem.scan import GridScan
    from abtem.transfer import Vortex

    gpts, extent, n, dz = 32, 20.0, 4, 2.0
    l = 2
    semiangle_cutoff = 25.0

    potential = abtem.PotentialArray(
        np.random.RandomState(0).rand(n, gpts, gpts) * 5,
        slice_thickness=dz,
        extent=(extent, extent),
    )
    A = VectorPotentialArray(
        np.random.RandomState(1).randn(n, 3, gpts, gpts) * 0.3,
        slice_thickness=dz,
        extent=(extent, extent),
    )
    B = MagneticFieldArray(
        np.random.RandomState(2).randn(n, 3, gpts, gpts) * 3000,
        slice_thickness=dz,
        extent=(extent, extent),
    )

    probe = abtem.Probe(
        aperture=Vortex(quantum_number=l, semiangle_cutoff=semiangle_cutoff),
        energy=ENERGY,
        extent=extent,
        gpts=gpts,
    )

    # the built-in Vortex equals the hand-built construction
    built = to_numpy(probe.build(lazy=False).array)
    base = to_numpy(
        abtem.Probe(
            semiangle_cutoff=semiangle_cutoff,
            energy=ENERGY,
            extent=extent,
            gpts=gpts,
            soft=False,
        )
        .build(lazy=False)
        .array
    )
    wavelength = energy2wavelength(ENERGY)
    k = np.fft.fftfreq(gpts, d=extent / gpts)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    mask = np.where(
        np.hypot(KX, KY) <= semiangle_cutoff * 1e-3 / wavelength,
        np.exp(1j * l * np.arctan2(KY, KX)),
        0.0,
    )
    mask[0, 0] = 0.0  # the vortex phase singularity carries no amplitude
    reference = np.fft.ifft2(np.fft.fft2(base) * mask)
    overlap = np.abs(np.vdot(built, reference)) / (
        np.linalg.norm(built) * np.linalg.norm(reference)
    )
    assert abs(overlap - 1) < 1e-12

    # the on-axis bin is zero, so +l and -l probes at mirrored positions are
    # exact mirror images -- required for mirror-difference magnetic signals
    assert np.abs(np.fft.fft2(to_numpy(built))[..., 0, 0]).max() < 1e-12
    p, Mp = (7.0, 11.0), (11.0, 7.0)
    bp = to_numpy(probe.build(scan=[list(p)], lazy=False).array)
    probe_m = abtem.Probe(
        aperture=Vortex(quantum_number=-l, semiangle_cutoff=semiangle_cutoff),
        energy=ENERGY,
        extent=extent,
        gpts=gpts,
    )
    bm = to_numpy(probe_m.build(scan=[list(Mp)], lazy=False).array)
    constant = np.vdot(bp.T, bm) / np.vdot(bp.T, bp.T)
    assert np.abs(bm - constant * bp.T).max() / np.abs(bm).max() < 1e-12

    # scanned spinor batch through the Pauli solver, lazy matches eager
    scan = GridScan(start=(5, 5), end=(10, 10), gpts=(2, 2), endpoint=False)
    eager = pauli_multislice(
        probe.build(scan=scan, lazy=False).to_spinor((1, 0)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    )
    lazy = pauli_multislice(
        probe.build(scan=scan, lazy=True).to_spinor((1, 0)),
        potential,
        vector_potential=A,
        magnetic_field=B,
    ).compute()

    assert eager.shape == (2, 2, 2, gpts, gpts)
    assert np.abs(to_numpy(eager.array) - to_numpy(lazy.array)).max() < 1e-12
    # the off-diagonal Zeeman terms populated the spin-down channel
    assert np.abs(to_numpy(eager.array)[1]).max() > 0


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_average_field_zeeman_without_periodic_field(device):
    """average_field applies its constant Zeeman term also when no periodic
    magnetic_field is given (only the periodic Zeeman part is omitted)."""
    gpts, extent = 32, 20.0
    n_slices, dz = 50, 10.0
    B0 = 2.0  # T

    potential = vacuum_potential(n_slices, gpts, extent, dz)
    A, _ = zero_fields(n_slices, gpts, extent, dz)

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent, device=device)
        .build(lazy=False)
        .to_spinor((1, 0))
    )

    with pytest.warns(UserWarning, match="magnetic_field"):
        out = pauli_multislice(
            spinor, potential, vector_potential=A, average_field=(B0, 0, 0)
        )

    wavelength = energy2wavelength(ENERGY)
    theta = e_over_hbar * B0 * wavelength / (2 * np.pi) * (n_slices * dz)

    s = spin_expectation(to_numpy(out.array))
    assert abs(s[1] + theta) < 1e-7


def test_fields_bundle_defaults():
    """pauli_multislice(fields=...) picks up the potential, both field
    arrays and average_field from the bundle, matching the explicit call;
    an explicit zero average_field suppresses the bundle's uniform field."""
    from abtem.magnetism.gpaw import GPAWMagneticFields

    gpts, extent, n, dz = 32, 20.0, 10, 2.0
    potential = abtem.PotentialArray(
        np.random.RandomState(0).rand(n, gpts, gpts) * 10,
        slice_thickness=dz,
        extent=(extent, extent),
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
    average_field = np.array([2.0, 1.0, 0.5])

    fields = GPAWMagneticFields(
        potential=potential,
        vector_potential=A,
        magnetic_field=B,
        average_field=average_field,
    )

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent)
        .build(lazy=False)
        .to_spinor((1, 0))
    )

    out_bundle = pauli_multislice(spinor.copy(), fields=fields)
    out_explicit = pauli_multislice(
        spinor.copy(),
        potential,
        vector_potential=A,
        magnetic_field=B,
        average_field=average_field,
    )
    assert np.array_equal(to_numpy(out_bundle.array), to_numpy(out_explicit.array))

    out_zero = pauli_multislice(spinor.copy(), fields=fields, average_field=(0, 0, 0))
    out_none = pauli_multislice(
        spinor.copy(), potential, vector_potential=A, magnetic_field=B
    )
    assert np.array_equal(to_numpy(out_zero.array), to_numpy(out_none.array))


def test_mismatched_slice_thickness_raises():
    """Equal slice counts but different slice thicknesses would silently
    mis-scale the field rates, so they must be rejected."""
    gpts, extent, n = 16, 10.0, 4
    potential = vacuum_potential(n, gpts, extent, 1.0)
    A, B = zero_fields(n, gpts, extent, 2.0)

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent)
        .build(lazy=False)
        .to_spinor((1, 0))
    )

    with pytest.raises(ValueError, match="slice thickness"):
        pauli_multislice(spinor, potential, vector_potential=A, magnetic_field=B)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_rectangular_grid(device):
    """Spin precession on a rectangular grid with unequal samplings,
    exercising the non-square handling of the stencils, fields and extent."""
    gpts, extent = (24, 40), (12.0, 30.0)
    n_slices, dz = 20, 10.0
    B0 = 2e4

    potential = abtem.PotentialArray(
        np.zeros((n_slices,) + gpts), slice_thickness=dz, extent=extent
    )
    A = VectorPotentialArray(
        np.zeros((n_slices, 3) + gpts), slice_thickness=dz, extent=extent
    )
    B_array = np.zeros((n_slices, 3) + gpts)
    B_array[:, 0] = B0 * dz
    B = MagneticFieldArray(B_array, slice_thickness=dz, extent=extent)

    spinor = (
        abtem.PlaneWave(energy=ENERGY, gpts=gpts, extent=extent, device=device)
        .build(lazy=False)
        .to_spinor((1, 0))
    )
    out = pauli_multislice(spinor, potential, vector_potential=A, magnetic_field=B)

    wavelength = energy2wavelength(ENERGY)
    theta = e_over_hbar * B0 * wavelength / (2 * np.pi) * (n_slices * dz)
    s = spin_expectation(to_numpy(out.array))
    assert np.allclose(s, [0.0, -np.sin(theta), np.cos(theta)], atol=1e-6)


def test_auto_rotation_recorded():
    """The rotation that rotate_field applies to the vector components is
    recorded on the builder, so gpaw_magnetic_fields can transform the
    separately computed average_field consistently."""
    from abtem.magnetism.gpaw import (
        _ROTATION_Y_INTO_Z,
        MagnetizationVectorPotential,
    )

    n = 16
    cell = np.diag([8.0, 8.0, 8.0])
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # collinear spin density varying along x: A = curl solve gives A_y only,
    # so the auto rotation must pick the y-into-z swap
    spin_density = np.broadcast_to(
        (1.0 + 0.5 * np.cos(x))[:, None, None], (n, n, n)
    ).copy()

    builder = MagnetizationVectorPotential(
        spin_density, cell, gpts=n, slice_thickness=0.5, rotate_field="auto"
    )
    built = builder.build()

    assert builder._resolved_rotation_matrix is not None
    assert np.allclose(builder._resolved_rotation_matrix, _ROTATION_Y_INTO_Z)

    # the built A indeed carries its magnetic component along z
    array = np.asarray(built.array)
    assert np.abs(array[:, 2]).max() > 10 * np.abs(array[:, :2]).max()

    # without rotation nothing is recorded
    builder_none = MagnetizationVectorPotential(
        spin_density, cell, gpts=n, slice_thickness=0.5, rotate_field=None
    )
    builder_none.build()
    assert builder_none._resolved_rotation_matrix is None


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
        potential.num_slices, gpts, potential.extent[0], potential.slice_thickness
    )
    out_zero = pauli_multislice(
        spinor.copy(), potential, vector_potential=A_zero, magnetic_field=B_zero
    )
    difference = np.abs(out_array - to_numpy(out_zero.array)).max()
    assert difference > 1e-8

    assert np.abs(to_numpy(out_zero.array)[1]).max() == 0
    assert np.abs(out_array[1]).max() > 0


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_split_matches_series(device):
    """The split-step method agrees with the per-slice-exact series method
    up to the expected Strang splitting error, which shrinks with slice
    thickness."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 1)
    set_magnetic_moments(atoms, np.tile([0.0, 0.0, 2.2], (len(atoms), 1)))

    def run(method, dz):
        potential = abtem.Potential(
            atoms, gpts=64, slice_thickness=dz, device=device
        ).build(lazy=False)
        A = VectorPotential(atoms, gpts=64, slice_thickness=dz, device=device).build(
            lazy=False
        )
        B = MagneticField(atoms, gpts=64, slice_thickness=dz, device=device).build(
            lazy=False
        )
        probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=25, device=device)
        probe.grid.match(potential)
        spinor = probe.build(lazy=False).to_spinor((1, 0))
        out = pauli_multislice(
            spinor,
            potential,
            vector_potential=A,
            magnetic_field=B,
            average_field=(0, 0, 2.2),
            algorithm=ALGORITHMS[method],
        )
        return to_numpy(out.array)

    # sanity: at coarse slicing the two methods agree at the few-percent
    # level (the residual mixes split's Strang error with the series
    # method's dz-independent finite-difference dispersion, so an exact
    # convergence rate against the series is not testable here)
    dz_coarse = 2.87 / 2
    err_coarse = np.abs(run("split", dz_coarse) - run("series", dz_coarse)).max()
    scale = np.abs(run("series", dz_coarse)).max()
    assert err_coarse / scale < 0.05

    # Cauchy self-convergence of the split method with slice thickness
    e1 = np.abs(run("split", 2.87 / 2) - run("split", 2.87 / 4)).max()
    e2 = np.abs(run("split", 2.87 / 4) - run("split", 2.87 / 8)).max()
    assert e2 < e1


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_split_exit_planes(device):
    """The split method's Strang bookkeeping (trailing half-propagation
    completed on a copy at exit planes) gives exit waves identical to
    independent runs truncated at the same thickness."""
    gpts, extent, dz = 32, 20.0, 2.0
    n = 6
    rng = np.random.default_rng(11)

    V = rng.random((n, gpts, gpts)) * 30.0
    A_array = rng.standard_normal((n, 3, gpts, gpts)) * 0.5
    B_array = rng.standard_normal((n, 3, gpts, gpts)) * 100.0

    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=20, gpts=gpts, extent=extent, device=device
    )
    spinor_input = probe.build(lazy=False).to_spinor((0.8, 0.6j))

    full = abtem.PotentialArray(
        V, slice_thickness=dz, extent=(extent, extent), exit_planes=2
    )
    A = VectorPotentialArray(
        A_array, slice_thickness=dz, extent=(extent, extent)
    )
    B = MagneticFieldArray(B_array, slice_thickness=dz, extent=(extent, extent))

    out = pauli_multislice(
        spinor_input.copy(),
        full,
        vector_potential=A,
        magnetic_field=B,
        algorithm=FourierMultislice(),
    )
    out_array = to_numpy(out.array)

    # exit_planes=2 includes the entrance plane (-1) at index 0
    assert to_numpy(out.array).shape[0] == 4
    for i, k in enumerate((2, 4, 6), start=1):
        truncated = abtem.PotentialArray(
            V[:k], slice_thickness=dz, extent=(extent, extent)
        )
        A_k = VectorPotentialArray(
            A_array[:k], slice_thickness=dz, extent=(extent, extent)
        )
        B_k = MagneticFieldArray(
            B_array[:k], slice_thickness=dz, extent=(extent, extent)
        )
        reference = pauli_multislice(
            spinor_input.copy(),
            truncated,
            vector_potential=A_k,
            magnetic_field=B_k,
            algorithm=FourierMultislice(),
        )
        assert (
            np.abs(out_array[i] - to_numpy(reference.array)).max() < 1e-12
        ), f"exit plane at {k} slices does not match an independent run"


@pytest.mark.parametrize("method", ["series", "split"])
def test_z_periodic_fields(method):
    """Fields covering one z-period of a repeating potential are cycled
    modulo their slice count, matching explicitly z-tiled fields; also a
    CrystalPotential smoke test for the same workflow."""
    from abtem.potentials.iam import CrystalPotential

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    set_magnetic_moments(atoms, np.tile([0.0, 0.0, 2.2], (len(atoms), 1)))

    gpts_cell, xy_reps, z_reps = 16, 3, 4
    dz = 2.87 / 2
    gpts = gpts_cell * xy_reps

    A_unit = VectorPotential(atoms, gpts=gpts_cell, slice_thickness=dz).build(
        lazy=False
    ).tile((xy_reps, xy_reps, 1))
    B_unit = MagneticField(atoms, gpts=gpts_cell, slice_thickness=dz).build(
        lazy=False
    ).tile((xy_reps, xy_reps, 1))
    A_tiled = A_unit.tile((1, 1, z_reps))
    B_tiled = B_unit.tile((1, 1, z_reps))

    unit_potential = abtem.Potential(
        atoms * (xy_reps, xy_reps, 1), gpts=gpts, slice_thickness=dz
    ).build(lazy=False)
    crystal = CrystalPotential(unit_potential, (1, 1, z_reps))

    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=25, gpts=gpts, extent=2.87 * xy_reps
    )
    spinor_input = probe.build(lazy=False).to_spinor((1, 0))

    out_periodic = pauli_multislice(
        spinor_input.copy(),
        crystal,
        vector_potential=A_unit,
        magnetic_field=B_unit,
        average_field=(0, 0, 2.2),
        algorithm=ALGORITHMS[method],
    )
    out_tiled = pauli_multislice(
        spinor_input.copy(),
        crystal,
        vector_potential=A_tiled,
        magnetic_field=B_tiled,
        average_field=(0, 0, 2.2),
        algorithm=ALGORITHMS[method],
    )

    a1, a2 = to_numpy(out_periodic.array), to_numpy(out_tiled.array)
    assert np.abs(a1 - a2).max() < 1e-13
    assert np.all(np.isfinite(a1))

    # mismatched (non-divisor) slice counts still fail loudly
    A_bad = A_unit.tile((1, 1, 3))
    with pytest.raises(ValueError, match="divides"):
        pauli_multislice(
            spinor_input.copy(),
            crystal,
            vector_potential=A_bad,
            magnetic_field=B_unit,
            algorithm=ALGORITHMS[method],
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_field_builders_respect_device(device):
    """QuasiDipoleProjections.integrate_on_grid (the default integrator for
    VectorPotential/MagneticField) must return an array on the requested
    device, not silently ignore it. This directly covers a real GPU bug
    (integrate_on_grid always built on host via a hardcoded np.zeros,
    unconditionally returning NumPy even for device='gpu') that surfaced
    as a `ValueError: non-scalar numpy.ndarray cannot be used for fill`
    deep inside `generate_slices`' single-atomic-number fast path, for any
    single-element structure -- e.g. plain bcc Fe -- since that path skips
    the (correctly device-aware) explicit `xp.zeros` allocation used when
    multiple species or multiple slices are combined per chunk."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    set_magnetic_moments(atoms, np.tile([0.0, 0.0, 2.2], (len(atoms), 1)))

    A = VectorPotential(atoms, gpts=16, slice_thickness=1.435, device=device).build(
        lazy=False
    )
    B = MagneticField(atoms, gpts=16, slice_thickness=1.435, device=device).build(
        lazy=False
    )

    A_array, B_array = to_numpy(A.array), to_numpy(B.array)
    assert np.all(np.isfinite(A_array))
    assert np.all(np.isfinite(B_array))
    assert np.abs(A_array).max() > 0
    assert np.abs(B_array).max() > 0

    if device == "gpu":
        import cupy as cp

        assert isinstance(A.array, cp.ndarray)
        assert isinstance(B.array, cp.ndarray)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_magnetization_field_builders_respect_device(device):
    """_MagnetizationMagnetics.generate_slices (shared by GPAWVectorPotential/
    GPAWMagneticField and their DFT-agnostic counterparts
    MagnetizationVectorPotential/MagnetizationMagneticField) computes
    everything through host-only FFT routines
    (calculate_vector_potential_from_magnetization, curl_fourier -- both
    hardcoded np.fft, independent of `device`) and never moved the result
    to the requested device before yielding it -- a second, independent
    instance of the same bug class as test_field_builders_respect_device,
    in the GPAW-density field-construction path rather than the
    quasi-dipole atomistic one. Exercised here via the DFT-agnostic
    builders (synthetic magnetization), avoiding the cost of a real GPAW
    calculation while covering the exact code path that was broken."""
    from abtem.magnetism.gpaw import (
        MagnetizationMagneticField,
        MagnetizationVectorPotential,
    )

    cell = np.diag([2.87, 2.87, 2.87])
    gpts3d = (16, 16, 16)
    rng = np.random.default_rng(3)
    magnetization = np.zeros((3,) + gpts3d)
    magnetization[2] = rng.random(gpts3d)  # collinear, along z

    A = MagnetizationVectorPotential(
        magnetization, cell, gpts=16, slice_thickness=1.435, device=device
    ).build(lazy=False)
    B = MagnetizationMagneticField(
        magnetization, cell, gpts=16, slice_thickness=1.435, device=device
    ).build(lazy=False)

    A_array, B_array = to_numpy(A.array), to_numpy(B.array)
    assert np.all(np.isfinite(A_array))
    assert np.all(np.isfinite(B_array))
    assert np.abs(A_array).max() > 0
    assert np.abs(B_array).max() > 0

    if device == "gpu":
        import cupy as cp

        assert isinstance(A.array, cp.ndarray)
        assert isinstance(B.array, cp.ndarray)
