import ase.build
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem import FrozenPhonons, PhaseScramblePlasmons, PlaneWave, Potential
from abtem.inelastic.plasmons import (
    estimate_plasmon_parameters,
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
@pytest.mark.parametrize("lazy", [True, False])
def test_plasmons_none_is_noop(device, lazy):
    """Passing plasmons=None must reproduce the plain multislice result exactly."""
    wave, potential, _ = _setup(device)
    detector = abtem.PixelatedDetector(max_angle=40)

    ref = wave.multislice(potential, detector).compute()
    out = wave.multislice(potential, detector, plasmons=None).compute()

    assert np.allclose(_to_numpy(ref.array), _to_numpy(out.array))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_conserve_total_intensity(device):
    """The non-unitary plasmon operator is renormalized to conserve the incident
    electron count: the exit-wave total intensity must equal the entrance total (per
    configuration) and stay close to the plasmon-free exit (which differs only by the
    small anti-aliasing bandlimit loss)."""
    wave, potential, _ = _setup(device)

    exit_ref = wave.multislice(potential).compute()
    exit_pl = wave.multislice(potential, plasmons=_plasmons()).compute()

    i_ref = float((np.abs(_to_numpy(exit_ref.array)) ** 2).sum())
    i_pl = float((np.abs(_to_numpy(exit_pl.array)) ** 2).sum())

    # renormalization keeps the plasmon exit intensity within a couple percent of the
    # (bandlimited) plasmon-free exit -- in particular it does not blow up.
    assert i_pl == pytest.approx(i_ref, rel=2e-2)


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
def test_plasmons_order_resolved_output_shape(device):
    """max_loss_order produces an output with a leading plasmon-order axis."""
    wave, potential, _ = _setup(device, num_configs=3, nz=10)
    detector = abtem.PixelatedDetector(max_angle=40)

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=2,
    )
    result = wave.multislice(potential, detector, plasmons=plasmons).compute()

    arr = _to_numpy(result.array)
    assert arr.ndim == 3
    assert arr.shape[0] == 3  # orders 0, 1, 2


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_order_resolved_zero_loss_matches_elastic(device):
    """The zero-loss order pattern should have the same shape as the elastic
    pattern (attenuated but not broadened)."""
    wave, potential, _ = _setup(device, num_configs=4, nz=10)
    detector = abtem.PixelatedDetector(max_angle=40)

    d_elastic = _to_numpy(
        wave.multislice(potential, detector).compute().array
    )

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=2,
    )
    result = wave.multislice(potential, detector, plasmons=plasmons).compute()
    d_zero_loss = _to_numpy(result.array[0])

    # Normalise both to unit peak and compare shapes — zero-loss should
    # closely resemble the elastic pattern.
    d_elastic_norm = d_elastic / d_elastic.max()
    d_zero_norm = d_zero_loss / d_zero_loss.max()
    assert np.corrcoef(d_elastic_norm.ravel(), d_zero_norm.ravel())[0, 1] > 0.99


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_realspace_conserve_total_intensity(device):
    """The plasmon operator is a real-space multiplication, so it composes with
    the real-space (finite-difference) multislice exactly as with the Fourier
    algorithm: the renormalized exit intensity stays close to the plasmon-free
    real-space exit."""
    wave, potential, _ = _setup(device, nz=10)

    algorithm = RealSpaceMultislice()
    exit_ref = wave.multislice(potential, algorithm=algorithm).compute()
    exit_pl = wave.multislice(
        potential, plasmons=_plasmons(), algorithm=algorithm
    ).compute()

    i_ref = float((np.abs(_to_numpy(exit_ref.array)) ** 2).sum())
    i_pl = float((np.abs(_to_numpy(exit_pl.array)) ** 2).sum())

    assert i_pl == pytest.approx(i_ref, rel=2e-2)


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
        assert arr_realspace[n].sum() == pytest.approx(
            arr_fourier[n].sum(), rel=1e-3
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_order_resolved_rejects_full_expansion(device):
    """Backscattering (expansion_scope='full') is not defined per loss order, so
    combining it with order-resolved plasmons must raise rather than silently
    mishandle the backscattered component."""
    wave, potential, _ = _setup(device, num_configs=2, nz=6)
    detector = abtem.PixelatedDetector(max_angle=40)

    plasmons = PhaseScramblePlasmons(
        mean_free_path=1050.0,
        excitation_energy=16.7,
        critical_angle=19.1,
        seed=7,
        max_loss_order=2,
    )

    with pytest.raises(NotImplementedError):
        wave.multislice(
            potential,
            detector,
            plasmons=plasmons,
            algorithm=RealSpaceMultislice(expansion_scope="full"),
        ).compute()


def test_estimate_plasmon_parameters_silicon():
    """The free-electron plasmon energy is accurate for Si (~16.7 eV); the
    critical angle and mean free path are positive, physically-sized estimates."""
    si = ase.build.bulk("Si", cubic=True)
    E_p, theta_c, lambda_p = estimate_plasmon_parameters(
        si, 200e3, valence_electrons=4
    )
    # Plasmon energy is the reliable estimate — within a few percent of 16.7 eV.
    assert E_p == pytest.approx(16.7, abs=0.5)
    # theta_c and lambda_p are order-of-magnitude free-electron estimates.
    assert 1.0 < theta_c < 50.0          # mrad
    assert 200.0 < lambda_p < 5000.0     # Å


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
        critical_angle=19.1,     # calibrated override
        mean_free_path=1050.0,   # calibrated override
        max_loss_order=3,
    )
    assert plasmons.excitation_energy == pytest.approx(16.7, abs=0.5)  # estimated
    assert plasmons.critical_angle == 19.1       # override respected
    assert plasmons.mean_free_path == 1050.0     # override respected
    assert plasmons.max_loss_order == 3          # kwarg forwarded


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
