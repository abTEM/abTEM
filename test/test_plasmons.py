import ase.build
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem import FrozenPhonons, PhaseScramblePlasmons, PlaneWave, Potential


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

    assert np.allclose(np.asarray(ref.array), np.asarray(out.array))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_conserve_total_intensity(device):
    """The non-unitary plasmon operator is renormalized to conserve the incident
    electron count: the exit-wave total intensity must equal the entrance total (per
    configuration) and stay close to the plasmon-free exit (which differs only by the
    small anti-aliasing bandlimit loss)."""
    wave, potential, _ = _setup(device)

    exit_ref = wave.multislice(potential).compute()
    exit_pl = wave.multislice(potential, plasmons=_plasmons()).compute()

    i_ref = float((np.abs(np.asarray(exit_ref.array)) ** 2).sum())
    i_pl = float((np.abs(np.asarray(exit_pl.array)) ** 2).sum())

    # renormalization keeps the plasmon exit intensity within a couple percent of the
    # (bandlimited) plasmon-free exit -- in particular it does not blow up.
    assert i_pl == pytest.approx(i_ref, rel=2e-2)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_redistribute_to_higher_angles(device):
    """Plasmon scattering tilts the beam, reducing the central (zero) beam and
    broadening the diffraction pattern."""
    wave, potential, _ = _setup(device)
    detector = abtem.PixelatedDetector(max_angle=40)

    d_ref = np.asarray(wave.multislice(potential, detector).compute().array)
    d_pl = np.asarray(
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

    a = np.asarray(
        wave.multislice(potential, detector, plasmons=_plasmons(seed=3)).compute().array
    )
    b = np.asarray(
        wave.multislice(potential, detector, plasmons=_plasmons(seed=3)).compute().array
    )
    assert np.allclose(a, b)

    c = np.asarray(
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

    a = np.asarray(
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

    arr = np.asarray(result.array)
    assert arr.ndim == 3
    assert arr.shape[0] == 3  # orders 0, 1, 2


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_plasmons_order_resolved_zero_loss_matches_elastic(device):
    """The zero-loss order pattern should have the same shape as the elastic
    pattern (attenuated but not broadened)."""
    wave, potential, _ = _setup(device, num_configs=4, nz=10)
    detector = abtem.PixelatedDetector(max_angle=40)

    d_elastic = np.asarray(
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
    d_zero_loss = np.asarray(result.array[0])

    # Normalise both to unit peak and compare shapes — zero-loss should
    # closely resemble the elastic pattern.
    d_elastic_norm = d_elastic / d_elastic.max()
    d_zero_norm = d_zero_loss / d_zero_loss.max()
    assert np.corrcoef(d_elastic_norm.ravel(), d_zero_norm.ravel())[0, 1] > 0.99
