import ase
import ase.build
import numpy as np
import pytest
from utils import devices, synthetic_transition_potential, to_host_array

import abtem
from abtem.antialias import AntialiasAperture
from abtem.core.energy import energy2wavelength
from abtem.krylov import KrylovMultislice
from abtem.multislice import (
    FourierMultislice,
    FresnelPropagator,
    conventional_multislice_step,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def graphene_monolayer():
    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=6.0, axis=2)
    return atoms


def graphene_potential(slice_thickness, device="cpu"):
    return abtem.Potential(
        graphene_monolayer(),
        gpts=(48, 84),
        slice_thickness=slice_thickness,
        projection="finite",
        periodic=True,
        device=device,
    )


def beam_family_flux(array, energy, family_frequencies, width=0.01):
    """Intensity summed over the beams at the given spatial frequencies (1/Å)."""
    array = to_host_array(array)
    nx, ny = array.shape[-2:]
    intensity = np.abs(np.fft.fft2(array) / (nx * ny)) ** 2
    fx = np.fft.fftfreq(nx, 2.46 / nx)[:, None]
    fy = np.fft.fftfreq(ny, 4.2608 / ny)[None, :]
    radius = np.hypot(fx, fy)
    return [intensity[np.abs(radius - f) < width].sum() for f in family_frequencies]


@devices
@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_krylov_conserves_intensity(energy, device):
    # Oracle: for a real potential every slice operator is unitary on the band, so the
    # total intensity stays at the incident value (Parseval); rounding limits it, by
    # about one machine epsilon per slice.
    potential = graphene_potential(0.05, device)
    waves = abtem.PlaneWave(energy=energy, device=device).multislice(
        potential, algorithm=KrylovMultislice(), lazy=False
    )
    array = to_host_array(waves)
    intensity = (np.abs(np.fft.fft2(array)) ** 2).sum() / array.size**2
    assert abs(intensity - 1.0) < 2 * potential.num_slices * np.finfo(array.dtype).eps


@devices
def test_krylov_wave_has_no_components_beyond_the_cutoff(device):
    energy, cutoff = 1000.0, 0.8
    potential = graphene_potential(0.05, device)
    waves = abtem.PlaneWave(energy=energy, device=device).multislice(
        potential, algorithm=KrylovMultislice(propagating_cutoff=cutoff), lazy=False
    )
    array = to_host_array(waves)
    nx, ny = array.shape
    k = np.hypot(
        np.fft.fftfreq(nx, 2.46 / nx)[:, None], np.fft.fftfreq(ny, 4.2608 / ny)[None]
    )
    spectrum = np.abs(np.fft.fft2(array)) / array.size
    outside = k >= cutoff / energy2wavelength(energy)
    assert spectrum[outside].max() < 1e-6 * spectrum.max()
    assert spectrum[~outside & (k > 0)].max() > 1e-3  # and there is scattering inside


# Strong beams of a graphene monolayer (the three families of the hexagonal lattice at
# 0.469, 0.813 and 0.939 1/Å), as the flux they carry: the exact two-way solution of
# the Helmholtz equation (Numerov, 0.01 Å steps, propagating beams only), for the
# potential of graphene_potential(0.01) with finite projection.
EXACT_FAMILY_FLUX = {
    1000.0: [0.044630, 0.039530, 0.006040],
    200.0: [0.181624, 0.155618, 0.022262],
}
GRAPHENE_FAMILIES = [0.469, 0.813, 0.939]


@pytest.mark.float64
@pytest.mark.parametrize(
    "energy, split_step_error",
    [(1000.0, 0.05), (200.0, 0.15)],
)
def test_krylov_is_closer_to_the_exact_solution_than_split_step(
    energy, split_step_error
):
    # Oracle: the exact two-way solution restricted to the beams the Krylov step keeps
    # (the propagating ones), stored above. The split step is off by 9 % (1 keV) and up
    # to 18 % (200 eV) on the strong beams of graphene; the Krylov step by about 2 %.
    exact = np.array(EXACT_FAMILY_FLUX[energy])

    with abtem.config.set({"precision": "float64"}):
        potential = graphene_potential(0.01)

        def beam_errors(algorithm):
            waves = abtem.PlaneWave(energy=energy).multislice(
                potential, algorithm=algorithm, lazy=False
            )
            values = np.array(beam_family_flux(waves.array, energy, GRAPHENE_FAMILIES))
            return np.abs(values / exact - 1)

        krylov = beam_errors(KrylovMultislice())
        split = beam_errors(FourierMultislice())

    assert krylov.max() < 0.03
    assert split.max() > split_step_error


def test_krylov_refuses_unsupported_input():
    potential = graphene_potential(0.05)
    with pytest.raises(NotImplementedError, match="tilted"):
        abtem.PlaneWave(energy=1000.0, tilt=(5.0, 0.0)).multislice(
            potential, algorithm=KrylovMultislice(), lazy=False
        )

    waves = abtem.PlaneWave(energy=1000.0).match_grid(potential).build(lazy=False)
    step = KrylovMultislice().make_step(waves)
    transmission = potential.build(lazy=False)[0].transmission_function(energy=1000.0)
    with pytest.raises(TypeError, match="projected potential"):
        step(waves, transmission)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(propagating_cutoff=0.0),
        dict(propagating_cutoff=1.0),
        dict(tolerance=0.0),
        dict(max_iterations=1),
    ],
)
def test_krylov_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        KrylovMultislice(**kwargs)


class _CountingAlgorithm:
    """Neither of the two shipped algorithms: it provides its own step."""

    expansion_scope = "propagator"

    def __init__(self):
        self.steps = 0

    def make_step(self, waves, return_backscattered=False):
        antialias_aperture = AntialiasAperture()
        propagator = FresnelPropagator()

        def step(waves, potential_slice, next_slice=None):
            self.steps += 1
            return conventional_multislice_step(
                waves,
                potential_slice=potential_slice,
                antialias_aperture=antialias_aperture,
                propagator=propagator,
            )

        return step


@devices
def test_any_object_with_the_step_protocol_runs_through_multislice(device):
    # Oracle: the dispatch asks the algorithm for its step and nothing else, so a
    # counting wrapper around the Fourier step gives its result, once per slice.
    potential = graphene_potential(0.5, device)
    algorithm = _CountingAlgorithm()
    waves = abtem.PlaneWave(energy=1000.0, device=device).multislice(
        potential, algorithm=algorithm, lazy=False
    )
    reference = abtem.PlaneWave(energy=1000.0, device=device).multislice(
        potential, algorithm=FourierMultislice(), lazy=False
    )
    assert algorithm.steps == potential.num_slices
    np.testing.assert_array_equal(to_host_array(waves), to_host_array(reference))


def si_setup(slice_thickness=2.0):
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, projection="finite"
    )
    return atoms, potential


@pytest.mark.parametrize("lazy", [False, True])
def test_krylov_agrees_with_split_step_where_both_converge_for_a_scanned_probe(lazy):
    # Oracle: at 100 keV the nonparaxial terms are small, so the Krylov step and the
    # split step differ by 1-2 % (obliquity of the potential). The grid is far too
    # coarse for the propagating band, so this also guards the antialiasing limit:
    # without it the signal at 30-100 mrad is off by 4-12 %.
    _, potential = si_setup()
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    scan = np.array([[0.0, 0.0], [1.0, 1.5]])
    detector = abtem.AnnularDetector(inner=30, outer=100)

    def adf(algorithm):
        return probe.scan(
            potential, scan=scan, detectors=detector, algorithm=algorithm, lazy=lazy
        )

    result, reference = adf(KrylovMultislice()), adf(FourierMultislice())
    if lazy:
        result, reference = result.compute(), reference.compute()
    result, reference = np.asarray(result.array), np.asarray(reference.array)
    assert result.shape == reference.shape == (2,)
    np.testing.assert_allclose(result, reference, rtol=3e-2)


def test_krylov_lazy_and_eager_agree_for_a_frozen_phonon_energy_ensemble():
    atoms, _ = si_setup()
    potential = abtem.Potential(
        abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1),
        gpts=(32, 32),
        slice_thickness=2.0,
        projection="finite",
    )
    waves = abtem.PlaneWave(energy=[50e3, 100e3])

    def run(lazy):
        result = waves.multislice(potential, algorithm=KrylovMultislice(), lazy=lazy)
        return to_host_array(result.compute() if lazy else result)

    eager, lazy = run(False), run(True)
    assert eager.shape == (2, 2, 32, 32)
    np.testing.assert_allclose(lazy, eager, atol=1e-5)


def test_krylov_runs_in_the_core_loss_multislice():
    # Oracle: the single-channel core-loss signal at 100 keV, where the Krylov and the
    # split step agree to 4 % on this 32 x 32 grid (both are limited by it; the elastic
    # signal of the two converges to each other with the grid).
    atoms = list(
        abtem.FrozenPhonons(ase.build.bulk("Si", cubic=True), 1, sigmas=0.1, seed=1)
    )[0]
    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=atoms.cell[2, 2] / 3
    )
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    def signal(algorithm, lazy):
        measurement = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=np.array([[0.0, 0.0]]),
            detectors=abtem.AnnularDetector(inner=0.0, outer=None),
            algorithm=algorithm,
            lazy=lazy,
            sites=atoms,
        )
        return np.asarray(
            measurement.compute(progress_bar=False).array if lazy else measurement.array
        )

    reference = signal(FourierMultislice(), False)
    for lazy in (False, True):
        result = signal(KrylovMultislice(), lazy)
        assert result.shape == reference.shape
        assert np.abs(result - reference).max() < 6e-2 * np.abs(reference).max()
