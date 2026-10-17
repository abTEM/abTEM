"""``PixelatedDetector(components=...)``: the elastic, diffuse and total intensity of a
frozen-phonon ensemble from the ensemble-mean reduction (abTEM issue #454).

The reference is the route that keeps the exit wave of every configuration: a
``WavesDetector`` with ``FrozenPhonons(..., ensemble_mean=False)`` and
``elastic_diffuse_diffraction_patterns``. The detector route averages the cropped
amplitude and intensity of each configuration as the multislice tasks finish, and
forms the components after that average, so both routes must give the same numbers.
The sizes differ (3 configurations, 2 members of a second ensemble axis, a 48 x 48
grid, a 2 x 3 scan), so a component or axis in the wrong place shows up as a wrong
shape or wrong values.
"""

import ase
import dask.array as da
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem import distributions
from abtem.core.axes import FrozenPhononsAxis, OrdinalAxis
from abtem.core.backend import asnumpy
from abtem.detectors import PixelatedDetector, WavesDetector
from abtem.measurements import elastic_diffuse_diffraction_patterns
from abtem.waves import PlaneWave, Probe, Waves, reduce_ensemble

N_CONFIGS = 3
MAX_ANGLE = 40
COMPONENTS = ("total", "elastic", "diffuse")
# The reference keeps the frozen-phonon ensemble as individual wave functions.
pytestmark = pytest.mark.filterwarnings("ignore:ensemble_mean=False returns")


def _potential(ensemble_mean, num_configs=N_CONFIGS):
    atoms = ase.Atoms(
        "CO", positions=[(2.0, 2.0, 1.0), (5.5, 4.0, 2.5)], cell=(8.0, 8.0, 4.0)
    )
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=num_configs, sigmas=0.15, seed=7, ensemble_mean=ensemble_mean
    )
    return abtem.Potential(frozen_phonons, gpts=48, slice_thickness=2.0)


def _run(scan, device, lazy, potential, detectors, **probe_kwargs):
    """The multislice of a plane wave (``scan`` False) or a probe on a 2 x 3 scan."""
    if scan:
        probe = Probe(energy=100e3, semiangle_cutoff=20, device=device, **probe_kwargs)
        probe.grid.match(potential)
        grid_scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
        return probe.scan(
            potential, scan=grid_scan, detectors=detectors, lazy=lazy
        )
    return PlaneWave(energy=100e3, device=device, **probe_kwargs).multislice(
        potential, detectors, lazy=lazy
    )


def _detected(scan, device, lazy, detector, num_configs=N_CONFIGS, **probe_kwargs):
    potential = _potential(True, num_configs)
    return _run(scan, device, lazy, potential, detector, **probe_kwargs)


def _reference(scan, device, lazy, num_configs=N_CONFIGS, probe_kwargs=None, **kwargs):
    potential = _potential(False, num_configs)
    exit_waves = _run(
        scan, device, lazy, potential, WavesDetector(), **(probe_kwargs or {})
    )
    exit_waves = exit_waves.compute() if lazy else exit_waves
    return elastic_diffuse_diffraction_patterns(
        exit_waves, max_angle=MAX_ANGLE, parity="same", **kwargs
    )


def _array(measurement):
    if measurement.is_lazy:
        measurement = measurement.compute()
    return asnumpy(measurement.array)


def _assert_close(result, reference, rtol):
    assert result.shape == reference.shape
    reference = _array(reference)
    np.testing.assert_allclose(
        _array(result), reference, rtol=0, atol=rtol * np.abs(reference).max()
    )


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_detector_matches_the_exit_wave_route(scan, lazy, device):
    detector = PixelatedDetector(MAX_ANGLE, components=COMPONENTS)
    result = _detected(scan, device, lazy, detector)
    reference = _reference(scan, device, lazy)

    assert result.is_lazy == lazy
    assert result.ensemble_axes_metadata[0].values == COMPONENTS
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.metadata["frozen_phonon_component"] == list(COMPONENTS)
    assert result.metadata["num_configurations"] == N_CONFIGS
    assert "frozen_phonon_moments" not in result.metadata
    assert result.dtype == reference.dtype
    # The scan positions and the base axes are those of the reference.
    assert result.axes_metadata[1:] == reference.axes_metadata[1:]
    _assert_close(result, reference, rtol=2e-4)


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("reduction_dtype", [None, "float64"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_unbiased_and_float64_accumulation_match(
    scan, lazy, unbiased, reduction_dtype, device
):
    detector = PixelatedDetector(
        MAX_ANGLE,
        components="all",
        unbiased=unbiased,
        reduction_dtype=reduction_dtype,
    )
    result = _detected(scan, device, lazy, detector)
    reference = _reference(
        scan, device, lazy, unbiased=unbiased, reduction_dtype=reduction_dtype
    )

    assert result.metadata["unbiased"] is unbiased
    assert result.dtype == reference.dtype
    if reduction_dtype is None:
        _assert_close(result, reference, rtol=2e-4)
    else:
        # The exit waves are cast to complex128 in either route, so the two
        # differ by the roundoff of float64 sums alone.
        assert result.dtype == np.float64
        _assert_close(result, reference, rtol=1e-9)


@pytest.mark.parametrize("lazy", [False, True])
def test_unbiased_scales_the_diffuse_part_by_n_over_n_minus_one(lazy):
    biased = _detected(
        True, "cpu", lazy, PixelatedDetector(MAX_ANGLE, components="diffuse")
    )
    unbiased = _detected(
        True,
        "cpu",
        lazy,
        PixelatedDetector(MAX_ANGLE, components="diffuse", unbiased=True),
    )
    biased = _array(biased)
    np.testing.assert_allclose(
        _array(unbiased),
        biased * N_CONFIGS / (N_CONFIGS - 1),
        rtol=0,
        atol=1e-5 * np.abs(biased).max(),
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_components_follow_the_order_given_and_a_single_name_has_no_axis(lazy):
    everything = _detected(
        True, "cpu", lazy, PixelatedDetector(MAX_ANGLE, components="all")
    )
    stacked = _detected(
        True,
        "cpu",
        lazy,
        PixelatedDetector(MAX_ANGLE, components=("diffuse", "total")),
    )
    single = _detected(
        True, "cpu", lazy, PixelatedDetector(MAX_ANGLE, components="elastic")
    )

    assert stacked.ensemble_axes_metadata[0].values == ("diffuse", "total")
    np.testing.assert_allclose(
        _array(stacked)[0], _array(everything)[2], rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        _array(stacked)[1], _array(everything)[0], rtol=1e-5, atol=1e-8
    )

    assert single.shape == everything.shape[1:]
    assert single.metadata["frozen_phonon_component"] == "elastic"
    assert not any(
        isinstance(axis, OrdinalAxis) and axis.label == "component"
        for axis in single.ensemble_axes_metadata
    )
    np.testing.assert_allclose(
        _array(single), _array(everything)[1], rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_the_total_is_the_mean_intensity_of_a_plain_pixelated_detector(lazy):
    # Both detectors see the same exit waves in one multislice run.
    total, intensity = _detected(
        True,
        "cpu",
        lazy,
        [
            PixelatedDetector(MAX_ANGLE, components="total"),
            PixelatedDetector(MAX_ANGLE),
        ],
    )
    assert total.shape == intensity.shape
    np.testing.assert_allclose(
        _array(total),
        _array(intensity),
        rtol=0,
        atol=1e-5 * np.abs(_array(intensity)).max(),
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_other_ensemble_mean_axes_are_averaged_after_the_split(lazy, device):
    """A weighted defocus distribution is averaged incoherently: the weighted mean of
    the components of each defocus, not the components of the averaged amplitude."""
    values = np.array([-60.0, 0.0, 90.0])
    weights = np.array([0.5, 0.2, 0.3])
    defocus = distributions.from_values(values, weights=weights, ensemble_mean=True)

    detector = PixelatedDetector(MAX_ANGLE, components="all")
    result = _detected(True, device, lazy, detector, defocus=defocus)

    expected = 0.0
    for value, weight in zip(values, weights):
        reference = _reference(
            True, device, lazy, probe_kwargs=dict(defocus=float(value))
        )
        expected = expected + weight * _array(reference)
    expected = expected / weights.sum()

    assert result.shape == expected.shape
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=2e-4 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_a_potential_without_frozen_phonons_raises(lazy):
    detector = PixelatedDetector(MAX_ANGLE, components="all")
    atoms = _potential(True).frozen_phonons.atoms
    potential = abtem.Potential(atoms, gpts=48, slice_thickness=2.0)
    with pytest.raises(ValueError, match="FrozenPhononsAxis"):
        PlaneWave(energy=100e3).multislice(potential, detector, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_frozen_phonons_kept_as_an_ensemble_raise(lazy):
    detector = PixelatedDetector(MAX_ANGLE, components="all")
    with pytest.raises(ValueError, match="ensemble_mean=True"):
        PlaneWave(energy=100e3).multislice(_potential(False), detector, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_single_configuration_has_no_diffuse_part(lazy):
    detector = PixelatedDetector(MAX_ANGLE, components=("total", "elastic"))
    result = _array(_detected(False, "cpu", lazy, detector, num_configs=1))
    np.testing.assert_allclose(result[0], result[1], rtol=1e-5, atol=1e-8)

    for kwargs in (
        dict(components="diffuse"),
        dict(components="elastic", unbiased=True),
    ):
        with pytest.raises(ValueError, match="at least 2"):
            _detected(
                False, "cpu", lazy, PixelatedDetector(MAX_ANGLE, **kwargs), 1
            )


def _exit_waves(lazy, ensemble_mean=True, seed=3):
    """Configurations ψ_j = μ + noise_j with a frozen-phonon axis followed by a second
    ensemble axis (2 members)."""
    rng = np.random.default_rng(seed)
    shape = (N_CONFIGS, 2, 16, 20)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    array = array + 2 * np.ones(shape[1:], dtype=np.complex64)
    if lazy:
        array = da.from_array(array, chunks=(1, 1, 16, 20))
    return Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            FrozenPhononsAxis(_ensemble_mean=ensemble_mean),
            OrdinalAxis(label="member", values=(0, 1)),
        ],
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("unbiased", [False, True])
def test_detecting_waves_directly_and_reducing(lazy, unbiased):
    """One chunk per configuration and member when lazy, a tree reduction."""
    detector = PixelatedDetector("cutoff", components="all", unbiased=unbiased)
    raw = detector.detect(_exit_waves(lazy))
    assert "frozen_phonon_moments" in raw.metadata

    result = reduce_ensemble(raw)
    assert result.is_lazy == lazy
    reference = elastic_diffuse_diffraction_patterns(
        _exit_waves(False, ensemble_mean=False),
        max_angle="cutoff",
        parity="same",
        unbiased=unbiased,
    )
    # the members of the second axis keep their own components
    assert result.shape == (3, 2) + reference.base_shape
    _assert_close(result, reference, rtol=1e-5)


def test_detecting_waves_directly_without_a_mean_raises_on_reduction():
    detector = PixelatedDetector("cutoff", components="all")
    raw = detector.detect(_exit_waves(False, ensemble_mean=False))
    with pytest.raises(ValueError, match="ensemble_mean=True"):
        reduce_ensemble(raw)


def test_the_detector_validates_its_arguments_and_copies():
    with pytest.raises(ValueError, match="incoherent"):
        PixelatedDetector(MAX_ANGLE, components="incoherent")
    with pytest.raises(ValueError, match="components"):
        PixelatedDetector(MAX_ANGLE, components=())
    with pytest.raises(ValueError, match="reciprocal_space"):
        PixelatedDetector(MAX_ANGLE, components="all", reciprocal_space=False)
    with pytest.raises(ValueError, match="resample"):
        PixelatedDetector(MAX_ANGLE, components="all", resample="uniform")
    with pytest.raises(ValueError, match="apply to components"):
        PixelatedDetector(MAX_ANGLE, unbiased=True)
    with pytest.raises((ValueError, TypeError), match="reduction_dtype"):
        PixelatedDetector(MAX_ANGLE, components="all", reduction_dtype="float16")

    detector = PixelatedDetector(
        MAX_ANGLE, components=("diffuse",), unbiased=True, reduction_dtype="float64"
    )
    copied = detector.copy()
    assert copied.components == ("diffuse",)
    assert copied.unbiased
    assert copied.reduction_dtype == np.float64
    assert copied == detector
    assert PixelatedDetector(MAX_ANGLE).components is None


# ---------------------------------------------------------------------------------
# WavesDetector(average_frozen_phonons=True): the configuration-averaged wave <ψ>
# ---------------------------------------------------------------------------------


def _mean_wave_reference(scan, device, lazy, probe_kwargs=None):
    """<ψ> by hand: the exit waves of every configuration, averaged afterwards."""
    exit_waves = _run(
        scan, device, lazy, _potential(False), WavesDetector(), **(probe_kwargs or {})
    )
    fp_axis = [
        isinstance(ax, FrozenPhononsAxis) for ax in exit_waves.ensemble_axes_metadata
    ].index(True)
    exit_waves = exit_waves.compute() if lazy else exit_waves
    return exit_waves, np.mean(_array(exit_waves), axis=fp_axis, dtype=np.complex128)


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_averaged_waves_detector_matches_the_mean_of_the_exit_waves(scan, lazy, device):
    detector = WavesDetector(average_frozen_phonons=True)
    result = _detected(scan, device, lazy, detector)
    exit_waves, expected = _mean_wave_reference(scan, device, lazy)

    assert isinstance(result, Waves)
    assert result.is_lazy == lazy
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.shape == expected.shape
    assert result.shape == exit_waves.shape[1:]
    assert result.dtype == exit_waves.dtype
    assert result.metadata["num_configurations"] == N_CONFIGS
    assert "frozen_phonon_average" not in result.metadata
    assert result.axes_metadata == exit_waves.axes_metadata[1:]
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_averaged_waves_in_double_precision(lazy, device):
    detector = WavesDetector(average_frozen_phonons=True, reduction_dtype="float64")
    result = _detected(True, device, lazy, detector)
    _, expected = _mean_wave_reference(True, device, lazy)

    assert result.dtype == np.complex128
    # the waves of the configurations are float32; the mean of them is not rounded
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_the_default_waves_detector_keeps_every_configuration(lazy):
    detector = WavesDetector()
    assert not detector.average_frozen_phonons
    result = _detected(True, "cpu", lazy, detector)
    assert result.shape[0] == N_CONFIGS
    assert isinstance(result.ensemble_axes_metadata[0], FrozenPhononsAxis)
    assert "frozen_phonon_average" not in result.metadata


@pytest.mark.parametrize("lazy", [False, True])
def test_averaged_waves_keep_other_ensemble_axes(lazy):
    """Other axes with an ensemble mean are not averaged for waves, as before."""
    defocus = distributions.from_values(
        np.array([-60.0, 90.0]), weights=np.array([0.5, 0.5]), ensemble_mean=True
    )
    result = _detected(
        True,
        "cpu",
        lazy,
        WavesDetector(average_frozen_phonons=True),
        defocus=defocus,
    )
    exit_waves, expected = _mean_wave_reference(
        True, "cpu", lazy, probe_kwargs=dict(defocus=defocus)
    )
    assert result.shape == expected.shape
    assert result.shape[0] == 2
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_averaged_waves_without_frozen_phonons_raise(lazy):
    atoms = _potential(True).frozen_phonons.atoms
    potential = abtem.Potential(atoms, gpts=48, slice_thickness=2.0)
    with pytest.raises(ValueError, match="FrozenPhononsAxis"):
        PlaneWave(energy=100e3).multislice(
            potential, WavesDetector(average_frozen_phonons=True), lazy=lazy
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_averaged_waves_of_frozen_phonons_kept_as_an_ensemble_raise(lazy):
    with pytest.raises(ValueError, match="ensemble_mean=True"):
        PlaneWave(energy=100e3).multislice(
            _potential(False), WavesDetector(average_frozen_phonons=True), lazy=lazy
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_averaging_waves_directly_is_done_by_reduce_ensemble(lazy):
    detector = WavesDetector(average_frozen_phonons=True)
    raw = detector.detect(_exit_waves(lazy))
    assert raw.shape == (N_CONFIGS, 2, 16, 20)

    result = reduce_ensemble(raw)
    assert result.shape == (2, 16, 20)
    assert result.is_lazy == lazy
    expected = np.mean(_array(_exit_waves(False)), axis=0)
    np.testing.assert_allclose(_array(result), expected, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="ensemble_mean=True"):
        reduce_ensemble(detector.detect(_exit_waves(lazy, ensemble_mean=False)))


def test_waves_detector_validates_its_arguments_and_copies():
    with pytest.raises(ValueError, match="average_frozen_phonons"):
        WavesDetector(reduction_dtype="float64")
    with pytest.raises(TypeError, match="average_frozen_phonons"):
        WavesDetector(average_frozen_phonons="yes")
    detector = WavesDetector(average_frozen_phonons=True, reduction_dtype="float64")
    copied = detector.copy()
    assert copied.average_frozen_phonons
    assert copied.reduction_dtype == np.float64
    assert copied == detector
