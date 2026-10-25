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
from utils import gpu, requires_mps, skip_float64_on_single_precision_arrays

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
        return probe.scan(potential, scan=grid_scan, detectors=detectors, lazy=lazy)
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
    assert result.dtype == reference.dtype
    # The scan positions and the base axes are those of the reference.
    assert result.axes_metadata[1:] == reference.axes_metadata[1:]
    _assert_close(result, reference, rtol=2e-4)


# Double precision. The float32 accumulation is the case of the test above.
@pytest.mark.float64
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_float64_accumulation_matches_the_exit_wave_route(lazy, device):
    skip_float64_on_single_precision_arrays(device, "summed in float64")
    detector = PixelatedDetector(MAX_ANGLE, components="all", reduction_dtype="float64")
    result = _detected(True, device, lazy, detector)
    reference = _reference(True, device, lazy, reduction_dtype="float64")

    # The exit waves are cast to complex128 in either route, so the two differ by the
    # roundoff of float64 sums alone (1e-16 relative; the float32 routes differ by
    # 1e-5, so 1e-9 cannot hold if the sums were still accumulated in float32).
    assert result.dtype == reference.dtype == np.float64
    _assert_close(result, reference, rtol=1e-9)


def _sample_variance_components(exit_waves):
    """total, unbiased elastic and unbiased diffuse intensity by hand, from the
    complex amplitudes of every configuration: the diffuse intensity is the sample
    variance (ddof=1) of the real and imaginary part, the total the mean intensity, and
    the elastic intensity what remains of the total. It shares no code with the
    N / (N - 1) of the detector."""
    fp_axis = [
        isinstance(ax, FrozenPhononsAxis) for ax in exit_waves.ensemble_axes_metadata
    ].index(True)
    amplitude = _array(
        exit_waves.diffraction_patterns(
            max_angle=MAX_ANGLE, parity="same", return_complex=True, fftshift=True
        )
    )
    total = np.mean(np.abs(amplitude) ** 2, axis=fp_axis)
    diffuse = np.var(amplitude.real, axis=fp_axis, ddof=1) + np.var(
        amplitude.imag, axis=fp_axis, ddof=1
    )
    return np.stack([total, total - diffuse, diffuse])


def _assert_unbiased_components(result, exit_waves):
    expected = _sample_variance_components(exit_waves)
    assert result.metadata["unbiased"] is True
    assert result.shape == expected.shape
    # Margin: the unbiased diffuse part is N / (N - 1) = 1.5 times the biased one, so
    # a factor missing or inverted (1 or 2/3) is off by 30-50 % of the maximum, against
    # the 2e-4 of the float32 routes.
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=2e-4 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_unbiased_components_are_the_sample_variance_of_the_amplitudes(lazy, device):
    detector = PixelatedDetector(MAX_ANGLE, components="all", unbiased=True)
    result = _detected(True, device, lazy, detector)
    exit_waves = _run(True, device, False, _potential(False), WavesDetector())
    _assert_unbiased_components(result, exit_waves)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_components_follow_the_order_given_and_a_single_name_has_no_axis(lazy, device):
    everything = _detected(
        True, device, lazy, PixelatedDetector(MAX_ANGLE, components="all")
    )
    stacked = _detected(
        True,
        device,
        lazy,
        PixelatedDetector(MAX_ANGLE, components=("diffuse", "total")),
    )
    single = _detected(
        True, device, lazy, PixelatedDetector(MAX_ANGLE, components="elastic")
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
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_total_is_the_mean_intensity_of_a_plain_pixelated_detector(lazy, device):
    # Both detectors see the same exit waves in one multislice run.
    total, intensity = _detected(
        True,
        device,
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
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_components_of_several_energies_have_the_pixel_count_of_the_highest(
    lazy, device
):
    """As for any max_angle in mrad (#503), the energies are cropped alike, to the
    pixels of the highest energy, so that they stack along the energy axis."""
    energies = [80e3, 200e3]
    potential = _potential(True)
    components = PlaneWave(energy=energies, device=device).multislice(
        potential, PixelatedDetector(MAX_ANGLE, components="all"), lazy=lazy
    )
    plain = PlaneWave(energy=energies, device=device).multislice(
        potential, PixelatedDetector(MAX_ANGLE), lazy=lazy
    )
    highest = PlaneWave(energy=energies[-1], device=device).multislice(
        potential, PixelatedDetector(MAX_ANGLE), lazy=lazy
    )

    assert components.shape == (3,) + plain.shape
    assert plain.shape[-2:] == highest.shape[-2:]
    np.testing.assert_allclose(
        _array(components)[0], _array(plain), rtol=0, atol=1e-5 * _array(plain).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_a_single_configuration_has_no_diffuse_part(lazy, device):
    detector = PixelatedDetector(MAX_ANGLE, components=("total", "elastic"))
    result = _array(_detected(False, device, lazy, detector, num_configs=1))
    np.testing.assert_allclose(result[0], result[1], rtol=1e-5, atol=1e-8)

    for kwargs in (
        dict(components="diffuse"),
        dict(components="elastic", unbiased=True),
    ):
        with pytest.raises(ValueError, match="at least 2"):
            _detected(False, device, lazy, PixelatedDetector(MAX_ANGLE, **kwargs), 1)


def _exit_waves(lazy, ensemble_mean=True, seed=3, mean=2.0, device="cpu"):
    """Configurations ψ_j = μ + noise_j with a frozen-phonon axis followed by a second
    ensemble axis (2 members)."""
    rng = np.random.default_rng(seed)
    shape = (N_CONFIGS, 2, 16, 20)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    array = array + mean * np.ones(shape[1:], dtype=np.complex64)
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
    ).copy_to_device(device)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_detecting_waves_directly_and_reducing(lazy, device):
    """One chunk per configuration and member when lazy, a tree reduction."""
    detector = PixelatedDetector("cutoff", components="all")
    raw = detector.detect(_exit_waves(lazy, device=device))

    result = reduce_ensemble(raw)
    assert result.is_lazy == lazy
    reference = elastic_diffuse_diffraction_patterns(
        _exit_waves(False, ensemble_mean=False),
        max_angle="cutoff",
        parity="same",
    )
    # the members of the second axis keep their own components
    assert result.shape == (3, 2) + reference.base_shape
    _assert_close(result, reference, rtol=1e-5)


# Double precision.
@pytest.mark.float64
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_a_weak_diffuse_signal_needs_the_float64_reduction(lazy, device):
    skip_float64_on_single_precision_arrays(device, "summed in float64")
    """The diffuse part is the difference of two large numbers where it is far below the
    Bragg peaks: a strong mean wave (300 against noise of 1 per pixel) puts 1e10 in the
    central pixel against a diffuse part of a few hundred, which the single-precision
    reduction loses to rounding. ``reduction_dtype='float64'`` does not."""
    reference = _array(
        elastic_diffuse_diffraction_patterns(
            _exit_waves(False, ensemble_mean=False, mean=300.0),
            max_angle="cutoff",
            parity="same",
            reduction_dtype="float64",
        )
    )[2]
    scale = reference.max()
    assert scale > 0

    errors = {}
    for reduction_dtype in (None, "float64"):
        detector = PixelatedDetector(
            "cutoff", components="diffuse", reduction_dtype=reduction_dtype
        )
        raw = detector.detect(_exit_waves(lazy, mean=300.0, device=device))
        diffuse = _array(reduce_ensemble(raw))
        assert diffuse.shape[-2:] == reference.shape[-2:]
        errors[reduction_dtype] = np.abs(diffuse - reference).max() / scale

    # Measured 9e-10 against 0.67 for the single-precision reduction: float64 rounds at
    # 1e-16 of the 1e10 central pixel, and the bound leaves a margin of 10.
    assert errors["float64"] < 1e-8
    # The option is of use: the default precision is no better. (How much worse it is
    # is not pinned; a better float32 reduction should not fail this test.)
    assert errors["float64"] < errors[None]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", [pytest.param("mps", marks=requires_mps)])
def test_a_float64_reduction_is_refused_on_the_single_precision_backend(lazy, device):
    """Metal cannot hold the float64 sums; it is said so, not left to fail obscurely."""
    detector = PixelatedDetector(
        "cutoff", components="diffuse", reduction_dtype="float64"
    )
    with pytest.raises(RuntimeError, match="single-precision backend"):
        _array(reduce_ensemble(detector.detect(_exit_waves(lazy, device=device))))


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_detecting_waves_directly_without_a_mean_raises_on_reduction(device):
    detector = PixelatedDetector("cutoff", components="all")
    raw = detector.detect(_exit_waves(False, ensemble_mean=False, device=device))
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
    assert result.axes_metadata == exit_waves.axes_metadata[1:]
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


# Double precision.
@pytest.mark.float64
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_averaged_waves_in_double_precision(lazy, device):
    skip_float64_on_single_precision_arrays(device, "averaged in float64")
    detector = WavesDetector(average_frozen_phonons=True, reduction_dtype="float64")
    result = _detected(True, device, lazy, detector)
    _, expected = _mean_wave_reference(True, device, lazy)

    assert result.dtype == np.complex128
    # the waves of the configurations are float32; the mean of them is not rounded
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_default_waves_detector_keeps_every_configuration(lazy, device):
    detector = WavesDetector()
    assert not detector.average_frozen_phonons
    result = _detected(True, device, lazy, detector)
    assert result.shape[0] == N_CONFIGS
    assert isinstance(result.ensemble_axes_metadata[0], FrozenPhononsAxis)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_averaged_waves_keep_other_ensemble_axes(lazy, device):
    """Other axes with an ensemble mean are not averaged for waves, as before."""
    defocus = distributions.from_values(
        np.array([-60.0, 90.0]), weights=np.array([0.5, 0.5]), ensemble_mean=True
    )
    result = _detected(
        True,
        device,
        lazy,
        WavesDetector(average_frozen_phonons=True),
        defocus=defocus,
    )
    exit_waves, expected = _mean_wave_reference(
        True, device, lazy, probe_kwargs=dict(defocus=defocus)
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
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_averaging_waves_directly_is_done_by_reduce_ensemble(lazy, device):
    detector = WavesDetector(average_frozen_phonons=True)
    raw = detector.detect(_exit_waves(lazy, device=device))
    assert raw.shape == (N_CONFIGS, 2, 16, 20)

    result = reduce_ensemble(raw)
    assert result.shape == (2, 16, 20)
    assert result.is_lazy == lazy
    expected = np.mean(_array(_exit_waves(False)), axis=0)
    np.testing.assert_allclose(_array(result), expected, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="ensemble_mean=True"):
        reduce_ensemble(
            detector.detect(_exit_waves(lazy, ensemble_mean=False, device=device))
        )


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


# ---------------------------------------------------------------------------------
# The same options through the PRISM scan, eagerly (every S-matrix member is detected
# in turn and the frozen-phonon moments accumulated) and lazily
# ---------------------------------------------------------------------------------


def _prism_scan(
    device, lazy, detector, ensemble_mean=True, num_configs=N_CONFIGS, **kw
):
    s_matrix = abtem.SMatrix(
        energy=100e3,
        semiangle_cutoff=20,
        potential=_potential(ensemble_mean, num_configs),
        interpolation=1,
        device=device,
    )
    grid_scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
    return s_matrix.scan(scan=grid_scan, detectors=detector, lazy=lazy, **kw)


def _prism_reference(device, lazy, ctf=None, **kwargs):
    """The exit waves of every configuration of the PRISM scan, reduced by hand."""
    exit_waves = _prism_scan(
        device, lazy, WavesDetector(), ensemble_mean=False, ctf=ctf
    )
    exit_waves = exit_waves.compute() if lazy else exit_waves
    return exit_waves, elastic_diffuse_diffraction_patterns(
        exit_waves, max_angle=MAX_ANGLE, parity="same", **kwargs
    )


@pytest.mark.parametrize(
    "lazy, disable_s_matrix_chunks",
    [(False, "auto"), (True, False), (True, True)],
)
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_components_match_the_exit_wave_route(
    lazy, disable_s_matrix_chunks, device
):
    detector = PixelatedDetector(MAX_ANGLE, components="all")
    kwargs = {} if not lazy else dict(disable_s_matrix_chunks=disable_s_matrix_chunks)
    result = _prism_scan(device, lazy, detector, **kwargs)
    _, reference = _prism_reference(device, lazy)

    assert result.is_lazy == lazy
    assert result.ensemble_axes_metadata[0].values == COMPONENTS
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.metadata["num_configurations"] == N_CONFIGS
    _assert_close(result, reference, rtol=2e-4)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_unbiased_components_are_the_sample_variance_of_the_amplitudes(
    lazy, device
):
    # The eager scan accumulates the members one by one, and N is the number of
    # members, not the length of an axis: its own route to the factor.
    detector = PixelatedDetector(MAX_ANGLE, components="all", unbiased=True)
    result = _prism_scan(device, lazy, detector)
    exit_waves = _prism_scan(device, False, WavesDetector(), ensemble_mean=False)
    _assert_unbiased_components(result, exit_waves)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_components_of_a_single_configuration(lazy, device):
    # N = 1: elastic = total, and the diffuse part is refused, as for multislice
    detector = PixelatedDetector(MAX_ANGLE, components=("total", "elastic"))
    result = _array(_prism_scan(device, lazy, detector, num_configs=1))
    np.testing.assert_allclose(result[0], result[1], rtol=1e-5, atol=1e-8)

    with pytest.raises(ValueError, match="at least 2"):
        _prism_scan(
            device,
            lazy,
            PixelatedDetector(MAX_ANGLE, components="diffuse"),
            num_configs=1,
        )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_components_average_other_axes_after_the_split(lazy, device):
    """A weighted defocus distribution is averaged incoherently, after the frozen
    phonons are averaged coherently, member by member in the eager scan."""
    values = np.array([-60.0, 90.0])
    weights = np.array([0.7, 0.3])
    defocus = distributions.from_values(values, weights=weights, ensemble_mean=True)

    result = _prism_scan(
        device,
        lazy,
        PixelatedDetector(MAX_ANGLE, components="all"),
        ctf={"defocus": defocus},
    )

    expected = 0.0
    for value, weight in zip(values, weights):
        _, reference = _prism_reference(device, lazy, ctf={"defocus": float(value)})
        expected = expected + weight * _array(reference)
    expected = expected / weights.sum()

    assert result.shape == expected.shape
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=2e-4 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_prism_components_guards(lazy):
    detector = PixelatedDetector(MAX_ANGLE, components="all")
    with pytest.raises(ValueError, match="ensemble_mean=True"):
        _prism_scan("cpu", lazy, detector, ensemble_mean=False)

    atoms = _potential(True).frozen_phonons.atoms
    s_matrix = abtem.SMatrix(
        energy=100e3,
        semiangle_cutoff=20,
        potential=abtem.Potential(atoms, gpts=48, slice_thickness=2.0),
        interpolation=1,
    )
    with pytest.raises(ValueError, match="FrozenPhononsAxis"):
        s_matrix.scan(scan=(2.0, 2.0), detectors=detector, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_averaged_waves_match_the_mean_of_the_exit_waves(lazy, device):
    result = _prism_scan(device, lazy, WavesDetector(average_frozen_phonons=True))
    exit_waves, _ = _prism_reference(device, lazy)
    expected = np.mean(_array(exit_waves), axis=0, dtype=np.complex128)

    assert isinstance(result, Waves)
    assert result.is_lazy == lazy
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.shape == expected.shape == exit_waves.shape[1:]
    assert result.metadata["num_configurations"] == N_CONFIGS
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_averaged_waves_without_the_option_keep_every_configuration(lazy, device):
    result = _prism_scan(device, lazy, WavesDetector())
    assert result.shape[0] == N_CONFIGS


@pytest.mark.parametrize("lazy", [False, True])
def test_prism_averaged_waves_guards(lazy):
    with pytest.raises(ValueError, match="ensemble_mean=True"):
        _prism_scan(
            "cpu", lazy, WavesDetector(average_frozen_phonons=True), ensemble_mean=False
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_upsample_refuses_components_but_averages_waves(device):
    # The compressed S-matrix stitches patterns by angle, which the components do
    # not survive: refused at the call, not left to fail in the reduction.
    s_matrix = abtem.SMatrix(
        energy=100e3,
        semiangle_cutoff=20,
        potential=_potential(True),
        interpolation=2,
        upsample=True,
        device=device,
    )
    scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
    with pytest.raises(NotImplementedError, match="upsample"):
        s_matrix.scan(
            scan=scan, detectors=PixelatedDetector(MAX_ANGLE, components="all")
        )

    result = s_matrix.scan(
        scan=scan, detectors=WavesDetector(average_frozen_phonons=True)
    )
    every = abtem.SMatrix(
        energy=100e3,
        semiangle_cutoff=20,
        potential=_potential(False),
        interpolation=2,
        upsample=True,
        device=device,
    ).scan(scan=scan, detectors=WavesDetector())
    expected = np.mean(_array(every), axis=0, dtype=np.complex128)
    assert result.shape == expected.shape
    np.testing.assert_allclose(
        _array(result), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


# ---------------------------------------------------------------------------------
# The same options through the PRISM scan of several energies
# ---------------------------------------------------------------------------------

ENERGIES = [100e3, 200e3]


def _prism_scan_energies(device, lazy, detector, ensemble_mean=True):
    s_matrix = abtem.SMatrix(
        energy=ENERGIES,
        semiangle_cutoff=20,
        potential=_potential(ensemble_mean),
        interpolation=1,
        device=device,
    )
    grid_scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
    return s_matrix.scan(scan=grid_scan, detectors=detector, lazy=lazy)


def _prism_energy_reference(device, lazy, energy, **kwargs):
    """The exit waves of every configuration of the PRISM scan at one energy."""
    s_matrix = abtem.SMatrix(
        energy=energy,
        semiangle_cutoff=20,
        potential=_potential(False),
        interpolation=1,
        device=device,
    )
    grid_scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
    exit_waves = s_matrix.scan(scan=grid_scan, detectors=WavesDetector(), lazy=lazy)
    exit_waves = exit_waves.compute() if lazy else exit_waves
    return exit_waves, elastic_diffuse_diffraction_patterns(
        exit_waves, max_angle=MAX_ANGLE, parity="same", **kwargs
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_components_of_several_energies(lazy, device):
    """Each energy has the true number of configurations, and the energies are
    cropped to the pixels of the highest, as for multislice (#503)."""
    result = _prism_scan_energies(
        device, lazy, PixelatedDetector(MAX_ANGLE, components="all")
    )
    plain = _prism_scan_energies(device, lazy, PixelatedDetector(MAX_ANGLE))
    _, reference = _prism_energy_reference(device, lazy, ENERGIES[-1])

    assert result.is_lazy == lazy
    assert result.ensemble_axes_metadata[0].values == tuple(ENERGIES)
    assert result.ensemble_axes_metadata[1].values == COMPONENTS
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.metadata["num_configurations"] == N_CONFIGS
    assert result.shape[2:] == reference.shape[1:]
    # the highest energy is the exit-wave route, every energy has the mean intensity
    # of the plain detector as its total
    _assert_close(result[-1], reference, rtol=2e-4)
    total = _array(result)[:, 0]
    np.testing.assert_allclose(
        total, _array(plain), rtol=0, atol=2e-4 * np.abs(_array(plain)).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_averaged_waves_of_several_energies(lazy, device):
    result = _prism_scan_energies(
        device, lazy, WavesDetector(average_frozen_phonons=True)
    )

    assert isinstance(result, Waves)
    assert result.is_lazy == lazy
    assert not any(
        isinstance(ax, FrozenPhononsAxis) for ax in result.ensemble_axes_metadata
    )
    assert result.metadata["num_configurations"] == N_CONFIGS
    for i, energy in enumerate(ENERGIES):
        exit_waves, _ = _prism_energy_reference(device, lazy, energy)
        expected = np.mean(_array(exit_waves), axis=0, dtype=np.complex128)
        assert result.shape[1:] == expected.shape
        np.testing.assert_allclose(
            _array(result)[i], expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_result_of_a_scan_is_not_averaged_again(device, lazy):
    """The detector marks its raw output as still to be averaged over the frozen
    phonons (metadata read by reduce_ensemble). The scans return it averaged, and a
    mark left on the result would make a further reduce_ensemble fail or split the
    components twice: it must leave the result unchanged."""
    components = PixelatedDetector(MAX_ANGLE, components="all")
    waves = WavesDetector(average_frozen_phonons=True)
    results = {
        "multislice components": _detected(True, device, lazy, components),
        "multislice waves": _detected(True, device, lazy, waves),
        "PRISM components": _prism_scan(device, lazy, components),
        "PRISM waves": _prism_scan(device, lazy, waves),
        "PRISM components of several energies": _prism_scan_energies(
            device, lazy, components
        ),
        "PRISM waves of several energies": _prism_scan_energies(device, lazy, waves),
    }
    for name, result in results.items():
        again = reduce_ensemble(result)
        assert again.shape == result.shape, name
        np.testing.assert_array_equal(_array(again), _array(result), err_msg=name)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_the_reduction_of_a_multi_energy_member_defers_the_averaged_waves(device):
    """The one test of private API: no public route reaches it. SMatrix.scan splits
    the energies before the frozen-phonon members (a spy on
    SMatrixArray._reduce_per_energy sees no call from it, eager or lazy), and the
    public reduce of a member has no frozen-phonon reduction to defer. Only the
    eager PRISM scan reduces a built multi-energy member with
    ``_defer_frozen_phonon_reduction``, so that the member keeps its frozen-phonon
    axis (of length 1) behind the energy axis for the mean over the members."""
    potential = _potential(True)
    scan = abtem.GridScan((1, 1), (3, 4), gpts=(2, 3))
    members = []
    for _, _, member in potential.generate_blocks(1):
        s_matrix = abtem.SMatrix(
            energy=ENERGIES,
            semiangle_cutoff=20,
            potential=member.item(),
            interpolation=1,
            device=device,
        ).build(lazy=False)
        members.append(
            s_matrix._reduce(
                scan=scan,
                detectors=WavesDetector(average_frozen_phonons=True),
                _defer_frozen_phonon_reduction=True,
            )
        )
    assert len(members) == N_CONFIGS
    assert members[0].ensemble_axes_metadata[0].values == tuple(ENERGIES)

    expected = _prism_scan_energies(
        device, False, WavesDetector(average_frozen_phonons=True)
    )
    mean = np.mean([_array(member) for member in members], axis=0)
    # without the deferral each energy is reduced early, over a frozen-phonon axis
    # of length 1, and the member has no such axis for the mean over the members
    mean = mean[:, 0]
    assert mean.shape == expected.shape
    np.testing.assert_allclose(
        mean,
        _array(expected),
        rtol=0,
        atol=1e-5 * np.abs(_array(expected)).max(),
    )
