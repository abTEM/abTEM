"""Independent-oracle physics tests of the wave-propagation methods.

Every expected value in this module comes from an oracle that does not share
the code path under test: an analytic result (the Fresnel propagator is a
defocus, the shift theorem, Parseval's theorem), a symmetry or conservation law
(unitarity of multislice for a real potential, frame invariance of a beam
tilt), or an independent method (PRISM vs multislice, real-space vs Fourier
multislice, Bloch waves vs multislice). None of the expected values are pasted
outputs of the code, and the tolerances are derived from the numerics, as
stated at each assertion.

The cells are orthorhombic and non-square and the atoms sit at general
positions, so the projected structure has no mirror, inversion or two-fold
axis: an x/y swap or a flipped sign changes the result instead of mapping it
onto itself.
"""

import ase
import numpy as np
import pytest
from utils import gpu, to_host_array

import abtem
from abtem.core.energy import energy2wavelength
from abtem.multislice import FourierMultislice

devices = pytest.mark.parametrize("device", ["cpu", gpu])

# The config precision of the double-precision tests; floating point bounds
# below are expressed in units of the corresponding machine epsilon.
_EPS = {"float32": np.finfo(np.float32).eps, "float64": np.finfo(np.float64).eps}


def _general_position_cell(symbols="GaNO"):
    """Three atoms at general positions of an orthorhombic, non-square cell.

    No projection of this cell has a mirror, an inversion centre or a two-fold
    axis, so Friedel pairs and +/-x (and x/y) are all inequivalent.
    """
    return ase.Atoms(
        symbols,
        scaled_positions=[[0.1, 0.2, 0.0], [0.45, 0.35, 0.33], [0.7, 0.8, 0.66]],
        cell=[4.1, 4.7, 3.3],
        pbc=True,
    )


def _vacuum(extent=(20.0, 26.0), thickness=60.0, gpts=(100, 130), slice_thickness=5.0,
            device="cpu"):
    return abtem.Potential(
        ase.Atoms(cell=[*extent, thickness]),
        gpts=gpts,
        slice_thickness=slice_thickness,
        device=device,
    )


def _roundoff_bound(num_slices, gpts, precision):
    """Accumulated round-off of ``num_slices`` multislice steps.

    Each step is one pointwise product and one FFT pair; an FFT of N points has
    a relative round-off of order eps * log2(N), and the steps accumulate
    linearly in the worst case.
    """
    return num_slices * _EPS[precision] * np.log2(np.prod(gpts))


# ---------------------------------------------------------------------------
# 1. Free-space propagation
# ---------------------------------------------------------------------------


@devices
@pytest.mark.parametrize("order", ["exact", 1, 2])
@pytest.mark.filterwarnings("ignore:Maximum propagator phase error:UserWarning")
def test_plane_wave_unchanged_by_vacuum(order, device):
    # Oracle: a plane wave along the optic axis is an eigenfunction of the
    # free-space propagator with eigenvalue exp(2 pi i T / lambda); abTEM
    # drops that global phase (the propagator phase is sqrt(1 - x) - 1), so
    # the wave is unchanged. Bound: round-off of the slices.
    with abtem.config.set({"precision": "float64"}):
        vacuum = _vacuum(device=device)
        plane_wave = abtem.PlaneWave(energy=100e3, normalize=False, device=device)
        plane_wave = plane_wave.match_grid(vacuum)

        exit_wave = to_host_array(
            plane_wave.multislice(
                vacuum, lazy=False, algorithm=FourierMultislice(order=order)
            )
        )

    bound = _roundoff_bound(vacuum.num_slices, vacuum.gpts, "float64")
    np.testing.assert_allclose(exit_wave, 1.0, rtol=0, atol=bound)


@devices
@pytest.mark.parametrize("order", ["exact", 1, 2])
@pytest.mark.filterwarnings("ignore:Maximum propagator phase error:UserWarning")
def test_vacuum_propagation_is_a_defocus_change(order, device):
    # Oracle: in the paraxial limit the Fresnel propagator over a distance T
    # is exp(-i pi lambda k^2 T), and abTEM's aberration phase for a defocus
    # df (C10 = -df) is exp(+i pi lambda k^2 df) (transfer.py: chi =
    # 2 pi / lambda * alpha^2 C10 / 2, applied as exp(-i chi)). Hence a probe
    # with defocus d0 propagated by T in the +z direction must be the probe
    # with defocus d0 - T: a positive defocus focuses the probe at depth d0
    # below the entrance surface. This pins the sign of the propagator.
    semiangle_cutoff = 15.0
    d0, thickness = 40.0, 60.0
    energy = 100e3
    with abtem.config.set({"precision": "float64"}):
        vacuum = _vacuum(thickness=thickness, device=device)
        probe = abtem.Probe(
            energy=energy, semiangle_cutoff=semiangle_cutoff, defocus=d0,
            device=device,
        ).match_grid(vacuum)
        propagated = to_host_array(
            probe.multislice(
                vacuum, lazy=False, algorithm=FourierMultislice(order=order)
            )
        )
        refocused = to_host_array(
            abtem.Probe(
                energy=energy, semiangle_cutoff=semiangle_cutoff,
                defocus=d0 - thickness, device=device,
            )
            .match_grid(vacuum)
            .build(lazy=False)
        )

    # The order-1 propagator *is* the paraxial one, so it matches to
    # round-off. The exact (and order-2) propagator differ from the defocus by
    # the non-paraxial part of the phase, (2 pi T / lambda)
    # |sqrt(1 - x) - 1 + x / 2| with x = (lambda k)^2, which is largest at the
    # (soft-edged, hence one angular pixel wider) aperture edge. Since
    # |exp(i d) - 1| <= |d|, Parseval bounds the relative L2 error by it.
    wavelength = energy2wavelength(energy)
    bound = _roundoff_bound(vacuum.num_slices, vacuum.gpts, "float64")
    if order != 1:
        alpha = semiangle_cutoff * 1e-3 + wavelength / min(vacuum.extent)
        x = alpha**2
        bound += 2 * np.pi * thickness / wavelength * abs(np.sqrt(1 - x) - 1 + x / 2)

    error = np.linalg.norm(propagated - refocused) / np.linalg.norm(refocused)
    assert error <= bound, (error, bound)


# ---------------------------------------------------------------------------
# 2. Beam tilt
# ---------------------------------------------------------------------------


def _first_harmonic_shift(intensity, reference, extent):
    """The translation that maps ``reference`` onto ``intensity``.

    By the shift theorem, the first Fourier harmonic of a periodic function
    translated by a picks up the phase exp(-2 pi i a / L); for an exact
    translation this recovers a to round-off, whatever the shape.
    """
    shifts = []
    for axis, length in enumerate(extent):
        sum_axes = tuple(i for i in range(intensity.ndim) if i != axis)
        n = intensity.shape[axis]
        harmonic = np.exp(-2j * np.pi * np.arange(n) / n)
        f = (intensity.sum(sum_axes) * harmonic).sum()
        f0 = (reference.sum(sum_axes) * harmonic).sum()
        shifts.append(-np.angle(f / f0) * length / (2 * np.pi))
    return np.array(shifts)


@devices
@pytest.mark.parametrize("tilt", [(20.0, 0.0), (0.0, -15.0), (12.0, 17.0)])
def test_tilt_translates_beam_by_thickness_times_tan_theta(tilt, device):
    # Oracle: geometry. A beam travelling at angle theta to the optic axis is
    # displaced laterally by T tan(theta) over a depth T, in the direction of
    # its transverse wave vector (+x for a positive x-tilt). In vacuum the
    # tilted propagation must be an exact translation of the untilted one, so
    # the shift theorem measures the displacement to round-off -- well below
    # the T (tan(theta) - theta) = 1.6e-4 A difference of the small-angle
    # model at 20 mrad. The cell is non-square, so an x/y swap is caught.
    thickness = 60.0
    with abtem.config.set({"precision": "float64"}):
        vacuum = _vacuum(thickness=thickness, device=device)
        kwargs = dict(energy=100e3, semiangle_cutoff=25, device=device)
        untilted = abtem.Probe(**kwargs).match_grid(vacuum)
        tilted = abtem.Probe(**kwargs, tilt=tilt).match_grid(vacuum)

        reference = np.abs(to_host_array(untilted.multislice(vacuum, lazy=False))) ** 2
        intensity = np.abs(to_host_array(tilted.multislice(vacuum, lazy=False))) ** 2

    shift = _first_harmonic_shift(intensity, reference, vacuum.extent)
    expected = thickness * np.tan(np.array(tilt) * 1e-3)

    np.testing.assert_allclose(shift, expected, rtol=0, atol=1e-9)


@devices
@pytest.mark.parametrize("n", [2, 3])
def test_tilted_plane_wave_matches_explicit_phase_ramp(n, device):
    # Oracle: frame invariance. abTEM represents a tilted beam in the frame
    # co-moving with it: the wave carries no phase ramp and the tilt enters
    # through the propagator, so its diffraction spot stays at k = 0. The
    # same physics in the lab frame is an untilted simulation of the plane
    # wave exp(2 pi i k_t x) with k_t = sin(theta) / lambda, whose diffraction
    # pattern is the tilted-frame one moved to k = k_t (~ theta / lambda). We
    # take k_t = n / L_x so that it falls on a grid point, and compare beam
    # intensities after moving the lab pattern back by n pixels.
    energy = 200e3
    wavelength = energy2wavelength(energy)
    atoms = _general_position_cell() * (1, 1, 12)
    with abtem.config.set({"precision": "float64"}):
        potential = abtem.Potential(
            atoms, gpts=(82, 94), slice_thickness=0.5, projection="infinite",
            device=device,
        )
        k_t = n / potential.extent[0]
        theta = np.arcsin(wavelength * k_t) * 1e3  # mrad

        x = np.arange(potential.gpts[0]) * potential.sampling[0]
        ramp = np.exp(2j * np.pi * k_t * x)[:, None] * np.ones(potential.gpts)
        lab = abtem.Waves(
            ramp, energy=energy, sampling=potential.sampling
        ).copy_to_device(device)
        lab = to_host_array(lab.multislice(potential).compute())

        def tilted(tilt):
            plane_wave = abtem.PlaneWave(energy=energy, tilt=tilt, device=device)
            plane_wave = plane_wave.match_grid(potential)
            return to_host_array(plane_wave.multislice(potential, lazy=False))

        comoving = tilted((theta, 0.0))
        mirrored = tilted((-theta, 0.0))

    orders = [(h, k) for h in range(-4, 5) for k in range(-4, 5)]

    def beams(wave, offset=0):
        intensity = np.abs(np.fft.fft2(wave)) ** 2
        intensity /= intensity.sum()
        return np.array([intensity[(h + offset) % wave.shape[0], k] for h, k in orders])

    lab_intensity = np.abs(np.fft.fft2(lab)) ** 2
    # the lab-frame zero-order spot sits at k_t, i.e. index n along kx
    assert np.unravel_index(lab_intensity.argmax(), lab_intensity.shape) == (n, 0)
    lab_beams = beams(lab, offset=n)

    def r1(beams):
        return np.abs(beams - lab_beams).sum() / lab_beams.sum()

    # The frames differ (i) at second order of the propagator expanded about
    # k_t: the lab frame has the curvature 1 / cos^3(theta), the co-moving
    # frame 1, a phase error of at most 2 pi T lambda k^2 (3/4) (lambda k_t)^2
    # on the compared beams, which changes the (normalised) intensities by at
    # most twice that in R1; and (ii) through the antialias aperture, centred
    # on k = 0 in one frame and on k_t in the other, which only touches beams
    # at the aperture edge that carry < 1e-4 of the intensity at this
    # sampling (the residual falls with finer sampling).
    k_max = np.hypot(4 / atoms.cell[0, 0], 4 / atoms.cell[1, 1])
    phase_error = (
        2 * np.pi * potential.thickness * wavelength * k_max**2
        * 0.75 * (wavelength * k_t) ** 2
    )
    bound = 2 * phase_error + 1e-4
    assert r1(beams(comoving)) < bound
    # the structure is sensitive to the tilt sign: the opposite tilt is far off
    assert r1(beams(mirrored)) > 10 * bound


# ---------------------------------------------------------------------------
# 3. Unitarity
# ---------------------------------------------------------------------------

# The antialias aperture is a projection that discards the intensity scattered
# beyond 2/3 of the Nyquist frequency, and band-limiting the transmission
# function makes it non-unimodular; both are deliberate losses. A cutoff of
# 2 / sampling / 2 lies beyond the corner of the Fourier grid, so it disables
# both and leaves the multislice operators exactly unitary.
_NO_ANTIALIAS = {"antialias.cutoff": 2.0, "antialias.taper": 0.0}


def _unitarity_potential(device):
    return abtem.Potential(
        _general_position_cell() * (2, 2, 6),
        gpts=(72, 84),
        slice_thickness=0.5,
        device=device,
    )


@devices
@pytest.mark.parametrize("order", ["exact", 1, 2])
@pytest.mark.parametrize("wave", ["plane_wave", "probe"])
@pytest.mark.filterwarnings("ignore:Maximum propagator phase error:UserWarning")
def test_multislice_conserves_intensity(wave, order, device):
    # Oracle: for a real potential the transmission function exp(i sigma V)
    # has unit modulus and the free-space propagator is a pure phase, so
    # multislice is unitary and conserves the total intensity (Parseval) to
    # round-off.
    with abtem.config.set({"precision": "float64", **_NO_ANTIALIAS}):
        potential = _unitarity_potential(device)
        if wave == "plane_wave":
            builder = abtem.PlaneWave(energy=200e3, device=device)
            kwargs = {}
        else:
            builder = abtem.Probe(energy=200e3, semiangle_cutoff=25, device=device)
            kwargs = {"scan": [(3.1, 4.3)]}
        builder = builder.match_grid(potential)

        incident = to_host_array(builder.build(lazy=False, **kwargs))
        exit_wave = to_host_array(
            builder.multislice(
                potential, lazy=False, algorithm=FourierMultislice(order=order),
                **kwargs,
            )
        )

    ratio = (np.abs(exit_wave) ** 2).sum() / (np.abs(incident) ** 2).sum()
    bound = _roundoff_bound(potential.num_slices, potential.gpts, "float64")
    assert abs(ratio - 1) < bound, (ratio - 1, bound)


@devices
@pytest.mark.parametrize(
    "precision",
    [
        "float32",
        pytest.param(
            "float64",
            marks=pytest.mark.xfail(
                strict=True,
                reason="SMatrix ignores config['precision']: the plane-wave "
                "expansion is always built in complex64 (float32 wave vectors "
                "and coordinates in prism/utils.py prism_wave_vectors and "
                "plane_waves, and in SMatrix._build_s_matrix), so a float64 "
                "PRISM run loses ~7e-6 of the intensity over 40 slices, the "
                "single-precision round-off, instead of ~1e-14",
            ),
        ),
    ],
)
def test_prism_conserves_intensity(precision, device):
    # Oracle: PRISM at interpolation 1 is a linear combination of plane waves
    # each propagated by (unitary) multislice, so the reduced exit wave has
    # the intensity of the incident probe, to the round-off of the precision.
    position = (3.1, 4.3)
    with abtem.config.set({"precision": precision, **_NO_ANTIALIAS}):
        potential = _unitarity_potential(device)
        probe = abtem.Probe(energy=200e3, semiangle_cutoff=25, device=device)
        incident = to_host_array(
            probe.match_grid(potential).build(scan=[position], lazy=False)
        )
        s_matrix = abtem.SMatrix(
            potential=potential, energy=200e3, semiangle_cutoff=25,
            interpolation=1, downsample=False, device=device,
        )
        exit_wave = to_host_array(
            s_matrix.build(lazy=False).reduce(scan=abtem.CustomScan([position]))
        )

    ratio = (np.abs(exit_wave) ** 2).sum() / (np.abs(incident) ** 2).sum()
    bound = _roundoff_bound(potential.num_slices, potential.gpts, precision)
    assert abs(ratio - 1) < bound, (ratio - 1, bound)


# ---------------------------------------------------------------------------
# 6. PRISM scans vs multislice probe scans
# ---------------------------------------------------------------------------


@devices
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_prism_scan_matches_probe_scan(lazy, device):
    # Oracle: an independent method. At interpolation 1 the PRISM probe at r_p
    # is sum_k c_k exp(-2 pi i k.r_p) S_k, which by linearity of multislice is
    # exactly the multislice exit wave of the probe at r_p -- position by
    # position, not just up to a permutation or reflection of the scan. The
    # structure has no symmetry and the scan is non-square and offset, so a
    # sign or transposition error in the position phases moves the probes to
    # inequivalent sites.
    potential = abtem.Potential(
        _general_position_cell() * (2, 2, 2), gpts=(64, 72), slice_thickness=1.1,
        device=device,
    )
    scan = abtem.GridScan(start=(0.3, 0.7), end=(6.1, 8.2), gpts=(3, 5))
    detectors = [
        abtem.AnnularDetector(inner=0, outer=15),
        abtem.AnnularDetector(inner=40, outer=120),
        abtem.WavesDetector(),
    ]

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    reference = probe.match_grid(potential).scan(
        potential, scan=scan, detectors=detectors, lazy=False
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1,
        downsample=False, device=device,
    )
    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy)

    # single-precision round-off of the slices plus that of the sum over the
    # plane waves
    bound = _roundoff_bound(potential.num_slices, potential.gpts, "float32") + (
        _EPS["float32"] * np.sqrt(len(s_matrix))
    )
    for measurement, expected in zip(measurements, reference):
        measurement, expected = to_host_array(measurement), to_host_array(expected)
        assert measurement.shape == expected.shape
        assert expected.shape[:2] == (3, 5)
        np.testing.assert_allclose(
            measurement, expected, rtol=0, atol=bound * np.abs(expected).max()
        )
