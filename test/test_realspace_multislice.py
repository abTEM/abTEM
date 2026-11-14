import os

import ase
import ase.build
import numpy as np
import pytest
from utils import devices, gpu, requires_gpu, to_host_array

import abtem
from abtem.core.energy import energy2wavelength
from abtem.finite_difference import (
    _series_prefactor,
    full_series,
    propagator_taylor_series,
)
from abtem.multislice import FourierMultislice, RealSpaceMultislice

# Suppress Numba performance warnings during tests; the series of the full operator is
# deprecated (and tested here until it is removed)
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning",
    "ignore::UserWarning",
    "ignore:RealSpaceMultislice\\(expansion_scope='full'\\):FutureWarning",
)


def create_sto_atoms():
    """Create a SrTiO3 unit cell and supercell."""
    unit_cell = ase.Atoms(
        symbols="SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        cell=[3.9127, 3.9127, 3.9127],
        pbc=True,
    )
    atoms = unit_cell * (2, 2, 4)
    return atoms, unit_cell


@pytest.fixture
def test_system(request):
    """Create complete test system with atoms, potential, probe, and scans."""
    # Get device from parametrization if available, otherwise default to "cpu"
    device = getattr(request, "param", "cpu")

    atoms, unit_cell = create_sto_atoms()

    # Standard potential
    potential = abtem.Potential(
        atoms,
        gpts=(80, 80),
        slice_thickness=0.75,
        projection="finite",
        device=device,
    )

    # Potential with exit planes
    potential_exit_planes = abtem.Potential(
        atoms,
        gpts=(80, 80),
        slice_thickness=0.75,
        exit_planes=1,
        projection="finite",
        device=device,
    )

    # Probe
    probe = abtem.Probe(
        semiangle_cutoff=20,
        energy=30e3,
        device=device,
    ).match_grid(potential)

    # Scans
    single_point_scan = [[0, 0]]
    grid_scan = abtem.GridScan(
        start=(0, 0),
        end=(unit_cell.cell[0, 0], unit_cell.cell[1, 1]),
        gpts=2,
    )

    return {
        "atoms": atoms,
        "unit_cell": unit_cell,
        "potential": potential,
        "potential_exit_planes": potential_exit_planes,
        "probe": probe,
        "single_point_scan": single_point_scan,
        "grid_scan": grid_scan,
        "device": device,
    }


class TestLazyVsEager:
    """Test that lazy and eager computations produce similar results."""

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.parametrize(
        "algorithm",
        [
            FourierMultislice(order=1),
            FourierMultislice(order=2),
            FourierMultislice(order='exact'),
            # the lazy/eager plumbing does not depend on the real-space order
            pytest.param(RealSpaceMultislice(order=3), marks=pytest.mark.slow),
        ],
    )
    def test_lazy_vs_eager_single_point(self, test_system, algorithm):
        """Test that lazy and eager give similar results for single point."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        # Lazy computation
        lazy_result = probe.multislice(
            potential=potential,
            scan=scan,
            algorithm=algorithm,
        ).compute()

        # Eager computation
        eager_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=algorithm,
        )

        # Check shapes match
        assert lazy_result.array.shape == eager_result.array.shape

        # Check values are close (allowing for numerical differences)
        np.testing.assert_allclose(
            to_host_array(lazy_result.array),
            to_host_array(eager_result.array),
            rtol=1e-5,
            atol=1e-8,
        )


class TestFourierMultislice:
    """Test FourierMultislice algorithm with various configurations."""

    def test_fourier_invalid_order(self, test_system):
        """Test that FourierMultislice rejects invalid orders."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        with pytest.raises(ValueError, match="Only order 1, 2, and 'exact' are supported"):
            probe.multislice(
                potential=potential,
                scan=scan,
                lazy=False,
                algorithm=FourierMultislice(order=3),  # type: ignore
            )

    @devices
    def test_fourier_conjugate_transpose(self, device):
        # Two exact identities for a plane wave, whose array is real and which the
        # propagator leaves unchanged. A step is t then P (transmit, propagate):
        # - conjugate=True conjugates t and reverses the propagation, so each step
        #   is the conjugate of the plain step of the conjugate wave, and the run
        #   is the conjugate of the plain run;
        # - transpose=True is P then t, so P (t_n P ... t_1 P) = t_0 P (t_n P ...)
        #   with t_0 = 1: the transposed run with a trailing vacuum slice is the
        #   plain run.
        # Measured: <= 2.4e-7 and 0 against the 1e-6 bound; ignoring the flag
        # gives 4.1e-2 and 1.4e-3.
        rng = np.random.default_rng(0)
        array = 10 * rng.random((3, 32, 28)).astype(np.float32)
        vacuum = np.zeros_like(array[:1])

        def run(array, **kwargs):
            with abtem.config.set({"device": device}):
                potential = abtem.PotentialArray(
                    array, slice_thickness=1.5, extent=(8.0, 7.0)
                )
                return to_host_array(
                    abtem.PlaneWave(energy=100e3)
                    .multislice(
                        potential, lazy=False, algorithm=FourierMultislice(**kwargs)
                    )
                    .array
                )

        for order in (1, 2, "exact"):
            plain = run(array, order=order)
            np.testing.assert_allclose(
                run(array, order=order, conjugate=True),
                np.conj(plain),
                rtol=0,
                atol=1e-6,
                err_msg=f"conjugate, order {order}",
            )
            np.testing.assert_allclose(
                run(np.concatenate([array, vacuum]), order=order, transpose=True),
                plain,
                rtol=0,
                atol=1e-6,
                err_msg=f"transpose, order {order}",
            )


class TestRealSpaceMultislice:
    """Test RealSpaceMultislice algorithm with various configurations."""

    @devices
    def test_realspace_derivative_accuracy_and_max_terms(self, device):
        # A harmonic exp(2 pi i m x / N) through vacuum: the paraxial Fourier
        # propagator is exact for it, so the order-1 real-space error is that of
        # the finite-difference Laplacian, which falls with its accuracy.
        # Measured (m = 4 of N = 24): 3.6e-2, 5.0e-3, 8.2e-4, 1.5e-4 at accuracy
        # 2, 4, 6, 8; each step at least 5.5x, against the asserted 3x. An
        # ignored accuracy gives a ratio of 1.
        from abtem.finite_difference import NotConvergedError

        gpts, extent = (24, 20), (6.0, 5.0)
        harmonic = np.exp(2j * np.pi * 4 * np.arange(gpts[0]) / gpts[0])
        array = np.broadcast_to(harmonic[:, None], gpts).astype(np.complex64)

        def run(algorithm):
            with abtem.config.set({"device": device}):
                vacuum = abtem.PotentialArray(
                    np.zeros((4,) + gpts, np.float32),
                    slice_thickness=2.0,
                    extent=extent,
                )
                waves = abtem.Waves(array.copy(), energy=100e3, extent=extent)
                waves = waves.copy_to_device(device)
                return to_host_array(
                    waves.multislice(vacuum, algorithm=algorithm).array
                )

        reference = run(FourierMultislice(order=1))
        errors = [
            np.abs(
                run(RealSpaceMultislice(order=1, derivative_accuracy=accuracy))
                - reference
            ).max()
            for accuracy in (2, 4, 6, 8)
        ]
        for coarse, fine in zip(errors, errors[1:]):
            assert fine < coarse / 3, errors

        # the exponential series needs more than 3 terms
        with pytest.raises(NotConvergedError, match="in 3terms"):
            run(RealSpaceMultislice(max_terms=3))


class TestOperatorSeries:
    """The per-slice operator series on a plane-wave eigenstate.

    The Laplacian of a plane wave is a number, so the series can be compared with
    the exact forward wavenumber offset without a finite-difference stencil. The
    functions are plain arithmetic on the arrays they are given, so the tests are
    not parametrised over devices.
    """

    wavelength = energy2wavelength(30e3)
    K = 2 * np.pi / wavelength
    thickness = 0.2

    def series_offset(self, function, order, laplacian, potential):
        # per-slice phase rate (rad / Angstrom), the series divided by i * thickness
        wave = np.ones(1, dtype=complex)
        args = (wave, lambda w: laplacian * w, np.full(1, potential))
        if function is full_series:
            result = function(*args, order, self.wavelength, self.thickness)
        else:
            result = function(wave, order, *args[1:], self.wavelength, self.thickness)
        return (result / (1j * self.thickness))[0].real

    @pytest.mark.parametrize(
        "function, sine_angle, potential",
        [
            (full_series, 0.3, 0.0),
            (full_series, 0.3, 1.5),
            (full_series, 0.0, 1.5),
            (propagator_taylor_series, 0.3, 0.0),
        ],
    )
    def test_error_falls_with_every_order_up_to_six(
        self, function, sine_angle, potential
    ):
        # K (sqrt(1 + 2C / K) - 1) with C = Laplacian / (2K) + potential. The
        # propagator scope expands only the Laplacian part. Measured relative
        # errors for the 300 mrad vacuum case, orders 1 to 6: 2.2e-2, 9.9e-4,
        # 5.4e-5, 3.3e-6, 2.2e-7, 1.5e-8; each order gains at least 14x, against
        # the asserted 5x. Coefficients that are not the binomial ones stall the
        # error, for example a geometric series at 1.2e-5 from order 4 on.
        laplacian = -((self.K * np.sin(sine_angle)) ** 2)
        c_laplacian = laplacian / (2 * self.K)
        c_total = c_laplacian + potential
        if function is full_series:
            exact = self.K * (np.sqrt(1 + 2 * c_total / self.K) - 1)
        else:
            exact = self.K * (np.sqrt(1 + 2 * c_laplacian / self.K) - 1) + potential
        errors = [
            abs(self.series_offset(function, n, laplacian, potential) - exact)
            for n in range(1, 7)
        ]
        for lower, higher in zip(errors[:-1], errors[1:]):
            assert higher < lower / 5, errors
        assert errors[-1] < 1e-7 * abs(exact), errors

    @pytest.mark.parametrize("energy", [30e3, 200e3])
    def test_coefficients_are_the_binomial_ones(self, energy):
        # coefficient of C^i in K (sqrt(1 + 2C / K) - 1), times K^(i - 1)
        wavelength = energy2wavelength(energy)
        k = 2 * np.pi / wavelength
        binomial = {2: -1 / 2, 3: 1 / 2, 4: -5 / 8, 5: 7 / 8, 6: -21 / 16}
        for i, coefficient in binomial.items():
            np.testing.assert_allclose(
                _series_prefactor(i, wavelength) * k ** (i - 1),
                coefficient,
                rtol=1e-12,
            )

    @pytest.mark.parametrize(
        "potential, offsets",
        [
            (0.0, (-3.931207462702469, -4.017037704469395, -4.020785576069488)),
            (1.5, (-2.431207462702469, -2.464034557479313, -2.4649210455047244)),
        ],
    )
    def test_orders_one_to_three_of_full_series_are_unchanged(self, potential, offsets):
        # 300 mrad plane wave at 30 keV. The coefficients of orders 1 to 3 are
        # 1, -1/(2K), 1/(2K^2), so these hard-coded values do not depend on the
        # coefficients of the higher orders.
        laplacian = -((self.K * np.sin(0.3)) ** 2)
        for order, offset in enumerate(offsets, start=1):
            result = self.series_offset(full_series, order, laplacian, potential)
            np.testing.assert_allclose(result, offset, rtol=1e-13)

    def test_orders_one_to_three_of_propagator_series_are_unchanged(self):
        laplacian = -((self.K * np.sin(0.3)) ** 2)
        offsets = (-2.431207462702469, -2.5170377044693955, -2.520785576069488)
        for order, offset in enumerate(offsets, start=1):
            result = self.series_offset(propagator_taylor_series, order, laplacian, 1.5)
            np.testing.assert_allclose(result, offset, rtol=1e-13)


class TestOutputShapes:
    """Test that output shapes match expected dimensions."""

    def test_exit_planes_grid_scan_shape(self, test_system):
        """The exit planes lead, then the scan, then the grid; the first plane is
        the entrance plane, which holds the incident probe at every position."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        grid_scan = test_system["grid_scan"]

        result = probe.multislice(
            potential=potential_exit_planes,
            scan=grid_scan,
            lazy=False,
            algorithm=RealSpaceMultislice(),
        )

        expected_shape = (
            (potential_exit_planes.num_exit_planes,)
            + grid_scan.shape
            + potential_exit_planes.gpts
        )
        assert result.array.shape == expected_shape
        # the entrance plane is copied before any step: exactly equal
        np.testing.assert_array_equal(
            to_host_array(result.array[0]),
            to_host_array(probe.build(scan=grid_scan, lazy=False).array),
        )


class TestDetectors:
    """Test multislice with various detector configurations."""

    def test_multiple_detectors(self, test_system):
        """Test with multiple detectors."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
        ]
        results = probe.multislice(
            potential=potential,
            scan=scan,
            detectors=detectors,
            lazy=False,
        )

        # Should return a list of measurements
        assert isinstance(results, list)
        assert len(results) == 2


@pytest.mark.filterwarnings("ignore:return_backscattered is deprecated:FutureWarning")
class TestDeprecatedReturnBackscattered:
    """The deprecated return_backscattered: the backscattered waves at every exit
    plane, which need an exit plane after every slice."""

    def test_backscattering_requires_full_expansion(self, test_system):
        """Test that backscattering requires expansion_scope='full'."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        with pytest.raises(
            ValueError,
            match="Backscattering contributions require expansion_scope='full'",
        ):
            probe.multislice(
                potential=potential_exit_planes,
                scan=scan,
                lazy=False,
                algorithm=RealSpaceMultislice(order=3),
                return_backscattered=True,
            )

    def test_backscattering_requires_exit_planes(self, test_system):
        """Test that backscattering requires exit_planes."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        with pytest.raises(
            ValueError,
            match="Backscattering contributions require potential.exit_planes",
        ):
            probe.multislice(
                potential=potential,
                scan=scan,
                lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
                return_backscattered=True,
            )

    def test_backscattering_requires_the_entrance_plane(self):
        # the back-propagation sums the backscattered waves into the entrance
        # plane: a plane after every slice is not enough without it
        atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
        potential = abtem.Potential(
            atoms,
            gpts=(24, 20),
            slice_thickness=atoms.cell[2, 2] / 6,
            exit_planes=tuple(range(6)),
        )
        with pytest.raises(ValueError, match="starting with the entrance plane -1"):
            _deprecated_backscattered_arrays(potential, False)

    # One tuple comparison checks every spelling, before the lazy/eager split;
    # the lazy case checks that the error surfaces from the dask task.
    @pytest.mark.parametrize(
        "exit_planes, lazy",
        [
            (2, False),
            ((-1, 0, 1, 2, 3, 4), False),  # all but the last plane
            (np.array([-1, 1, 3, 5]), True),
        ],
        ids=["int", "tuple", "array-lazy"],
    )
    def test_backscattering_requires_an_exit_plane_after_every_slice(
        self, exit_planes, lazy
    ):
        atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
        potential = abtem.Potential(
            atoms,
            gpts=(24, 20),
            slice_thickness=atoms.cell[2, 2] / 6,
            exit_planes=exit_planes,
        )
        with pytest.raises(ValueError, match="after every slice"):
            _deprecated_backscattered_arrays(potential, lazy)

    def test_backscattering_accepts_every_exit_plane_as_a_sequence(self):
        # Three random slices on a small grid: the run is cheap, and a sequence
        # spelling of every plane must give the result of exit_planes=1.
        array = np.random.default_rng(0).random((3, 16, 12)).astype(np.float32)

        def potential(exit_planes):
            return abtem.PotentialArray(
                array, slice_thickness=1.0, extent=(4.0, 3.0), exit_planes=exit_planes
            )

        reference = _deprecated_backscattered_arrays(potential(1), False)[1]
        assert np.abs(reference[0]).max() > 0
        for exit_planes in (tuple(range(-1, 3)), list(range(-1, 3)), np.arange(-1, 3)):
            spelled = potential(exit_planes)
            assert tuple(spelled.exit_planes) == potential(1).exit_planes
            np.testing.assert_array_equal(
                _deprecated_backscattered_arrays(spelled, False)[1],
                reference,
                err_msg=type(exit_planes).__name__,
            )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_backscattered_wave_crosses_each_slab_slice_once(self, lazy, device):
        # Slab over vacuum: the only source is the slab/vacuum interface at plane 3
        # (after slice 2). A laterally uniform backward wave crossing a uniform
        # slice of projected potential V (in V Å) turns its phase by sigma * V and
        # keeps its modulus, so each plane above the interface differs from the one
        # below by exactly sigma, whichever sign the backward step is given.
        array = np.zeros((7, 24, 20), dtype=np.float32)
        array[:3] = 1.0
        with abtem.config.set({"device": device}):
            potential = abtem.PotentialArray(
                array, slice_thickness=1.0, extent=(6.0, 5.0), exit_planes=1
            )
            _, backscattered = _deprecated_backscattered_arrays(potential, lazy)

        source = backscattered[3]
        assert np.abs(source).min() > 0
        for plane in range(3):
            np.testing.assert_allclose(
                np.abs(backscattered[plane]), np.abs(source), rtol=1e-4
            )
            step = np.angle(backscattered[plane] * np.conj(backscattered[plane + 1]))
            np.testing.assert_allclose(
                np.abs(step), abtem.core.energy.energy2sigma(100e3), rtol=1e-2
            )
        assert not np.any(backscattered[4:])


def _random_slices(exit_planes):
    # three random slices on a small grid: cheap, and every slice reflects
    array = np.random.default_rng(0).random((3, 16, 12)).astype(np.float32)
    return abtem.PotentialArray(
        array, slice_thickness=1.0, extent=(4.0, 3.0), exit_planes=exit_planes
    )

    @pytest.mark.parametrize("lazy", [False, True])
    def test_the_deprecated_output_of_two_configurations_and_detectors(self, lazy):
        # guards the deprecated output until it is removed: each configuration of a
        # frozen-phonon potential gives the outputs of its run alone
        atoms = ase.build.bulk("Si", cubic=True)
        frozen_phonons = abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1)
        kwargs = dict(gpts=(24, 20), slice_thickness=5.43 / 3, exit_planes=1)
        detectors = [abtem.AnnularDetector(0, 20), abtem.PixelatedDetector()]
        algorithm = RealSpaceMultislice(expansion_scope="full")

        def run(potential, lazy):
            results = abtem.PlaneWave(energy=100e3).multislice(
                potential,
                detectors=detectors,
                algorithm=algorithm,
                return_backscattered=True,
                lazy=lazy,
            )
            if lazy:
                results = [
                    r.compute(scheduler="synchronous", progress_bar=False)
                    for r in results
                ]
            return [to_host_array(r.array) for r in results]

        members = [
            run(abtem.Potential(config, **kwargs), False) for config in frozen_phonons
        ]
        result = run(abtem.Potential(frozen_phonons, **kwargs), lazy)
        assert len(result) == 3
        for i, output in enumerate(result):
            expected = np.stack([member[i] for member in members])
            np.testing.assert_allclose(
                output, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
            )

    def test_a_lazy_deprecated_call_warns_once(self):
        # the tasks of a lazy run rebuild the transform, and must not warn again
        with pytest.warns(FutureWarning) as record:
            outputs = abtem.PlaneWave(energy=100e3).multislice(
                _random_slices(1),
                algorithm=RealSpaceMultislice(expansion_scope="full"),
                return_backscattered=True,
                lazy=True,
            )
            for output in outputs:
                output.compute(scheduler="synchronous", progress_bar=False)
        messages = [str(r.message) for r in record]
        assert sum("return_backscattered" in m for m in messages) == 1


@pytest.mark.filterwarnings("ignore:return_backscattered is deprecated:FutureWarning")
class TestDeprecatedExpansionScope:
    """The deprecated expansion_scope: each value warns at the caller's line, and the
    algorithms compare equal as before."""

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: RealSpaceMultislice(expansion_scope="full"), "series of the full"),
            (lambda: RealSpaceMultislice(expansion_scope="propagator"), "leave it out"),
            (lambda: FourierMultislice(expansion_scope="propagator"), "no effect"),
        ],
    )
    def test_an_expansion_scope_warns_at_the_callers_line(self, make, match):
        with pytest.warns(FutureWarning, match=match) as record:
            make()
        assert all(os.path.samefile(r.filename, __file__) for r in record)

    def test_the_algorithms_compare_as_before(self):
        with pytest.warns(FutureWarning):
            assert RealSpaceMultislice() == RealSpaceMultislice(
                expansion_scope="propagator"
            )
            assert FourierMultislice() == FourierMultislice(
                expansion_scope="propagator"
            )
        assert RealSpaceMultislice().expansion_scope == "propagator"
        assert FourierMultislice().expansion_scope == "propagator"

    @pytest.mark.parametrize(
        "make",
        [
            lambda: RealSpaceMultislice(expansion_scope="bogus"),
            lambda: FourierMultislice(expansion_scope="full"),
        ],
    )
    def test_an_unknown_expansion_scope_is_refused(self, make):
        with pytest.raises(ValueError, match="expansion_scope"):
            make()


class TestBackscattering:
    """The wave reflected at the entrance, measured by detectors with
    reflected=True."""

    @pytest.mark.parametrize(
        "algorithm", [FourierMultislice(), RealSpaceMultislice(order=2)]
    )
    def test_reflected_detectors_need_an_algorithm_that_reflects(self, algorithm):
        with pytest.raises(ValueError, match="needs an algorithm that reflects"):
            abtem.PlaneWave(energy=100e3).multislice(
                _random_slices(None),
                detectors=abtem.WavesDetector(reflected=True),
                algorithm=algorithm,
            )

    def test_reflected_detectors_are_refused_where_nothing_reflects(self):
        potential = _random_slices(None)
        s_matrix = abtem.SMatrix(potential=potential, energy=100e3, semiangle_cutoff=20)
        with pytest.raises(ValueError, match="cannot be used here"):
            s_matrix.scan(detectors=abtem.AnnularDetector(0, 20, reflected=True))

    def test_the_multislice_function_refuses_what_the_transform_refuses(self):
        # multislice_and_detect is also called directly, without MultisliceTransform
        from abtem.multislice import (
            MultisliceTransform,
            multislice_and_detect,
            transition_potential_multislice_and_detect,
        )

        potential = _random_slices(None)
        waves = abtem.PlaneWave(energy=100e3).match_grid(potential).build(lazy=False)
        reflected = [abtem.WavesDetector(reflected=True)]
        with pytest.raises(ValueError, match="needs an algorithm that reflects"):
            multislice_and_detect(waves, potential, reflected, RealSpaceMultislice())
        with pytest.warns(FutureWarning):
            full = RealSpaceMultislice(expansion_scope="full")
        with pytest.raises(ValueError, match="not both"):
            multislice_and_detect(
                waves, _random_slices(1), reflected, full, return_backscattered=True
            )
        with pytest.raises(ValueError, match="only for the elastic multislice"):
            MultisliceTransform(
                potential,
                reflected,
                multislice_func=transition_potential_multislice_and_detect,
                algorithm=full,
            )

    def test_the_automatic_batch_counts_the_waves_a_reflected_detector_keeps(self):
        # a reflected detector keeps the backscattered wave of every slice of a block
        from abtem.core.chunks import estimate_scan_batch_size
        from abtem.waves import _max_batch_with_exit_planes

        potential = abtem.Potential(
            ase.build.bulk("Si", cubic=True) * (1, 1, 6), gpts=64, slice_thickness=0.5
        )
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20).match_grid(potential)
        annular = abtem.AnnularDetector(0, 20)
        assert _max_batch_with_exit_planes(probe, potential, [annular], "auto") == (
            "auto"
        )
        batch = _max_batch_with_exit_planes(
            probe,
            potential,
            [annular, abtem.AnnularDetector(0, 20, reflected=True)],
            "auto",
        )
        unreduced = estimate_scan_batch_size(probe.gpts, probe.dtype, "cpu")
        # the kept waves, and one more for the two annular outputs
        assert batch == unreduced // (potential.num_slices + 1)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_the_reflected_wave_is_minus_the_deprecated_one_at_the_entrance(
        self, lazy
    ):
        # the deprecated output is the backscattered component B psi, the reflected
        # wave is -B psi
        with pytest.warns(FutureWarning, match="return_backscattered is deprecated"):
            _, old = _deprecated_backscattered_arrays(_random_slices(1), lazy)
        transmitted, reflected = _multislice_arrays(_random_slices(None), lazy)
        assert reflected.shape == transmitted.shape
        assert np.abs(reflected).max() > 0
        np.testing.assert_allclose(reflected, -old[0], rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_a_reflected_detector_has_no_exit_plane_axis(self, lazy):
        results = abtem.PlaneWave(energy=100e3).multislice(
            _random_slices(1),
            detectors=[
                abtem.WavesDetector(),
                abtem.WavesDetector(reflected=True),
                abtem.PixelatedDetector(reflected=True),
            ],
            algorithm=RealSpaceMultislice(expansion_scope="full"),
            lazy=lazy,
        )
        transmitted, reflected, pattern = results
        assert transmitted.shape == (4, 16, 12)  # the entrance and three planes
        assert reflected.shape == (16, 12)
        assert not reflected.ensemble_axes_metadata
        assert reflected.metadata["reflected"] and pattern.metadata["reflected"]
        assert "reflected" not in transmitted.metadata
        if lazy:
            pattern, reflected = pattern.compute(), reflected.compute()
        np.testing.assert_allclose(
            to_host_array(pattern.array),
            to_host_array(abtem.PixelatedDetector().detect(reflected).array),
            rtol=1e-5,
            atol=1e-12,
        )

    @pytest.mark.parametrize("frozen_phonons", [False, True])
    def test_a_reflected_annular_scan_is_lazy_as_it_is_eager(self, frozen_phonons):
        # the reflected output drops the exit-plane axis and the wave's own axes
        atoms = ase.build.bulk("Si", cubic=True)
        if frozen_phonons:
            atoms = abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1)
        potential = abtem.Potential(
            atoms, gpts=(24, 20), slice_thickness=5.43 / 3, exit_planes=1
        )
        detectors = [
            abtem.AnnularDetector(inner=0, outer=20),
            abtem.AnnularDetector(inner=0, outer=20, reflected=True),
        ]
        # three positions: no scan axis has the size of the two configurations
        scan = abtem.GridScan(start=(0, 0), end=(2, 2), gpts=(3, 1))
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
        algorithm = RealSpaceMultislice(expansion_scope="full")
        eager = probe.scan(
            potential, scan=scan, detectors=detectors, algorithm=algorithm, lazy=False
        )
        lazy = probe.scan(
            potential, scan=scan, detectors=detectors, algorithm=algorithm, lazy=True
        )
        assert eager[0].shape == (4, 3, 1) and eager[1].shape == (3, 1)
        for e, measurement in zip(eager, lazy):
            assert measurement.shape == e.shape
            # synchronously, as the other tests here: Numba's workqueue layer aborts
            # on parallel stencils from several threads
            computed = measurement.compute(scheduler="synchronous", progress_bar=False)
            np.testing.assert_allclose(
                to_host_array(computed.array), to_host_array(e.array), rtol=1e-5
            )

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.slow
    def test_backscattering_with_detectors(self, test_system):
        """Detectors of the transmitted and of the reflected wave in one run measure
        each what they would measure of that wave alone."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        def detectors(reflected):
            return [
                abtem.PixelatedDetector(reflected=reflected),
                abtem.AnnularDetector(inner=30, outer=100, reflected=reflected),
                abtem.WavesDetector(reflected=reflected),
            ]

        results = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            detectors=detectors(False) + detectors(True),
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
        )

        assert isinstance(results, (list, tuple))
        assert len(results) == 6
        transmitted, reflected = results[2], results[5]
        assert isinstance(reflected, abtem.Waves)
        # the transmitted waves have an exit-plane axis, the reflected wave none
        assert reflected.shape == transmitted.shape[1:]
        assert np.abs(to_host_array(reflected.array)).max() > 0
        # float32 rounding only: on the GPU the annular sum differs by 1.7e-7
        # relative between the two routes; another detector or the backscattered
        # waves would differ by order 1.
        for detector, measurement, waves in zip(
            detectors(False) * 2, results, [transmitted] * 3 + [reflected] * 3
        ):
            actual = to_host_array(measurement.array)
            expected = to_host_array(detector.detect(waves).array)
            atol = 1e-12
            if test_system["device"] != "cpu":
                # The two routes Fourier transform the exit wave separately, and
                # float32 rounding of an FFT is an error relative to the
                # strongest amplitude, not to each pixel: a pixel of intensity
                # 1e-6 in a pattern reaching down to 1e-19 can differ by 1e-5
                # relative (Metal, 3 of 31768 pixels). Its intensity error is at
                # most 2 sqrt(I) * eps * sqrt(max I) <= 2 eps max I, so scale
                # atol to the peak, with a margin of 10 for the FFT's error growth.
                eps = np.finfo(np.float32).eps
                atol = max(atol, 20 * eps * np.abs(expected).max())
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=atol,
                err_msg=type(detector).__name__,
            )


def _multislice_arrays(
    potential, lazy, scan=None, backscattered=True, potential_chunk_size="auto"
):
    """The arrays of a full-expansion real-space multislice: the transmitted
    waves, and with `backscattered` also the wave reflected at the entrance."""
    detectors = [abtem.WavesDetector()]
    if backscattered:
        detectors.append(abtem.WavesDetector(reflected=True))
    kwargs = dict(
        lazy=lazy,
        detectors=detectors,
        potential_chunk_size=potential_chunk_size,
        algorithm=RealSpaceMultislice(expansion_scope="full"),
    )
    return _computed(potential, kwargs, scan, backscattered, lazy)


def _deprecated_backscattered_arrays(potential, lazy):
    """The transmitted waves and the backscattered waves at every exit plane, of the
    deprecated return_backscattered."""
    kwargs = dict(
        lazy=lazy,
        return_backscattered=True,
        algorithm=RealSpaceMultislice(expansion_scope="full"),
    )
    return _computed(potential, kwargs, None, True, lazy)


def _computed(potential, kwargs, scan, several, lazy):
    if scan is None:
        result = abtem.PlaneWave(energy=100e3).multislice(potential, **kwargs)
    else:
        result = abtem.Probe(energy=100e3, semiangle_cutoff=20).multislice(
            potential, scan=scan, **kwargs
        )
    result = result if several else [result]
    if lazy:
        result = [
            r.compute(scheduler="synchronous", progress_bar=False) for r in result
        ]
    return [to_host_array(r) for r in result]


class TestBackscatteringEnsemble:
    """The backscattered waves of an ensemble potential are those of each
    configuration run on its own."""

    @pytest.mark.parametrize(
        "num_configs, slices_per_cell, scan, lazy, device",
        [
            pytest.param(
                *case,
                lazy,
                device.values[0] if device is gpu else device,
                marks=[pytest.mark.slow] * (lazy and case[0] == 3)
                + (list(gpu.marks) if device is gpu else []),
            )
            for case in [
                (2, 2, None),  # 3 exit planes
                (3, 2, None),  # as many configurations as exit planes
                (2, 3, None),  # 4 exit planes
                (3, 2, [(1.0, 1.0), (2.0, 3.0)]),  # two probe positions
            ]
            # lazy device runs are covered by TestFullExpansionChunking
            for lazy, device in ((False, "cpu"), (False, gpu), (True, "cpu"))
        ],
    )
    def test_frozen_phonons(self, num_configs, slices_per_cell, scan, lazy, device):
        atoms = ase.build.bulk("Si", cubic=True)
        frozen_phonons = abtem.FrozenPhonons(atoms, num_configs, sigmas=0.1, seed=1)
        kwargs = dict(
            gpts=(24, 20),
            slice_thickness=atoms.cell[2, 2] / slices_per_cell,
            exit_planes=1,
        )

        with abtem.config.set({"device": device}):
            members = [
                _multislice_arrays(abtem.Potential(config, **kwargs), False, scan)
                for config in frozen_phonons
            ]
            result = _multislice_arrays(
                abtem.Potential(frozen_phonons, **kwargs), lazy, scan
            )

        for i, output in enumerate(result):
            expected = np.stack([member[i] for member in members])
            assert output.shape == expected.shape
            np.testing.assert_allclose(
                output, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
            )

    @pytest.mark.parametrize(
        "num_frozen_phonons, lazy",
        [
            pytest.param(
                num_frozen_phonons,  # 5 exit planes
                lazy,
                marks=[pytest.mark.slow] if lazy and num_frozen_phonons == 5 else [],
            )
            for num_frozen_phonons in (2, 5)
            for lazy in (False, True)
        ],
    )
    def test_crystal_potential(self, num_frozen_phonons, lazy):
        atoms = ase.build.bulk("Si", cubic=True)

        def crystal():
            unit = abtem.Potential(
                abtem.FrozenPhonons(atoms, 4, sigmas=0.1, seed=1),
                gpts=(24, 20),
                slice_thickness=atoms.cell[2, 2] / 2,
            )
            return abtem.CrystalPotential(
                unit,
                (2, 1, 2),
                num_frozen_phonons=num_frozen_phonons,
                seeds=tuple(range(1, num_frozen_phonons + 1)),
                exit_planes=1,
            )

        members = []
        for _, _, member in crystal().generate_blocks():
            member = member.item()
            chunks = list(member.generate_chunked_slices())
            slices = abtem.PotentialArray(
                np.concatenate([chunk.array for chunk in chunks]),
                slice_thickness=[t for chunk in chunks for t in chunk.slice_thickness],
                sampling=member.sampling,
                exit_planes=1,
            )
            members.append(_multislice_arrays(slices, False))

        result = _multislice_arrays(crystal(), lazy)

        for i, output in enumerate(result):
            expected = np.stack([member[i] for member in members])
            assert output.shape == expected.shape
            np.testing.assert_allclose(
                output, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
            )


def _full_expansion_potentials():
    atoms = ase.build.bulk("Si", cubic=True)
    grid = dict(gpts=(24, 20), slice_thickness=atoms.cell[2, 2] / 3)  # 6 slices

    def displaced(atoms):
        return list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]

    return {
        "potential": lambda exit_planes: abtem.Potential(
            displaced(atoms * (1, 1, 2)), exit_planes=exit_planes, **grid
        ),
        "frozen_phonons": lambda exit_planes: abtem.Potential(
            abtem.FrozenPhonons(atoms * (1, 1, 2), 4, sigmas=0.1, seed=1),
            exit_planes=exit_planes,
            **grid,
        ),
        "crystal_potential": lambda exit_planes: abtem.CrystalPotential(
            abtem.Potential(abtem.FrozenPhonons(atoms, 4, sigmas=0.1, seed=1), **grid),
            (1, 1, 2),
            num_frozen_phonons=3,
            seeds=(1, 2, 3),
            exit_planes=exit_planes,
        ),
    }


class TestFullExpansionChunking:
    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("backscattered", [False, True])
    # Lazy chunking is covered by the lazy single potential; lazy == eager for
    # the ensembles is asserted by TestBackscatteringEnsemble.
    @pytest.mark.parametrize(
        "name, lazy",
        [
            (name, lazy)
            for name in _full_expansion_potentials()
            for lazy in ((False, True) if name == "potential" else (False,))
        ],
    )
    def test_independent_of_the_potential_chunk_size(
        self, name, backscattered, lazy, device
    ):
        make = _full_expansion_potentials()[name]

        def arrays(chunk_size):
            with abtem.config.set({"device": device}):
                return _multislice_arrays(
                    make(exit_planes=2),
                    lazy,
                    backscattered=backscattered,
                    potential_chunk_size=chunk_size,
                )

        expected = arrays(6)
        # 4 does not divide the 6 slices; the lazy runs are slow, so skip chunk size 2
        for chunk_size in (1, 4) if lazy else (1, 2, 4):
            result = arrays(chunk_size)
            for array, reference in zip(result, expected):
                assert array.shape == reference.shape
                np.testing.assert_allclose(
                    array,
                    reference,
                    rtol=0,
                    atol=1e-6 * np.abs(reference).max(),
                    err_msg=f"chunk size {chunk_size}",
                )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_vacuum_in_front_delays_the_reflection_by_its_round_trip(
        self, lazy, device
    ):
        # A laterally uniform slab under 4 or under 1 slice of vacuum: the only source
        # is its front face, and a laterally uniform backward wave crosses vacuum
        # unchanged, so the two reflections differ by the round trip of the 3 Å
        # between the faces, exp(2 i k 3 Å).
        def reflected(vacuum_slices):
            array = np.zeros((vacuum_slices + 3, 24, 20), dtype=np.float32)
            array[vacuum_slices:] = 1.0
            potential = abtem.PotentialArray(
                array, slice_thickness=1.0, extent=(6.0, 5.0)
            )
            return _multislice_arrays(potential, lazy)[1]

        with abtem.config.set({"device": device}):
            deep, shallow = reflected(4), reflected(1)

        k = 2 * np.pi / abtem.core.energy.energy2wavelength(100e3)
        assert np.abs(shallow).min() > 0
        np.testing.assert_allclose(
            deep, shallow * np.exp(2j * k * 3.0), rtol=0, atol=1e-4 * np.abs(deep).max()
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_backscattering_between_identical_slices_is_zero(self, lazy, device):
        # One slice per unit cell: every slice equals the next, so the correction
        # term (the difference between consecutive slices) vanishes exactly.
        atoms = ase.build.bulk("Si", cubic=True)
        displaced = list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]
        with abtem.config.set({"device": device}):
            potential = abtem.Potential(
                displaced * (1, 1, 3),
                gpts=(24, 20),
                slice_thickness=atoms.cell[2, 2],
                exit_planes=1,
            )
            transmitted, backscattered = _multislice_arrays(potential, lazy)
            (alone,) = _multislice_arrays(potential, lazy, backscattered=False)

        np.testing.assert_array_equal(transmitted, alone)
        assert not np.any(backscattered)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_the_backscattered_wave_of_a_flat_slab_is_the_airy_reflection(self, device):
        # Oracle: the reflection of a medium of wave number k2 = sqrt(k^2 + 2 k sigma
        # V0) for a wave along (tan x, tan y, 1), with the longitudinal wave numbers
        # k_z of the two media; the reflections of the faces add with the round trip
        # exp(2 i k2_z d). The amplitude is compared, not |r|^2: it is r times the
        # round trip exp(2 i k1_z z0) to the front face at z0, so a wrong sign or
        # depth, a missing step through the slab (d up to 2.5 Å) or the untilted k for
        # a tilted member would fail.
        energy, v0, dz, vacuum = 1000.0, 10.0, 0.01, 20
        k = 2 * np.pi / abtem.core.energy.energy2wavelength(energy)
        k2 = k**2 + 2 * k * abtem.core.energy.energy2sigma(energy) * v0
        tilts_x = [0.0, 100.0]

        for n in (2, 6, 18, 250):
            profile = np.r_[np.zeros(vacuum), np.full(n, v0 * dz), np.zeros(vacuum)]
            with abtem.config.set({"device": device}):
                potential = abtem.PotentialArray(
                    np.ones((len(profile), 8, 8), np.float32) * profile[:, None, None],
                    slice_thickness=dz,
                    sampling=0.5,
                )
                reflected = abtem.PlaneWave(
                    energy=energy, tilt=(tilts_x, 0.0)
                ).multislice(
                    potential,
                    detectors=abtem.WavesDetector(reflected=True),
                    algorithm=RealSpaceMultislice(order=1, expansion_scope="full"),
                    lazy=False,
                )
            stored = to_host_array(reflected.array).mean(axis=(-2, -1))

            expected = []
            for tilt in tilts_x:
                transverse = (k * np.sin(tilt / 1e3)) ** 2
                k1z, k2z = np.sqrt(k**2 - transverse), np.sqrt(k2 - transverse)
                r12 = (k1z - k2z) / (k1z + k2z)
                phase = np.exp(2j * k2z * n * dz)
                r = r12 * (1 - phase) / (1 - r12**2 * phase)
                expected.append(r * np.exp(2j * k1z * vacuum * dz))
            np.testing.assert_allclose(
                stored, expected, rtol=0.03, err_msg=f"{n} slices"
            )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_backscattering_of_a_slab_followed_by_vacuum(self, device):
        atoms = ase.build.bulk("Si", cubic=True)
        slab = list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]
        slab.positions[:, 2] += 2.0

        def run(cells, backscattered=True):
            # the same slices of the slab, with more or less vacuum behind it
            behind = slab.copy()
            behind.cell[2, 2] = cells * atoms.cell[2, 2]
            potential = abtem.Potential(
                behind, gpts=(24, 20), slice_thickness=atoms.cell[2, 2] / 3
            )
            return _multislice_arrays(potential, False, backscattered=backscattered)

        with abtem.config.set({"device": device}):
            transmitted, reflected = run(3)
            (alone,) = run(3, backscattered=False)
            _, more_vacuum = run(4)

        np.testing.assert_array_equal(transmitted, alone)
        assert np.abs(reflected).max() > 0
        # nothing comes back from the vacuum behind the slab
        np.testing.assert_allclose(
            more_vacuum, reflected, rtol=0, atol=1e-5 * np.abs(reflected).max()
        )


class TestAlgorithmComparison:
    """Test that different algorithms produce reasonable results."""

    @pytest.mark.slow
    def test_higher_orders_differ(self, test_system):
        """Test that higher orders produce different results."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        order1_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=1),
        )

        order3_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3),
        )

        # Shapes should match
        assert order1_result.array.shape == order3_result.array.shape

        # Results should be different (if identical, something's wrong)
        assert not np.allclose(
            to_host_array(order1_result.array), to_host_array(order3_result.array), rtol=1e-10
        )

    def test_fourier_order2_differs_from_order1(self, test_system):
        """Test that Fourier order 2 differs from order 1."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        order1_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=1),
        )

        order2_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=2),
        )

        # Shapes should match
        assert order1_result.array.shape == order2_result.array.shape

        # Results should be different
        assert not np.allclose(
            to_host_array(order1_result.array), to_host_array(order2_result.array), rtol=1e-10
        )


class TestComplexWorkflows:
    """Test complex multislice workflows."""

    @pytest.mark.slow
    def test_realspace_with_scan_and_detectors(self, test_system):
        """Test RealSpace multislice with scan and detectors."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        grid_scan = test_system["grid_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
        ]

        results = probe.multislice(
            potential=potential,
            scan=grid_scan,
            detectors=detectors,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3),
        )

        assert isinstance(results, list)
        assert len(results) == len(detectors)


class TestStencilNumericalAccuracy:
    """Verify the fast stencils match the scipy reference implementation."""

    @devices
    @pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
    def test_laplace_stencil_matches_scipy_reference(self, accuracy, device):
        """Compare the fast Laplacian stencils against scipy.ndimage.convolve."""
        from abtem.core.backend import get_array_module
        from abtem.finite_difference import (
            _laplace_operator_func_slow,
            _laplace_operator_stencil,
        )

        prefactor = np.complex64(1.0 + 0.5j)
        rng = np.random.default_rng(42)
        a = (
            rng.standard_normal((2, 24, 24))
            + 1j * rng.standard_normal((2, 24, 24))
        ).astype(np.complex64)

        ref = np.stack(
            [
                _laplace_operator_func_slow(accuracy, prefactor)(a[m])
                for m in range(a.shape[0])
            ]
        )
        xp = get_array_module(device)
        result = _laplace_operator_stencil(
            accuracy, prefactor, mode="wrap", dtype=np.complex64, device=device
        )(xp.asarray(a))

        np.testing.assert_allclose(
            to_host_array(result),
            ref,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Stencil mismatch at accuracy={accuracy} on {device}",
        )

    @requires_gpu
    def test_gpu_stencil_rejects_non_complex_dtype(self):
        """The raw GPU kernel only ships complex specializations; a real array
        must raise instead of silently reinterpreting the buffer.

        CUDA-only: it reaches for cupy directly rather than going through the
        device parametrization, so it cannot stand in for another accelerator.
        The Metal stencil's own rejection is covered by the shared
        dtype check in ``_laplace_operator_stencil``."""
        import cupy as cp

        from abtem.finite_difference import _laplace_operator_stencil

        stencil = _laplace_operator_stencil(
            4, 1.0, mode="wrap", dtype=np.complex64, device="gpu"
        )
        with pytest.raises(TypeError, match="complex64 or complex128"):
            stencil(cp.ones((8, 8), dtype=cp.float32))
