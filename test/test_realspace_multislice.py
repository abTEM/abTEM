import ase
import ase.build
import numpy as np
import pytest
from utils import devices, gpu, requires_gpu, to_host_array

import abtem
from abtem.multislice import FourierMultislice, RealSpaceMultislice

# Suppress Numba performance warnings during tests
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning",
    "ignore::UserWarning"
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


class TestBackscattering:
    """Test backscattering calculations."""

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
                algorithm=RealSpaceMultislice(order=3, expansion_scope="propagator"),
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
            _multislice_arrays(potential, False)

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
            _multislice_arrays(potential, lazy)

    def test_backscattering_accepts_every_exit_plane_as_a_sequence(self):
        # Three random slices on a small grid: the run is cheap, and a sequence
        # spelling of every plane must give the result of exit_planes=1.
        array = np.random.default_rng(0).random((3, 16, 12)).astype(np.float32)

        def potential(exit_planes):
            return abtem.PotentialArray(
                array, slice_thickness=1.0, extent=(4.0, 3.0), exit_planes=exit_planes
            )

        reference = _multislice_arrays(potential(1), False)[1]
        assert np.abs(reference[0]).max() > 0
        for exit_planes in (tuple(range(-1, 3)), list(range(-1, 3)), np.arange(-1, 3)):
            spelled = potential(exit_planes)
            assert tuple(spelled.exit_planes) == potential(1).exit_planes
            np.testing.assert_array_equal(
                _multislice_arrays(spelled, False)[1],
                reference,
                err_msg=type(exit_planes).__name__,
            )

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.slow
    def test_backscattering_with_detectors(self, test_system):
        """The backscattered waves follow the detectors' measurements, which are
        those of the transmitted waves."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
            abtem.WavesDetector(),
        ]
        results = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            detectors=detectors,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            return_backscattered=True,
        )

        # N detectors + 1 backscattered waves
        assert isinstance(results, (list, tuple))
        assert len(results) == len(detectors) + 1
        *measurements, transmitted, backscattered = results
        assert isinstance(backscattered, abtem.Waves)
        assert backscattered.shape == transmitted.shape
        assert np.abs(to_host_array(backscattered.array[0])).max() > 0
        # float32 rounding only: on the GPU the annular sum differs by 1.7e-7
        # relative between the two routes; another detector or the backscattered
        # waves would differ by order 1.
        for detector, measurement in zip(detectors, measurements):
            np.testing.assert_allclose(
                to_host_array(measurement.array),
                to_host_array(detector.detect(transmitted).array),
                rtol=1e-5,
                atol=1e-12,
                err_msg=type(detector).__name__,
            )


def _multislice_arrays(
    potential, lazy, scan=None, backscattered=True, potential_chunk_size="auto"
):
    """The arrays of a full-expansion real-space multislice: the transmitted
    waves, and with `backscattered` also the backscattered waves."""
    algorithm = RealSpaceMultislice(expansion_scope="full")
    kwargs = dict(
        lazy=lazy,
        return_backscattered=backscattered,
        potential_chunk_size=potential_chunk_size,
        algorithm=algorithm,
    )
    if scan is None:
        result = abtem.PlaneWave(energy=100e3).multislice(potential, **kwargs)
    else:
        result = abtem.Probe(energy=100e3, semiangle_cutoff=20).multislice(
            potential, scan=scan, **kwargs
        )
    result = result if backscattered else [result]
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
                # backscattering needs an exit plane after every slice
                return _multislice_arrays(
                    make(exit_planes=1 if backscattered else 2),
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
    def test_backscattered_wave_crosses_vacuum_unchanged(self, lazy, device):
        # Vacuum over a laterally uniform slab: the only source is the interface
        # at plane 4 (after slice 3). A laterally uniform backward wave crosses
        # vacuum unchanged, so it reaches every plane above the interface as the
        # source itself; crossing a slab slice would rotate its phase.
        array = np.zeros((7, 24, 20), dtype=np.float32)
        array[4:] = 1.0
        with abtem.config.set({"device": device}):
            potential = abtem.PotentialArray(
                array, slice_thickness=1.0, extent=(6.0, 5.0), exit_planes=1
            )
            _, backscattered = _multislice_arrays(potential, lazy)

        source = backscattered[4]
        assert np.abs(source).min() > 0
        np.testing.assert_allclose(
            backscattered[:4],
            np.broadcast_to(source, backscattered[:4].shape),
            rtol=0,
            atol=1e-5 * np.abs(source).max(),
        )
        assert not np.any(backscattered[5:])

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
            _, backscattered = _multislice_arrays(potential, lazy)

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
    def test_backscattering_of_a_slab_followed_by_vacuum(self, device):
        atoms = ase.build.bulk("Si", cubic=True)
        slab = list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]
        slab.positions[:, 2] += 2.0
        slab.cell[2, 2] = 3 * atoms.cell[2, 2]
        with abtem.config.set({"device": device}):
            potential = abtem.Potential(
                slab, gpts=(24, 20), slice_thickness=atoms.cell[2, 2] / 3, exit_planes=1
            )
            transmitted, backscattered = _multislice_arrays(potential, False)
            (alone,) = _multislice_arrays(potential, False, backscattered=False)

        np.testing.assert_array_equal(transmitted, alone)
        per_plane = np.abs(backscattered).max(axis=(-2, -1))
        # nothing comes back from the vacuum behind the slab
        assert per_plane[:5].all() and not per_plane[5:].any()


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
