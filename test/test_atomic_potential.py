import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import abtem
from abtem.integrals import (
    _MAX_CACHE_ENTRIES,
    _MAX_SCATTERING_FACTOR_ENTRIES,
    GaussianProjectionIntegrals,
    QuadratureProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
    _DeviceArrayCache,
)
from utils import assert_array_matches_device, gpu

# from abtem.integrals import GaussianProjectionIntegrals
from abtem.parametrizations import (
    KirklandParametrization,
    LobatoParametrization,
    PengParametrization,
)


@given(atomic_number=st.integers(min_value=1, max_value=98))
@pytest.mark.parametrize(
    "parametrization_a",
    [
        LobatoParametrization(),
        KirklandParametrization(),
        PengParametrization(),
    ],
    ids=["lobato", "kirkland", "peng"],
)
@pytest.mark.parametrize(
    "parametrization_b",
    [LobatoParametrization(), KirklandParametrization(), PengParametrization()],
    ids=["lobato", "kirkland", "peng"],
)
def test_parametrizations(atomic_number, parametrization_a, parametrization_b):
    k = np.linspace(0, 5, 100)
    assert np.allclose(
        parametrization_a.projected_scattering_factor(atomic_number)(k),
        parametrization_b.projected_scattering_factor(atomic_number)(k),
        atol=10,
        rtol=0.1,
    )
    r = np.linspace(0.2, 5, 100)
    assert np.allclose(
        parametrization_a.projected_potential(atomic_number)(r),
        parametrization_b.projected_potential(atomic_number)(r),
        atol=20,
        rtol=0.1,
    )


# @pytest.mark.parametrize('parameters',
#                          [{'gaussian_projection_integrals':
#                                GaussianProjectionIntegrals(correction_parametrization=None),
#                            'parametrization':
#                                PengParametrization()
#                            },
#                           {'gaussian_projection_integrals':
#                                GaussianProjectionIntegrals(correction_parametrization='lobato'),
#                            'parametrization': LobatoParametrization()
#                            }
#                           ], ids=['uncorrected', 'corrected'])
# def test_gaussian_projection_integrals(parameters):
#     parametrization = parameters['parametrization']
#     gaussian_projection_integrals = parameters['gaussian_projection_integrals']
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.1, 0.1)
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#     a = -np.inf
#     b = np.inf
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build('C', gpts, sampling)
#
#     projections = gaussian_scattering_factors.integrate_on_grid(
#         positions,
#         a,
#         b,
#         gpts,
#         sampling,
#         fourier_space=True)
#
#     k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
#     analytical = parametrization.projected_scattering_factor(symbol)(k ** 2)
#     assert np.allclose(analytical, projections[0].real, rtol=1e-6)
#
#
# @pytest.mark.parametrize('fourier_space', [True, False])
# def test_finite_gaussian_projection_integrals(fourier_space):
#     gaussian_projection_integrals = GaussianProjectionIntegrals(correction_parametrization=None)
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.02, 0.02)
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#     a = -.2
#     b = .2
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build('C', gpts, sampling)
#
#     projections = gaussian_scattering_factors.integrate_on_grid(
#         positions,
#         a,
#         b,
#         gpts,
#         sampling,
#         fourier_space=fourier_space)
#
#     if fourier_space:
#         k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
#         analytical = PengParametrization().finite_projected_scattering_factor(symbol)(k, a, b)
#         assert np.allclose(analytical, projections[0].real, rtol=1e-6)
#
#     else:
#         r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False)
#         analytical = PengParametrization().finite_projected_potential(symbol)(r, a, b)
#         assert np.allclose(analytical[:len(r) // 2], projections[0][:len(r) // 2], rtol=1e-6, atol=5)


# @pytest.mark.parametrize('parametrization',
#                          [LobatoParametrization(),
#                           KirklandParametrization()])
# def test_quadrature(parametrization):
#     quadrature = QuadratureProjectionIntegrals(parametrization, quad_order=20, cutoff_tolerance=1e-9)
#
#     symbol = 'Au'
#     gpts = (256, 256)
#     sampling = (0.05, 0.05)
#     xp = np
#     a = -20
#     b = 20
#
#     positions = xp.array([[0, 0, 0]], dtype=xp.float32)
#
#     table = quadrature.build_integral_table(symbol, min(sampling) / 2)
#
#     integrated = table.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False).astype(np.float32)
#     analytical = parametrization.projected_potential(symbol)(r[1:])
#
#     assert np.allclose(analytical, integrated[0, 1:], atol=2)
#

# def test_finite_projections():
#     quadrature = ProjectionQuadratureRule('lobato', quad_order=8, cutoff_tolerance=1e-4)
#     gaussian_projection_integrals = GaussianProjectionIntegrals()
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.05, 0.05)
#     a = .1
#     b = 1
#
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#
#     table = quadrature.build_integral_table(symbol, min(sampling) / 2)
#
#     quadrature_potential = table.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build(symbol, gpts, sampling)
#
#     gaussian_potential = gaussian_scattering_factors.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     assert np.allclose(quadrature_potential[0, :gpts[1] // 2], gaussian_potential[0, :gpts[1] // 2], atol=2)


class TestScatteringFactorCacheKey:
    """``get_scattering_factor`` cached on the chemical symbol alone.

    The cached array depends on the grid and on the device it was allocated
    on as well, so reusing one integrator across two potentials served the
    first potential's array to the second: a broadcast error for a new grid,
    and a numpy array handed to a cupy kernel for a new device. ``integrator``
    is a documented ``Potential`` parameter, so sharing one is ordinary use.
    """

    @staticmethod
    def _atoms():
        import ase.build

        return ase.build.bulk("Si", cubic=True)

    def test_second_grid_is_not_served_the_first_grids_array(self):
        integrator = ScatteringFactorProjectionIntegrals()
        for gpts, sampling in (((64, 64), (0.125, 0.125)), ((128, 128), (0.0625,) * 2)):
            array = integrator.get_scattering_factor("Si", gpts, sampling, "cpu")
            assert array.shape == gpts

    def test_potentials_on_two_grids_may_share_an_integrator(self):
        atoms = self._atoms()
        shared = ScatteringFactorProjectionIntegrals()
        for gpts in ((64, 64), (128, 128)):
            got = abtem.Potential(
                atoms, gpts=gpts, slice_thickness=1.0, integrator=shared
            ).build(lazy=False)
            reference = abtem.Potential(
                atoms,
                gpts=gpts,
                slice_thickness=1.0,
                integrator=ScatteringFactorProjectionIntegrals(),
            ).build(lazy=False)
            assert np.array_equal(got.array, reference.array)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_array_lands_on_the_requested_device(self, device):
        integrator = ScatteringFactorProjectionIntegrals()
        # Warm the cache on the cpu first: the array served for ``device``
        # must still be the one belonging to ``device``.
        integrator.get_scattering_factor("Si", (64, 64), (0.125, 0.125), "cpu")
        array = integrator.get_scattering_factor("Si", (64, 64), (0.125, 0.125), device)
        assert_array_matches_device(array, device)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_potential_may_share_an_integrator_across_devices(self, device):
        atoms = self._atoms()
        shared = ScatteringFactorProjectionIntegrals()
        abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=1.0, integrator=shared, device="cpu"
        ).build(lazy=False)
        got = abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=1.0, integrator=shared, device=device
        ).build(lazy=False)
        reference = abtem.Potential(
            atoms,
            gpts=(64, 64),
            slice_thickness=1.0,
            integrator=ScatteringFactorProjectionIntegrals(),
            device=device,
        ).build(lazy=False)
        assert np.allclose(
            np.asarray(got.to_cpu().array), np.asarray(reference.to_cpu().array)
        )

    def test_cache_is_bounded(self):
        """A full key admits an entry per grid, so it needs a bound."""
        integrator = ScatteringFactorProjectionIntegrals()
        for n in range(_MAX_SCATTERING_FACTOR_ENTRIES + 8):
            integrator.get_scattering_factor("Si", (8 + n,) * 2, (0.1, 0.1), "cpu")
        assert len(integrator.scattering_factors) <= _MAX_SCATTERING_FACTOR_ENTRIES

    def test_cache_is_least_recently_used_not_first_in_first_out(self):
        """A hit must refresh recency, or a sweep gets no benefit from it.

        This needs *more* distinct keys than the bound: with exactly maxsize
        keys nothing is ever evicted and FIFO and LRU are indistinguishable.
        """
        import collections

        integrator = ScatteringFactorProjectionIntegrals()
        recomputations: dict = collections.Counter()
        original = integrator._calculate_scattering_factor_on_device

        def counting(symbol, gpts, sampling, device_key):
            recomputations[tuple(gpts)] += 1
            return original(symbol, gpts, sampling, device_key)

        integrator._calculate_scattering_factor_on_device = counting

        kept = (8, 8)
        integrator.get_scattering_factor("Si", kept, (0.1, 0.1), "cpu")
        # Insert well past the bound, touching `kept` between each insertion so
        # it stays the most recently used entry. Under FIFO it would be evicted
        # first regardless; under LRU it survives.
        for n in range(_MAX_SCATTERING_FACTOR_ENTRIES + 8):
            integrator.get_scattering_factor("Si", (16 + n,) * 2, (0.1, 0.1), "cpu")
            integrator.get_scattering_factor("Si", kept, (0.1, 0.1), "cpu")
        assert len(integrator.scattering_factors) <= _MAX_SCATTERING_FACTOR_ENTRIES
        # Presence at the end proves nothing: under FIFO `kept` is evicted
        # repeatedly, but the very next lookup misses and re-inserts it at the
        # tail, so it is present either way. What separates LRU from FIFO is
        # how often it had to be *recomputed*.
        assert recomputations[kept] == 1, (
            f"`kept` was recomputed {recomputations[kept]} times; under LRU a "
            "touched entry is never evicted, so once is the only right answer"
        )

    def test_concurrent_access_does_not_race(self):
        """Hand-rolled dict eviction raced: two threads evicting the same key
        gave KeyError, and iteration could see the dict resized underneath."""
        import sys
        import threading

        errors = []

        def hammer(integrator, seed):
            try:
                for n in range(4000):
                    g = 8 + ((seed * 7919 + n * 13) % 200)
                    integrator.get_scattering_factor(
                        "Si", (g, g), (0.1, 0.1), "cpu"
                    )
            except Exception as exc:  # noqa: BLE001 -- report, don't mask
                errors.append(exc)

        integrator = ScatteringFactorProjectionIntegrals()
        # A short switch interval makes the window reachable in seconds
        # instead of relying on luck.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [
                threading.Thread(target=hammer, args=(integrator, i))
                for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)

        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"


class TestScatteringFactorIntegratorIsAPlainObject:
    """The cache must not make the integrator unpicklable, unequal or uncopyable.

    A per-instance ``functools.lru_cache`` around a bound method was tried and
    broke all three: the wrapper resolves by ``__qualname__`` so neither pickle
    nor cloudpickle can serialise it (which disables dask's processes scheduler
    and distributed, and therefore the multi-GPU path the cache exists to
    serve); ``EqualityMixin`` compares ``__dict__`` and a wrapper has identity
    equality; and ``_lru_cache_wrapper.__deepcopy__`` returns ``self``, so a
    "copy" shared the original's cache and stayed bound to the original.
    """

    @staticmethod
    def _warmed():
        integrator = ScatteringFactorProjectionIntegrals()
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), "cpu")
        return integrator

    def _potential(self, **kwargs):
        import ase.build

        return abtem.Potential(
            ase.build.bulk("Si", cubic=True),
            gpts=(64, 64),
            slice_thickness=1.0,
            **kwargs,
        )

    def test_pickles(self):
        import pickle

        assert isinstance(
            pickle.loads(pickle.dumps(self._warmed())),
            ScatteringFactorProjectionIntegrals,
        )

    def test_potential_cloudpickles(self):
        cloudpickle = pytest.importorskip("cloudpickle")
        assert isinstance(
            cloudpickle.loads(cloudpickle.dumps(self._potential())), abtem.Potential
        )

    def test_builds_under_the_processes_scheduler(self):
        """The processes scheduler round-trips the graph through pickle."""
        import dask

        with dask.config.set(scheduler="processes"):
            built = self._potential().build(lazy=True).compute(progress_bar=False)
        assert built.array.shape == (6, 64, 64)

    def test_tokenizes_deterministically(self):
        """dask falls back to a random token when pickling fails."""
        import dask

        assert len({dask.base.tokenize(self._potential()) for _ in range(3)}) == 1

    def test_two_fresh_integrators_compare_equal(self):
        assert ScatteringFactorProjectionIntegrals() == (
            ScatteringFactorProjectionIntegrals()
        )
        assert self._potential() == self._potential()

    def test_deepcopy_is_independent(self):
        """A copy starts cold and cannot disturb the original.

        The clone's cache is empty rather than a copy of the original's:
        __getstate__ drops caches, and deepcopy goes through it. That is the
        point -- a cache is per-worker state, not part of the object's value --
        so what this test pins is independence in both directions, not that the
        contents were carried over.
        """
        import copy

        original = self._warmed()
        before = len(original.scattering_factors)
        assert before > 0

        clone = copy.deepcopy(original)
        assert clone.scattering_factors is not original.scattering_factors
        assert len(clone.scattering_factors) == 0

        clone.get_scattering_factor("Si", (32, 32), (0.1, 0.1), "cpu")
        assert len(clone.scattering_factors) == 1
        assert len(original.scattering_factors) == before

    def test_cached_arrays_are_freed_without_a_cyclic_collection(self):
        """The wrapper made instance -> cache -> bound method -> instance."""
        import gc
        import weakref

        gc.disable()
        try:
            integrator = self._warmed()
            reference = weakref.ref(integrator)
            del integrator
            assert reference() is None
        finally:
            gc.enable()

    @pytest.mark.parametrize(
        "device", ["cpu", np, pytest.param(np.zeros(3), id="array")]
    )
    def test_device_may_be_any_form_get_array_module_accepts(self, device):
        """``device`` is not part of the key, so it need not be hashable."""
        integrator = ScatteringFactorProjectionIntegrals()
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), device)
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), "cpu")
        # All spellings of the same physical device share one canonical entry.
        assert len(integrator.scattering_factors) == 1


class TestIntegratorCaches:
    """Caches in integrals.py, keyed on less than their value depends on.

    ``QuadratureProjectionIntegrals._tables`` was keyed on the element alone
    while the table is built from ``sampling``; every key omitted the precision
    config, through which every cached value is built. Two further caches in
    ``GaussianProjectionIntegrals`` computed a key, missed, recomputed and never
    stored, and are removed rather than repaired.
    """

    @staticmethod
    def _atoms(a=5.65):
        import ase.build

        # Two elements on purpose. Note this is necessary but not sufficient:
        # a potential-level shared-vs-fresh comparison cannot detect a
        # symbol-blind key either way, because both sides serve the first
        # element's array to the second and therefore agree. Only the
        # method-level probes below catch that one.
        return ase.build.bulk("GaAs", crystalstructure="zincblende", a=a, cubic=True)

    def _build(self, integrator, gpts=None, sampling=None, a=5.65, **kwargs):
        """Build a potential.

        `sampling` is honoured. An earlier version of this helper accepted it
        and dropped it on the floor, so every case that meant to vary sampling
        independently of the grid silently varied neither -- and the lattice
        constant knob is what makes "independently" possible at all, since with
        a fixed cell sampling is just extent/gpts.
        """
        grid = {}
        if gpts is not None:
            grid["gpts"] = gpts
        if sampling is not None:
            grid["sampling"] = sampling
        if not grid:
            grid["gpts"] = (64, 64)
        return abtem.Potential(
            self._atoms(a=a), slice_thickness=1.0,
            integrator=integrator, **grid, **kwargs,
        ).build(lazy=False)

    def test_integral_tables_are_keyed_on_sampling(self):
        """Keyed on the element alone, a sampling sweep was ~54 % wrong."""
        shared = QuadratureProjectionIntegrals()
        self._build(shared, gpts=(32, 32))
        got = self._build(shared, gpts=(128, 128))
        reference = self._build(QuadratureProjectionIntegrals(), gpts=(128, 128))
        assert np.array_equal(got.array, reference.array)
        assert len(shared._tables) == 4  # two elements x two samplings

    # ------------------------------------------------------------------
    # Key-completeness, probed one component at a time at the *method* level.
    #
    # The potential-level tests above cannot do this job. A shared-vs-fresh
    # comparison of two full builds is blind to a key that drops the chemical
    # symbol, because both sides serve the first element's array to the second
    # and therefore agree with each other. Calling the cached method directly,
    # with exactly one component changed from the warmed call, is what makes
    # each component observable.

    @staticmethod
    def _scattering_factor(
        integrator, symbol="Ga", gpts=(64, 64), sampling=(0.10, 0.10),
        precision="float32",
    ):
        with abtem.config.set({"precision": precision}):
            return np.asarray(
                integrator.get_scattering_factor(symbol, gpts, sampling, "cpu")
            )

    @pytest.mark.parametrize(
        "component, other",
        [
            ("symbol", "As"),
            ("gpts", (96, 96)),
            ("sampling", (0.13, 0.13)),
            ("precision", "float64"),
        ],
    )
    def test_no_scattering_factor_key_component_may_be_dropped(
        self, component, other
    ):
        shared = ScatteringFactorProjectionIntegrals()
        self._scattering_factor(shared)
        got = self._scattering_factor(shared, **{component: other})
        reference = self._scattering_factor(
            ScatteringFactorProjectionIntegrals(), **{component: other}
        )
        assert got.shape == reference.shape
        assert got.dtype == reference.dtype
        assert np.array_equal(got, reference)

    @staticmethod
    def _table(integrator, symbol="Ga", sampling=(0.10, 0.10), precision="float32"):
        with abtem.config.set({"precision": precision}):
            table = integrator.get_integral_table(symbol, sampling)
        return np.asarray(table.values), np.asarray(table.radial_gpts)

    @pytest.mark.parametrize(
        "component, other",
        [("symbol", "As"), ("sampling", (0.13, 0.13)), ("precision", "float64")],
    )
    def test_no_integral_table_key_component_may_be_dropped(self, component, other):
        shared = QuadratureProjectionIntegrals()
        self._table(shared)
        got = self._table(shared, **{component: other})
        reference = self._table(
            QuadratureProjectionIntegrals(), **{component: other}
        )
        assert all(np.array_equal(a, b) for a, b in zip(got, reference))

    @staticmethod
    def _gaussian_on_grid(
        integrator, gpts=(64, 64), sampling=(0.10, 0.10), precision="float32"
    ):
        import ase.build

        atoms = ase.build.bulk("Si", cubic=True)
        with abtem.config.set({"precision": precision, "fft": "numpy"}):
            return np.asarray(
                integrator.integrate_on_grid(
                    atoms, a=0.0, b=1.0, gpts=gpts, sampling=sampling, device="cpu"
                )
            )

    @pytest.mark.parametrize(
        "component, other",
        [
            ("gpts", (96, 96)),
            ("sampling", (0.13, 0.13)),
            ("precision", "float64"),
        ],
    )
    def test_no_sinc_key_component_may_be_dropped(self, component, other):
        """The sinc is the only cached value on this path.

        GaussianProjectionIntegrals' other two caches are deleted in this
        commit, so its gaussians and corrections are recomputed on every call;
        anything a shared integrator gets wrong here is the sinc.
        """
        shared = GaussianProjectionIntegrals()
        self._gaussian_on_grid(shared)
        got = self._gaussian_on_grid(shared, **{component: other})
        reference = self._gaussian_on_grid(
            GaussianProjectionIntegrals(), **{component: other}
        )
        assert got.shape == reference.shape
        assert np.array_equal(got, reference)

    def test_the_sorted_disk_is_not_served_across_precisions(self):
        """The disk is sized int(ceil(cutoff / min(sampling))), and `cutoff` is
        precision-dependent, so the key needs precision even though the disk
        itself is integer offsets.

        The sampling below is not arbitrary: cases where the two precisions
        want different radii have to be solved for (cutoff32/m <= s <
        cutoff64/m), not scanned for. Si wants radius 18 at float32 and 19 at
        float64 here, and serving the float32 disk to the float64 build is
        wrong by 4.6e-05 on a peak of 312.
        """
        import ase

        sampling = 0.281458955844481
        atoms = ase.Atoms(
            "Si", positions=[(2.0, 2.0, 0.5)], cell=(sampling * 80, sampling * 80, 1.0)
        )

        def build(integrator, precision):
            with abtem.config.set({"precision": precision, "fft": "numpy"}):
                return np.asarray(
                    abtem.Potential(
                        atoms, sampling=(sampling, sampling), slice_thickness=1.0,
                        integrator=integrator,
                    ).build(lazy=False).array
                )

        shared = QuadratureProjectionIntegrals()
        build(shared, "float32")
        got = build(shared, "float64")
        reference = build(QuadratureProjectionIntegrals(), "float64")
        assert np.array_equal(got, reference)

    def test_the_sorted_disk_is_not_served_across_elements(self):
        """The disk is sized from the element's own cutoff.

        A potential-level shared-vs-fresh comparison cannot see this: both
        sides would serve the first element's disk to the second and agree.
        What makes it observable is a cell whose *lower* atomic number has the
        *smaller* cutoff -- H (3.3499 A, radius 34 at this sampling) is
        processed before Au (5.0342 A, radius 51), so a symbol-blind key hands
        Au a disk too small by 17 pixels and truncates it.
        """
        import ase

        atoms = ase.Atoms(
            "HAu", positions=[(3.0, 3.0, 0.5), (9.0, 9.0, 0.5)], cell=(12.8, 12.8, 1.0)
        )
        with abtem.config.set({"fft": "numpy"}):
            got = np.asarray(
                abtem.Potential(
                    atoms, sampling=(0.1, 0.1), slice_thickness=1.0,
                    integrator=QuadratureProjectionIntegrals(),
                ).build(lazy=False).array
            )
            # The oracle is the same cell built one element at a time, where
            # no sharing can occur: their sum is what the mixed build must be.
            separate = sum(
                np.asarray(
                    abtem.Potential(
                        ase.Atoms(
                            symbol, positions=[position], cell=(12.8, 12.8, 1.0)
                        ),
                        sampling=(0.1, 0.1), slice_thickness=1.0,
                        integrator=QuadratureProjectionIntegrals(),
                    ).build(lazy=False).array
                )
                for symbol, position in (("H", (3.0, 3.0, 0.5)), ("Au", (9.0, 9.0, 0.5)))
            )
        assert np.abs(got - separate).max() < 1e-6 * np.abs(separate).max()

    def test_sorted_disks_are_bounded(self):
        integrator = QuadratureProjectionIntegrals()
        for n in range(_MAX_CACHE_ENTRIES + 8):
            self._build(integrator, gpts=(32 + 2 * n,) * 2)
        assert len(integrator._sorted_disks) <= _MAX_CACHE_ENTRIES
        assert len(integrator._tables) <= _MAX_CACHE_ENTRIES

    @pytest.mark.parametrize(
        "integrator_class",
        [ScatteringFactorProjectionIntegrals, QuadratureProjectionIntegrals],
    )
    def test_a_cache_is_not_served_across_precisions(self, integrator_class):
        """Every cached value is built through get_dtype.

        The difference is ~1e-6 relative, which np.allclose with default
        tolerances reports as equal -- so this must compare exactly.
        """
        shared = integrator_class()
        with abtem.config.set({"precision": "float32"}):
            self._build(shared)
        with abtem.config.set({"precision": "float64"}):
            got = self._build(shared)
            reference = self._build(integrator_class())
        assert got.array.dtype == reference.array.dtype
        assert np.array_equal(got.array, reference.array)

    def test_results_do_not_depend_on_cache_state(self):
        """Vary the grid and the sampling independently, not together."""
        shared = ScatteringFactorProjectionIntegrals()
        cases = [
            dict(gpts=(64, 64)),
            dict(gpts=(96, 96)),
            dict(gpts=(64, 64)),
            # Same gpts as case 1, different sampling (via the lattice
            # constant); then same sampling as case 1, different gpts.
            dict(gpts=(64, 64), a=7.20),
            dict(sampling=(5.65 / 64, 5.65 / 64), a=8.475),
        ]
        for case in cases:
            got = self._build(shared, **case)
            reference = self._build(
                ScatteringFactorProjectionIntegrals(), **case
            )
            assert np.array_equal(got.array, reference.array)

    def test_caches_restored_from_an_older_pickle_still_work(self):
        """_sinc_cache post-dates PR #269, and a restored plain dict has no put."""
        import pickle

        missing = GaussianProjectionIntegrals()
        missing.__dict__.pop("_sinc_cache", None)
        restored = pickle.loads(pickle.dumps(missing))
        assert isinstance(restored._sinc_cache, _DeviceArrayCache)

        # A state dict as an *older* abTEM would have written it: the cache is
        # a plain dict, with .get but no .put, so the miss path -- not the hit
        # path -- raised AttributeError. It has to be built by hand rather than
        # round-tripped, because __getstate__ now drops caches, so no pickle
        # this version writes carries one.
        legacy_state = QuadratureProjectionIntegrals().__dict__.copy()
        legacy_state["_tables"] = {("Ga", (0.1, 0.1), "float32"): "entry"}
        restored = QuadratureProjectionIntegrals.__new__(QuadratureProjectionIntegrals)
        restored.__setstate__(legacy_state)
        assert isinstance(restored._tables, _DeviceArrayCache)
        # Empty, not carried over: those entries are keyed in the old shape,
        # and serving a value found under an incomplete key is the defect this
        # commit fixes. Recomputing them costs a miss.
        assert len(restored._tables) == 0
        # The miss path is the one that used to raise AttributeError.
        self._build(restored)

    def test_the_dead_gaussian_caches_are_gone(self):
        integrator = GaussianProjectionIntegrals()
        assert not hasattr(integrator, "_gaussians")
        assert not hasattr(integrator, "_corrections")

    def test_the_integrator_stays_picklable_comparable_and_copyable(self):
        """The three things a per-instance lru_cache broke in #388."""
        import copy
        import pickle

        used, fresh = QuadratureProjectionIntegrals(), QuadratureProjectionIntegrals()
        self._build(used)
        assert isinstance(
            pickle.loads(pickle.dumps(used)), QuadratureProjectionIntegrals
        )
        assert used == fresh
        assert copy.deepcopy(used)._tables is not used._tables

    def test_concurrent_access_serves_correct_values(self):
        """Not just "nothing raised": check what the cache hands back."""
        import sys
        import threading

        integrator = ScatteringFactorProjectionIntegrals()
        combinations = [("Ga", (16 + n,) * 2, (0.1, 0.1)) for n in range(150)]
        truth = {
            c: ScatteringFactorProjectionIntegrals().get_scattering_factor(*c, "cpu")
            for c in combinations[:8]
        }
        errors = []

        def hammer(seed):
            try:
                for n in range(1500):
                    combination = combinations[(seed * 7919 + n * 13) % len(combinations)]
                    got = integrator.get_scattering_factor(*combination, "cpu")
                    expected = truth.get(combination)
                    if expected is not None and not np.array_equal(got, expected):
                        errors.append(f"wrong value for {combination}")
            except Exception as exc:  # noqa: BLE001 -- report, don't mask
                errors.append(exc)

        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-9)
        try:
            threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
        assert len(integrator._scattering_factors) <= _MAX_CACHE_ENTRIES
