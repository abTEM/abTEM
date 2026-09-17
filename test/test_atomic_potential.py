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

    def test_the_public_caches_still_behave_like_the_dicts_they_replaced(self):
        """`tables` and `scattering_factors` are public and were plain dicts.

        Keeping them dict-like is the entire reason the container is a Mapping,
        so the two-argument `get(key, default)` has to work: a one-argument
        override shadows Mapping.get and turns ordinary dict usage into a
        TypeError.
        """
        integrator = ScatteringFactorProjectionIntegrals()
        self._build(integrator)
        cache = integrator.scattering_factors
        key = next(iter(cache))

        assert cache.get(key) is not None
        assert cache.get(("absent",)) is None
        assert cache.get(("absent",), "fallback") == "fallback"
        assert key in cache
        assert len(list(cache.keys())) == len(cache)
        assert len(dict(cache.items())) == len(cache)
        assert cache[key] is cache.get(key)

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

    def test_the_gaussian_caches_are_keyed_and_transient(self):
        """The parent commit deleted these as unusable -- never storing, no
        precision in the key, and shipped into every task graph. They are back
        with the key they needed, and excluded from pickling."""
        import pickle

        integrator = GaussianProjectionIntegrals()
        integrator.get_gaussians("Si", (64, 64), (0.1, 0.1))
        assert len(integrator._gaussians) == 1
        assert len(pickle.dumps(integrator)) == len(
            pickle.dumps(GaussianProjectionIntegrals())
        )

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
        # deepcopy goes through __getstate__, which drops the caches, so the
        # copy starts empty -- and must be its own object, not a shared one.
        clone = copy.deepcopy(used)
        assert clone._tables is not used._tables
        assert len(clone._tables) == 0

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


class TestGaussianProjectionIntegralsUsable:
    """The class was reachable but unfinished.

    It is the only integrator that is both finite (z-resolved within a slice)
    and periodic -- ``ScatteringFactorProjectionIntegrals`` is periodic but
    z-unresolved, ``QuadratureProjectionIntegrals`` is z-resolved but needs
    padding. Two of its constructor parameters -- `parametrization` and
    `gaussian_parametrization` -- were validated, stored and never used
    (`cutoff_tolerance` was live), `integrate_on_grid` accepted a
    `fourier_space` flag it never read, and it could not run on GPU at all.
    """

    @staticmethod
    def _atoms():
        import ase.build

        return ase.build.bulk("Si", cubic=True)

    def _build(self, integrator, device="cpu", gpts=(128, 128)):
        return np.asarray(
            abtem.core.backend.asnumpy(
                abtem.Potential(
                    self._atoms(), gpts=gpts, slice_thickness=1.0,
                    integrator=integrator, device=device,
                ).build(lazy=False).array
            )
        )

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_builds_on_both_devices_and_they_agree(self, device):
        """It failed on GPU with TypeError: Unsupported type numpy.ndarray."""
        if device == "cpu":
            pytest.skip("the cpu case would compare a build with itself")
        got = self._build(GaussianProjectionIntegrals(), device=device)
        reference = self._build(GaussianProjectionIntegrals(), device="cpu")
        assert got.shape == reference.shape
        assert got.dtype == reference.dtype
        # 1.1e-07 on this cell and grid. The bound is 2e-6 rather than
        # something tighter because the deviation is a property of the cell,
        # not of the code: across 100 combinations of element, grid and slice
        # thickness the worst float32 case reached 5.7e-07, every one of them
        # on a non-power-of-two grid, so 5e-7 sat 14 % from failing.
        #
        # No tolerance here can guard the device transfer cast, contrary to
        # what an earlier version of this comment claimed: dropping the cast
        # changes GPU bytes but leaves this difference identical to seven
        # significant figures, because it moves CPU and GPU apart in the same
        # direction. The cast's effect is bytes moved -- see the commit message.
        scale = np.abs(reference).max()
        assert np.abs(got - reference).max() < 2e-6 * scale

    def test_the_correction_parametrization_is_used(self):
        """It was stored and ignored; the module default was used instead."""
        lobato = self._build(GaussianProjectionIntegrals(parametrization="lobato"))
        kirkland = self._build(GaussianProjectionIntegrals(parametrization="kirkland"))
        assert not np.array_equal(lobato, kirkland)

    def test_the_gaussian_parametrization_is_used(self):
        """Two Gaussian-form parametrizations must give different potentials.

        The first version of this test fell back to varying the *correction*
        parametrization, making it a duplicate of the test above and leaving
        half the wiring untested. peng_low/peng_high are real alternatives,
        13.8 % apart.
        """
        low = self._build(
            GaussianProjectionIntegrals(gaussian_parametrization=self._peng("low"))
        )
        high = self._build(
            GaussianProjectionIntegrals(gaussian_parametrization=self._peng("high"))
        )
        assert not np.array_equal(low, high)

    @staticmethod
    def _peng(variant):
        from abtem.parametrizations import PengParametrization

        return PengParametrization(parameters=f"peng_{variant}.json")

    def test_get_gaussians_uses_the_configured_parametrization(self):
        """Isolates one of the two wirings.

        The end-to-end test above cannot: gaussian_projection_weights is also
        wired, so its difference alone makes the potentials differ even when
        get_gaussians still ignores self's parametrization.
        """
        low = GaussianProjectionIntegrals(gaussian_parametrization=self._peng("low"))
        high = GaussianProjectionIntegrals(gaussian_parametrization=self._peng("high"))
        assert not np.array_equal(
            low.get_gaussians("Si", (64, 64), (0.1, 0.1)),
            high.get_gaussians("Si", (64, 64), (0.1, 0.1)),
        )

    def test_projection_weights_use_the_configured_parametrization(self):
        """Isolates the _integrate_gaussians wiring.

        Calling the module-level gaussian_projection_weights with an explicit
        parametrization tests nothing -- that keyword already existed. Drive it
        through the integrator with get_gaussians pinned, so only the weights
        wiring can produce a difference.
        """
        from unittest.mock import patch

        low = GaussianProjectionIntegrals(gaussian_parametrization=self._peng("low"))
        high = GaussianProjectionIntegrals(gaussian_parametrization=self._peng("high"))
        # Three things vary with gaussian_parametrization: the gaussians, the
        # weights, and get_corrections' long_range. Pin the other two, or this
        # passes for the wrong reason.
        pinned_g = low.get_gaussians("Si", (64, 64), (0.1, 0.1))
        pinned_c = low.get_corrections("Si", (64, 64), (0.1, 0.1))

        def build(integrator):
            with patch.object(
                type(integrator), "get_gaussians", lambda *a, **k: pinned_g
            ), patch.object(
                type(integrator), "get_corrections", lambda *a, **k: pinned_c
            ):
                return self._build(integrator, gpts=(64, 64))

        assert not np.array_equal(build(low), build(high))

    def test_the_correction_long_range_follows_the_gaussian_parametrization(self):
        """The correction must subtract exactly the Gaussian field that was
        added, so get_corrections' long_range has to be the *gaussian*
        parametrization. Reverting that alone changes the potential by 27.6 %
        and nothing else in this class detects it."""
        from abtem.integrals import correction_projected_scattering_factors

        integrator = GaussianProjectionIntegrals(
            gaussian_parametrization=self._peng("low")
        )
        got = integrator.get_corrections("Si", (64, 64), (0.1, 0.1))
        wired = correction_projected_scattering_factors(
            "Si", (64, 64), (0.1, 0.1),
            short_range=integrator.correction_parametrization,
            long_range=integrator.gaussian_parametrization,
        )
        default_long_range = correction_projected_scattering_factors(
            "Si", (64, 64), (0.1, 0.1),
            short_range=integrator.correction_parametrization,
        )
        assert np.array_equal(got, wired)
        assert not np.array_equal(got, default_long_range)

    def test_an_element_the_parametrization_lacks_does_not_reject_it(self):
        """The first form check validated the fixed element "C", so any
        parametrization without carbon was rejected outright -- the shipped
        peng_ionic.json among them.

        This guards the *validator*, not a production path: no build reaches an
        ionic symbol, because integrate_on_grid derives symbols from
        chemical_symbols[number], and peng_ionic in a real build dies earlier
        in cutoff(). The per-element rework is justified by
        test_a_bad_entry_is_caught_at_the_element_that_uses_it, which is a live
        path; this one pins that validation follows the element asked for.
        """
        from abtem.parametrizations import PengParametrization

        integrator = GaussianProjectionIntegrals(
            gaussian_parametrization=PengParametrization(
                parameters="peng_ionic.json"
            )
        )
        assert integrator.get_gaussians("O--", (32, 32), (0.1, 0.1)).shape[0] == 5

    @pytest.mark.parametrize(
        "centre, width, label",
        [
            (20.0, 0.5, "outside the old [0.01, 4.0] window"),
            (0.0744, 1e-4, "between two of the old 32 sample points"),
        ],
    )
    def test_a_non_gaussian_feature_is_caught_wherever_it_sits(
        self, centre, width, label
    ):
        """The form check has to look at the grid the build uses.

        Sampling a fixed linspace(0.01, 4.0, 32) covered 2.3 % of the k^2 a
        128^2 build evaluates and 0.1 % of a 512^2 one, with the 32 points
        0.13 apart. Both gaps were reachable, and a parametrization that
        cleared the check through either built a potential several per cent to
        tens of per cent wrong -- from the very check meant to prevent exactly
        that.
        """
        import ase.build

        from abtem.parametrizations import PengParametrization

        class Bumped(PengParametrization):
            """Peng's Gaussian sum times a narrow bump: not a Gaussian sum."""

            def projected_scattering_factor(self, symbol, *args, **kwargs):
                base = super().projected_scattering_factor(symbol, *args, **kwargs)

                def factor(k2):
                    k2 = np.asarray(k2)
                    return base(k2) * (
                        1.0 + 3.0 * np.exp(-((k2 - centre) ** 2) / width)
                    )

                return factor

        integrator = GaussianProjectionIntegrals(gaussian_parametrization=Bumped())
        with pytest.raises(ValueError, match="not a superposition of Gaussians"):
            abtem.Potential(
                ase.build.bulk("Si", cubic=True), gpts=(128, 128),
                slice_thickness=1.0, integrator=integrator,
            ).build(lazy=False)

    def test_get_corrections_validates_the_parametrization_too(self):
        """get_corrections uses gaussian_parametrization as its long_range
        term, so a non-Gaussian one gives a plausible array rather than an
        error. It is only safe by call order today -- integrate_on_grid happens
        to call get_gaussians first -- and get_corrections is public."""
        integrator = GaussianProjectionIntegrals(gaussian_parametrization="lobato")
        with pytest.raises(ValueError, match="not a superposition of Gaussians"):
            integrator.get_corrections("Si", (64, 64), (0.1, 0.1))

    def test_the_form_check_verdict_does_not_depend_on_precision(self):
        """Accept or reject must not be a function of abtem.config.

        scaled_parameters is float64 while Parametrization._get_function casts
        to get_dtype, so comparing one against the other made the verdict
        precision-dependent: a user script that worked at float64 raised at
        float32. The fixture is a pair of near-cancelling terms, which is where
        the two dtypes disagree most.
        """
        import copy

        from abtem.parametrizations import PengParametrization
        from abtem.integrals import _validate_gaussian_form

        # Five terms, so the table and Peng's own five-term function agree, and
        # the last two cancel almost exactly: a genuine Gaussian superposition
        # that float32 cannot sum accurately. The amplitude is chosen so the
        # float32 evaluation error lands above the tolerance while the exact
        # deviation is zero -- 1e4 is already enough, 1e5 leaves margin.
        parameters = copy.deepcopy(PengParametrization().parameters)
        a, b = list(parameters["Si"][0]), list(parameters["Si"][1])
        parameters["Si"] = [a[:3] + [1e5, -1e5], b[:3] + [0.30, 0.30000001]]
        cancelling = PengParametrization(parameters=parameters)

        verdicts = {}
        for precision in ("float32", "float64"):
            with abtem.config.set({"precision": precision}):
                try:
                    _validate_gaussian_form(cancelling, "Si", (64, 64), (0.1, 0.1))
                    verdicts[precision] = "accepted"
                except ValueError:
                    verdicts[precision] = "rejected"
        # Same verdict at both, and the right one: it *is* a Gaussian sum.
        # Comparing float64 parameters against a float32 own-function rejected
        # it at float32 and accepted it at float64.
        assert verdicts == {"float32": "accepted", "float64": "accepted"}, verdicts

    def test_a_bad_entry_is_caught_at_the_element_that_uses_it(self):
        """Validating one fixed element let a parametrization whose carbon
        entry is sound and whose silicon entry is not through silently -- the
        extra terms were added to the field while get_corrections and cutoff
        still saw five, a ~10 % error, finite everywhere."""
        import copy

        from abtem.parametrizations import PengParametrization

        parameters = copy.deepcopy(PengParametrization().parameters)
        parameters["Si"] = [list(row) + [1.0] for row in parameters["Si"]]
        integrator = GaussianProjectionIntegrals(
            gaussian_parametrization=PengParametrization(parameters=parameters)
        )
        # The untouched element is unaffected...
        assert integrator.get_gaussians("Ga", (32, 32), (0.1, 0.1)).shape[0] == 5
        # ...and the tampered one is refused where it is used.
        with pytest.raises(ValueError, match="superposition of Gaussians for 'Si'"):
            integrator.get_gaussians("Si", (32, 32), (0.1, 0.1))

    @pytest.mark.parametrize("name", ["lobato", "kirkland"])
    def test_a_non_gaussian_parametrization_is_refused(self, name):
        """It was accepted and gave a potential 4.8x too large, finite
        everywhere -- a dead parameter turned into a silent physics error."""
        integrator = GaussianProjectionIntegrals(gaussian_parametrization=name)
        with pytest.raises(ValueError, match="superposition of Gaussians"):
            integrator.get_gaussians("Si", (64, 64), (0.1, 0.1))

    def test_the_gaussian_count_follows_the_parametrization(self):
        """The loop bound was hardcoded to 5 while the source became
        configurable, so a sixth Gaussian was silently dropped.

        Behavioural rather than an `inspect.getsource` check, which would fail
        a correct `zip(gaussians, weights)` refactor and pass a broken loop
        body. The fixture is Peng's silicon entry with its last Gaussian split
        into three identical thirds: the same function, seven terms. With the
        bound following the parametrization the split is invisible; hardcoded
        to five it drops two thirds of the last Gaussian.
        """
        import copy

        from abtem.parametrizations import PengParametrization

        class NTermPeng(PengParametrization):
            """Peng, but its own scattering factor honours every term.

            Needed because the shipped PengParametrization hardcodes five in
            `scattering_factor_k2`, so a seven-term table is -- correctly --
            rejected by the form check as self-inconsistent. That is exactly
            why the loop bound cannot be reached with any shipped
            parametrization: real n-term support has to fix Peng first.
            """

            def projected_scattering_factor(self, symbol, *args, **kwargs):
                parameters = np.asarray(
                    self.scaled_parameters(symbol, "projected_scattering_factor")
                )

                def factor(k2):
                    k2 = np.asarray(k2)
                    return (
                        parameters[0][:, None]
                        * np.exp(-parameters[1][:, None] * k2.ravel()[None])
                    ).sum(0).reshape(k2.shape)

                return factor

        parameters = copy.deepcopy(PengParametrization().parameters)
        a, b = parameters["Si"][0], parameters["Si"][1]
        parameters["Si"] = [
            list(a[:-1]) + [a[-1] / 3.0] * 3,
            list(b[:-1]) + [b[-1]] * 3,
        ]
        split = GaussianProjectionIntegrals(
            gaussian_parametrization=NTermPeng(parameters=parameters)
        )
        assert split.get_gaussians("Si", (64, 64), (0.1, 0.1)).shape[0] == 7

        # The five-term reference goes through the same class, so the only
        # difference between the two builds is the term count.
        reference_integrator = GaussianProjectionIntegrals(
            gaussian_parametrization=NTermPeng()
        )
        with abtem.config.set({"fft": "numpy"}):
            got = self._build(split)
            reference = self._build(reference_integrator)
        assert np.abs(got - reference).max() < 1e-5 * np.abs(reference).max()

    def test_the_defaults_are_unchanged_by_the_wiring(self):
        """The defaults equal the module defaults the code used before.

        Pinned to the numpy FFT: fftw's FFTW_MEASURE picks plans by wall-clock
        benchmark, so abTEM is run-to-run nondeterministic on some grids and a
        bit-identity assertion would flake.
        """
        with abtem.config.set({"fft": "numpy"}):
            default = self._build(GaussianProjectionIntegrals())
            explicit = self._build(
                GaussianProjectionIntegrals(
                    parametrization="lobato", gaussian_parametrization="peng"
                )
            )
        assert np.array_equal(default, explicit)

    def test_integrate_on_grid_accepts_no_parameter_it_ignores(self):
        """`fourier_space` was accepted and never read.

        It is a leftover of the commented-out test_finite_gaussian_projection_
        integrals in this file. No caller passes it -- iam.py:968 is the only
        one, and the abstract FieldIntegrator.integrate_on_grid does not
        declare it -- so asking for a reciprocal-space result silently returned
        a real-space one. Removed; the request is now a loud TypeError.

        The oracle is the two sibling integrators, not the abstract base: all
        three concrete ones renamed the first argument to `atoms` and the base
        still calls it `positions`.
        """
        import inspect

        def params(cls):
            return list(inspect.signature(cls.integrate_on_grid).parameters)

        assert params(GaussianProjectionIntegrals) == params(
            ScatteringFactorProjectionIntegrals
        ) == params(QuadratureProjectionIntegrals)

        integrator = GaussianProjectionIntegrals()

        with pytest.raises(TypeError):
            integrator.integrate_on_grid(
                self._atoms(),
                a=0.0,
                b=1.0,
                gpts=(32, 32),
                sampling=(0.1, 0.1),
                fourier_space=True,
            )

    def test_the_host_path_does_not_cast_the_parametrization_arrays(
        self, monkeypatch
    ):
        """`if xp is not np` around the device cast is load-bearing.

        The parametrization arrays are float64 whatever the run's precision.
        Casting them to float32 is right on the way to a device -- it halves
        the transfer -- and wrong on the host, where it changes the CPU result
        and so breaks the bit-identity with dev that this commit rests on.

        Both arrays have to be cast to see it, and the cell matters: casting
        the gaussians alone is invisible on every cell tried, and casting both
        is invisible on GaAs at any grid and on Si and Au except at 97x131.
        Diamond shows it at every grid tried, which is why it is the fixture --
        picked by searching for a cell that discriminates rather than by
        assuming one does.
        """
        import ase.build

        import abtem.integrals

        atoms = ase.build.bulk("C", cubic=True)

        def build(cast):
            integrator = GaussianProjectionIntegrals()
            if cast:
                original_gaussians = integrator.get_gaussians
                original_weights = abtem.integrals.gaussian_projection_weights
                integrator.get_gaussians = lambda *a, **k: np.asarray(
                    original_gaussians(*a, **k), dtype=np.float32
                )
                monkeypatch.setattr(
                    abtem.integrals,
                    "gaussian_projection_weights",
                    lambda *a, **k: np.asarray(
                        original_weights(*a, **k), dtype=np.float32
                    ),
                )
            with abtem.config.set({"precision": "float32", "fft": "numpy"}):
                return np.asarray(
                    abtem.Potential(
                        atoms, gpts=(64, 64), slice_thickness=1.0,
                        integrator=integrator,
                    ).build(lazy=False).array
                )

        plain = build(cast=False)
        cast = build(cast=True)
        assert not np.array_equal(plain, cast), (
            "casting the parametrization arrays to float32 changed nothing on "
            "this cell, so this test cannot tell whether the host path casts"
        )

    @pytest.mark.parametrize("method", ["get_gaussians", "get_corrections"])
    @pytest.mark.parametrize(
        "component, other",
        [("symbol", "As"), ("gpts", (96, 96)), ("sampling", (0.13, 0.13))],
    )
    def test_no_parametrization_cache_key_component_may_be_dropped(
        self, method, component, other
    ):
        """Both caches, every component -- not just the gaussians' precision.

        The corrections cache had no key coverage at all, and neither cache's
        sampling component was exercised by any test.
        """
        base = dict(symbol="Ga", gpts=(64, 64), sampling=(0.10, 0.10))
        probe = {**base, component: other}

        shared = GaussianProjectionIntegrals()
        getattr(shared, method)(**base)
        got = np.asarray(getattr(shared, method)(**probe))
        reference = np.asarray(
            getattr(GaussianProjectionIntegrals(), method)(**probe)
        )
        assert got.shape == reference.shape
        assert np.array_equal(got, reference)

    @pytest.mark.parametrize("method", ["get_gaussians", "get_corrections"])
    def test_neither_parametrization_cache_is_served_across_precisions(self, method):
        args = dict(symbol="Ga", gpts=(64, 64), sampling=(0.10, 0.10))
        shared = GaussianProjectionIntegrals()
        with abtem.config.set({"precision": "float32"}):
            getattr(shared, method)(**args)
        with abtem.config.set({"precision": "float64"}):
            served = np.asarray(getattr(shared, method)(**args))
            correct = np.asarray(
                getattr(GaussianProjectionIntegrals(), method)(**args)
            )
        assert np.array_equal(served, correct)

    def test_the_parametrization_arrays_are_cached(self):
        """integrate_on_grid runs once per slice per species and recomputed
        both arrays each time -- 46x redundant on a 46-slice cell, and most of
        the build time."""
        import ase.build

        atoms = ase.build.bulk(
            "GaAs", crystalstructure="zincblende", a=5.65, cubic=True
        ) * (1, 1, 4)
        integrator = GaussianProjectionIntegrals()
        abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=1.0, integrator=integrator
        ).build(lazy=False)
        # One entry per (element, grid, sampling, precision) -- two elements.
        assert len(integrator._gaussians) == 2
        assert len(integrator._corrections) == 2

    def test_caches_are_not_shipped_into_the_task_graph(self):
        """abTEM pickles integrators into every task, so a populated cache
        would ride along to every worker -- 12.6x graph inflation when these
        caches were first made to store."""
        import pickle

        used = GaussianProjectionIntegrals()
        self._build(used)
        assert len(used._gaussians) > 0

        payload = pickle.dumps(used)
        assert len(payload) == len(pickle.dumps(GaussianProjectionIntegrals()))

        restored = pickle.loads(payload)
        assert len(restored._gaussians) == 0
        # and it must still work after the round trip
        self._build(restored)

    def test_a_cache_is_not_served_across_precisions(self):
        """Every cached value reaches get_dtype through spatial_frequencies."""
        integrator = GaussianProjectionIntegrals()
        with abtem.config.set({"precision": "float32"}):
            integrator.get_gaussians("Ga", (64, 64), (0.1, 0.1))
        with abtem.config.set({"precision": "float64"}):
            served = integrator.get_gaussians("Ga", (64, 64), (0.1, 0.1))
            correct = GaussianProjectionIntegrals().get_gaussians(
                "Ga", (64, 64), (0.1, 0.1)
            )
        # Not a dtype assertion: both are float64 at either precision, so
        # comparing dtypes passes whether or not the key carries precision.
        # The values are what differ.
        assert np.array_equal(served, correct)
