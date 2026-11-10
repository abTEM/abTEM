import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from dask.base import is_dask_collection
from hypothesis import given
from utils import assert_array_objects_equal, si_cubic_atoms

from abtem.array import ArrayObject
from abtem.core.backend import asnumpy


@given(data=st.data())
@pytest.mark.parametrize(
    "copyable",
    [
        abtem_st.atoms,
        abtem_st.frozen_phonons,
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.potential,
        abtem_st.potential_array,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        # abtem_st.composite_wave_transform,
        abtem_st.ctf,
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.waves,
        abtem_st.s_matrix,
        # # # prism_st.random_s_matrix,
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
    ],
)
def test_copy_equals(data, copyable):
    original = data.draw(copyable())
    copied = original.copy()

    # `copy` returning `self` would satisfy every equality check below.
    assert copied is not original

    for path, array in _ndarray_attributes(original):
        copied_array = _get_path(copied, path)
        assert not np.may_share_memory(array, copied_array), (
            f"copy shares the buffer of {'.'.join(map(str, path))}"
        )

    if isinstance(original, ArrayObject):
        # `==` has no verdict for lazy objects.
        assert_array_objects_equal(copied, original)
        _assert_copy_does_not_share_the_array(original)
    else:
        assert copied == original


def _assert_copy_does_not_share_the_array(original):
    """Writing into a copy's array must leave the original's untouched."""
    # to_cpu() first: it returns a new object, whereas compute() works in
    # place and would leave `original` eager and its lazy copy behind.
    eager = original.to_cpu()
    eager.compute()
    copied = eager.copy()
    assert copied.array is not eager.array

    before = asnumpy(eager.array).copy()
    copied.array[...] = copied.array * 2 + 1
    assert not np.array_equal(asnumpy(copied.array), before)  # the write landed
    np.testing.assert_array_equal(asnumpy(eager.array), before)


def _ndarray_attributes(obj, path=(), seen=None, max_depth=4):
    """(path, array) for every numpy array reachable from ``obj`` through
    instance attributes, dicts, lists and tuples."""
    if seen is None:
        seen = set()
    if id(obj) in seen or len(path) > max_depth:
        return
    seen.add(id(obj))

    if isinstance(obj, np.ndarray):
        if obj.size:
            yield path, obj
        return
    if is_dask_collection(obj):
        # Unmaterialised; the eager copy is checked by writing into it.
        return
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        items = vars(obj).items()
    else:
        return
    for key, value in items:
        yield from _ndarray_attributes(value, path + (key,), seen, max_depth)


def _get_path(obj, path):
    for key in path:
        if isinstance(obj, (dict, list, tuple)):
            obj = obj[key]
        else:
            obj = vars(obj)[key]
    return obj


class TestEqualityDiscriminates:
    """`test_copy_equals` above compares an object with itself.

    A self-comparison cannot fail for an over-permissive ``__eq__`` -- and
    ``safe_equality`` was over-permissive in two independent ways, so nothing
    in the suite noticed. These tests compare objects that genuinely DIFFER.
    """

    @staticmethod
    def _atoms(dx=0.0):
        atoms = si_cubic_atoms()
        if dx:
            atoms.positions[0] += [dx, 0.0, 0.0]
        return atoms

    def _potential(self, dx=0.0, **kwargs):
        import abtem

        kwargs.setdefault("gpts", (64, 64))
        kwargs.setdefault("slice_thickness", 1.0)
        return abtem.Potential(self._atoms(dx), **kwargs)

    def test_potentials_with_different_atomic_positions_are_not_equal(self):
        """`safe_equality` tested `equal is False`, which only catches the
        `False` singleton. `ase.Atoms.__eq__` returns `numpy.bool_`, and
        `np.False_ is False` is False -- so the difference was dropped and two
        physically different potentials compared equal."""
        assert self._potential() != self._potential(dx=1.234)

    def test_potentials_with_identical_atoms_are_still_equal(self):
        """The fix must not make everything unequal."""
        assert self._potential() == self._potential()

    @pytest.mark.parametrize(
        "kwargs", [{"gpts": (32, 32)}, {"slice_thickness": 2.0}], ids=["gpts", "dz"]
    )
    def test_other_differences_are_still_caught(self, kwargs):
        assert self._potential() != self._potential(**kwargs)

    def test_a_built_potential_equals_an_identical_unbuilt_one(self):
        """`_sliced_atoms` is populated lazily by get_sliced_atoms(), so a
        built potential stopped comparing equal to an identical unbuilt one --
        equality depended on whether a result had been computed from it."""
        import abtem

        built, unbuilt = self._potential(), self._potential()
        with abtem.config.set({"fft": "numpy"}):
            built.build(lazy=False)
        assert built == unbuilt

    def test_an_empty_ensemble_attribute_does_not_short_circuit_the_rest(self):
        """The EmptyEnsemble branch did `return True` for the whole object
        rather than `continue` for that attribute, so anything declared after
        one was never compared.

        Latent in abtem today -- nothing instantiates EmptyEnsemble or its
        subclass EmptyTransform -- so this is demonstrated on a local class,
        which is also the only way to pin it against reintroduction.
        """
        from abtem.core.ensemble import EmptyEnsemble
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self, tail):
                self._ensemble = EmptyEnsemble()  # declared FIRST
                self._tail = tail

        assert Holder("one") == Holder("one")
        assert Holder("one") != Holder("two")

    def test_an_object_missing_an_attribute_is_not_equal(self):
        """An object restored without an attribute the other has (for example
        unpickled from an older version) differs from it."""
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self):
                self._a = 1.0
                self._b = 2.0

        complete, incomplete = Holder(), Holder()
        del incomplete.__dict__["_b"]

        assert complete != incomplete

    def test_an_attribute_whose_comparison_raises_is_a_difference(self):
        """`list.__eq__` reduces each pairwise `==` of arrays to a bool, which
        numpy refuses for arrays of more than one element."""
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self):
                self._arrays = [np.zeros(3)]

        assert Holder() != Holder()

    def test_a_built_potential_equals_an_unbuilt_one_inside_a_container(self):
        """`_eq_exclude` is applied by `EqualityMixin.__eq__`, which compares a
        nested `EqualityMixin` attribute too, so the invariant above holds for
        an operand held by another object: the same potential held by an
        SMatrix keeps comparing equal once it has been built."""
        import abtem

        built, unbuilt = self._potential(), self._potential()
        s_built = abtem.SMatrix(potential=built, energy=100e3, semiangle_cutoff=20)
        s_unbuilt = abtem.SMatrix(potential=unbuilt, energy=100e3, semiangle_cutoff=20)
        with abtem.config.set({"fft": "numpy"}):
            built.build(lazy=False)
        assert s_built == s_unbuilt

    def test_a_held_object_compares_as_it_does_on_its_own_in_both_orders(self):
        """A nested object is compared with `==`, so a subclass of the other
        operand's class is dispatched as at the top level, and the verdict does
        not depend on which of two held objects is on the left. A
        `PotentialArray` and a `TransmissionFunction` (its subclass) with the
        same array differ, whichever way round. Device-independent."""
        import abtem
        from abtem.multislice import MultisliceTransform

        transmission = self._potential(gpts=(16, 16)).build(lazy=False)
        transmission = transmission.transmission_function(100e3)
        potential = abtem.PotentialArray(
            transmission.array.copy(),
            slice_thickness=transmission.slice_thickness,
            extent=transmission.extent,
        )

        assert potential != transmission
        assert transmission != potential
        assert MultisliceTransform(potential) != MultisliceTransform(transmission)
        assert MultisliceTransform(transmission) != MultisliceTransform(potential)

    def test_a_nested_object_is_compared_once(self, monkeypatch):
        """Each nested `EqualityMixin` attribute is compared once per path, so
        the number of `safe_equality` calls grows with the number of nested
        objects, not with 2 ** depth."""
        import collections

        import abtem
        import abtem.core.utils as utils

        real_safe_equality = utils.safe_equality
        parents = []
        calls = collections.Counter()

        def counting_safe_equality(a, b, exclude=()):
            calls[(parents[-1] if parents else None, id(a), id(b))] += 1
            parents.append((id(a), id(b)))
            try:
                return real_safe_equality(a, b, exclude)
            finally:
                parents.pop()

        monkeypatch.setattr(utils, "safe_equality", counting_safe_equality)

        a = abtem.SMatrix(
            potential=self._potential(), energy=100e3, semiangle_cutoff=20
        )
        b = abtem.SMatrix(
            potential=self._potential(), energy=100e3, semiangle_cutoff=20
        )
        assert a == b
        assert len(calls) > 1
        assert max(calls.values()) == 1, [k for k, n in calls.items() if n > 1]

    def test_a_built_crystal_potential_equals_an_identical_unbuilt_one(self):
        """CrystalPotential creates its own `_sliced_atoms` and descends from
        `_PotentialBuilder`, not from `_FieldBuilderFromAtoms`, so it needs the
        exclusion declared in its own right -- inheriting it through Potential
        does not reach it."""
        import abtem

        built = abtem.CrystalPotential(self._potential(), repetitions=(1, 1, 2))
        unbuilt = abtem.CrystalPotential(self._potential(), repetitions=(1, 1, 2))

        # get_sliced_atoms(), not build(): build() populates the *unit*
        # potential's cache, while this populates the crystal's own. It is the
        # public entry point, and the core-loss driver calls it on the caller's
        # potential to extract EELS sites.
        built.get_sliced_atoms()
        assert built.__dict__["_sliced_atoms"] is not None

        assert built == unbuilt

    @pytest.mark.parametrize(
        "holder", ["SMatrix", "MultisliceTransform", "CrystalPotential"]
    )
    def test_an_object_holding_a_crystal_potential_tells_two_mosaics_apart(
        self, holder
    ):
        """Crystals built from seeds one apart have different root seeds, so an
        object holding one of them differs from an object holding the other.
        Device-independent."""
        import abtem
        from abtem.multislice import MultisliceTransform

        def crystal(seed):
            frozen_phonons = abtem.FrozenPhonons(
                self._atoms(), 2, sigmas=0.1, seed=(seed, 2_000_000)
            )
            unit = abtem.Potential(frozen_phonons, gpts=(16, 16), slice_thickness=1.0)
            return abtem.CrystalPotential(unit, (2, 2, 1))

        def hold(potential):
            if holder == "SMatrix":
                return abtem.SMatrix(
                    potential=potential, energy=100e3, semiangle_cutoff=20
                )
            if holder == "MultisliceTransform":
                return MultisliceTransform(potential)
            return abtem.CrystalPotential(potential, (1, 1, 2), _root_seed=5)

        a, b = crystal(1_000_000), crystal(1_000_001)
        assert a != b
        assert hold(a) != hold(b)
        assert hold(a) == hold(a.copy())

    def test_held_slice_indexed_atoms_equal_an_identical_twin(self):
        """`SliceIndexedAtoms.__eq__` compares its list of index arrays with
        `np.array_equal`; the generic rule compares the list with `list.__eq__`,
        which raises for arrays of more than one element. An object holding the
        slices, with no exclusion of them, gives the slices' own verdict.
        Device-independent."""
        from abtem.core.utils import EqualityMixin
        from abtem.slicing import SliceIndexedAtoms

        class Holder(EqualityMixin):
            def __init__(self, atoms, slice_thickness):
                self._sliced_atoms = SliceIndexedAtoms(atoms, slice_thickness)

        assert Holder(self._atoms(), 1.0) == Holder(self._atoms(), 1.0)
        assert Holder(self._atoms(), 1.0) != Holder(self._atoms(), 0.5)

    @pytest.mark.parametrize(
        "kind", ["FrozenPhonons", "Potential", "CrystalPotential", "NoiseTransform"]
    )
    def test_seeds_a_few_parts_per_million_apart_are_different(self, kind):
        """Integers compare exactly. `np.allclose` takes seeds that differ by one
        in a million as equal, so objects that draw different random numbers
        compared equal. Device-independent."""
        import abtem
        from abtem.noise import NoiseTransform

        seeds = (1_000_000, 2_000_000)
        near = (1_000_001, 2_000_000)

        def make(seeds):
            atoms = self._atoms()
            if kind == "NoiseTransform":
                return NoiseTransform(dose=1e3, seeds=seeds)
            frozen_phonons = abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=seeds)
            if kind == "FrozenPhonons":
                return frozen_phonons
            unit = abtem.Potential(
                frozen_phonons, gpts=(16, 16), slice_thickness=atoms.cell[2, 2] / 4
            )
            if kind == "Potential":
                return unit
            unit = abtem.Potential(
                abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1),
                gpts=(16, 16),
                slice_thickness=atoms.cell[2, 2] / 4,
            )
            return abtem.CrystalPotential(unit, (2, 2, 1), seeds=seeds)

        assert make(seeds) != make(near)
        assert make(seeds) == make(seeds)

    @pytest.mark.parametrize(
        "a, b",
        [
            ({"energy": 100_000}, {"energy": 100e3}),
            ({"sampling": 1, "gpts": None}, {"sampling": 1.0, "gpts": None}),
            ({"semiangle_cutoff": 20}, {"semiangle_cutoff": 20.0}),
        ],
        ids=["energy", "sampling", "semiangle_cutoff"],
    )
    def test_an_integer_and_the_same_float_still_compare_equal(self, a, b):
        """Only a pair of integers compares exactly; an integer against a float
        keeps the tolerance of `np.allclose`. Device-independent."""
        import abtem

        def probe(**kwargs):
            kwargs = {"semiangle_cutoff": 20, "gpts": 64, "extent": 10, **kwargs}
            return abtem.Probe(**{k: v for k, v in kwargs.items() if v is not None})

        assert probe(**a) == probe(**b)

    def test_integers_compare_by_value_whatever_the_container(self):
        """A list and a tuple of the same integers are equal, as are an array and
        a list, and differing integers are not. Device-independent."""
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self, values):
                self._values = values

        assert Holder([7]) == Holder((7,))
        assert Holder(np.array([7, 8])) == Holder([7, 8])
        assert Holder(1_000_000) != Holder(1_000_001)
        assert Holder((1_000_000, 2)) != Holder((1_000_001, 2))
        assert Holder(True) != Holder(False)

    def test_equality_does_not_execute_a_dask_graph(self):
        """Refusing to compare lazy content must not run the graphs: the
        comparison raises before anything is computed."""
        import dask
        import dask.array as da

        from abtem.measurements import Images

        calls = []
        real_compute = dask.base.compute

        def counting_compute(*args, **kwargs):
            calls.append(args)
            return real_compute(*args, **kwargs)

        dask.base.compute = counting_compute
        try:
            a = Images(da.zeros((16, 16), dtype="float32"), sampling=0.1)
            b = Images(da.ones((16, 16), dtype="float32"), sampling=0.1)
            with pytest.raises(TypeError):
                a == b
        finally:
            dask.base.compute = real_compute

        assert not calls, f"`==` executed {len(calls)} dask graph(s)"

    @staticmethod
    def _images(array):
        from abtem.measurements import Images

        return Images(array, sampling=0.1)

    def test_lazy_objects_with_different_content_do_not_compare(self):
        """A dask value is never the `False` singleton, so `==` used to skip a
        lazy attribute and every lazy object compared equal to every other.
        Lazy content now has no verdict, and `==` and `!=` say so."""
        import dask.array as da

        from abtem.core.utils import LazyComparisonError

        zeros = self._images(da.zeros((16, 12), dtype="float32"))
        ones = self._images(da.ones((16, 12), dtype="float32"))

        with pytest.raises(LazyComparisonError, match=r"\.compute\(\)"):
            zeros == ones
        with pytest.raises(LazyComparisonError, match=r"\.compute\(\)"):
            zeros != ones
        assert issubclass(LazyComparisonError, TypeError)

    def test_a_lazy_object_does_not_compare_with_an_eager_one(self):
        """The content is lazy on one side only, in either order."""
        import dask.array as da

        lazy = self._images(da.zeros((16, 12), dtype="float32"))
        eager = self._images(np.zeros((16, 12), dtype="float32"))

        with pytest.raises(TypeError, match=r"\.compute\(\)"):
            lazy == eager
        with pytest.raises(TypeError, match=r"\.compute\(\)"):
            eager == lazy

    def test_a_lazy_object_compares_after_compute(self):
        import dask.array as da

        zeros = self._images(da.zeros((16, 12), dtype="float32")).compute()
        ones = self._images(da.ones((16, 12), dtype="float32")).compute()
        zeros_again = self._images(da.zeros((16, 12), dtype="float32")).compute()

        assert zeros == zeros_again
        assert zeros != ones

    def test_the_very_same_dask_array_compares_equal(self):
        """Two objects that hold the same dask array (the same object, or a
        copy with the same name) have the same content by construction, so the
        array compares equal and the other attributes decide."""
        import dask.array as da

        array = da.zeros((16, 12), dtype="float32")
        a = self._images(array)

        assert a == self._images(array)
        assert not a != self._images(array)
        assert a == a.copy()
        assert a.copy().array is not a.array
        assert a.copy().array.name == a.array.name

        different_sampling = type(a)(array, sampling=0.2)
        assert a != different_sampling

    def test_nested_lazy_content_does_not_compare(self):
        """An object that holds a lazy `EqualityMixin` raises through the
        nested comparison instead of reporting the objects unequal."""
        import abtem

        def s_matrix():
            potential = abtem.Potential(
                self._atoms(), gpts=(16, 16), slice_thickness=2.0
            ).build(lazy=True)
            return abtem.SMatrix(potential=potential, energy=100e3, semiangle_cutoff=20)

        with pytest.raises(TypeError, match=r"\.compute\(\)"):
            s_matrix() == s_matrix()

    def test_lazy_content_inside_a_list_does_not_compare(self):
        """A lazy object held in a list raises through `list.__eq__` as it does
        when held directly, instead of counting as a difference."""
        import dask.array as da

        from abtem.core.utils import EqualityMixin, LazyComparisonError

        class Holder(EqualityMixin):
            def __init__(self, images):
                self._images = [images]

        a = Holder(self._images(da.zeros((16, 12), dtype="float32")))
        b = Holder(self._images(da.ones((16, 12), dtype="float32")))

        with pytest.raises(LazyComparisonError, match=r"\.compute\(\)"):
            a == b
        with pytest.raises(LazyComparisonError, match=r"\.compute\(\)"):
            a != b

    def test_a_difference_decides_before_lazy_content(self):
        """Any attribute that differs makes lazy objects unequal, wherever it
        sits in `__dict__`; only otherwise does lazy content raise."""
        import dask.array as da

        from abtem.core.utils import LazyComparisonError
        from abtem.measurements import IndexedDiffractionPatterns

        def patterns(array, hkl):
            return IndexedDiffractionPatterns(
                array=array,
                miller_indices=np.array(hkl),
                reciprocal_lattice_vectors=np.eye(3),
            )

        lazy = patterns(da.zeros(3, dtype="float32"), [[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        other_hkl = [[1, 0, 0], [0, 1, 0], [2, 0, 0]]
        # `_miller_indices` comes after `_array` in IndexedDiffractionPatterns
        assert lazy != patterns(da.ones(3, dtype="float32"), other_hkl)
        assert lazy != patterns(np.zeros(3, dtype="float32"), other_hkl)

        images = self._images(da.zeros((16, 12), dtype="float32"))
        transposed = self._images(np.zeros((12, 16), dtype="float32"))
        assert images != transposed
        assert transposed != images
        assert not images == transposed
        with pytest.raises(LazyComparisonError):
            images == self._images(np.zeros((16, 12), dtype="float32"))

    def test_lazy_content_in_a_nested_object_decides_last(self):
        """An `SMatrix` with a lazy potential differs by its energy; with the
        same energy only the lazy potential is left, and it raises."""
        import abtem
        from abtem.core.utils import LazyComparisonError

        def s_matrix(energy):
            potential = abtem.Potential(
                self._atoms(), gpts=(16, 16), slice_thickness=2.0
            ).build(lazy=True)
            return abtem.SMatrix(
                potential=potential, energy=energy, semiangle_cutoff=20
            )

        assert s_matrix(100e3) != s_matrix(200e3)
        assert not s_matrix(100e3) == s_matrix(200e3)
        with pytest.raises(LazyComparisonError):
            s_matrix(100e3) == s_matrix(100e3)

    def test_slice_indexed_atoms_equals_an_identical_twin(self):
        """`_slice_index` is a list of integer arrays, one per slice.
        `list.__eq__` reduces each pairwise `==` (itself an array) to a bool,
        which numpy refuses for anything but a length-1 array -- the generic
        safe_equality catches that ValueError and reports "unequal", so this
        class never compared equal to anything, including an identical twin.
        """
        from abtem.slicing import SliceIndexedAtoms

        a, b = SliceIndexedAtoms(self._atoms(), 1.0), SliceIndexedAtoms(self._atoms(), 1.0)
        assert a == b

    def test_slice_indexed_atoms_with_different_binning_are_not_equal(self):
        """The fix must not make every SliceIndexedAtoms equal regardless of
        content -- different slice thicknesses bin the same atoms differently
        and must still compare unequal."""
        from abtem.slicing import SliceIndexedAtoms

        a = SliceIndexedAtoms(self._atoms(), 1.0)
        b = SliceIndexedAtoms(self._atoms(), 0.5)
        assert a != b

    def test_magnetic_field_equals_its_own_copy(self):
        """`QuasiDipoleProjections` (the integrator `MagneticField`/
        `VectorPotential` hold as `_integrator`) was a plain class with no
        `__eq__`, so it fell back to object identity and every `MagneticField`
        compared unequal to its own copy, including an untouched one."""
        from abtem.magnetism.iam import MagneticField

        atoms = self._atoms()
        m = MagneticField(atoms, gpts=(32, 32), slice_thickness=1.0)
        assert m == m.copy()

    def test_magnetic_field_with_different_atoms_is_not_equal(self):
        from abtem.magnetism.iam import MagneticField

        a = MagneticField(self._atoms(), gpts=(32, 32), slice_thickness=1.0)
        b = MagneticField(self._atoms(dx=1.234), gpts=(32, 32), slice_thickness=1.0)
        assert a != b

    def test_quasi_dipole_integrator_ignores_its_table_cache(self):
        """`_tables` is populated lazily by `get_integral_table`, one entry
        per element on first use -- the same shape of defect `_sliced_atoms`
        had for `Potential`, and doubly so: an unpopulated cache made a used
        integrator stop comparing equal to a fresh one, and a *populated* one
        made two integrators that cached the identical element compare
        unequal anyway, since dict.__eq__ on numpy-array values hits the same
        ValueError the list case above does.
        """
        from abtem.magnetism.iam import MagneticField

        atoms = self._atoms()
        atoms.numbers[:] = 26  # Fe, present in the Lyon parametrization
        m1 = MagneticField(atoms, gpts=(32, 32), slice_thickness=1.0)
        m2 = m1.copy()
        assert m1 == m2

        m1._integrator.get_integral_table("Fe")
        assert m1.__dict__["_integrator"].__dict__["_tables"]
        assert not m2.__dict__["_integrator"].__dict__["_tables"]
        assert m1 == m2

        m2._integrator.get_integral_table("Fe")
        assert m1 == m2

    def test_transition_potential_array_ignores_its_device_cache(self):
        """`_local_potential_device_cache` is populated lazily by
        `_local_potential_on_device()`, a normal side effect of core-loss
        multislice -- the same shape of defect `_sliced_atoms` had for
        `Potential`. Two transition potentials with identical arrays stopped
        comparing equal once one of them had been used on a device."""
        import numpy as np

        from abtem.core.axes import OrdinalAxis
        from abtem.inelastic.core_loss import TransitionPotentialArray

        def make():
            rng = np.random.default_rng(0)
            array = (
                rng.standard_normal((2, 16, 16))
                + 1j * rng.standard_normal((2, 16, 16))
            ).astype(np.complex64)
            return TransitionPotentialArray(
                Z=14,
                array=array,
                energy=100e3,
                extent=4.0,
                ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
                metadata={"Z": 14, "n": 1, "l": 0},
            )

        tp1, tp2 = make(), make()
        assert tp1 == tp2

        tp1._local_potential_on_device(np.zeros((16, 16), dtype=np.complex64))
        assert tp1.__dict__["_local_potential_device_cache"] is not None
        assert tp2.__dict__["_local_potential_device_cache"] is None

        assert tp1 == tp2

    @pytest.mark.parametrize(
        "a, b",
        [
            ((), (1,)),
            ([], [5]),
            (7, (7, 7)),
            (7, (7,)),
            ((5,), (5, 5)),
            ((1.0,), (1.0, 1.0)),
            (np.arange(7), np.tile(np.arange(7), (7, 1))),
            (np.zeros(3), np.zeros((2, 3))),
        ],
        ids=[
            "empty-vs-one",
            "empty-list-vs-one",
            "number-vs-pair",
            "number-vs-one",
            "one-vs-two",
            "float-one-vs-two",
            "int-array-vs-tile",
            "float-array-vs-stack",
        ],
    )
    def test_values_of_different_shapes_are_different(self, a, b):
        """`np.equal` and `np.allclose` broadcast, so `(5,)` equalled `(5, 5)`
        and an empty sequence equalled anything. The array pairs differ in size
        along both axes. Device-independent."""
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self, value):
                self._value = value

        assert Holder(a) != Holder(b)
        assert Holder(b) != Holder(a)

    @pytest.mark.parametrize(
        "a, b",
        [
            ([7], (7,)),
            (np.arange(3), [0, 1, 2]),
            (((1, 2), (3,)), ((1, 2), (3,))),
            ((1.0,), (1.0 + 1e-9,)),
            (np.zeros((2, 3)), np.zeros((2, 3))),
        ],
        ids=["list-vs-tuple", "array-vs-list", "ragged", "float-tolerance", "array"],
    )
    def test_values_of_the_same_shape_still_compare_equal(self, a, b):
        """The shape check adds no difference between values of one shape: a list
        and a tuple, an array and a list, identical ragged tuples and floats
        within `np.allclose`'s tolerance stay equal, and different ragged tuples
        stay different. Device-independent."""
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self, value):
                self._value = value

        assert Holder(a) == Holder(b)
        assert Holder(((1, 2), (3,))) != Holder(((1, 2), (4,)))

    @pytest.mark.parametrize("kind", ["FrozenPhonons", "NoiseTransform"])
    def test_one_seed_against_two_equal_seeds_is_different(self, kind):
        """A `FrozenPhonons` of one configuration and one of two with the same
        seed, and noise transforms with seeds `(5,)` and `(5, 5)`, draw
        different numbers of random streams. Device-independent."""
        import abtem
        from abtem.noise import NoiseTransform

        def make(seeds):
            if kind == "NoiseTransform":
                return NoiseTransform(dose=1e3, seeds=seeds)
            return abtem.FrozenPhonons(
                self._atoms(), len(seeds), sigmas=0.1, seed=seeds
            )

        assert make((5,)) != make((5, 5))
        assert make((5, 5)) == make((5, 5))

    @pytest.mark.parametrize("kind", ["PlaneWave", "Probe"])
    def test_a_one_value_energy_list_equals_the_value(self, kind):
        """A one-value energy list is stored as its value, so it builds the
        same waves as the value and compares equal to it in both orders; two
        energies still make an `EnergyAxis`. Device-independent."""
        import abtem
        from abtem.core.axes import EnergyAxis

        def make(energy):
            if kind == "PlaneWave":
                return abtem.PlaneWave(energy=energy, gpts=16, extent=4)
            return abtem.Probe(energy=energy, semiangle_cutoff=20, gpts=16, extent=4)

        value, listed = make(100e3), make([100e3])
        assert value == listed
        assert listed == value
        assert value.ensemble_shape == listed.ensemble_shape == ()
        np.testing.assert_array_equal(
            value.build(lazy=False).array, listed.build(lazy=False).array
        )

        pair = make([100e3, 200e3])
        assert pair != value
        assert isinstance(pair.ensemble_axes_metadata[0], EnergyAxis)

    def test_the_energy_setter_stores_a_one_value_list_as_its_value(self):
        """Device-independent."""
        from abtem.waves import EnergyEnsemble

        ensemble = EnergyEnsemble(100e3)
        ensemble.energy = [200e3]

        assert ensemble == EnergyEnsemble(200e3)
        assert EnergyEnsemble(200e3) == ensemble
