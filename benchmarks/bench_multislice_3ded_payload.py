"""Benchmark: one orientation of a 3DED-style multislice rotation series, stage by stage.

Times the stages of what py3DED's `setup_multislice` runs for each orientation of a
rotation series: the rotated and cropped atoms (host), the slicing of the atoms (host),
the potential slices (per chunk, on the device), the propagation (transmission function,
antialiasing, propagator) and the detection (window, FFT, crop, copy to the host). The
reference is the unmodified `pw.multislice(...).compute()`; the instrumented run
synchronises the device around every timed call, so use it for the proportions, not for the
totals (the ratio of the two says how much the asynchronous launches of the device hide).

The structure and settings can come from a py3DED simulation config (a JSON file, whose
`cif_file` is read relative to it): `box_size_x`, `box_size_y`, `thickness`, `sampling`,
`slice_thickness`, `exit_planes`, `energy`, `thermal_sigmas`, `projection`, `precision`,
`fft`, `rotation_axis_orientation`, `rotation_min`, `rotation_max`, `g_max_store`, `margin`,
`window_func`, `vacancies`, `centering`, `sg_max`, `g_max` and `integration_radius`.
Every option below overrides the config; without a config the defaults are a tiny
baddeleyite box that runs in seconds on a CPU. Resolution is a physics decision and is
never changed here; `--also SAMPLING SLICE` only measures another pair next to the first.

Besides the stage times, the report answers three questions about the machinery:
  * is the scheduler synchronous (does any host work overlap the device work)?
  * are the integrator caches (scattering factor) kept across the ensemble blocks, or built
    again for every block (copies drop the caches of `_CacheStateMixin`)?
  * is the slicing of the atoms rebuilt for every chunk of potential slices?

Usage
-----
    python benchmarks/bench_multislice_3ded_payload.py [STRUCTURE] [options]

    STRUCTURE  "baddeleyite" (default; ZrO2, P21/c), the path of a file ASE reads (a CIF),
               or the path of a py3DED config (.json)

    --device cpu|gpu         (default cpu; a config's "device" is not used)
    --box X Y                size of the box [A] (default 20 20, or the config's)
    --thickness Z            thickness of the box [A] (default 30, or the config's)
    --sampling S             real-space sampling [A] (default 0.1, or the config's)
    --slice-thickness T      [A] (default 2.0, or the config's)
    --planes N               `exit_planes` as abTEM takes it: a measurement every N slices
                             and after the last (default 3, or the config's); the
                             number of detections is in the report
    --cif PATH               the structure, in place of the config's `cif_file` (the
                             example configs name files that py3DED does not ship)
    --also SAMPLING SLICE    also run this pair of sampling and slice thickness [A]
                             (repeatable), e.g. `--also 0.04 0.8` next to a 0.02/0.4 run
    --angle DEG              tilt of the orientation (default: the middle of the tilt range)
    --orientations N         orientations in the run (default 1; they are one ensemble
                             block each, and the report is per run)
    --axis DEG               azimuth of the tilt axis (default 0, or the config's)
    --chunk-size N|auto      potential.slice-chunk-size (default "auto", as py3DED)
    --to-cpu                 the detector copies each exit plane to the host (py3DED: off)
    --index                  also time `index_diffraction_spots` on the result
    --no-reference           skip the uninstrumented run (halves the run time)
    --energy, --thermal-sigma, --projection, --precision, --margin, --window-func,
    --g-max-store, --fft     as the config keys (defaults: 200e3 eV, 0.078 A, infinite,
                             float32, 0.0 A, hann, 4.0 1/A, fftw)

Production-size run on a GPU (one orientation of the ZrO2 config, 8000 x 8000 grid, 2500
slices; the results are also written to benchmarks/results/, which is git-ignored):
    python benchmarks/bench_multislice_3ded_payload.py ../py3DED/examples/realistic_cases/ZrO2.json --cif ../py3DED/examples/realistic_cases/ZrO2_baddeleyit_VESTA.cif --device gpu --index --also 0.04 0.8
"""

import argparse
import inspect
import json
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import abtem

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("structure", nargs="?", default="baddeleyite")
parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--box", type=float, nargs=2)
parser.add_argument("--thickness", type=float)
parser.add_argument("--sampling", type=float)
parser.add_argument("--slice-thickness", type=float)
parser.add_argument("--planes", type=int)
parser.add_argument("--cif")
parser.add_argument("--also", type=float, nargs=2, action="append", default=[])
parser.add_argument("--angle", type=float)
parser.add_argument("--orientations", type=int, default=1)
parser.add_argument("--axis", type=float)
parser.add_argument("--chunk-size", default="auto")
parser.add_argument("--to-cpu", action="store_true")
parser.add_argument("--index", action="store_true")
parser.add_argument("--no-reference", action="store_true")
parser.add_argument("--energy", type=float)
parser.add_argument("--thermal-sigma", type=float)
parser.add_argument("--projection")
parser.add_argument("--precision", choices=["float32", "float64"])
parser.add_argument("--margin", type=float)
parser.add_argument("--window-func")
parser.add_argument("--g-max-store", type=float)
parser.add_argument("--fft")
args = parser.parse_args()

config = {}
structure = args.structure
if structure.endswith(".json"):
    config = json.loads(Path(structure).read_text())
    structure = str(Path(structure).parent / config["cif_file"])
if args.cif:
    structure = args.cif
if structure != "baddeleyite" and not Path(structure).exists():
    parser.error(f"{structure} does not exist; give the structure with --cif")


def setting(option, key, default):
    """The option if given, else the config's value of `key`, else the default."""
    value = getattr(args, option)
    return value if value is not None else config.get(key, default)


device = args.device
box = tuple(
    args.box or (config.get("box_size_x", 20.0), config.get("box_size_y", 20.0))
)
thickness = setting("thickness", "thickness", 30.0)
planes = setting("planes", "exit_planes", 3)
cases = [
    (
        setting("sampling", "sampling", 0.1),
        setting("slice_thickness", "slice_thickness", 2.0),
    )
] + [tuple(pair) for pair in args.also]
energy = setting("energy", "energy", 200e3)
precision = setting("precision", "precision", "float32")
projection = setting("projection", "projection", "infinite")
margin = setting("margin", "margin", 0.0)
window_func = setting("window_func", "window_func", "hann")
g_max_store = setting("g_max_store", "g_max_store", 4.0)
fft = setting("fft", "fft", "fftw")
axis = setting("axis", "rotation_axis_orientation", 0.0)
tilt_min = config.get("rotation_min", 0.0)
tilt_max = config.get("rotation_max", 4.0)
n_orientations = args.orientations
angles = (
    np.array([args.angle if args.angle is not None else (tilt_min + tilt_max) / 2] * 1)
    if n_orientations == 1
    else np.linspace(tilt_min, tilt_max, n_orientations)
)
chunk_size = args.chunk_size if args.chunk_size == "auto" else int(args.chunk_size)
vacancies = config.get("vacancies", 0.0)


def load_atoms():
    if structure == "baddeleyite":
        from ase.spacegroup import crystal

        return crystal(
            ["Zr", "O", "O"],
            basis=[
                (0.2754, 0.0395, 0.2083),
                (0.0700, 0.3317, 0.3447),
                (0.4423, 0.7569, 0.4792),
            ],
            spacegroup=14,
            cellpar=[5.151, 5.212, 5.317, 90, 99.23, 90],
        )
    from ase.io import read

    return read(structure)


def synchronize():
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


# --------------------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------------------


class Profiler:
    """Times calls with the device synchronised around each one, as inclusive and as
    exclusive time (the inclusive time less that of the timed calls inside it), counts
    calls and records the top-level intervals per thread to find overlapping work.

    The name of a timed call starts with its category (`host`, `potential`, `propagation`,
    `detection`), which is where its exclusive time is counted.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.local = threading.local()
        self.stats = {}
        self.events = []
        self.counts = Counter()
        self.sets = defaultdict(set)
        self.keyed = Counter()
        self.patched = []

    def stack(self):
        if not hasattr(self.local, "stack"):
            self.local.stack = []
        return self.local.stack

    def current_category(self):
        stack = self.stack()
        return stack[-1][0].split(":")[0] if stack else "other"

    def wrap(self, owner, attribute, name=None, on_call=None):
        """Replace `owner.attribute` (function, static or class method, or module
        function) by a timed wrapper; `on_call(args, kwargs)` records what a call is
        given. An attribute that is not there is skipped."""
        static = inspect.getattr_static(owner, attribute, None)
        if static is None:
            return
        function = (
            static.__func__
            if isinstance(static, (staticmethod, classmethod))
            else static
        )
        label = (
            name
            or f"{owner.__name__ if hasattr(owner, '__name__') else owner}.{attribute}"
        )
        profiler = self

        def timed(*call_args, **call_kwargs):
            if on_call is not None:
                on_call(call_args, call_kwargs)
            stack = profiler.stack()
            synchronize()
            frame = [label, 0.0]
            stack.append(frame)
            start = time.perf_counter()
            try:
                return function(*call_args, **call_kwargs)
            finally:
                synchronize()
                end = time.perf_counter()
                stack.pop()
                elapsed = end - start
                if stack:
                    stack[-1][1] += elapsed
                with profiler.lock:
                    record = profiler.stats.setdefault(label, [0, 0.0, 0.0])
                    record[0] += 1
                    record[1] += elapsed
                    record[2] += elapsed - frame[1]
                    if not stack:
                        profiler.events.append((start, end, threading.get_ident()))

        timed.__wrapped__ = function
        if isinstance(static, staticmethod):
            replacement = staticmethod(timed)
        elif isinstance(static, classmethod):
            replacement = classmethod(timed)
        else:
            replacement = timed
        self.patched.append((owner, attribute, static, attribute in vars(owner)))
        setattr(owner, attribute, replacement)

    def count(self, owner, attribute, key, record=None, keyed=False):
        """Count the calls of `owner.attribute` under `key`; `record(self_or_arg0)`
        returns something to collect in a set under the same key; with `keyed`, the
        (object, first argument) of each call is counted, to find a value computed
        again for the same key."""

        def on_call(call_args, call_kwargs):
            with self.lock:
                self.counts[key] += 1
                if record is not None and call_args:
                    self.sets[key].add(record(call_args[0]))
                if keyed and len(call_args) > 1:
                    self.keyed[(id(call_args[0]), call_args[1])] += 1

        static = inspect.getattr_static(owner, attribute, None)
        if static is None:
            return
        function = (
            static.__func__
            if isinstance(static, (staticmethod, classmethod))
            else static
        )

        def counted(*call_args, **call_kwargs):
            on_call(call_args, call_kwargs)
            return function(*call_args, **call_kwargs)

        self.patched.append((owner, attribute, static, attribute in vars(owner)))
        setattr(
            owner,
            attribute,
            staticmethod(counted)
            if isinstance(static, staticmethod)
            else classmethod(counted)
            if isinstance(static, classmethod)
            else counted,
        )

    def restore(self):
        for owner, attribute, static, own in reversed(self.patched):
            if own:
                setattr(owner, attribute, static)
            else:  # inherited: remove the wrapper rather than copy the method down
                delattr(owner, attribute)
        self.patched = []

    def overlap_seconds(self):
        """The time over which more than one thread was inside a timed call."""
        points = sorted(
            [(s, 1) for s, e, t in self.events] + [(e, -1) for s, e, t in self.events]
        )
        # a thread's own top-level intervals never overlap, so the number of intervals
        # open at once is the number of threads at work
        active, previous, overlap = 0, 0.0, 0.0
        for time_point, step in points:
            if active > 1:
                overlap += time_point - previous
            active += step
            previous = time_point
        return overlap

    def threads(self):
        return len({t for s, e, t in self.events})


def instrument(profiler):
    """Wrap the stages of the multislice pipeline in `profiler`."""
    import abtem.detectors as detectors
    import abtem.integrals as integrals
    import abtem.multislice as multislice
    import abtem.potentials.iam as iam
    import abtem.rotation_series as rotation_series
    import abtem.slicing as slicing
    import abtem.waves as waves
    from abtem.antialias import AntialiasAperture
    from abtem.core import chunks

    wrap = profiler.wrap
    field_builder = iam._FieldBuilderFromAtoms

    # host: the rotated and cropped atoms
    wrap(rotation_series, "_rotate_and_crop_to_cell", "host: rotate + crop atoms")
    # host: the slicing of the atoms (the wrap, the binning into slices, ...)
    wrap(field_builder, "_prepare_atoms", "host: slice atoms (_prepare_atoms)")
    wrap(field_builder, "_transform_atoms", "host: transform atoms")
    wrap(slicing.SliceIndexedAtoms, "__init__", "host: bin atoms into slices")
    wrap(slicing.SlicedAtoms, "__init__", "host: bin atoms into slices (finite)")
    wrap(slicing.SliceIndexedAtoms, "get_atoms_in_slices", "host: atoms of a slice")
    wrap(slicing.SlicedAtoms, "get_atoms_in_slices", "host: atoms of a slice (finite)")

    # potential: the chunks and the slices; a build with `lazy=False` is one chunk
    def chunk_range(call_args, call_kwargs):
        first = call_kwargs.get(
            "first_slice", call_args[1] if len(call_args) > 1 else 0
        )
        last = call_kwargs.get(
            "last_slice", call_args[2] if len(call_args) > 2 else None
        )
        lazy = call_kwargs.get("lazy", call_args[4] if len(call_args) > 4 else None)
        if lazy is False:
            with profiler.lock:
                profiler.counts["chunk builds"] += 1
                profiler.sets["chunk ranges"].add((first, last))

    wrap(
        field_builder,
        "build",
        "potential: build of a chunk (assembly)",
        on_call=chunk_range,
    )
    wrap(
        integrals.ScatteringFactorProjectionIntegrals,
        "integrate_on_grid",
        "potential: integrate_on_grid (glue)",
    )
    wrap(
        integrals.QuadratureProjectionIntegrals,
        "integrate_on_grid",
        "potential: integrate_on_grid (glue, quadrature)",
    )
    wrap(
        integrals.ScatteringFactorProjectionIntegrals,
        "_calculate_scattering_factor_on_device",
        "potential: scattering factor (cache miss)",
    )
    wrap(
        integrals.QuadratureProjectionIntegrals,
        "_calculate_integral_table",
        "potential: integral table (cache miss)",
    )
    wrap(
        integrals, "superpose_deltas", "potential: superpose deltas (upload + scatter)"
    )
    wrap(integrals, "sinc", "potential: sinc (recomputed per call)")
    wrap(integrals, "fft2", "potential: fft2 of the atoms")
    wrap(integrals, "ifft2", "potential: ifft2 of the slice")
    wrap(
        integrals,
        "interpolate_radial_functions_cuda",
        "potential: radial kernel (quadrature)",
    )
    wrap(
        integrals,
        "_threaded_interpolate_radial_functions",
        "potential: radial kernel (quadrature, host)",
    )

    # propagation
    wrap(multislice, "conventional_multislice_step", "propagation: step (glue)")
    wrap(
        iam.PotentialArray,
        "transmission_function",
        "propagation: transmission function",
    )
    wrap(AntialiasAperture, "bandlimit", "propagation: antialias (fft convolve)")
    wrap(iam.TransmissionFunction, "transmit", "propagation: transmit")
    wrap(
        multislice.FresnelPropagator,
        "propagate",
        "propagation: propagate (fft convolve)",
    )
    wrap(
        multislice.FresnelPropagator,
        "_calculate_array",
        "propagation: propagator kernel (cache miss)",
    )

    # detection
    wrap(multislice, "_update_measurements", "detection: update measurements (glue)")
    wrap(
        detectors.PixelatedDetector,
        "_calculate_new_array",
        "detection: detector (glue)",
    )
    wrap(waves.Waves, "window", "detection: window (crop + taper)")
    wrap(
        waves.Waves,
        "diffraction_patterns",
        "detection: diffraction patterns (fft + crop)",
    )
    wrap(waves.Waves, "to_cpu", "detection: waves to_cpu")

    # counters: caches and rebuilds
    profiler.count(field_builder, "__init__", "potential objects", record=id)
    profiler.count(
        field_builder, "_from_partitioned_args_func", "ensemble blocks built"
    )
    profiler.count(field_builder, "_prepare_atoms", "slicings of the atoms")
    profiler.count(
        integrals._CacheStateMixin, "__setstate__", "integrator copies (caches dropped)"
    )
    profiler.count(
        integrals._DeviceArrayCache,
        "put",
        "cache entries computed",
        record=id,
        keyed=True,
    )
    profiler.count(chunks, "estimate_potential_chunk_size", "chunk size estimates")


@contextmanager
def gpu_memory(record, profiler=None):
    """The peak of the CuPy pool in use [MB] over the block (and where it was reached) in
    record['peak_mb'], and the pool's reserved memory in record['reserved_mb']."""
    if device != "gpu":
        yield
        return
    import cupy as cp

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    class PeakHook(cp.cuda.MemoryHook):
        peak = 0
        where = "other"
        by_category = {}

        def malloc_postprocess(self, **kwargs):
            used = pool.used_bytes()
            category = profiler.current_category() if profiler else "other"
            self.by_category[category] = max(self.by_category.get(category, 0), used)
            if used > self.peak:
                self.peak, self.where = used, category

    hook = PeakHook()
    synchronize()
    hook.peak = pool.used_bytes()
    with hook:
        yield
        synchronize()
    record["peak_mb"] = round(hook.peak / 1e6)
    record["peak_reached_in"] = hook.where
    record["peak_mb_by_category"] = {
        key: round(value / 1e6) for key, value in hook.by_category.items()
    }
    record["reserved_mb"] = round(pool.total_bytes() / 1e6)


# --------------------------------------------------------------------------------------
# the pipeline, as py3DED's setup_multislice builds it
# --------------------------------------------------------------------------------------


def set_config(slice_chunk_size):
    abtem.config.set(
        {
            "precision": precision,
            "device": device,
            "fft": fft,
            "diagnostics.task_progress": False,
            "diagnostics.progress_bar": False,
            # as py3DED's set_abtem_config
            "grid.round-to-fast-fft": True,
            "potential.slice-chunk-size": slice_chunk_size,
        }
    )


def make_supercell(unit_cell, z):
    from abtem.atoms import cut_disk

    supercell = cut_disk(
        unit_cell,
        (box[0], box[1], z),
        rotation_axis=axis,
        rotation_range=(float(np.min(angles)), float(np.max(angles))),
    )
    if vacancies > 0.0:  # as py3DED's add_vacancies
        np.random.seed(1337)
        supercell = supercell[np.random.rand(len(supercell)) > vacancies]
    return supercell


def make_pipeline(supercell, sampling, slice_thickness, exit_planes):
    """The lazy multislice of the rotation series, as py3DED's `setup_multislice`."""
    from abtem.core.energy import energy2wavelength
    from abtem.detectors import WindowedPixelatedDetector
    from abtem.rotation_series import rotated_atoms_ensemble

    atoms_ensemble = rotated_atoms_ensemble(supercell, angles, rotation_axis=axis)
    species = sorted(set(supercell.get_chemical_symbols()))
    thermal_sigma = setting("thermal_sigma", "thermal_sigmas", 0.078)
    parametrization = abtem.parametrizations.LobatoParametrization(
        sigmas={symbol: thermal_sigma for symbol in species}
    )
    potential = abtem.potentials.iam.Potential(
        atoms_ensemble,
        sampling=sampling,
        exit_planes=exit_planes,
        slice_thickness=slice_thickness,
        parametrization=parametrization,
        projection=projection,
    )
    plane_wave = abtem.waves.PlaneWave(energy=energy)
    max_angle = g_max_store * energy2wavelength(energy) * 1e3
    detector = WindowedPixelatedDetector(
        max_angle=max_angle * 1.2,
        to_cpu=args.to_cpu,
        reciprocal_space=True,
        margin=margin,
        window_func=window_func,
    )
    return plane_wave.multislice(potential=potential, detectors=detector), potential


def failure(error):
    frames = traceback.extract_tb(error.__traceback__)
    keep = [f for f in frames if "/abtem/" in f.filename.replace("\\", "/")]
    return {
        "failed": f"{type(error).__name__}: {str(error)[:200]}",
        "where": [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in keep[-8:]],
    }


def run_case(unit_cell, sampling, slice_thickness):
    """Time the stages of one pair of sampling and slice thickness."""
    result = {"sampling": sampling, "slice_thickness": slice_thickness}
    print(
        f"\n=== sampling {sampling} A, slice thickness {slice_thickness} A ===",
        flush=True,
    )
    set_config(chunk_size)

    # the host work of the rotation series is not part of this case: the supercell is cut once
    start = time.perf_counter()
    supercell = make_supercell(unit_cell, thickness)
    result["supercell"] = {
        "atoms": len(supercell),
        "cut_disk_seconds": round(time.perf_counter() - start, 3),
        "cell": [round(float(v), 2) for v in np.diag(supercell.cell)],
    }
    print(f"  supercell: {result['supercell']}", flush=True)

    # warm up on a few slices of the same grid: compiles the kernels, plans the FFTs
    warm = make_supercell(unit_cell, min(thickness, 4 * slice_thickness))
    start = time.perf_counter()
    make_pipeline(warm, sampling, slice_thickness, 2)[0].compute()
    synchronize()
    print(
        f"  warm-up on {len(warm)} atoms: {time.perf_counter() - start:.2f} s",
        flush=True,
    )

    graph_start = time.perf_counter()
    lazy, potential = make_pipeline(supercell, sampling, slice_thickness, planes)
    graph_seconds = time.perf_counter() - graph_start
    n_slices = len(potential)
    result.update(
        gpts=list(potential.gpts),
        n_slices=n_slices,
        n_exit_planes=len(potential.exit_planes),
        n_orientations=len(angles),
        graph_seconds=round(graph_seconds, 3),
    )
    print(
        f"  grid {potential.gpts}, {n_slices} slices, {len(potential.exit_planes)} exit "
        f"planes, {len(angles)} orientation(s); graph built in {graph_seconds:.2f} s",
        flush=True,
    )

    slice_count = n_slices * len(angles)

    if not args.no_reference:
        record = result["reference"] = {}
        try:
            with gpu_memory(record):
                start = time.perf_counter()
                lazy.compute()
                synchronize()
                record["seconds"] = round(time.perf_counter() - start, 3)
            record["slices_per_second"] = round(slice_count / record["seconds"], 2)
        except Exception as error:  # out of memory has a different class per backend
            record.update(failure(error))
        print(f"  reference compute: {record}", flush=True)
        lazy, potential = make_pipeline(supercell, sampling, slice_thickness, planes)

    profiler = Profiler()
    scheduler_calls = []
    instrument(profiler)
    if device == "gpu":
        import abtem.array as array_module

        resolve = array_module._resolve_gpu_scheduler

        def resolve_recording(kwargs):
            resolved = resolve(kwargs)
            scheduler_calls.append(str(resolved.get("scheduler")))
            return resolved

        array_module._resolve_gpu_scheduler = resolve_recording
    record = result["instrumented"] = {}
    try:
        with gpu_memory(record, profiler):
            start = time.perf_counter()
            output = lazy.compute()
            synchronize()
            wall = time.perf_counter() - start
    except Exception as error:  # out of memory has a different class per backend
        record.update(failure(error))
        output = None
        wall = None
    finally:
        profiler.restore()
        if device == "gpu":
            array_module._resolve_gpu_scheduler = resolve
    if wall is not None:
        report(result, record, profiler, wall, slice_count, scheduler_calls)
    else:
        print(f"  instrumented compute: {record}", flush=True)

    if output is not None:
        post_process(result, output, unit_cell)
    return result


def report(result, record, profiler, wall, slice_count, scheduler_calls):
    rows = {}
    categories = Counter()
    for name, (calls, inclusive, exclusive) in profiler.stats.items():
        category = name.split(":")[0]
        categories[category] += exclusive
        rows[name] = {
            "calls": calls,
            "inclusive_s": round(inclusive, 3),
            "exclusive_s": round(exclusive, 3),
            "exclusive_ms_per_slice": round(1e3 * exclusive / slice_count, 3),
            "fraction": round(exclusive / wall, 4),
        }
    categories["other"] = max(wall - sum(categories.values()), 0.0)
    record.update(
        seconds=round(wall, 3),
        slices_per_second=round(slice_count / wall, 2),
        categories={
            key: {"seconds": round(value, 3), "fraction": round(value / wall, 4)}
            for key, value in sorted(categories.items(), key=lambda item: -item[1])
        },
        functions=dict(sorted(rows.items(), key=lambda item: -item[1]["exclusive_s"])),
    )

    print(
        f"\n  instrumented compute: {wall:.2f} s, {record['slices_per_second']} slices/s"
    )
    if "reference" in result and "seconds" in result["reference"]:
        reference = result["reference"]["seconds"]
        print(
            f"    reference (unsynchronised) {reference:.2f} s -> synchronising costs "
            f"{wall / reference:.2f}x"
        )
    print("\n  by category (exclusive time, device synchronised around every call):")
    for key, value in record["categories"].items():
        print(
            f"    {key:<14}{value['seconds']:>10.2f} s {100 * value['fraction']:>6.1f} %"
        )
    print("\n  by call (exclusive time):")
    print(f"    {'':<56}{'calls':>8}{'excl. s':>10}{'ms/slice':>10}{'%':>7}")
    for name, row in record["functions"].items():
        print(
            f"    {name:<56}{row['calls']:>8}{row['exclusive_s']:>10.2f}"
            f"{row['exclusive_ms_per_slice']:>10.3f}{100 * row['fraction']:>7.1f}"
        )

    # the scheduler and the caches
    overlap = profiler.overlap_seconds()
    machinery = record["machinery"] = {
        "scheduler": scheduler_calls or "not recorded on the CPU",
        "threads_in_timed_calls": profiler.threads(),
        "seconds_with_overlapping_calls": round(overlap, 3),
        "counts": dict(profiler.counts),
        "potential_objects": len(profiler.sets["potential objects"]),
        "caches_filled": len(profiler.sets["cache entries computed"]),
        "chunks": profiler.sets["chunk ranges"],
        "cache_entries_computed_more_than_once": sum(
            n > 1 for n in profiler.keyed.values()
        ),
    }
    machinery["slices_per_chunk"] = sorted(
        {last - first for first, last in machinery.pop("chunks") if last is not None}
    )
    print("\n  machinery:")
    for key, value in machinery.items():
        print(f"    {key}: {value}")
    if device == "gpu":
        synchronous = (
            set(scheduler_calls) == {"synchronous"} and profiler.threads() <= 1
        )
    else:
        synchronous = profiler.threads() <= 1
    print(
        "    -> host and device work "
        + (
            "never overlap: the scheduler is synchronous and one thread runs every call"
            if synchronous
            else "overlap for part of the run (several threads in timed calls); the "
            "fractions are of the wall time and may sum to more than 100 %"
        )
    )
    blocks = max(len(angles), 1)
    slicings = profiler.counts["slicings of the atoms"]
    print(
        f"    -> the atoms were sliced {slicings} time(s) for {blocks} orientation(s) and "
        f"{profiler.counts['chunk builds']} chunk build(s)"
    )
    misses = sum(
        row["exclusive_s"]
        for name, row in rows.items()
        if "(cache miss)" in name and "propagator" not in name
    )
    machinery["seconds_in_integrator_cache_misses"] = round(misses, 3)
    print(
        f"    -> {misses:.2f} s of the run went to filling the integrator caches, which "
        "a run over many orientations pays once"
    )
    print(
        f"    -> {profiler.counts['integrator copies (caches dropped)']} integrator "
        f"cop(ies) dropped the caches; {profiler.counts['cache entries computed']} cache "
        f"entr(ies) computed"
    )

    if "peak_mb" in record:
        print(
            f"\n  peak GPU memory in use {record['peak_mb']} MB (reached in "
            f"{record['peak_reached_in']}); by category {record['peak_mb_by_category']}; "
            f"pool reserved {record['reserved_mb']} MB"
        )


def post_process(result, output, unit_cell):
    """Stages after the compute: the copy to the host and the indexing of the spots."""
    post = result["after_compute"] = {}
    try:
        start = time.perf_counter()
        host = output.to_cpu()
        synchronize()
        post["to_cpu_seconds"] = round(time.perf_counter() - start, 3)
        post["result_mb"] = round(host.array.nbytes / 1e6, 1)
        if args.index:
            from abtem.rotation_series import rotation_series_orientation_matrices

            matrices = rotation_series_orientation_matrices(angles, rotation_axis=axis)
            n_placeholders = len(host.ensemble_shape) - 1
            matrices = matrices.reshape(len(angles), *([1] * n_placeholders), 3, 3)
            start = time.perf_counter()
            indexed = host.index_diffraction_spots(
                cell=unit_cell,
                orientation_matrices=matrices,
                centering=config.get("centering", "P"),
                sg_max=config.get("sg_max", 1.0),
                g_max=config.get("g_max", 8.0) / 2,
                radius=config.get("integration_radius", 0.02),
            )
            indexed.crop(k_max=g_max_store)
            post["index_diffraction_spots_seconds"] = round(
                time.perf_counter() - start, 3
            )
    except Exception as error:
        post.update(failure(error))
    print(f"\n  after the compute: {post}", flush=True)


def main():
    unit_cell = load_atoms()
    print(
        f"{structure} ({unit_cell.get_chemical_formula()}) device={device} box={box} x "
        f"{thickness} A, tilt {list(np.round(angles, 3))} deg about {axis} deg, "
        f"{planes} exit planes, energy {energy} eV, precision {precision}, projection "
        f"{projection}, chunk size {chunk_size}, window {window_func} (margin {margin} A), "
        f"to_cpu {args.to_cpu}",
        flush=True,
    )
    results = {
        "structure": structure,
        "device": device,
        "box": list(box),
        "thickness": thickness,
        "angles": [float(a) for a in angles],
        "axis": axis,
        "planes": planes,
        "energy": energy,
        "precision": precision,
        "projection": projection,
        "chunk_size": chunk_size,
        "window_func": window_func,
        "margin": margin,
        "to_cpu": args.to_cpu,
        "abtem": abtem.__version__,
        "cases": [],
    }
    for sampling, slice_thickness in cases:
        results["cases"].append(run_case(unit_cell, sampling, slice_thickness))

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    first = cases[0]
    path = output / (
        f"ms3ded_{Path(structure).stem}_{device}_{box[0]:g}x{box[1]:g}x{thickness:g}"
        f"_s{first[0]:g}_t{first[1]:g}_{precision}.json"
    )
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
