"""Benchmark: a 3DED-style Bloch-wave rotation series, stage by stage.

Times the stages of a tilt series of one crystal -- structure factor and beam
selection, the dynamical diffraction patterns (the eigenproblem; this is the path of a
3DED multislice-vs-Bloch-wave comparison), and the exit waves synthesised with the
formula `plane_wave_expansion` replaced ("old") and with it ("new"). A stage that runs
out of memory is recorded as failed instead of ending the run, so a payload that does
not fit still reports the stages before it.

The settings can come from a py3DED simulation config (a JSON file, whose `cif_file`
is read relative to it): `energy`, `thermal_sigmas`, `centering`, `sg_max`, `g_max`,
`use_wave_eq`, `precision`, `rotation_axis_orientation` (the azimuth of the tilt axis),
`rotation_min`, `rotation_max`, `thickness` and `exit_planes`. Every option below
overrides the config; without a config the defaults are those of the options.

Usage
-----
    python benchmarks/bench_bloch_3ded_payload.py STRUCTURE [options]

    STRUCTURE  "baddeleyite" (ZrO2, P21/c), the path of a file ASE reads (a CIF), or
               the path of a py3DED config (.json)

    --device cpu|gpu|mps     (default cpu; a config's "device" is not used)
    --tilts N                number of tilts, evenly spread over the tilt range, as a
                             zxz rotation series (axis, tilt, -axis) (default 5; a
                             config has hundreds)
    --tilt-range MIN MAX     [deg] (default 0 4, or the config's)
    --axis DEG               azimuth of the tilt axis [deg] (default 0, or the config's)
    --gpts N                 grid points per side of the exit waves (default 256)
    --split                  split the diffraction-pattern stage into the structure
                             matrix, the eigendecomposition (`eigh`) and the rest
                             (synchronising the device around each, so use it for
                             one diagnostic run, not for the totals; the times are sums over
                             the threads that run the tilts, so with several tilts they can
                             exceed the wall time: read them as proportions)
    --old                    also time the old synthesis, whose (beams x gpts**2 x
                             thicknesses x 16 bytes) basis must fit
    --g-max, --sg-max, --energy, --thermal-sigma, --centering, --precision,
    --thickness, --planes    as the config keys (defaults: 3.0 1/A, 0.2 1/A, 200e3 eV,
                             0.078 A, F, float32, 500 A, 5)

Production-size runs on a GPU (the results are also written to
benchmarks/results/, which is git-ignored):
    python benchmarks/bench_bloch_3ded_payload.py ../py3DED/examples/realistic_cases/Al2SiO5_Kyanite.json --device gpu --tilts 3
"""

import argparse
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import abtem
from abtem.bloch import BlochWaves, StructureFactor, dynamical
from abtem.core.backend import asnumpy
from abtem.core.complex import complex_exponential

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("structure")
parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "mps"])
parser.add_argument("--tilts", type=int, default=5)
parser.add_argument("--tilt-range", type=float, nargs=2)
parser.add_argument("--axis", type=float)
parser.add_argument("--gpts", type=int, default=256)
parser.add_argument("--old", action="store_true")
parser.add_argument("--split", action="store_true")
parser.add_argument("--g-max", type=float)
parser.add_argument("--sg-max", type=float)
parser.add_argument("--energy", type=float)
parser.add_argument("--thermal-sigma", type=float)
parser.add_argument("--centering")
parser.add_argument("--precision", choices=["float32", "float64"])
parser.add_argument("--thickness", type=float)
parser.add_argument("--planes", type=int)
args = parser.parse_args()

config = {}
structure = args.structure
if structure.endswith(".json"):
    config = json.loads(Path(structure).read_text())
    structure = str(Path(structure).parent / config["cif_file"])


def setting(option, key, default):
    """The option if given, else the config's value of `key`, else the default."""
    value = getattr(args, option)
    return value if value is not None else config.get(key, default)


device = args.device
n = args.gpts
n_tilts = args.tilts
with_old = args.old
g_max = setting("g_max", "g_max", 3.0)
sg_max = setting("sg_max", "sg_max", 0.2)
energy = setting("energy", "energy", 200e3)
thermal_sigma = setting("thermal_sigma", "thermal_sigmas", 0.078)
centering = setting("centering", "centering", "F")
precision = setting("precision", "precision", "float32")
axis = setting("axis", "rotation_axis_orientation", 0.0)
tilt_range = args.tilt_range or (
    config.get("rotation_min", 0.0),
    config.get("rotation_max", 4.0),
)
thickness = setting("thickness", "thickness", 500.0)
planes = setting("planes", "exit_planes", 5)
use_wave_eq = config.get("use_wave_eq", True)
thicknesses = np.linspace(thickness / planes, thickness, planes)
tilts = np.linspace(tilt_range[0], tilt_range[1], n_tilts)


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


def old_plane_wave_expansion(amplitudes, g, x, y, z):
    """The plane wave basis and reduction that `plane_wave_expansion` replaced."""
    plane_waves_x = complex_exponential(
        2 * np.pi * g[None, :, 0, None, None] * x[None, None, :, None]
    )
    plane_waves_y = complex_exponential(
        2 * np.pi * g[None, :, 1, None, None] * y[None, None, None, :]
    )
    plane_waves_z = complex_exponential(
        2 * np.pi * g[None, :, 2, None, None] * z[..., None, None, None]
    )
    basis = plane_waves_x * plane_waves_y * plane_waves_z
    if not z.ndim:
        basis = basis[0]
    return (amplitudes[..., None, None] * basis).sum(-3)


new_plane_wave_expansion = dynamical.plane_wave_expansion


def synchronize():
    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif device == "mps":
        import torch

        torch.mps.synchronize()


@contextmanager
def peak_gpu_memory(record):
    """The peak of the CuPy pool in use [MB] over the block, in record['peak_mb']."""
    if device != "gpu":
        yield
        return
    import cupy as cp

    pool = cp.get_default_memory_pool()

    class PeakHook(cp.cuda.MemoryHook):
        peak = 0

        def malloc_postprocess(self, **kwargs):
            self.peak = max(self.peak, pool.used_bytes())

    hook = PeakHook()
    synchronize()
    hook.peak = pool.used_bytes()
    with hook:
        yield
        synchronize()
    record["peak_mb"] = round(hook.peak / 1e6)


@contextmanager
def split_timers(record):
    """Record the time spent in the structure matrix and in `eigh` over the block, in
    record['split'] [s], as sums over every call; the rest is the total minus these."""
    if device not in ("cpu", "gpu"):
        yield
        return

    totals = {"structure matrix": 0.0, "eigh": 0.0}

    def timed(name, function):
        def wrapper(*args, **kwargs):
            synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            synchronize()
            totals[name] += time.perf_counter() - start
            return result

        return wrapper

    # dynamical._eigh, not xp.linalg.eigh: the eigendecomposition calls the private
    # in-place solvers of CuPy and SciPy where it can
    structure_matrix = dynamical.calculate_structure_matrix
    eigh = dynamical._eigh
    dynamical.calculate_structure_matrix = timed("structure matrix", structure_matrix)
    dynamical._eigh = timed("eigh", eigh)
    try:
        yield
    finally:
        dynamical.calculate_structure_matrix = structure_matrix
        dynamical._eigh = eigh
        record["split"] = {name: round(value, 3) for name, value in totals.items()}


def stage(name, function, results, warm_up=False):
    """Time `function`, recording failures (out of memory) instead of raising."""
    record = {}
    results[name] = record
    try:
        if warm_up:
            function()
            synchronize()
        with peak_gpu_memory(record):
            start = time.perf_counter()
            value = function()
            synchronize()
            record["seconds"] = round(time.perf_counter() - start, 3)
    except Exception as error:  # out of memory has a different class per backend
        record["failed"] = f"{type(error).__name__}: {str(error)[:200]}"
        # where in the stage it failed: the abtem frames (the innermost frames of an
        # out-of-memory error are the allocator's), then the innermost frame
        frames = traceback.extract_tb(error.__traceback__)
        keep = [f for f in frames if "/abtem/" in f.filename.replace("\\", "/")]
        record["where"] = [
            f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in keep[-8:] + frames[-1:]
        ]
        value = None
    print(f"  {name:<28} {record}", flush=True)
    return value


abtem.config.set({"precision": precision})
atoms = load_atoms()
print(
    f"{structure} ({atoms.get_chemical_formula()}) device={device} g_max={g_max} "
    f"sg_max={sg_max} gpts={n}x{n} tilts={n_tilts} in {tuple(tilt_range)} deg about "
    f"{axis} deg, {planes} thicknesses to {thickness} A, centering {centering}, "
    f"energy {energy} eV, precision {precision}",
    flush=True,
)
abtem.config.set({"diagnostics.progress_bar": False, "diagnostics.task_progress": False})
results = {
    "structure": structure,
    "device": device,
    "g_max": g_max,
    "sg_max": sg_max,
    "gpts": n,
    "n_tilts": n_tilts,
    "precision": precision,
    "energy": energy,
    "centering": centering,
    "thermal_sigma": thermal_sigma,
    "axis": axis,
    "tilt_range": list(tilt_range),
    "thickness": thickness,
    "planes": planes,
}
stages = results["stages"] = {}

bloch_waves = stage(
    "structure factor + beams",
    lambda: BlochWaves(
        structure_factor=StructureFactor(
            atoms,
            g_max=g_max,
            parametrization="lobato",
            thermal_sigma=thermal_sigma,
            centering=centering,
            device=device,
        ),
        energy=energy,
        sg_max=sg_max,
        use_wave_eq=use_wave_eq,
        device=device,
    ),
    stages,
)
if bloch_waves is not None:
    results["n_beams"] = len(bloch_waves.hkl)
    print(f"  {results['n_beams']} beams selected", flush=True)
    angles = np.array([[axis, tilt, -axis] for tilt in tilts])
    rotated = bloch_waves.rotate("zxz", angles, degrees=True)

    stage(
        "diffraction patterns",
        lambda: rotated.calculate_diffraction_patterns(thicknesses).compute(),
        stages,
    )
    if args.split:
        # a second, instrumented run: the timers synchronise the device, which the
        # run above must not pay for
        split_record = {}
        with split_timers(split_record):
            stage(
                "diffraction patterns (split)",
                lambda: rotated.calculate_diffraction_patterns(thicknesses).compute(),
                stages,
            )
        stages["diffraction patterns (split)"].update(split_record)
        print(f"  split: {split_record.get('split')}", flush=True)

    for label, function in (
        (("old", old_plane_wave_expansion) if with_old else (None, None)),
        ("new", new_plane_wave_expansion),
    ):
        if label is None:
            continue
        dynamical.plane_wave_expansion = function
        try:
            stage(
                f"exit waves, {label} synthesis",
                lambda: rotated.calculate_exit_waves(
                    thicknesses, gpts=(n, n), lazy=False
                ),
                stages,
                warm_up=True,
            )
        finally:
            dynamical.plane_wave_expansion = new_plane_wave_expansion

output = Path(__file__).parent / "results"
output.mkdir(exist_ok=True)
path = output / f"3ded_{Path(structure).stem}_{device}_g{g_max}_sg{sg_max}_{n}_{n_tilts}_{precision}.json"
path.write_text(json.dumps(results, indent=2))
print(f"written to {path}")
