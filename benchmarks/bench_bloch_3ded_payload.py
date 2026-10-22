"""Benchmark: a 3DED-style Bloch-wave rotation series, stage by stage.

Times the stages of a tilt series of one crystal -- structure factor and beam
selection, the dynamical diffraction patterns (the eigenproblem; this is the path of a
3DED multislice-vs-Bloch-wave comparison), and the exit waves synthesised with the
formula `plane_wave_expansion` replaced ("old") and with it ("new"). A stage that runs
out of memory is recorded as failed instead of ending the run, so a payload that does
not fit still reports the stages before it.

Usage
-----
    python benchmarks/bench_bloch_3ded_payload.py STRUCTURE [device] [g_max] [gpts] [n_tilts] [old] [sg_max] [precision]

    STRUCTURE  "baddeleyite" (ZrO2, P21/c) or the path of a file ASE reads (a CIF)
    device     cpu (default), gpu or mps
    g_max      reciprocal-space cutoff of the structure factor [1/A] (default 3.0)
    gpts       grid points of the exit waves per side (default 256)
    n_tilts    tilts about x, 1 degree apart, as a zxz rotation series (default 5)
    old        1 (default) also times the old synthesis; 0 skips it, which it must
               when its (beams x gpts**2 x thicknesses x 16 bytes) basis does not fit
    sg_max     maximum excitation error of the selected beams [1/A] (default 0.2); more
               beams are selected the larger it is
    precision  float32 (default) or float64

Production-size runs on a GPU (the results are also written to
benchmarks/results/, which is git-ignored):
    python benchmarks/bench_bloch_3ded_payload.py baddeleyite gpu 8.0 256 5 0
    python benchmarks/bench_bloch_3ded_payload.py kyanite.cif gpu 8.0 256 5 0
"""

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import abtem
from abtem.bloch import BlochWaves, StructureFactor, dynamical
from abtem.core.backend import asnumpy
from abtem.core.complex import complex_exponential

structure = sys.argv[1] if len(sys.argv) > 1 else "baddeleyite"
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
g_max = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
n_tilts = int(sys.argv[5]) if len(sys.argv) > 5 else 5
with_old = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
sg_max = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
precision = sys.argv[8] if len(sys.argv) > 8 else None
energy = 200e3
thicknesses = np.linspace(100, 500, 5)


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
        value = None
    print(f"  {name:<28} {record}", flush=True)
    return value


if precision is not None:
    abtem.config.set({"precision": precision})
atoms = load_atoms()
print(
    f"{structure} ({atoms.get_chemical_formula()}) device={device} g_max={g_max} "
    f"sg_max={sg_max} gpts={n}x{n} tilts={n_tilts} "
    f"precision={abtem.config.get('precision')}",
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
    "precision": abtem.config.get("precision"),
}
stages = results["stages"] = {}

bloch_waves = stage(
    "structure factor + beams",
    lambda: BlochWaves(
        structure_factor=StructureFactor(
            atoms, g_max=g_max, parametrization="lobato", thermal_sigma=0.078,
            device=device,
        ),
        energy=energy,
        sg_max=sg_max,
        use_wave_eq=True,
        device=device,
    ),
    stages,
)
if bloch_waves is not None:
    results["n_beams"] = len(bloch_waves.hkl)
    print(f"  {results['n_beams']} beams selected", flush=True)
    angles = np.array([[0.0, tilt, 0.0] for tilt in range(n_tilts)])
    rotated = bloch_waves.rotate("zxz", angles, degrees=True)

    stage(
        "diffraction patterns",
        lambda: rotated.calculate_diffraction_patterns(thicknesses).compute(),
        stages,
    )

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
path = output / f"3ded_{Path(structure).stem}_{device}_g{g_max}_sg{sg_max}_{n}_{n_tilts}_{results['precision']}.json"
path.write_text(json.dumps(results, indent=2))
print(f"written to {path}")
