"""
Benchmark: marginal cost of running multiple angular (radial) detectors in a
single transition-potential (core-loss) multislice pass.

Before the shared-FFT cache (see ``Waves._cached_diffraction_pattern_fft`` in
abtem/waves.py), every radial detector (``AnnularDetector``,
``FlexibleAnnularDetector``, ``SegmentedDetector``, ``PixelatedDetector``)
recomputed its own 2D FFT of the wave function array via
``Waves.diffraction_patterns`` -> ``_diffraction_pattern``, even when several
such detectors are evaluated against the exact same (eager) array in the same
detection pass (as happens once per detector, per site batch, per slice in
``transition_potential_multislice_and_detect``). This benchmark reproduces
that scenario and reports the wall-clock cost of adding a second (and third)
``AnnularDetector`` to the pass.

Usage:
    python benchmarks/benchmark_shared_diffraction_pattern.py
    python benchmarks/benchmark_shared_diffraction_pattern.py --device gpu
"""

import argparse
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from ase.build import bulk

import abtem
from abtem import AnnularDetector, GridScan, Potential, Probe
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.multislice import transition_potential_multislice_and_detect

try:
    import cupy as cp
except ImportError:
    cp = None


def _disable_shared_diffraction_pattern_cache():
    """Monkeypatch away the shared-FFT optimization so the *same* installed
    abtem can also produce a true pre-optimization baseline, without needing
    a separate checkout/branch.

    ``Waves.diffraction_patterns`` calls
    ``abtem.core.fft.get_shared_diffraction_pattern_fft``, imported into the
    ``abtem.waves`` module namespace. Replacing that name there (not in
    ``abtem.core.fft``, which every other module's own already-bound import
    wouldn't see) makes every diffraction-pattern computation always
    recompute its FFT, regardless of whether a
    ``waves._share_diffraction_pattern_fft()`` block is active -- i.e.
    exactly the behavior before this optimization existed.
    """
    import abtem.waves as _waves_mod

    def _always_recompute(waves, normalize, compute):
        return compute()

    _waves_mod.get_shared_diffraction_pattern_fft = _always_recompute


def _make_scene(device, gpts=96, num_slices=3, n_transitions=16):
    # A small Si cell repeated/sliced to give exactly `num_slices` potential
    # slices, matching the "3 slices" scenario in the task description.
    atoms = bulk("Si", cubic=True)
    slice_thickness = atoms.cell[2, 2] / num_slices

    potential = Potential(
        atoms,
        gpts=gpts,
        slice_thickness=slice_thickness,
        device=device,
    )
    assert potential.num_slices == num_slices, potential.num_slices

    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((n_transitions, *potential.gpts))
        + 1j * rng.standard_normal((n_transitions, *potential.gpts))
    ).astype(np.complex64)
    if device == "gpu":
        array = cp.asarray(array)

    transition_potentials = TransitionPotentialArray(
        Z=14,
        array=array,
        energy=100e3,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )

    scan = GridScan(
        start=(0, 0),
        end=(1, 1),
        fractional=True,
        potential=potential,
        endpoint=False,
        gpts=(4, 4),
    )

    # Eager batch of probe wave functions at all 4x4 scan positions -- this
    # is exactly the ``waves`` that ``Waves.transition_potential_multislice``
    # would build internally and hand to
    # ``transition_potential_multislice_and_detect`` per dask block.
    waves = probe.build(scan=scan, lazy=False)

    return waves, potential, transition_potentials


def _run(waves, potential, transition_potentials, detectors):
    # Call the low-level driver directly rather than going through
    # ``Probe.transition_potential_scan``: with more than one detector, the
    # latter hits a pre-existing, unrelated bug in
    # ``Waves.transition_potential_multislice`` (it asserts the
    # per-transition-potential result is a single Waves/BaseMeasurements,
    # but ``apply_transform`` returns a plain list/ComputableList once there
    # is more than one detector). That bug is orthogonal to the optimization
    # under test here -- this driver is precisely the hot loop named in the
    # task ("transition_potential_multislice_and_detect ... calls
    # detector.detect(scattered_waves) in a loop over detectors").
    return transition_potential_multislice_and_detect(
        waves=waves.copy(),
        potential=potential,
        transition_potential=transition_potentials,
        detectors=detectors,
        sites=None,
    )


def _time_it(label, fn, n_repeats, n_warmup=1):
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    best = min(times)
    print(f"  {label:45s} best={best:.4f}s  all={['%.4f' % t for t in times]}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--gpts", type=int, default=96)
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--transitions", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.device == "gpu" and cp is None:
        print("cupy not available, skipping GPU benchmark")
        return

    waves, potential, tp = _make_scene(
        args.device, gpts=args.gpts, num_slices=args.slices,
        n_transitions=args.transitions,
    )

    print(
        f"device={args.device} gpts={args.gpts} slices={args.slices} "
        f"scan=4x4 transitions={args.transitions}\n"
    )

    configs = {
        "one AnnularDetector": [AnnularDetector(inner=0, outer=20)],
        "two AnnularDetectors": [
            AnnularDetector(inner=0, outer=20),
            AnnularDetector(inner=20, outer=40),
        ],
        "three AnnularDetectors": [
            AnnularDetector(inner=0, outer=20),
            AnnularDetector(inner=20, outer=40),
            AnnularDetector(inner=40, outer=60),
        ],
    }

    print("-- with shared-FFT cache (current behavior) --")
    cached = {
        label: _time_it(label, lambda d=detectors: _run(waves, potential, tp, d), args.repeats)
        for label, detectors in configs.items()
    }

    print("\n-- pre-cache baseline (each detector recomputes its own FFT) --")
    _disable_shared_diffraction_pattern_cache()
    baseline = {
        label: _time_it(label, lambda d=detectors: _run(waves, potential, tp, d), args.repeats)
        for label, detectors in configs.items()
    }

    labels = list(configs)
    print()
    print(f"  {'':24s} {'baseline':>10s} {'cached':>10s} {'speedup':>9s}")
    for label in labels:
        b, c = baseline[label], cached[label]
        print(f"  {label:24s} {b:9.4f}s {c:9.4f}s {b / c:8.2f}x")

    print()
    b1, b2, b3 = (baseline[l] for l in labels)
    c1, c2, c3 = (cached[l] for l in labels)
    print("  marginal cost of an extra detector, relative to 1 detector's own time:")
    print(
        f"    baseline : 2nd detector {(b2 - b1) / b1 * 100:+.1f}%   "
        f"3rd detector {(b3 - b2) / b1 * 100:+.1f}%"
    )
    print(
        f"    cached   : 2nd detector {(c2 - c1) / c1 * 100:+.1f}%   "
        f"3rd detector {(c3 - c2) / c1 * 100:+.1f}%"
    )


if __name__ == "__main__":
    main()
