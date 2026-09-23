"""STEM scans."""

from abtem_bench.fixtures import detector_angles, multislice_kwargs, silicon
from abtem_bench.registry import Tier, Tolerance, Variant, case


def _nominal_bytes(p) -> int:
    n_slices = int(round(5.43 * p.reps[2] / 2.0))
    return p.gpts[0] * p.gpts[1] * (4 * n_slices + 16 * int(p.max_batch))


@case(
    "stem.multidetector",
    tags={"stem", "multislice", "detectors", "v1.1"},
    tiers={
        "quick": Tier(
            gpts=(256, 256),
            reps=(4, 4, 8),
            scan=(6, 6),
            max_batch=8,
            chunk=4,
            timeout=300,
        ),
        "standard": Tier(
            gpts=(1024, 1024),
            reps=(8, 8, 15),
            scan=(16, 16),
            max_batch=8,
            chunk=8,
            timeout=1800,
        ),
        "large": Tier(
            gpts=(4096, 4096),
            reps=(20, 20, 75),
            scan=(8, 8),
            max_batch=8,
            chunk=8,
            devices=("gpu",),
        ),
    },
    variants={
        "order1": Variant(algorithm_order=1, compare_as="default"),
        "eager": Variant(lazy=False),
        "auto": Variant(max_batch="auto", chunk=None, flag=False),
    },
    outputs=("bf", "adf", "segmented"),
    tolerance=Tolerance(),
    nominal_bytes=_nominal_bytes,
    consistency=(("default", "eager"),),
)
def stem_multidetector(p, device):
    """Probe scan over Si with a bright-field disk, an annular dark-field ring
    and a segmented detector evaluated in one pass."""
    from abtem import AnnularDetector, GridScan, Potential, Probe, SegmentedDetector

    potential = Potential(
        silicon(p.reps), gpts=p.gpts, slice_thickness=2.0, device=device
    )
    probe = Probe(
        energy=200e3,
        semiangle_cutoff=20,
        gpts=p.gpts,
        extent=potential.extent,
        device=device,
    )
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=p.scan)
    a = detector_angles(probe)
    detectors = [
        AnnularDetector(inner=0, outer=0.2 * a),
        AnnularDetector(inner=0.5 * a, outer=0.9 * a),
        SegmentedDetector(
            inner=0.2 * a, outer=0.5 * a, nbins_radial=2, nbins_azimuthal=4
        ),
    ]
    kwargs = multislice_kwargs(p)

    def run():
        result = probe.scan(
            potential,
            scan=scan,
            detectors=detectors,
            lazy=p.lazy,
            max_batch=p.max_batch,
            **kwargs,
        )
        if p.lazy:
            result = result.compute()
        return list(result)

    return run
