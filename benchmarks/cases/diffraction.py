"""Convergent-beam diffraction."""

from abtem_bench.fixtures import multislice_kwargs, srtio3
from abtem_bench.registry import Tier, Tolerance, Variant, case


@case(
    "diffraction.cbed",
    tags={"diffraction", "multislice", "v1.1"},
    tiers={
        "quick": Tier(
            gpts=(256, 256), reps=(4, 4, 10), chunk=4, max_angle=80.0, timeout=120
        ),
        "standard": Tier(
            gpts=(1024, 1024), reps=(8, 8, 40), chunk=8, max_angle=80.0, timeout=900
        ),
        "large": Tier(
            gpts=(4096, 4096),
            reps=(20, 20, 120),
            chunk=8,
            max_angle=80.0,
            devices=("gpu",),
        ),
    },
    variants={
        "order1": Variant(algorithm_order=1, compare_as="default"),
    },
    outputs=("cbed",),
    tolerance=Tolerance(),
)
def diffraction_cbed(p, device):
    """A single 200 keV probe through SrTiO3; the CBED pattern spans several
    decades of intensity, which is what the relative-error mask is for."""
    from abtem import Potential, Probe

    potential = Potential(
        srtio3(p.reps), gpts=p.gpts, slice_thickness=1.0, device=device
    )
    probe = Probe(
        energy=200e3,
        semiangle_cutoff=20,
        gpts=p.gpts,
        extent=potential.extent,
        device=device,
    )
    kwargs = multislice_kwargs(p)

    def run():
        exit_wave = probe.multislice(potential, lazy=p.lazy, **kwargs)
        if p.lazy:
            exit_wave = exit_wave.compute()
        return {
            "cbed": exit_wave.diffraction_patterns(
                max_angle=p.max_angle, block_direct=False
            )
        }

    return run
