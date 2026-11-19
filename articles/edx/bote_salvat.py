"""Bote-Salvat (2008) analytical parameterisation of electron-impact
inner-shell ionisation cross sections.

Data and formula ported from NIST's public-domain
`usnistgov/BoteSalvatICX.jl <https://github.com/usnistgov/BoteSalvatICX.jl>`_
(src/xione.jl, fetched 2026-09-05), which implements:

    D. Bote and F. Salvat, "Calculations of inner-shell ionization by electron
    impact with the distorted-wave and plane-wave Born approximations",
    Phys. Rev. A 77, 042701 (2008).

    D. Bote, F. Salvat, A. Jablonski, C.J. Powell, "Cross sections for
    ionization of K, L and M shells of atoms by impact of electrons and
    positrons with energies up to 1 GeV: Analytical formulas", At. Data Nucl.
    Data Tables 95, 871 (2009). (Full coefficient tables.)

`bote_salvat_2008.json` (this directory) holds the per-element coefficients,
parsed out of the Julia source with the parser this docstring's git history
points to. Used in EDX_SCOPING.md section 6c-ter for a direct, single-energy
absolute cross-section comparison against abTEM's transition-potential
theory -- the comparison that neither Bethe slope method (section 6c-bis) can
make for a heavy edge.

Subshell index: 1=K, 2=L1, 3=L2, 4=L3, 5=M1, 6=M2, 7=M3, 8=M4, 9=M5 -- the
same order as xraydb's IUPAC level names, which the ``str`` form of
``subshell`` below uses directly.
"""

import json
from pathlib import Path

import numpy as np

_TABLE = {
    int(k): v
    for k, v in json.loads(
        (Path(__file__).parent / "bote_salvat_2008.json").read_text()
    ).items()
}

_A0_CM = 5.291772108e-9  # Bohr radius [cm], as used in xione.jl
_REV_EV = 5.10998918e5  # electron rest energy [eV], as used in xione.jl

SUBSHELL_INDEX = {
    "K": 1, "L1": 2, "L2": 3, "L3": 4,
    "M1": 5, "M2": 6, "M3": 7, "M4": 8, "M5": 9,
}


def edge_energy(z: int, subshell: str | int) -> float:
    """Bote-Salvat's own tabulated edge energy [eV] for (Z, subshell)."""
    idx = SUBSHELL_INDEX[subshell] if isinstance(subshell, str) else subshell
    return _TABLE[z]["edge"][idx - 1]


def has_edge(z: int, subshell: str | int) -> bool:
    """Whether Bote-Salvat tabulates this (Z, subshell) at all.

    Some elements omit deep-valence subshells (e.g. Cu has no M4/M5 entry,
    since 3d is valence there, not core) -- this is not a data gap, just
    outside what "inner-shell ionisation" was tabulated for.
    """
    idx = SUBSHELL_INDEX[subshell] if isinstance(subshell, str) else subshell
    return idx <= len(_TABLE[z]["edge"])


def ionization_cross_section(
    z: int,
    subshell: str | int,
    energy_eV: float,
    edge_energy_eV: float | None = None,
) -> float:
    """Total ionisation cross section [cm^2] for a single atom.

    Direct port of xione.jl's ``ionizationcrosssection``.

    Parameters
    ----------
    z : int
        Atomic number, 1-99.
    subshell : str or int
        IUPAC level name ("K", "L1", ..., "M5") or Bote-Salvat's 1-9 index.
    energy_eV : float
        Incident electron kinetic energy [eV].
    edge_energy_eV : float, optional
        Edge energy [eV]. Defaults to Bote-Salvat's own tabulated value; pass
        an independent one (e.g. from xraydb) to test sensitivity to the
        edge-energy convention.
    """
    idx = SUBSHELL_INDEX[subshell] if isinstance(subshell, str) else subshell
    datum = _TABLE[z]
    if edge_energy_eV is None:
        edge_energy_eV = datum["edge"][idx - 1]

    over_v = energy_eV / edge_energy_eV
    if over_v <= 1.0:
        return 0.0

    i = idx - 1
    if over_v <= 16.0:
        a = datum["A"][i]
        opu = 1.0 / (1.0 + over_v)
        ffitlo = a[0] + a[1] * over_v + opu * (a[2] + opu**2 * (a[3] + opu**2 * a[4]))
        xione = (over_v - 1.0) * (ffitlo / over_v) ** 2
    else:
        beta2 = (energy_eV * (energy_eV + 2.0 * _REV_EV)) / (energy_eV + _REV_EV) ** 2
        x = np.sqrt(energy_eV * (energy_eV + 2.0 * _REV_EV)) / _REV_EV
        g = datum["G"][i]
        ffitup = (
            ((2.0 * np.log(x) - beta2) * (1.0 + g[0] / x))
            + g[1]
            + g[2] * np.sqrt(_REV_EV / (energy_eV + _REV_EV))
            + g[3] / x
        )
        factr = datum["Anlj"][i] / beta2
        xione = (factr * over_v / (over_v + datum["Be"][i])) * ffitup

    return 4.0 * np.pi * _A0_CM**2 * xione


if __name__ == "__main__":
    # Sanity checks -- see EDX_SCOPING.md 6c-ter for the abTEM comparison.
    assert ionization_cross_section(29, "K", 8000.0) == 0.0
    assert ionization_cross_section(29, "K", 9500.0) > 0.0

    e_k = edge_energy(29, "K")
    s_lo = ionization_cross_section(29, "K", e_k * 15.999)
    s_hi = ionization_cross_section(29, "K", e_k * 16.001)
    print(
        f"Cu K seam at overV=16: {s_lo:.6e} vs {s_hi:.6e} "
        f"({abs(s_lo - s_hi) / s_lo:.4%} jump)"
    )

    print(f"\n{'E0 (keV)':>9} {'sigma (cm^2)':>13} {'sigma (A^2)':>13}")
    for E0_keV in [10, 20, 50, 100, 200, 300, 1000]:
        s = ionization_cross_section(29, "K", E0_keV * 1e3)
        print(f"{E0_keV:9d} {s:13.4e} {s * 1e16:13.4e}")
