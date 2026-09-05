# Scoping: EDX (energy-dispersive X-ray) simulation in abTEM

Status: scoping only, nothing implemented. Written against `dev` @ 6dd56250.

## Summary

The transition-potential machinery in `abtem/inelastic/core_loss.py` already contains
almost everything EDX needs. The two genuinely new pieces are:

1. **Atomic data** — fluorescence yields, emission-line energies and branching ratios,
   Coster–Kronig probabilities, and mass-attenuation coefficients. All of this is
   available from **`xraydb`** (MIT, pure Python, bundled SQLite). Verified below.
2. **An X-ray detector class** — solid angle + energy-dependent efficiency. This is the
   *easy* part; it is ~150 lines and a lookup table.

The two things that are **not** easy, and which dominate the work, are physics gaps in
the existing core-loss path rather than anything to do with the detector:

3. **Energy integration.** EDX collects the *whole* edge, while abTEM evaluated the
   transition potential at a single continuum energy. Implemented as `EnergyIntegral`
   (§4). Verifying the normalisation this rests on turned up a real bug in the
   continuum wavefunction that also affects existing EELS results (§6).
4. **Angular truncation.** EDX collects *all* scattering angles. Measured below: a
   30 mrad EELS aperture captures only ~33 % of the total ionisation at 100 keV; ~200 mrad
   is needed for 99 %. That places a hard constraint on real-space sampling.

There is a large architectural payoff hiding in (4), described in §5.

---

## 1. Fluorescence-yield data: `xraydb`

`xraydb` 4.5.8 — https://github.com/xraypy/XrayDB

| | |
|---|---|
| Licence | MIT (compatible with abTEM's GPL-3.0-or-later) |
| Purity | pure Python wheel (`py3-none-any`), no compiler |
| Deps | numpy, scipy, sqlalchemy, platformdirs |
| Size | 10 MB (bundled `xraydb.sqlite`, works fully offline) |
| Source data | Elam, Ravel & Sieber (2002); Chantler; Krause |

### Verified values (this machine, xraydb 4.5.8)

Per-**subshell** fluorescence yields, i.e. exactly the granularity `SubshellTransitions`
needs:

```
        K        L1       L2       L3
C    0.0014
O    0.0058
Si   0.0429   0.00003  0.00037  0.00038
Ti   0.2184   0.00047  0.0011   0.0015
Fe   0.3510   0.0010   0.0036   0.0063
Cu   0.4411   0.0016   0.0057   0.0110
Ag   0.8219   0.016    0.051    0.052
Au   0.9808   0.107    0.334    0.320
```

These match the standard Krause/Elam tabulation (ω_K(Cu) = 0.441, ω_K(Fe) = 0.351).

Emission lines with branching ratios, normalised to 1 per initial level:

```
Cu Ka3  7882.3 eV  0.000306   K -> L1
Cu Ka2  8026.7 eV  0.294269   K -> L2
Cu Ka1  8046.3 eV  0.577108   K -> L3
Cu Kb3  8901.7 eV  0.043517   K -> M2
Cu Kb1  8903.9 eV  0.084010   K -> M3
Cu Kb5  8974.0 eV  0.000790   K -> M4,5
```

Coster–Kronig: `ck_probability('Fe','L2','L3') = 0.42`, `('Fe','L1','L3') = 0.696`.
Natural line widths: `core_width('Cu','K') = 1.55 eV`.
Mass attenuation for the detector model: `mu_elam(element, energies)` → cm⁻¹.

### API surface actually needed

```python
xraydb.xray_edges(el)          # -> {'K': XrayEdge(energy, fyield, jump_ratio), ...}
xraydb.xray_lines(el, level)   # -> {'Ka1': XrayLine(energy, intensity, init, final), ...}
xraydb.ck_probability(el, i, f)
xraydb.core_width(el, edge)
xraydb.mu_elam(el, energies)   # detector window / dead-layer / active-layer model
```

### Alternatives considered

- **xraylib** — the other standard. Line intensities from Scofield (1974), generally
  better regarded than Elam's interpolations for relative line intensities. But it is a
  SWIG-wrapped C library: a binary dependency, conda-forge only in practice. Worse fit
  for abTEM's pip-installable story.
- **fisx / PyMca5** — full EDX quantification including detector response and secondary
  fluorescence. C++ dependency, much heavier, and most of it (matrix corrections,
  fitting) is irrelevant here.
- **hyperspy** — has an EDS line database but not per-subshell fluorescence yields, and
  depending on hyperspy from abTEM inverts the natural dependency direction.
- **Bundle the numbers ourselves** — the subset actually needed (ω, line energies,
  branching ratios, CK probabilities for Z = 1–98) is ~50 kB of JSON. There is precedent
  in `abtem/parametrizations/data/*.json` and `MANIFEST.in` already ships `*.json`. This
  removes the dependency entirely at the cost of losing upstream updates.

**Recommendation:** add `xraydb` to the existing `gpaw` extra, behind a thin
adapter module so that xraylib or a bundled table can be swapped in without
touching the detector code. A separate `edx` extra would be redundant: core-loss
already requires GPAW for the atomic wave functions, so anyone running EDX has
that extra installed already.

---

## 2. The physics chain

For an emission line *L* of element *Z*, per incident electron:

```
I_L = P_ion(subshell)  x  omega(subshell)  x  b_L  x  (Omega / 4pi)  x  eff(E_L)
```

- `P_ion` — ionisation probability from the transition potentials (abTEM), **integrated
  over all scattering angles and over all energy losses above threshold**.
- `omega` — fluorescence yield of the ionised subshell (xraydb).
- `b_L` — radiative branching ratio into line *L* (xraydb, sums to 1 per initial level).
- `Omega / 4pi` — geometric collection fraction, assuming isotropic emission.
- `eff(E_L)` — detector efficiency at the line energy.

Deliberately excluded, per the stated assumptions, but each worth a docstring note:
self-absorption in the specimen; emission anisotropy from aligned p-vacancies (small but
real in oriented crystals); secondary fluorescence.

### The subshell mapping problem

`SubshellTransitions(Z, n, l)` is non-relativistic — a single `2p` transition potential.
`xraydb` is j-resolved: L1 (2s), L2 (2p½), L3 (2p³⁄₂). Two consequences:

- ω for the abTEM `2p` channel must be the statistically weighted average
  `(1·ω_L2 + 2·ω_L3)/3`, and lines from L2 and L3 both draw on that one ionisation
  probability with the appropriate weights.
- Coster–Kronig transfer (L1→L2, L1→L3, L2→L3) redistributes vacancies *before* they
  radiate, and is large (0.42 for Fe L2→L3). It must be applied to the vacancy
  populations, not to the yields. For K edges this is all trivial; for L and M it is not.

### Detector needs to know the edge

`TransitionPotentialArray.scatter()` builds the scattered waves from
`waves._copy_kwargs()`, so the transition potential's `metadata` (`{"Z", "n", "l"}`) is
**not** propagated onto the scattered waves. A detector therefore cannot introspect which
edge produced them. Two options: inject the transition metadata into the scattered waves'
`metadata` dict in `scatter()` (small, clean, and useful beyond EDX), or require the user
to construct `XrayDetector(element=..., line=...)` consistently with the
`SubshellTransitions` — error-prone. Prefer the former.

---

## 3. Detector efficiency model — prototyped and working

A standard windowless/windowed SDD stack modelled from tabulated attenuation
coefficients. Note that `mu_elam` returns the *mass* attenuation coefficient in
cm^2/g; `material_mu` returns the linear coefficient in 1/cm and is the one to use:

```python
T = lambda f, t_cm, kind: np.exp(-xraydb.material_mu(f, E, kind=kind) * t_cm)
window = T('Al', 30e-7, 'total') * T('Si', 50e-7, 'total')   # contact + dead layer
active = 1 - T('Si', 450e-4, 'photo')                        # full-energy peak only
```

Attenuating layers use the total cross-section; the active layer uses
photo-absorption, since only a photo-absorption event contributes to the
full-energy peak.

Output (textbook-correct shape, reproduced by the unit tests):

```
                    windowless SDD    + 8 um Be window
C  K    277 eV          0.5514            0.0000
N  K    392 eV          0.7605            0.0000
O  K    525 eV          0.8725            0.0027
F  K    677 eV          0.9299            0.0529
Na K   1041 eV          0.9753            0.4411
Al K   1487 eV          0.9903            0.7537
Si K   1740 eV          0.9710            0.8197
P  K   2013 eV          0.9512            0.8535
Ti K   4510 eV          0.9941            0.9853
Cu K   8040 eV          0.9975            0.9958
Mo K  17479 eV          0.4768            0.4766
Ag K  22163 eV          0.2682            0.2681
```

The Si-K dip at 1740 eV (0.9903 at Al K down to 0.9710) is the Si absorption edge
of the dead layer appearing correctly, and the Be window cutting off below ~1 keV
matches real Be-window detectors. Both are good smoke tests for the
implementation.

Proposed API:

```python
XrayDetector(
    solid_angle=0.7,                 # sr
    efficiency=1.0,                  # float | callable(E_eV) | (E, eff) table
    lines="K",                       # or ("Ka",), ("Ka1","Ka2"), "all"
    ...
)

XrayDetector.from_sdd(
    solid_angle=0.7,
    window={"Be": 8e-4},             # cm
    contact={"Al": 30e-7},
    dead_layer={"Si": 50e-7},
    active_layer={"Si": 450e-4},
)
```

`from_sdd` is the only place `xraydb.mu_elam` is touched, so a user supplying their own
measured efficiency curve needs no optional dependency at all.

---

## 4. Gap 1: energy integration -- IMPLEMENTED

`EnergyIntegral(stop, start=1.0, num=16)` places Gauss-Legendre nodes in
`log(epsilon)` and supplies quadrature weights; `SubshellTransitions(..., epsilon=
EnergyIntegral(...))` then builds one transition per (l', m_l', epsilon). Because
distinct final states add incoherently, the weight is folded into the transition
potential *amplitude* as its square root, so every downstream consumer -- scatter,
detectors, PRISM -- integrates over the edge with no code change of its own.

The quadrature is exact to 1e-8 for constants, `1/eps` and power laws, which are
the shapes the integrand actually takes.

### Measured epsilon-dependence (Si K, 100 keV, 10 A / 256^2, angle-unrestricted)

```
eps (eV)    int |H|^2      rel.     wall time
     1.0    3.436e-02     1.000       3.6 s
     5.0    2.846e-02     0.828       2.5 s
    10.0    2.955e-02     0.860       2.5 s
    25.0    2.961e-02     0.862       2.6 s
    50.0    2.908e-02     0.846       2.5 s
   100.0    2.707e-02     0.788       2.4 s
   200.0    2.266e-02     0.660       2.8 s
   400.0    1.676e-02     0.488       2.8 s
   800.0    9.827e-03     0.286       2.9 s
  1600.0    4.192e-03     0.122       2.9 s
```

The integrand is nearly flat out to ~100 eV and decays only slowly beyond -- the
log-log slope reaches just -1.2 by eps = 1600 eV. The integral is dominated by the
tail, not the near-edge: truncating at 100 eV captures only ~14 % of the
1-1600 eV integral. Any EDX implementation that reuses an EELS-window-sized
epsilon range will be wrong by nearly an order of magnitude.

### Validation

Against 40 dense single-epsilon builds integrated by trapezoid over
1-4000 eV (Si K, 100 keV):

```
brute force, 40 single-eps builds   2.4226e+01   (105 s)
EnergyIntegral(num= 4)              2.3549e+01   -2.80%     (9 s)
EnergyIntegral(num= 8)              2.4106e+01   -0.50%    (16 s)
EnergyIntegral(num=16)              2.4096e+01   -0.54%    (39 s)
EnergyIntegral(num=24)              2.4081e+01   -0.60%    (52 s)
```

Converged by `num=8`, and the residual 0.5 % is the brute-force trapezoid's own
error on a convex integrand, not the quadrature's. **num = 8 to 16 is the
practical recommendation**, at a few times the cost of a single-energy build.

`stop` must still be convergence-tested per edge: the integrand's tail is long,
and the value above is for `stop = 4000 eV` on a 1839 eV edge.

## 5. Gap 2: angular truncation — and the architectural payoff

### Measured (Si K, ε = 25 eV, 100 keV, 0.039 Å sampling, antialias cutoff 314 mrad)

```
collection aperture    fraction of total ionisation
     10 mrad                    3.3 %
     20 mrad                   15.9 %
     30 mrad                   32.7 %
     50 mrad                   59.8 %
    100 mrad                   89.3 %
    200 mrad                   99.3 %
    314 mrad (full)           100.0 %
```

So EDX genuinely requires the full grid, and **the antialias cutoff must exceed ~200 mrad**
— roughly 0.04 Å real-space sampling at 100 keV. A typical STEM simulation at 0.05–0.1 Å
sampling would truncate the ionisation cross-section by 5–15 %. This is invisible in EELS
because a 30 mrad aperture sits far inside the grid cutoff. It gets worse at higher ε, so
the ε-integrated requirement is stricter still. This needs to be a documented convergence
parameter with a runtime warning.

### The payoff -- IMPLEMENTED

For an **angle-unrestricted** detector, Parseval turns the reciprocal-space
integral into a real-space one:

```
sum_f  int dk |FT(H_f psi)|^2   ==   int dr |psi(r)|^2 * [ sum_f |H_f(r)|^2 ]
```

**Verified numerically** to 1 part in 1e5 against `TransitionPotentialArray.scatter`.
The bracketed quantity is a single real-valued *effective local ionisation
potential* `mu(r)`, independent of the probe, exposed as
`TransitionPotentialArray.effective_ionization_potential(sites)` and consumed by
`effective_ionization_multislice_and_detect`.

**One subtlety that is easy to get wrong.** `mu` must be built by shifting the
transition potential *amplitudes* to the site positions and summing the
intensities afterwards, exactly as `scatter` does. Summing the intensity first
and then shifting it -- one convolution with the site structure factor, which
would genuinely be O(1) per slice -- is wrong: the intensity has twice the
bandwidth of the amplitude, so a sub-pixel Fourier shift of it rings, and `mu`
picks up negative values of about -9 % of its maximum. Building `mu` therefore
costs one FFT per site per final state in each slice.

That is still a large win, because `mu` **does not depend on the probe
position**: it is built once per slice and reused for the whole scan, and there
are no scattered waves to propagate. Measured against the existing
scattered-wave route (96 Si atoms, 12 slices, 128^2, single-channel):

```
   scan   n_transitions   scattered-wave   effective mu   speedup
   4x4                4          2.40 s         0.45 s      5.3x
   4x4               32          9.24 s         0.61 s     15.1x
   8x8                4          4.79 s         0.33 s     14.3x
   8x8               32         34.51 s         0.75 s     46.1x
```

The advantage grows with both the scan size and the number of final states, so
for a realistic EDX run -- a 32x32 scan with 8 quadrature nodes x 4 channels x m
sublevels -- it is several hundredfold. Thickness resolution falls out for free
from the slice-wise accumulation.

**Double-channelling is not an approximation here.** With no angular
restriction, the subsequent elastic propagation of the ejected-state wave is
unitary and cannot change the total count. It matters for EELS only because an
aperture breaks that.

### Agreement with the existing route

Against `transition_potential_multislice` with `AnnularDetector(0, None)`,
single-channel, on a band-limited transition potential:

```
t (A)      effective mu     scattered wave    ratio
  0.0        0.000e+00        0.000e+00         --
  4.1        2.2203e-10       2.2198e-10      1.00024
  8.2        4.5928e-10       4.5913e-10      1.00033
 12.2        7.0045e-10       7.0015e-10      1.00043
 16.3        9.4801e-10       9.4749e-10      1.00054
```

The residual is the antialiasing truncation, which only the scattered-wave route
suffers -- so the real-space value is the more accurate of the two. Note that
with a *white-noise* test potential the two differ by ~3x, because the
antialiasing disc covers only pi/9 of the square grid; a realistic transition
potential decays with k, which is what makes them agree.

## 6. Continuum normalisation -- VERIFIED AND FIXED

EDX is an inherently *absolute* technique, so the normalisation of
`TransitionPotential.build()` has to be trusted in a way relative EELS mapping
never required.

**The convention is sound.** The continuum states are energy-normalised,
`u(r) -> sin(kr + delta) / sqrt(pi k)` with the energy in Rydberg. Measured on
the returned wavefunctions, `A * sqrt(pi k) = 1.0000` -- so the transition
potential is genuinely differential in energy loss, and integrating over epsilon
is meaningful. `_calculate_overlap_integral` already divides by
`sqrt(units.Rydberg)`, so the form factor is per **eV**, and the quadrature
weights (also in eV) apply directly with no further conversion.

**The implementation was wrong in two regimes.** The asymptotic amplitude was
taken as `max(u)` over a fixed 20 Bohr grid. That fails when

1. the grid spans less than one oscillation, i.e. low `epsilon`, and
2. `l' >= 2`, where the envelope overshoots its asymptotic value near the
   turning point of the centrifugal barrier, so the maximum is attained well
   inside the atom -- at eps = 400 eV, l' = 3 the maximum sits at r = 0.76 Bohr
   with envelope 3.67 against an asymptotic 3.31.

Both are fixed: the grid now grows until its outer quarter is asymptotically
free, and the amplitude is read from the envelope `sqrt(u^2 + (u'/k)^2)` there.
Normalisation is now exact (1.000000) for every eps and l' tested, and the grid
only grows where it must -- eps >= 25 eV keeps the original 1e6 points.

### Effect on existing results (new / old intensity, Si)

```
   eps     l'=0     l'=1     l'=2     l'=3
   1.0    0.990    0.832    1.337    1.356
   5.0    1.040    1.015    1.177    1.324
  25.0    0.999    1.001    1.008    1.248
 100.0    1.000    1.000    1.045    1.222
 400.0    1.000    1.000    1.060    1.229
1600.0    1.000    1.000    1.093    1.239
```

- **K edges with `order=1`** use l' = 0, 1 only, so they are unchanged for
  eps >= 25 eV -- but at the **default `epsilon=1.0`** the dominant l' = 1
  channel was ~17 % too strong.
- **L edges with `order=1`** gain 4.5-9 % through the l' = 2 dipole channel.
- **M edges, or any edge with `order=2`**, gain ~22-24 % through l' = 3.

### 6c. Absolute validation against photoabsorption

Bethe asymptotics give `sigma * T = 4 pi a0^2 R M^2 ln(T) + const`, so the slope
of `sigma*T` against `ln(T)` is `4 pi a0^2 R M^2` with

    M^2 = integral (R/E) (df/dE) dE,    df/dE = sigma_photo(E) / (1.0975e-16 cm^2 eV)

`M^2` is computable from tabulated photoabsorption alone, entirely independently
of abTEM. It converges by `E_max = 20 x E_K`.

**The reference is sound across Z.** The extracted K-shell oscillator strength
`f_K` is 1.65 (C), 1.47 (Si), 1.31 (Ti), 1.24 (Cu) -- all approaching the ideal
2, varying smoothly, and nowhere near a factor of ten.

**Energy dependence: validated.** `sigma*T` versus `ln(T)` is linear to 0.2-1 %
once the simulation is converged, which is the Bethe form.

**Absolute scale.** A first pass suggested a factor-ten Z trend (C 3.92, Si 0.92,
Cu 0.38). Almost all of that turned out to be artifacts of the *validation
setup*, not of abTEM:

- **Light elements need a large cell.** An inelastic event is delocalised over
  `b ~ hbar v / dE`, which for the carbon K edge is 5 A at 300 keV. In an 8 A
  cell carbon's cross-section is 6 % high at 60 keV but **41 % high at 300 keV**,
  so the cell truncation biases the *slope*, not merely the magnitude. Converging
  the cell moves carbon's ratio from 4.82 (8 A) to 1.48 (24 A) to 1.41 (32 A),
  with the fit residual falling from 8.5 % to 0.7 %.
- **Heavy elements need a high `T/E_K`.** Bethe asymptotics are not reached at
  `T/E_K ~ 6-17`. Refitting copper over `T/E_K = 8.6-27.3` moves its ratio from
  0.375 to 0.581, and it is still climbing.

Converged picture, `order=1`:

```
 element   ratio abTEM / xraydb   note
       C          1.41            32 A cell, T/E_K 200-1000
      Si          0.90            stable over two fit ranges
      Cu          0.58            still not asymptotic, a lower bound
```

Adding the `order=2` multipoles (+11 % Si, +20 % Cu) brings silicon to ~1.0 and
copper to ~0.7. The residual spread is therefore roughly a factor of two, not
ten, and it is bounded by effects that are understood.

**Conclusion: no evidence of a structural error in abTEM's absolute scale.** What
remains is ordinary convergence, and users must converge the cell and `order`
per edge. `TransitionPotential.build` now warns when the cell is smaller than
four delocalisation lengths.

Caveat: the reference partitions photoabsorption with a constant edge-jump
factor, which is crude, and the Elam tables are least reliable at soft-X-ray
energies. A ~20-30 % uncertainty on `M^2(xraydb)` is plausible, which is the
same size as the remaining spread.

### 6c-bis. Why copper cannot be validated this way at all

The plan to "push copper to asymptotic `T/E_K`" turns out to be impossible, for
two independent reasons.

**1. `T` saturates.** The Bethe variable `T = m v^2 / 2` tends to `mc^2/2 =
255 keV` as the beam energy grows, so `T/E_K` for copper cannot exceed **28.5**
at any voltage. The non-relativistic asymptotic regime is unreachable for a
heavy edge; the earlier copper numbers (0.375, then 0.581 on a wider range) were
not converging toward anything.

**2. The relativistic form measures a term abTEM does not have.** The Fano plot
would sidestep the ceiling: `sigma beta^2` against `L = ln(beta^2 gamma^2) -
beta^2` is linear with slope `8 pi a0^2 R M^2 / mc^2`, and `L` spans 6.2 over
100 keV - 10 MeV against 1.1 for `ln T`. But that slope comes largely from the
transverse (Moller) interaction, and abTEM's transition potential is a
longitudinal `1/k^2` Coulomb treatment with a relativistic *mass* correction
only. Measured for silicon:

```
 E0 (keV)   beta^2       L    sigma (A^2)   sigma*beta^2
      100   0.3005  -1.145     2.1734e-05     6.5320e-06
      400   0.6854   0.093     1.1010e-05     7.5456e-06
     2000   0.9586   2.183     8.3190e-06     7.9745e-06
    10000   0.9976   5.048     8.0365e-06     8.0175e-06
```

`sigma beta^2` **saturates** at 8.02e-6 rather than growing linearly in `L` --
exactly what a non-retarded calculation should do. Fitting a Fano slope to it
therefore measures nothing physical, and unsurprisingly gives ratios of 0.28
(C), 0.16 (Si), 0.13 (Ti), 0.09 (Cu) with 5-8 % fit residuals.

Control: the fixed `stop = 4 E_K` accounts for only ~3 % of the cross-section,
equally at 100 keV and 10 MeV, so the saturation is not an artifact of the
integration range.

**Conclusion.** Neither slope method can validate a heavy edge: the
non-relativistic one because `T` saturates, the relativistic one because abTEM
is deliberately non-retarded. The silicon agreement in §6c stands -- `T/E_K`
reaches 123 there and the longitudinal form is the right one at those velocities
-- but copper needs a *direct* comparison at a single energy against an
independent tabulation (Bote & Salvat 2008, or Egerton's SIGMAK), which is not
available offline here. **Light and medium edges are validated; heavy edges are
not, and cannot be by this route.**

### Still to validate

- Total ionisation cross-sections against an independent source -- Bote & Salvat
  (2008) or Egerton's SIGMAK/SIGMAL -- for a few edges spanning Z and voltage.
  The chain currently gives sigma_K(Si) = 2.17e-5 A^2 = 2.2e-21 cm^2 at 100 keV;
  **this has not been checked against anything.**
- Reproduce a published absolute-scale measurement. Chen et al.,
  *Ultramicroscopy* 168 (2016) 7-16 reports counts/s/nA/srad and was itself
  validated against uSTEM.


## 6d. Joint EELS and EDX in one pass

The elastic multislice and the scattered waves are the expensive shared part;
EELS and EDX differ only in angular acceptance. Passing several detectors to a
single `transition_potential_scan` now works -- it previously raised an
`AssertionError` in `Waves.transition_potential_multislice`, which assumed a
single measurement.

```python
maps = probe.transition_potential_scan(
    potential, transitions, scan=scan,
    detectors=[abtem.AnnularDetector(inner=0.0, outer=30.0),   # EELS
               abtem.IonizationDetector()],              # EDX
)
```

`IonizationDetector()` sums the intensity of the scattered waves in real
space. For an angle-integrated measurement that is exactly the EDX signal, with
no FFT and -- unlike `AnnularDetector(0, None)` -- no truncation at the
antialiasing cutoff (measured 1.00008x its value).

Measured on a small case (96^2 grid, 12 slices, 4x4 scan, 16 transitions), warm:

```
EELS alone                 0.54 s
EDX alone (same route)     0.55 s
both in one pass           0.91 s
two separate runs          1.09 s
```

Profiled, the cost breaks down as (0.61 s total, one EELS aperture plus one
`IonizationDetector`):

```
detect                        0.461 s
  AnnularDetector array       0.387 s
    diffraction_patterns      0.285 s   <- the FFT
  IonizationDetector array    0.069 s
scatter                       0.109 s
```

So the diffraction-pattern FFT dominates, not per-call plumbing as an earlier
guess here had it. Marginal costs, warm:

```
EELS alone                  0.511 s
EDX alone                   0.202 s
EELS + EDX                  0.598 s   marginal +17 %
two EELS apertures          0.979 s   marginal +92 %   <- duplicate FFT
separate EELS + EDX runs    0.713 s
```

Adding EDX to an EELS run costs +17 %, and the joint pass is 16 % cheaper than
two separate runs, so that combination is already efficient and there is little
left to win. The **+92 % for a second angular detector** is the real
inefficiency: each radial detector computes its own diffraction pattern of the
same waves. Sharing it across detectors within one detection pass would make
extra angular detectors nearly free. That is a general abTEM issue rather than
an EDX one, and it touches shared machinery, so it is left as a separate change.

**For EDX alone the effective-potential route is far better** -- 1.8x on this
case, up to 46x for large scans. The crossover, measured:

```
    scan   joint pass   EELS + mu route
     2x2      0.27 s          0.49 s
     4x4      0.58 s          0.57 s
     8x8      2.82 s          2.66 s
   12x12      6.30 s          5.38 s
```

Below about a 4x4 scan the joint pass wins; above it, running EELS through the
scattered-wave route and EDX through the effective potential is faster, because
the latter does not scale with the number of probe positions.

## 6e. PRISM port

For an angle-unrestricted measurement the PRISM signal factorises. With
`psi_p = sum_k c_k(r_p) S_k`,

```
I(r_p) = integral |psi_p|^2 mu  =  c_p^dagger M c_p,    M_kk' = integral S_k^* mu S_k'
```

`M` does not depend on the probe position, so it is accumulated slice by slice
as the S-matrix propagates and the scan becomes one quadratic form per position.
Exposed as `SMatrix.ionization_scan`.

**Interpolation is supported**, and needed two corrections that a first pass got
wrong by 15x:

1. **Site selection.** The PRISM wave function is periodic with period
   `window_extent`, so a probe illuminates only the sites inside its own window;
   the copies elsewhere are aliasing artifacts. Integrating `mu` over the whole
   cell counts every copy against every site. The window test must be
   *half-open* on a signed, wrapped separation, or a site sitting on a boundary
   is claimed by two probes at once.
2. **Normalisation.** The reduction normalises each replica to unit intensity,
   so the cell carries `prod(interpolation)` times one probe's worth. This
   showed up as an exact factor of 4 at interpolation 2, and is confirmed
   directly on the elastic probe: total intensity 0.0287 at interpolation 1
   against 0.1149 at interpolation 2.

With both applied, against the multislice reference:

```
 interpolation   n_k   window (A)   mean ratio   max rel. diff
             1   129    10.86         1.0000        0.000
             2    37     5.43         0.9981        0.004
```

0.4 % at interpolation 2 -- better than the ~5 % the scattered-wave PRISM-EELS
drivers show there, because the effective potential is more localised than a
transition potential and there is no angular selection to sample.

Interpolation 1 uses the cheap single-matrix path (~4x faster than the
multislice route on a 6x6 scan); interpolation > 1 builds one matrix per site,
since the site set now depends on the probe position.

### Windowing the M build

`S^dagger diag(mu) S` only needs a window around each site, because `mu` is
localised on the atom. The window is sized from `mu` itself -- the smallest one
holding all but 0.1 % of it -- or pinned by the caller with `inelastic_crop`
(§6i). For a real Si K potential that is 3.75 A regardless of the grid, which
shrinks the matmul by 7-28x, the saving growing with cell size. A window that
would clip `mu` is refused: a delocalised potential simply keeps the whole grid.

Cropping trades one full-grid matmul for one per site, so when the sites'
windows together cover more than the grid the driver falls back to the batched
full-grid path. Without that fallback interpolation 1 regressed from 0.24 s to
3.14 s.

With windowing in place the `n_k^2` factor is **not** the bottleneck at
realistic beam counts -- the S-matrix propagation, linear in `n_k`, dominates:

```
 cutoff   n_k     time    t / n_k^2
     10    37    0.16 s    115 us
     15    69    0.26 s     55 us
     20   129    0.38 s     23 us
     30   277    0.79 s     10 us
```

`t / n_k^2` falls throughout, i.e. the measured scaling is near-linear (a 7.5x
increase in beams costs 4.9x the time). The quadratic term would only take over
above roughly `n_k ~ 500` on this grid, and windowing pushed that threshold up
by the window-shrink factor.

## 6f. Precision configuration

All of `core_loss.py` now honours `abtem.config['precision']`. The real-valued
geometry arrays -- site coordinates, sampling, wave vectors, sub-pixel shifts --
were pinned to `np.float32` regardless of the config. Because
`fft_shift_kernel` takes its output dtype from the positions handed to it, that
forced the whole downstream chain back to single precision even under
`precision="float64"`: the same NEP 50 promotion trap as the `build()` bug in
§6b, one layer up.

Verified end to end:

```
float32 config: waves complex64,  mu float32, effective-mu float32, scattered-wave float32
float64 config: waves complex128, mu float64, effective-mu float64, scattered-wave float64
```

with the two agreeing to 1.2e-6, i.e. float32 rounding. A test greps the module
for reintroduced literal dtypes.

## 6g. Cross-shell Coster-Kronig

Vacancies are now redistributed across the **whole shell**, not only within the
ionised subshell. Ionising 2s puts a vacancy in L1, and for iron 87 % of it
migrates to L2 and L3 and radiates from there. Treating the subshell alone put
the fluorescence yield of an L1 edge five times too low:

```
Fe 2s vacancies:  L1 0.130   L2 0.174   L3 0.696      omega 0.00514  (was 0.0010)
Ag 2s vacancies:  L1 0.310   L2 0.085   L3 0.605      omega 0.04076
```

The cascade uses the *direct* rates and walks the levels in order of decreasing
binding energy, so a vacancy transferred into an intermediate level goes on to
cascade from there. xraydb's default `total=True` already folds the cascade in,
and using it would double count; the identity
`f13_total = f13_direct + f12 * f23` holds exactly in the tabulation and is
asserted as a test.

Because the cascade is linear in the vacancy population, each subshell keeps an
independent coefficient, so several edges combine by a weighted sum:
`XrayDetector.to_counts_from_subshells({(2, 0): l1_map, (2, 1): l23_map}, "Ag")`.
Mixing shells is refused.

## 6h. Specimen self-absorption

`SpecimenAbsorption(formula, density=None, takeoff_angle=18.0)` attenuates the
emitted photons on the way out: a photon generated at depth `z` travels
`z / sin(takeoff)` to a detector on the entrance side, the usual STEM-EDX
geometry. `mu` comes from the tabulation already in use, so there is no new
dependency and the cost is one scalar per line per slice.

Both multislice drivers and the PRISM driver stamp the emission depth on the
waves, so absorption composes with everything. Measured on silicon, Si K-alpha:

```
 thickness    absorbed
     54 A       0.08 %
    163 A       0.22 %
    326 A       0.43 %
```

Small for a thin specimen, as expected, and linear in thickness there. Soft
lines are absorbed more than hard ones. The scattered-wave and PRISM routes
agree to five decimals with absorption on.

It is a single-scattering correction: absorbed photons are lost, and the
secondary fluorescence they excite is not modelled. It cannot be applied by
`XrayDetector.to_counts`, which receives a measurement already summed over
depth, and says so rather than silently ignoring it.

## 6i. Consistency with the merged PRISM-EELS work

PRs #289 (linear-scaling PRISM-EELS) and #286 (transition-potential
performance) are both merged and in this base. Two points of contact:

- **No duplication.** `SMatrix.transition_potential_scan` from #289 forms
  scattered waves and detects them at an angle; `SMatrix.ionization_scan` here
  forms none and is angle-integrated by construction. They answer different
  questions, and #289's route is what this work uses as its reference and as
  the documented fallback.
- **`inelastic_crop` is now shared.** #289 established it as a real-space side
  length in Angstrom, scalar or pair, clamped to the cell with a warning. The
  effective-potential driver first grew its own automatic, tolerance-based
  window in grid points -- a second concept for the same thing. It now takes
  `inelastic_crop` with identical semantics, and uses the automatic sizing only
  as the default when it is not given. A crop the caller pins is always
  honoured, even where the full-grid path would be cheaper, because in #289's
  semantics it is an accuracy control rather than a performance knob.

`ionization_scan` deliberately has no `double_channel` (unitarity makes it moot
for an angle-integrated measurement) and no `lazy` (the driver is eager, unlike
`Probe.ionization_scan`). Both are documented on the method.

## 6j. Lazy and eager

`SMatrix.ionization_scan` was eager-only; it now takes `lazy` and mirrors the
`_eager_` / `_lazy_` block-mapping pattern that
`SMatrix.transition_potential_scan` already uses, so the two behave the same
way. It also gained `detectors`, which the underlying driver always supported
but the method did not expose -- an `XrayDetector` can now go straight into the
PRISM route as it can into the multislice one.

Every entry point and feature is checked lazy against eager:

```
Probe.ionization_scan                        plain / exit planes / frozen phonons
PlaneWave.ionization_multislice              plain
SMatrix.ionization_scan                      interpolation 1 and 2
SMatrix.ionization_scan                      XrayDetector, self-absorption,
                                             inelastic_crop, frozen phonons
transition_potential_scan                    joint EELS + EDX, both detectors
```

All twelve agree to float32 rounding, with matching shapes.

One convention difference, inherited rather than introduced: with frozen
phonons the multislice route keeps the configuration axis while the PRISM route
averages over it, because `_eager_ionization_scan` follows the `_ensemble_mean`
handling of `SMatrix.transition_potential_scan`.

## 7. Status

| phase | state |
|---|---|
| 1. Atomic data (`xray_data.py`) | done |
| 2. `XrayDetector` (`xray.py`) | done |
| 3. Energy integration (`EnergyIntegral`) | done |
| 4. Effective-potential multislice | done, lazy + eager |
| 5. Validation | done (§6c); no structural error found |
| 6. PRISM port | done, interpolation supported (§6e) |

Public API: `abtem.XrayDetector`, `abtem.SDDEfficiency`,
`abtem.TabulatedEfficiency`, `abtem.EnergyIntegral`, `abtem.SubshellTransitions`,
`abtem.IonizationDetector`, `Probe.ionization_scan`,
`Waves.ionization_multislice`, `SMatrix.ionization_scan`. Optional extra `gpaw`.

Tests: `test/test_xray.py`, `test/test_energy_integral.py`,
`test/test_edx_multislice.py`.

### Remaining

- **Share the diffraction pattern across radial detectors** in one detection
  pass; a second angular detector currently costs +92 % because it recomputes
  the same FFT (§6d). General to abTEM, not specific to EDX.
- **Cross-shell Coster-Kronig for M shells** is implemented but untested
  against anything; only the L-shell identity is verified (§6g).
- **Heavy-edge absolute validation** needs a direct comparison against an
  independent tabulation (Bote & Salvat, or SIGMAK). Neither Bethe slope method
  can do it -- see §6c-bis.
- Reproduce a published absolute-scale measurement, e.g. Chen et al.,
  *Ultramicroscopy* 168 (2016) 7-16.
- Site coordinates in the PRISM helpers of `prism/` (outside `core_loss.py`)
  have not been audited for hardcoded dtypes.
