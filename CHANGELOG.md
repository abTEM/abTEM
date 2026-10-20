# Changelog

The changes of each release, newest first. The changes of the next release go under a
heading `## Upcoming: <version>`, which is renamed to the version when it is released.

## 1.0.11

A bugfix release on top of 1.0.10. Every fix was ported from `dev`. There are no
new features, no new defaults, and none of the post-release algorithm changes.
Plain potentials, plane-wave and probe multislice with default settings,
ensembles of unweighted parameters, zarr round trips and the configuration
defaults give bit-identical results to 1.0.10 at the default float32 precision
(checked on 66 scenarios). At `precision='float64'` the atom snapping of #390 changes
plain potentials by 1e-17 to 1e-15.

### Fixed: results that change

These corrections change numbers that 1.0.10 produced. The magnitudes quoted are
from small test systems and vary with the data.

**4D-STEM and diffraction-pattern measurements**
- Centre of mass and iCOM are divided by the captured intensity (#417); an all-zero pattern now gives NaN instead of 0.
- Default patterns from `Waves.diffraction_patterns` and `PixelatedDetector` are unshifted. Their coordinates, `azimuthal_average`, `bandlimit` and centre of mass applied centred coordinates to unshifted data and are now right (#463): azimuthal averages differ by up to 80 %.
- Radial binning and the default `FlexibleAnnularDetector` bins are exactly `step_size` wide (#469): default ADF/HAADF values move by 0.2 to 0.8 %, `radial_binning` by 13 to 33 %, more on very coarse k grids.
- `PolarMeasurements.integrate` azimuthal limits and bin selection (#459); radian/degree conversions (#458, were off by 1e6); `depth_profile` z sampling and `Images.coordinates` spacing (#461); `interpolate_line` near edges and on anisotropic grids (#470); `Images` reflected division `2 / images` (#470); periodic spline interpolation (#506, ~1e-4).
- A `FlexibleAnnularDetector` without an explicit `outer` takes the cutoff angle of the waves of each call; it used to keep the angle of the first waves it saw, so reusing it for another energy or grid binned the new waves wrongly (#434).
- `AnnularDetector(offset=...)` is applied (#324); `get_detector_region` returns percent as a float, as its label says (#283).
- `block_direct=True` blocks the whole bright-field disk instead of 1 mrad (#484); `block_direct()` without a cutoff blocks only the zero-angle pixel (#556).
- Reciprocal-space waves measure like the same waves in real space (#565, #566).
- Lazy `gaussian_source_size` is periodic at the scan edges, as the eager path (#485); `integrate_gradient` takes the minimum per image (#556).
- Seeded Poisson and scan noise give a different realisation for the same seed (same statistics; now independent of chunking) (#464, #475).
- Float32 data stays float32 in transfer functions, `integrate_gradient` and `center_of_mass` (#604, ~1e-7).

**PRISM**
- Aberrated `SMatrix.reduce`/`scan` results were wrong and now match `Probe` (#318; 17 to 75 % in the test cases).
- An eager frozen-phonon reduce is averaged over configurations like the lazy path (#576); PRISM no longer leaves its outer angle on the caller's detector (#491). At `precision='float64'` PRISM builds change by 1e-8 to 5e-8 (#556).

**Potentials and atoms**
- `orthogonalize_cell` followed by `Potential` no longer loses atoms near cell faces (#345, #390, #555): for wurtzite GaN 2 of 8 atoms were dropped (projected potential +33 % relative to 1.0.10).
- Non-periodic potentials keep atoms displaced past a face, and transformed non-periodic cuts give the periodic result (#460, #555).
- `CrystalPotential` honours `first_slice`/`last_slice` (#269); `GPAWPotential(repetitions=...)` values were wrong by about 100 % (#494).
- GPAW valence slices and Bloch projected potentials are integrated centred on their slices, which shifts each slice by dz/2 (5 % per slice for GPAW, 17 to 30 % for Bloch; whole-cell sums unchanged) (#574). `ChargeDensityPotential` slices are sampled at midpoints (#472; error against fine slicing drops from 0.5 to 4.2 % to at most 0.55 %).
- `GPAWParametrization` applies `charge` (#329; only cations raise, as in 1.0.10).
- IAM magnetic fields change by ~1e-6 (#543); field and vector-potential slices are labelled with the units of a slice integral (`ÅT`, `Å²T`).

**Bloch waves**
- Non-centrosymmetric structures were computed for the position-negated structure (#379): exit waves change (wurtzite by up to 1.27), zone-axis patterns only at rounding; centrosymmetric results are unchanged.
- The symmetrising metric of the eigenproblem is applied consistently (#448). The eigendecomposition path (diffraction patterns, exit waves) and the matrix-exponential path (`calculate_scattering_matrix`) disagreed off the zone axis and now agree to rounding. The standard form (`use_wave_eq=False`, still the default) now solves the textbook generalised eigenproblem exactly, and `use_wave_eq=True` no longer carries the standard form's metric. For Si at 100 keV and thicknesses of 100 to 600 Å the intensities change by R = 1.5e-3 to 2.2e-3 on the zone axis and 3e-4 to 1.3e-3 for a 7° tilt (R = Σ|ΔI| / ΣI over the beams), for both forms.
- Eager rotation ensembles return each orientation's own pattern (#450); A/B/C centred cells keep the correct reflections (#487). Scattering matrices on CuPy are accurate (#449).

**HRTEM and CTF**
- `set_aberrations` with `'scherzer'`: C10 sign and ordering (#471). `apply_ctf` no longer writes the waves' energy into the CTF, so a CTF can be reused at another energy (#489).

**Core-loss**
- Continuum normalisation (#364; for example K scan -5.8 % at 3 eV, several-fold below ~1 eV), the (2l+1) degeneracy double count (#368: L2,3 intensity drops 3×, M4,5 5×; K edges unchanged), site threshold and a spurious 1.02 radial factor (#468: K rises by 3 to 4 %, L2,3 falls by 22 to 27 %), and the vxc vacuum tail (-3.7 % K and -3.9 % L2,3 at 3 eV, +0.6 to +1.6 % at 10 eV).
- Single-channel frozen-phonon core loss was `num_configs` times too large or zero (#387).

**Backscattering**
- Back-propagation of backscattered waves is corrected (#562, relative L2 0.4 at the entrance plane) and ensembles work (#499).

**Weighted distributions (focal spread, tilt and other parameters)**
- Distribution weights are probabilities applied when the ensemble is reduced (#466). In 1.0.10 they multiplied the wave amplitude of each member: a STEM `Probe(defocus=gaussian(σ))` was normalised member by member, which erased the weights, so the focal spread was effectively a top-hat; an HRTEM Gaussian spread had an effective standard deviation of σ/√2 instead of σ; and the weights of tilt and aperture or envelope distributions were ignored. For a 200 kV, 25 mrad probe with `defocus=gaussian(σ, num_samples=11)`, the peak of the reduced intensity is 1.76 times the 1.0.10 value at σ = 25 Å and 2.21 times at σ = 50 Å (measurement from the PR). Frozen-phonon results are unchanged to the bit.
- `.weights` of `gaussian` distributions now sum to one (they had a unit sum of squares). `normalize='intensity'` and `'amplitude'` are deprecated, emit a `FutureWarning` and behave as the new default `'probability'`; this never changes a reduced result.
- `.sum()` and `.mean()` over an axis with distribution weights ignore the weights and now warn: use `reduce_ensemble()` for the weighted mean, or `reduce_ensemble(axis=...)` for an axis that was kept with `ensemble_mean=False`.
- Eager `SMatrix.scan` and `SMatrix.reduce` reduce the `ensemble_mean` axes of a `CTF`, like the lazy path (they returned an extra axis).
- Passing negative weights, or weights of the wrong length, to a distribution now raises `ValueError`.
- The weights are not applied to the rotation spreads of Bloch waves or to noise axes, which carry none (unchanged from 1.0.10). A zarr file with a weighted axis written by 1.0.11 cannot be read by 1.0.10; files without one can.

**Other results that change**
- Eager multislice of potentials with two or more ensemble axes wrote its results to scrambled positions (#333).
- `wrapped_crop_2d` for probe and crop windows outside the cell, for example PRISM scan positions several periods away (#389).
- With `projection="finite"` and sigmas given for only some elements, the other elements were dropped from the potential (#480).
- The GPAW magnetic `real_space` projection ignored `first_slice` and truncated plane counts (#544).
- `Waves.phase_shift` relabelled reciprocal-space waves as real space and widened complex64 to complex128 (#550); `tile(renormalize=True)` widened complex64 waves (#414).
- FFTW: `fftn` over three axes transformed the wrong axes, and strided input with `overwrite_x=True` raised (#558; FFTW results change by about 1e-7 on x86). A lazy `fft2_convolve` returned the wrong shape when the kernel axis was longer than a block (#577).
- `bandlimit` and `block_direct(radius=0)` kept the direct beam on some grids (#506); 1D `interpolate` of float64 and complex profiles lost its dtype (#427).
- `StructureFactorArray.get_projected_potential` honours `sampling=` and an integer `gpts` and accepts sequences of slice thicknesses; `IndexedDiffractionPatterns.positions_dict` had only three entries (#456).
- `PixelatedDetector(reciprocal_space=False)` declares real-space axes, `WavesDetector(gpts=...)` declares the right shape, and `DiffractionPatterns.block_direct` no longer grows its metadata on each call (#499).

**Distributed computations**
- The client's configuration reaches the workers of a `distributed` client (#346). *ab*TEM resolves configuration inside each task, and worker processes started fresh and saw only the YAML defaults, so any distributed computation with a non-default configuration silently used the defaults: most importantly, `precision='float64'` runs were computed in `float32`. This applies to any distributed client, CPU clusters included, and now also to `to_tiff` and `to_zarr(compute=False)`. Distributed results obtained with a non-default configuration are worth repeating.

### Fixed: errors and crashes

- `Images.scan_noise`, `apply_ctf` at a second energy, `number - Images`, lazy `gaussian_filter` with a constant boundary, lazy `normalize`, single-point `GridScan`, noisy near-orthogonal `Potential` cells, tilted eager Bloch exit waves with scalar thickness, `IndexedDiffractionPatterns.sort()`, core loss on 16 or 17 pixel grids, `AnnularDetector` over a `LineScan` with an extra ensemble axis, GPAW 26 new-style calculators, frozen-phonon `GPAWPotential`, ensemble backscattering, A/B/C centring, graphene repeated along c, and numba typing on some numba/numpy pairings no longer raise.
- `excitations_weights` (the plasmon Poisson weights) no longer fails because `np.math` is gone in NumPy 2.
- On CuPy 14.2, a lazy FFT in a dask worker thread no longer fails with `CUFFT_EXEC_FAILED` after an FFT of the same grid in another thread (see `cupy.fft-cache-size` below).

### Changed: behaviour

- abTEM reads its own configuration files, `~/.config/abtem` and `$ABTEM_CONFIG`, as well as dask's (#416); 1.0.10 ignored them. A copy of the default `abtem.yaml` kept there now takes effect, including `cupy.fft-cache-size: 0 MB`, which disables the cuFFT plan cache (see below).
- A built `TransitionPotentialArray` whose energy or grid disagrees with the waves raises `RuntimeError`; a run at a mismatched energy used to complete silently (#386). `PixelatedDetector(resample='uniform')` without reciprocal space raises `ValueError` (#499); `transition_potential_scan` with `expansion_scope="full"` raises `NotImplementedError` (#506).
- Arrays over 512 MiB are written to zarr with a multi-chunk layout, and a failed write removes the partial file (#363, #369). `GPAWParametrization` for Pm, Sm, Eu and Pu falls back to an unpolarised atom with a `UserWarning` instead of an `AssertionError` (#467).
- `LineScan.angle` reports degrees (#470).
- `SegmentedDetector(to_cpu=...)` defaults to `True`, as documented (#371; GPU runs return NumPy).
- Backscattering with exit planes that do not start at the entrance plane, or are not after every slice, raises `ValueError` instead of returning a wrong result (#562, #499).
- Exploded `show()` grids use 0.25 in between panels (#354).
- The configuration key `cupy.fft-cache-size`, which 1.0.10 ignored, is read and defaults to `auto`: the cuFFT plans cached in each thread may hold up to 25 % of the device memory. Use `0 MB` to disable the cache, a size such as `"2 GB"` for a fixed bound, or `-1` for no bound. A plan larger than the bound runs uncached, with a warning.
- Importing `abtem` no longer fails if `cupyx.scipy.ndimage` cannot be loaded (#381).

### Added: public names that the fixes needed

`OrdinalAxis.weights` and `OrdinalAxis.from_distribution`, and `reduce_ensemble(axis=)` (#466), `LinearAxis.__getitem__` (#344), `SliceIndexedAtoms(wrap=)` (#390), `WavesDetector.gpts` (#482), `offset` on `FlexibleAnnularDetector`/`SegmentedDetector` (#469), `ArrayObject.__radd__`/`__rsub__`/`__rpow__` and `__array_priority__` (#507), `abtem.core.backend.push_config_to_workers` (#346), a `use_wave_eq` argument (default `False`) of the module functions `calculate_dynamical_scattering` and `calculate_scattering_matrix` of `abtem.bloch.dynamical` (#448), `CrystalPotential.match_grid`, `ScanNoiseTransform.apply` and `ReciprocalSpaceAxis.coordinates`. `StructureFactorArray.get_projected_potential(lazy=None)` (#487) and the abstract `FieldIntegrator.integrate_on_grid(atoms=...)` signature (#428) changed.

### Not in this release

These are on `dev` and are not part of a bugfix release because they change default results or add features: the exact propagator as the default (#298), the Bloch-wave `use_wave_eq='exact'` form and default (#447, #448) and Bloch waves honouring `precision` (#449), the GPAW defaults (`rotate_field`, `integration_step`), the `exit_planes` output shape (#516), and all new features.

`SMatrix.scan` with a potential that has `exit_planes` still raises an error, as in 1.0.10 (fixed on `dev` in #398).

## 1.0.10

Features:

- Expanded real-space multislice with propagator- and fully-corrected algorithms, and backscattered waves ([PR #236](https://github.com/abTEM/abTEM/pull/236))
    - Related internal function name change: standard multislice is now properly called `FourierMultislice`
- Updated `BullseyeAperture` to use smoothed aperture edges/corners ([PR #266](https://github.com/abTEM/abTEM/pull/266))
- Logarithmic scale display for `DiffractionPatterns` and images ([PR #303](https://github.com/abTEM/abTEM/pull/303))
- Finite-projection integrals 7x faster on CPU ([PR #309](https://github.com/abTEM/abTEM/pull/309))

Documentation:

- Expanded tutorial on real-space multislice
- Depth-profile visualization of potentials in the walkthrough
- Single-file Zarr zip storage, logarithmic display scaling, soft Bullseye apertures, anisotropic
  Debye-Waller factors and B-factor conversion helpers
- All published notebooks verified to run with this release

Dependencies:

- NumPy 2.0 or newer is now required ([PR #245](https://github.com/abTEM/abTEM/pull/245))
- GitHub actions based on `uv` and now cover more versions (including Python 3.14) ([PR #308](https://github.com/abTEM/abTEM/pull/308))
- New branching structure: `dev` for development, `main` for releases (only via PRs from `dev`)
- Support for Zarr 3 `ZipStore` (requiring `zarr>=3.1`)
    - Added Zstandard compression of the arrays in the ZipStore at default level 4 (hat tip: quantEM) ([PR #252](https://github.com/abTEM/abTEM/pull/252))
- Deprecated `[gpu]` optional dependency (as just specifying `cupy` will not install the correct CUDA version)
- Moved `testing`, `docs`, and `dev` from optional dependencies to groups
- Narrow Dask version exclusion to !=2025.12.*,!=2026.1.0,!=2026.1.1 ([PR #285](https://github.com/abTEM/abTEM/pull/285))
- Declared `sympy` as an optional dependency (`core-loss` extra), required for core-loss EELS form
  factors ([PR #320](https://github.com/abTEM/abTEM/pull/320))

Bugfixes:

- Frozen-phonon ensemble handling ([PR #267](https://github.com/abTEM/abTEM/pull/267) & [PR #292](https://github.com/abTEM/abTEM/pull/292))
  - May also have resulted in incorrect behavior with `ensemble_mean = False` for e.g. defocus distributions
- `CrystalPotential` with frozen phonons bugs (especially bad on – luckily rare – eager compute) ([PR #306](https://github.com/abTEM/abTEM/pull/306))
- Minor bugs, unsafe patterns, and dead code ([PR #265](https://github.com/abTEM/abTEM/pull/265))
- Anisotropic Debye-Waller factors for Bloch waves ([PR #271](https://github.com/abTEM/abTEM/pull/271))
  - Added helper functions to convert between crystallographic B-factors and thermal sigmas
- Added missing `.calculate_exit_waves` for `BlochwaveEnsemble` ([PR #294](https://github.com/abTEM/abTEM/pull/294))
- Silent atom drop when `z`-position lands in `SliceIndexedAtoms` blind spot ([PR #273](https://github.com/abTEM/abTEM/pull/273))
- Early-exit bug in `orthogonalize_cell` ([PR #291](https://github.com/abTEM/abTEM/pull/291))
- Fixed broken tutorial workflow for core-loss filtered imaging ([PR #284](https://github.com/abTEM/abTEM/pull/284))
  - Minor performance improvements for `transition_potential_scan` ([PR #286](https://github.com/abTEM/abTEM/pull/286))
- `Bullseye` aperture: `ring_width` and `spoke_width` are now validated, rejecting values that previously
  produced a silently wrong (solid disk) aperture; docstring corrected to describe the actual fractional
  units introduced by the soft-edge redesign ([PR #319](https://github.com/abTEM/abTEM/pull/319))
- Unified the task-level progress bar config key on `diagnostics.task_progress` (Bloch-wave code paths
  previously read a different, non-functional key) ([PR #321](https://github.com/abTEM/abTEM/pull/321))

## 1.0.9

Dependencies:

- Support for `scipy>=1.7` and `cupy>=12`.
- Restricted Dask versions (`>=2022.12.1,!=2025.12.*,!=2026.1.*`) to avoid an issue with Numba in the latest ones

## 1.0.8

Starting the changelog with version 1.0.8.

Features:

- Fully featured Bloch-wave simulations
- Simple real-space multislice algorithm
- Core-loss filtered imaging
- Structured illumination (custom apertures and phase plates)

Documentation:

- Updated and fixed example gallery
- Appendix on convergence
- Expanded tutorial on orthogonal periodic supercells

Bugfixes:

- Numerous small bugfixes and improvements
