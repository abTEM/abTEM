# Changelog

All notable changes to *ab*TEM are listed here, newest first. Add the changes of
the next release under a heading `## [Unreleased]` and give it a version and date
when it is released. Releases before 1.0.11 are described on the
[GitHub releases page](https://github.com/abTEM/abTEM/releases).

## [1.0.11] - unreleased

A bugfix release on top of 1.0.10. Every fix was ported from `dev`. There are no
new features, no new defaults, and none of the post-release algorithm changes.
Plain potentials, plane-wave and probe multislice with default settings, probe
ensembles, zarr round trips and the configuration defaults give bit-identical
results to 1.0.10 (checked on 66 scenarios).

### Fixed: results that change

These corrections change numbers that 1.0.10 produced. The magnitudes quoted are
from small test systems and vary with the data.

**4D-STEM and diffraction-pattern measurements**
- Centre of mass and iCOM are divided by the captured intensity (#417); an all-zero pattern now gives NaN instead of 0.
- Default patterns from `Waves.diffraction_patterns` and `PixelatedDetector` are unshifted. Their coordinates, `azimuthal_average`, `bandlimit` and centre of mass applied centred coordinates to unshifted data and are now right (#463): azimuthal averages differ by up to 80 %.
- Radial binning and the default `FlexibleAnnularDetector` bins are exactly `step_size` wide (#469): default ADF/HAADF values move by 0.2 to 0.8 %, `radial_binning` by 13 to 33 %, more on very coarse k grids.
- `PolarMeasurements.integrate` azimuthal limits and bin selection (#459); radian/degree conversions (#458, were off by 1e6); `depth_profile` z sampling and `Images.coordinates` spacing (#461); `interpolate_line` near edges and on anisotropic grids (#470); `Images` reflected division `2 / images` (#470); periodic spline interpolation (#506, ~1e-4).
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
- `orthogonalize_cell` followed by `Potential` no longer loses atoms near cell faces (#345, #390, #555): for wurtzite GaN 2 of 8 atoms were dropped (projected potential +25 %).
- Non-periodic potentials keep atoms displaced past a face, and transformed non-periodic cuts give the periodic result (#460, #555).
- `CrystalPotential` honours `first_slice`/`last_slice` (#269); `GPAWPotential(repetitions=...)` values were wrong by about 100 % (#494).
- GPAW valence slices and Bloch projected potentials are integrated centred on their slices, which shifts each slice by dz/2 (5 % per slice for GPAW, 17 to 24 % for Bloch; whole-cell sums unchanged) (#574). `ChargeDensityPotential` slices are sampled at midpoints (#472; error against fine slicing drops from 0.5 to 4.2 % to at most 0.55 %).
- `GPAWParametrization` applies `charge` (#329; only cations raise, as in 1.0.10).
- IAM magnetic fields change by ~1e-6 (#543); field and vector-potential slices are labelled with the units of a slice integral (`Å T`, `Å² T`).

**Bloch waves**
- Non-centrosymmetric structures were computed for the position-negated structure (#379): exit waves change (wurtzite by up to 1.27), zone-axis patterns only at rounding; centrosymmetric results are unchanged.
- Eager rotation ensembles return each orientation's own pattern (#450); A/B/C centred cells keep the correct reflections (#487). Scattering matrices on CuPy are accurate (#449).

**HRTEM and CTF**
- `set_aberrations` with `'scherzer'`: C10 sign and ordering (#471). `apply_ctf` no longer writes the waves' energy into the CTF, so a CTF can be reused at another energy (#489).

**Core-loss**
- Continuum normalisation (#364; for example K scan -5.8 % at 3 eV, several-fold below ~1 eV), the (2l+1) degeneracy double count (#368: L2,3 intensity drops 3×, M4,5 5×; K edges unchanged), site threshold and a spurious 1.02 radial factor (#468: K -3.8 %, L2,3 -27 %), and the vxc vacuum tail (+3.7 % K, +3.9 % L2,3 below 10 eV).
- Single-channel frozen-phonon core loss was `num_configs` times too large or zero (#387).

**Backscattering**
- Back-propagation of backscattered waves is corrected (#562, relative L2 0.4 at the entrance plane) and ensembles work (#499).

### Fixed: errors and crashes

- `Images.scan_noise`, `apply_ctf` at a second energy, `number - Images`, lazy `gaussian_filter` with a constant boundary, lazy `normalize`, single-point `GridScan`, noisy near-orthogonal `Potential` cells, tilted eager Bloch exit waves with scalar thickness, `IndexedDiffractionPatterns.sort()`, core loss on 16 or 17 pixel grids, `AnnularDetector` over a `LineScan` with an extra ensemble axis, GPAW 26 new-style calculators, frozen-phonon `GPAWPotential`, ensemble backscattering, A/B/C centring, graphene repeated along c, and numba typing on some numba/numpy pairings no longer raise.
- `excitations_weights` (the plasmon Poisson weights) no longer fails because `np.math` is gone in NumPy 2.
- With in-process workers, the configuration plugin of #346 no longer replays its snapshot into the client and reverts configuration changes.
- On CuPy 14.2, a lazy FFT in a dask worker thread no longer fails with `CUFFT_EXEC_FAILED` after an FFT of the same grid in another thread (see `cupy.fft-cache-size` below).

### Changed: behaviour

- `LineScan.angle` reports degrees (#470).
- `SegmentedDetector(to_cpu=...)` defaults to `True`, as documented (#371; GPU runs return NumPy).
- Backscattering with exit planes that do not start at the entrance plane, or are not after every slice, raises `ValueError` instead of returning a wrong result (#562, #499); a radial detector with a non-zero offset and an outer angle beyond the grid raises `RuntimeError` (#469).
- Exploded `show()` grids use 0.25 in between panels (#354).
- The configuration key `cupy.fft-cache-size`, which 1.0.10 ignored, is read and defaults to `auto`: the cuFFT plans cached in each thread may hold up to 25 % of the device memory. Use `0 MB` to disable the cache, a size such as `"2 GB"` for a fixed bound, or `-1` for no bound. A plan larger than the bound runs uncached, with a warning.
- Results from a distributed client follow the client's configured precision (#346), including `to_tiff` and `to_zarr(compute=False)`. Importing `abtem` no longer fails if `cupyx.scipy.ndimage` cannot be loaded (#381).

### Added: public names that the fixes needed

`LinearAxis.__getitem__` (#344), `SliceIndexedAtoms(wrap=)` (#390), `WavesDetector.gpts` (#482), `offset` on `FlexibleAnnularDetector`/`SegmentedDetector` (#469), `ArrayObject.__radd__`/`__rsub__`/`__rpow__` and `__array_priority__` (#507), `abtem.core.backend.push_config_to_workers` (#346). `StructureFactorArray.get_projected_potential(lazy=None)` (#487) and the abstract `FieldIntegrator.integrate_on_grid(atoms=...)` signature (#428) changed.

### Not in this release

These are on `dev` and are not part of a bugfix release because they change default results or add features: the exact propagator as the default (#298), the Bloch-wave `use_wave_eq='exact'` default and Bloch waves honouring `precision` (#447, #448, #449), distribution weights applied as probabilities (#466), the GPAW defaults (`rotate_field`, `integration_step`), the `exit_planes` output shape (#516), and all new features.

`SMatrix.scan` with a potential that has `exit_planes` still raises an error, as in 1.0.10 (fixed on `dev` in #398).
