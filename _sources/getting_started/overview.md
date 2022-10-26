(getting_started:overview)=

# Overview
To reliably interpret the data from electron microscopy experiments, a careful comparison between theoretical models and
experiments are often required. The use of dynamical scattering simulations based on the multislice algorithm has now 
aided this process for multiple decades. As a result there are many excellent simulation codes, *abTEM* is another one, 
however our approach differs in many key aspects.

## How abTEM differs

### No GUI, Jupyter notebooks

abTEM does not implement a graphical user interface, instead focusing on making scripts and especially Jupyter notebooks
work. The Jupyter Notebook mix of code, explanatory text and visualizations immediately provides documentation of the
simulation, allowing others to understand, share and reproduce it. abTEM aids a visual workflow by implementing methods 
for creating quick visualizations of each subtask; for example a heatmap of the projected potential or the profile of 
the electron probe.

### Flexible, no fixed simulation modes

abTEM does not directly implement *any* of the common imaging modes, instead inviting the user to mix and match
objects to construct the desired simulation. This provides the user with an enormous flexibility to design the
simulation while abTEM takes care of the computational details. See below for examples.

### "Pure Python", but fast

Python has become the most popular programming language in physics, thanks in particular to its extensive base
of open-source libraries. In the electron microscopy community, this includes packages such as 
[HyperSpy](https://hyperspy.org/), [Libertem](https://libertem.github.io/LiberTEM/) and 
[Py4DSTEM](https://py4dstem.readthedocs.io/en/latest/). Effective use of the fast open-source libraries such 
as [NumPy](https://numpy.org/), [CuPy](https://cupy.dev/), [Numba](https://numba.pydata.org/) and 
[PyFFTW](https://pyfftw.readthedocs.io/en/latest/), allows abTEM to be extremely competitive in raw performance.

### Deploys anywhere, powered by Dask

abTEM is created from the ground-up to work with [Dask](https://www.dask.org/). Dask allows scaling from a single
laptop to hundreds of nodes at high-performance computing (HPC) facilities with minimal changes to the code.

### Open-source and open to contributions

The abTEM developers are not alone in seeing the necessity of open-source software in electron microscopy. We are 
striving to make abTEM an inviting project for new contributors. Everybody is invited to contribute in using and 
developing the code, see our [guide for contributors](library:contributing).

## The abTEM object model

abTEM implements objects representing physical concepts, for example, a plane wave function is represented
by the `PlaneWave` object and a STEM probe wave function is represented by the `Probe` object.

To simulate a (basic) HRTEM image, we need a `PlaneWave`, a `Potential` and a `CTF` representing the objective lens, the
image is the `intensity` of the exit wave after applying the `CTF`. We show how this could be implemented in code below,
change the tabs for other simulation modes.

We can turn the HRTEM simulation into a SAED simulation by getting rid of the `CTF` and calculating
the `diffraction_patterns` instead of the `intensity` from the exit wave. The SAED simulation becomes a CBED simulation 
by only changing the `PlaneWave` for a `Probe`, which can become a precession ED simulation by setting the `tilt` 
property of the `Probe`.

A STEM-ADF simulation requires defining the scan area using the `GridScan` and an `AnnularDetector` defining the
inner and outer detector angles. The STEM-ADF simulation is easily modified into a 4D-STEM simulation by swapping the 
`AnnularDetector` for a `PixelatedDetector`.

The code below demonstrate slightly simplified *ab*TEM code for simulating several common electron microscopy 
experiments. For clarity of expression we have ommited import statements and the atomic models is assumed to be given as
`atoms`. For real, runable code examples look [further in the documentation](getting_started:basic_examples).

`````{tab-set}

````{tab-item} HRTEM
```python
wave = PlaneWave(energy=80e3)

potential = Potential(atoms, sampling=.05)

ctf = CTF(semiangle_cutoff=30, Cs=1e-3 * 1e10, defocus="scherzer")

exit_wave = wave.multislice(potential)

measurement = exit_wave.apply_ctf(ctf).intensity()
```
````

````{tab-item} SAED
```python
wave = PlaneWave(energy=80e3)

potential = Potential(atoms, sampling=.05)



exit_wave = wave.multislice(potential)

measurement = exit_wave.diffraction_patterns()
```
````

````{tab-item} CBED
```python
wave = Probe(energy=80e3, semiangle_cutoff=30, Cs=1e-3 * 1e10, defocus="scherzer")

potential = Potential(atoms, sampling=.05)



exit_wave = wave.multislice(potential)

measurement = exit_wave.diffraction_patterns()
```
````

````{tab-item} PED
```python
wave = Probe(energy=80e3, semiangle_cutoff=30, Cs=1e-3 * 1e10, defocus="scherzer")

potential = Potential(atoms, sampling=.05)

wave.tilt = precession_tilts(precession_angle = 20, num_samples=60)

exit_wave = probe.multislice(potential)

measurement = exit_wave.diffraction_patterns()
```
````

````{tab-item} STEM-MAADF
```python
wave = Probe(energy=80e3, semiangle_cutoff=30, Cs=1e-3 * 1e10, defocus="scherzer")

potential = Potential(atoms, sampling=.05)

detector = AnnularDetector(inner=60, outer=120)

scan = GridScan()

measurement = probe.scan(potential, detectors=detector, scan=scan)
```
````

````{tab-item} 4D-STEM
```python
wave = Probe(energy=80e3, semiangle_cutoff=30, Cs=1e-3 * 1e10, defocus="scherzer")

potential = Potential(atoms, sampling=.05)

detector = PixelatedDetector()

scan = GridScan()

measurement = probe.scan(potential, detectors=detector, scan=scan)
```
````

`````

