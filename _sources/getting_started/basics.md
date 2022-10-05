# The basics

This is a short overview of abTEM geared towards new users. Hence, for details 

There is much more information contained in the rest of the documentation.

For users with previous experience in . This 

We normally import abTEM as follows:
```python
import abtem
```

## Atomic model
abTEM uses the atomic simulation environment for importing atomic models:


We can also procedurally create and modify structures: 


The multislice algorithm requires cell with orthogonal lattice vectors. You can make your unit cell orthogonal ``:

## Potential 
We create an electron static potential by 


We can include frozen phonons by wrapping 

## Wave functions
abTEM differs from codes for simulating electron diffraction by not directly implementing any common imaging modes 
(STEM, HRTEM etc.). You select

For simulating HRTEM and SAED you should define a plane wave:

For STEM and CBED you should 

## Setup simulation



For simulating  

## Execute computation
abTEM is lazily evaluated. The result from a computation isn’t computed until you ask for it. Instead, a Dask task graph for the computation is produced.

The array is a [Dask Array](https://docs.dask.org/en/stable/array.html), this means that it is "lazily" evaluated. When we create a lazy array, we only set up the tasks to compute the array, to execute those tasks we also have to call `compute`. After calling `compute` the Dask array becomes a NumPy array. 


## Postprocess 




## 
::::{grid}

:::{grid-item}
:outline:
:columns: 4
**Wave functions**  
`PlaneWave`  
`Probe`  
`Waves`
:::
:::{grid-item}
:outline:
:columns: 4
**Potentials**  
`Potential`  
`PotentialArray`  
`CrystalPotential`  
`ChargeDensityPotential`  
`GPAWPotential`  
:::
:::{grid-item}
:outline:
:columns: 4
**Detectors**  
`AnnularDetector`
`FlexibleAnnularDetector`
`SegmentedDetector`
`PixelatedDetector`
`WavesDetector`
:::
:::{grid-item}
:outline:
:columns: 4
**Scans**  
`GridScan`  
`LineScan`  
`CustomScan`  
:::
:::{grid-item}
:outline:
:columns: 4
**Measurements**  
`Images`  
`LineProfiles`  
`FourierSpaceLineProfiles`  
`DiffractionPatterns`  
`PolarMeasurements`  
:::
:::{grid-item}
:outline:
:columns: 4
**PRISM**  
`SMatrix`  
`SMatrixArray`  
`PartitionedSMatrix`  
:::
:::{grid-item}
:outline:
:columns: 4
**FrozenPhonons**  
`FrozenPhonons`  
`MDFrozenPhonons`  
:::
:::{grid-item}
:outline:
:columns: 4
**Transfer**  
`Aperture`  
`Aberrations`  
`CTF`  
:::
::::

