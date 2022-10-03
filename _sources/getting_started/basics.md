# The basics

This is a short overview of Dask geared towards new users. There is much more information contained in the rest of the documentation.

We normally import abTEM as follows:

The array is a [Dask Array](https://docs.dask.org/en/stable/array.html), this means that it is "lazily" evaluated. When we create a lazy array, we only set up the tasks to compute the array, to execute those tasks we also have to call `compute`. After calling `compute` the Dask array becomes a NumPy array. 



```{seealso}
See our walkthrough on [Dask in abTEM](walkthrough:dask) for details.
```


## The objects


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

## Mixing and matching

## Ensembles

## Interoperating

## Where to go from here?

