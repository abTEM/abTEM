# Installation

There are two ways to install the abTEM package, using pip or conda:

`````{tab-set}

````{tab-item} conda
Install abtem using conda:
```{code-block}
conda install -c conda-forge abtem
```
````
````{tab-item} pip
Install abtem using pip:
```{code-block}
pip install abtem
```

Alternatively, if you have git and want to use unreleased features, you can install directly from github:
```{code-block}
pip install git+https://github.com/abTEM/abTEM
```
````
`````

## Optional dependencies

### GPAW (not available on Windows)

Some features of abTEM, such as calculating potentials from DFT require a working installation
of [GPAW](https://wiki.fysik.dtu.dk/gpaw/index.html). See [here](https://wiki.fysik.dtu.dk/gpaw/install.html) for
detailed installation instructions.

`````{tab-set}
````{tab-item} conda
Install gpaw using conda:
```{code-block}
conda install -c conda-forge gpaw
```
````
````{tab-item} pip

```{code-block}
Install gpaw using pip:
pip install gpaw
```
Install the PAW datasets into the folder `<dir>` using this command:
```{code-block}
gpaw install-data <dir>
```
````
`````

### GPU (only Nvidia) 

GPU calculations with abTEM require a working installation of [CuPy](https://cupy.dev/).
See [here](https://docs.cupy.dev/en/stable/install.html) for detailed installation instructions.

`````{tab-set}

````{tab-item} conda
Install CuPy using conda:
```{code-block}
conda install -c conda-forge cupy
```
````
````{tab-item} pip
Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

Install CuPy using pip:
```{code-block}
pip install cupy-cuda*
```
where * should be substituted for the CUDA Toolkit version.
````
`````

### Development

See [our guide to contributing](library:contributing) for instructions on a development installation `abTEM`.