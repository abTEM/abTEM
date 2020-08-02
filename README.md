[![Gitter](https://badges.gitter.im/abTEM/community.svg)](https://gitter.im/abTEM/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# abTEM
ab initio Transmission Electron Microscopy

This program is under development and subject to changes.

## Dependencies

Strict dependencies:
* numpy
* scipy
* numba
* imageio
* h5py
* matplotlib
* ase
* tqdm

Only for GPU calculations:
* CuPy

Only for DFT potentials with GPAW:
* GPAW

Only for interactive graphical interfaces:
* Bokeh
* ipywidgets

Only for testing:
* pytest
* hypothesis

## Install

To install ``abtem`` with pip and the strict dependencies, run

    pip install abtem

If you want to make sure none of your existing dependencies get upgraded, you can also do

    pip install abtem --no-deps

### GPU (only CUDA)
GPU calculations with abTEM requires CUDA Toolkit and CuPy, see CuPy's installation guide for instructions.

### DFT with GPAW
Install GPAW according to the installation instructions. Note that GPAW is not officially supported on Windows.

### Install with Anaconda





