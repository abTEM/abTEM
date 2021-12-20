Installation
============

Install with pip
----------------

To install ``abtem`` with pip run::

    pip install abtem

This installs ``abtem`` with its strict dependencies (see below); some functionality requires additional packages.

To installl ``abtem`` using conda run::

    conda install -c conda-forge abtem

For a development installation::

    git clone https://github.com/jacobjma/abTEM
    cd abtem
    pip install -e .

GPU calculations (CUDA only)
****************************
GPU calculations with abTEM require CUDA Toolkit and CuPy.

`Download <https://developer.nvidia.com/cuda-10.2-download-archive>`_ and install CUDA toolkit 11 from NVIDIA's website. On Windows, you may be prompted to install Visual Studio for some features, but this is *not* required to use abTEM.

Install CuPy from pip::

    pip install cupy-cuda*

where * should be substituted for the CUDA Toolkit version. See `CuPy's installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for details.

DFT potentials with GPAW (optional)
***********************************

abTEM can simulate bond-sensitive
GPAW can be installed from pip (note that GPAW is not officially supported on Windows)::

    pip install GPAW

See `GPAW's installation guide <https://wiki.fysik.dtu.dk/gpaw/>`_ for more details.

Install with Anaconda
---------------------
The recommended installation for abTEM uses the Anaconda python distribution. First, `download and install Anaconda <`www.anaconda.com/download>`_. Then open a terminal (on Windows use the Anaconda prompt) and run::

    conda update conda
    conda create --name abtem python=3.8
    conda activate abtem
    pip install abtem

This creates a new virtual environment and installs abtem into it.

Install CUDA toolkit and CuPy
*****************************
Once the CUDA driver is correctly set up, you can install CuPy from the conda-forge channel::

    conda install -c conda-forge cupy

and conda will install pre-built CuPy and most of the optional dependencies for you, including CUDA toolkit.

Install Jupyter and add the IPython kernel
******************************************
To be able to use the conda environment in a Jupyter Notebook, you have to install IPython kernel::

    conda install jupyter
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=abtem

Dependencies
------------
Strict dependencies:

* `Numpy <https://www.numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `Numba <https://www.numba.org/>`_
* `pyfftw <https://hgomersall.github.io/pyFFTW/>`_
* `imageio <https://imageio.github.io/>`_
* `h5py <https://h5py.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `ase <https://wiki.fysik.dtu.dk/ase/>`_
* `tqdm <https://tqdm.github.io/>`_
* `psutil <https://github.com/giampaolo/psutil>`_

Only for GPU calculations:

* `CuPy <https://cupy.dev/>`_

Only for DFT potentials with GPAW:

* `GPAW <https://wiki.fysik.dtu.dk/gpaw/>`_

Only for interactive graphical interfaces:

* `bqplot <https://bqplot.readthedocs.io/en/latest/>`_
* `bqplot-image-gl <https://github.com/glue-viz/bqplot-image-gl>`_
* `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_

Only for testing:

* `pytest <http://www.pytest.org/>`_
