Installation
============

Installing abinitEM
-------------------
To install ``abinitEM`` with pip, run::

    pip install abinitEM

If you want to make sure none of your existing dependencies get upgraded, you
can also do::

    pip install abinitEM --no-deps

On the other hand, if you want to install ``abinitEM`` along with all of the available optional dependencies, you can
do::

    pip install abinitEM[all]

Requirements
------------
``abinitEM`` has the following strict requirements:

- `Python <https://www.python.org/>`_ 3.6 or later
- `Numpy <https://www.numpy.org/>`_ 1.16.0 or later
- `Numba <https://www.numba.org/>`_ 0.46 or later
- `pytest <http://www.pytest.org/>`_ 3.1 or later

``abinitEM`` also depends on other packages for optional features:

- `Jupyter <https://jupyter.org/>`_: To work with ``abinitEM`` in notebooks.
- `scipy <https://scipy.org/>`_: To power a variety of features in several modules.
- `pyFFTW <https://pyfftw.readthedocs.io/>`_: To speed up FFT's
- `h5py <https://h5py.org/>`_: To read/write objects from/to HDF5 files.
- `numexpr <https://numexpr.readthedocs.io/en/latest/>`_: To speed up the evaluation of some mathematical expressions.
- `matplotlib <https://matplotlib.org/>`_ 2.0 or later: To provide plotting functionality.

The custom graphical user interfaces of ``abinitEM`` depends on:

- `pillow <https://pillow.readthedocs.io/>`_
- `bqplot <https://bqplot.readthedocs.io/>`_
- `ipywidgets <https://ipywidgets.readthedocs.io/>`_


