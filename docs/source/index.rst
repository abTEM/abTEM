.. abTEM documentation master file, created by
   sphinx-quickstart on Fri Dec 27 18:19:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

abTEM: ab initio Transmission Electron Microscopy
====================================================

abTEM provides a Python API for running simulations of Transmission Electron Microscopy images. It is written entirely
in Python, which enables easy integration with first-principles codes and analysis tools accessible from Python,
and allows for a simple and intuitive user interface. The computationally demanding parts are implemented using
jit-compiled Numba code and high-performance libraries, maintaining speed while ensuring portability.

abTEM works with the Atomic Simulation Environment and the density functional theory code GPAW to provide an seamless
environment for simulating images from first principles.

>>> from ase import read
>>> from abTEM.waves import PlaneWaves
>>> atoms = read('SrTiO.cif')
>>> waves = PlaneWaves(sampling=0.1, energy=300e3)
>>> waves.multislice(atoms)
>>> waves.apply_ctf(defocus=200, focal_spread=40)
>>> waves.display_image(repeat=(5, 5))

.. toctree::
   :maxdepth: 1
   :caption: Main

   install
   how_to_use/introduction
   .. tutorials/introduction
   .. modules/api
   .. cite
   ../../examples/test

abTEM has been developed at the Faculty of Physics of the University of Vienna, Austria. Please consult the credits page
for information on how to cite abTEM. abTEM and its development are hosted on github. Bugs and feature requests are
ideally submitted via the github issue tracker.