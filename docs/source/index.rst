.. abTEM documentation master file, created by
   sphinx-quickstart on Fri Dec 27 18:19:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

abiniTEM: ab initio Transmission Electron Microscopy
====================================================

abiniTEM provides a Python API for running simulations of Transmission Electron Microscopy images.

>>> from ase import read
>>> from abiniTEM.waves import PlaneWaves
>>> atoms = read('SrTiO.cif')
>>> waves = PlaneWaves(sampling=0.1, energy=300e3)
>>> waves.multislice(atoms)
>>> waves.apply_ctf(defocus=200, focal_spread=40)
>>> waves.display_image(repeat=(5, 5))

abiniTEM works with the Atomic Simulation Environment and the density functional theory code GPAW


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorials/introduction
   modules/api
   cite
   ../../examples/test
