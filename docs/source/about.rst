About
=====

abTEM has been developed at the Faculty of Physics of the University of Vienna, Austria. Please consult the credits page
for information on how to cite abTEM. abTEM and its development are hosted on github. Bugs and feature requests are
ideally submitted via the github issue tracker.

How to cite abTEM
-----------------
If you use abTEM and its implemented methods in your research, we kindly ask that you cite the relevant publications::

abTEM implementation (tentative reference)::

    @Article{abtem,
    author = {Jacob Madsen and Toma Susi},
    title = {The abTEM code: transmission electron microscopy from first principles [version 2; peer review: 2 approved]
    },
    journal = {Open Research Europe},
    volume = {1},
    year = {2021},
    number = {24},
    doi = {10.12688/openreseurope.13015.2}
    }

ab initio electrostatic potentials from GPAW::

    @article{gpawpotentials,
    title = {Efficient first principles simulation of electron scattering factors for transmission electron microscopy},
    author = {Toma Susi and Jacob Madsen and Ursula Ludacka and Jens Jørgen Mortensen and Timothy J. Pennycook and Zhongbo Lee and Jani Kotakoski and Ute Kaiser and Jannik C. Meyer},
    journal = {Ultramicroscopy},
    volume = {197},
    pages = {16-22},
    year = {2019},
    }

The (default) potential parametrization by Ivan Lobato and Dirk Van Dyck::

    @article{lobato-parameters,
        title = {An accurate parameterization for the scattering factors, electron densities and electrostatic potentials for neutral atoms that obey all physical constraints},
        author = {Ivan Lobato and Dirk Van Dyck},
        journal = {Acta Crystallographica Section A},
        year = {2014},
        pages = {636-649},
        volume = {70}
    }

The potential parametrization by Earl J. Kirkland::

    @book{kirkland-parameters,
        author = {Earl J. Kirkland},
        title = {Advanced computing in electron microscopy},
        publisher = {Springer},
        year = {2010},
        edition = {2},
        isbn = {978-1-4419-6532-5}
    }

The PRISM algorithm by Colin Ophus::

    @article{prism,
        title = {A fast image simulation algorithm for scanning transmission electron microscopy},
        author = {Colin Ophus},
        journal = {Advanced Structural and Chemical Imaging},
        year = {2017},
        paper = {1},
        volume = {3}
    }

The atomic simulation environment (ASE) for setting up atomic structures::

    @article{ase,
        author={Ask Hjorth Larsen and Jens Jørgen Mortensen and Jakob Blomqvist and Ivano E Castelli and Rune Christensen and Marcin Dułak and Jesper Friis and Michael N Groves and Bjørk Hammer and Cory Hargus and Eric D Hermes and Paul C Jennings and Peter Bjerre Jensen and James Kermode and John R Kitchin and Esben Leonhard Kolsbjerg and Joseph Kubal and Kristen Kaasbjerg and Steen Lysgaard and Jón Bergmann Maronsson and Tristan Maxson and Thomas Olsen and Lars Pastewka and Andrew Peterson and Carsten Rostgaard and Jakob Schiøtz and Ole Schütt and Mikkel Strange and Kristian S Thygesen and Tejs Vegge and Lasse Vilhelmsen and Michael Walter and Zhenhua Zeng and Karsten W Jacobsen},
        title = {The atomic simulation environment — a Python library for working with atoms},
        journal = {Journal of Physics: Condensed Matter},
        volume = {29},
        number = {27},
        pages = {273002},
        year = {2017},
    }

GPAW for calculating DFT potentials::

    @article{gpaw,
        author = {J. J. Mortensen and L. B. Hansen and K. W. Jacobsen},
        title = {Real-space grid implementation of the projector augmented wave method},
        year = {2005},
        volume = {71},
        number = {3},
        pages = {035109},
        journal = {Phys. Rev. B},
    }

Linear-scaling algorithm for calculation of inelastic transitions by Hamish G. Brown et al::

    @article{PhysRevResearch.1.033186,
      title = {Linear-scaling algorithm for rapid computation of inelastic transitions in the presence of multiple electron scattering},
      author = {Hamish G. Brown and Jim Ciston and Colin Ophus},
      journal = {Phys. Rev. Research},
      volume = {1},
      issue = {3},
      pages = {033186},
      numpages = {10},
      year = {2019},
      month = {Dec},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.1.033186},
    }

Linear-scaling algorithm for calculation of inelastic transitions by  Brown et al::

    @article{PhysRevResearch.1.033186,
      title = {Linear-scaling algorithm for rapid computation of inelastic transitions in the presence of multiple electron scattering},
      author = {Hamish G. Brown and Jim Ciston and Colin Ophus},
      journal = {Phys. Rev. Research},
      volume = {1},
      issue = {3},
      pages = {033186},
      numpages = {10},
      year = {2019},
      month = {Dec},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.1.033186},
    }

