(getting_started:overview)=
# Overview

`abTEM` is an open-source code for simulation code designed to meet these demands by integrating the ability to
calculate
the potential via density functional theory (DFT) with a flexible modular software design.

See our 

## Why abTEM?

and
allows for a simple and intuitive user interface.

The computationally demanding parts are implemented using jit-compiled
Numba code and high-performance libraries, maintaining speed while ensuring portability.

abTEM works with the Atomic Simulation Environment and the density functional theory code GPAW to provide a seamless
environment for simulating images from first principles.

### The abTEM object model

abTEM differs from most other codes by not directly implementing any common imaging modes (STEM, HRTEM etc.), but
invites the user to mix and match objects to construct the desired simulation .
This design patterns is inspired from object-oriented scientific codes, particularly ASE

The objects are based on models and concepts that should be familiar to physicists, for example, the `PlaneWave` object
represents a plane wave function and the `Potential` an electrostatic potential. See a list of the most important
objects in 

## Integration

## Speed and scale
