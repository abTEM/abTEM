"""Module to describe projection integrals of radial parametrizations."""
from abtem.integrals.gaussians import GaussianProjectionIntegrals
from abtem.integrals.infinite import InfinitePotentialProjections
from abtem.integrals.quadrature import ProjectionQuadratureRule

named_integrators = {'gaussian': GaussianProjectionIntegrals,
                     'infinite': InfinitePotentialProjections,
                     'quadrature': ProjectionQuadratureRule
                     }
