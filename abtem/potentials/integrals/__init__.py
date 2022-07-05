from abtem.potentials.integrals.gaussians import GaussianProjectionIntegrals
from abtem.potentials.integrals.infinite import InfinitePotentialProjections
from abtem.potentials.integrals.quadrature import ProjectionQuadratureRule

named_integrators = {'gaussian': GaussianProjectionIntegrals,
                     'infinite': InfinitePotentialProjections,
                     'quadrature': ProjectionQuadratureRule
                     }
