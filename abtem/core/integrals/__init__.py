from abtem.core.integrals.gaussians import GaussianProjectionIntegrals
from abtem.core.integrals.infinite import InfinitePotentialProjections
from abtem.core.integrals.quadrature import ProjectionQuadratureRule

named_integrators = {'gaussian': GaussianProjectionIntegrals,
                     'infinite': InfinitePotentialProjections,
                     'quadrature': ProjectionQuadratureRule
                     }
