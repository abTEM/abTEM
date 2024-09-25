import math

import numpy as np

from abtem.core.backend import cp

# ### expm ###
b = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
]

th13 = 5.37


def expm(a: np.ndarray) -> np.ndarray:
    """Compute the matrix exponential.

    Parameters
    ----------
    a : ndarray, 2D

    Returns
    -------
    matrix exponential of `a`

    Notes
    -----
    Uses (a simplified) version of Algorithm 2.3 of [1]_:
    a [13 / 13] Pade approximant with scaling and squaring.

    Simplifications:

        * we always use a [13/13] approximate
        * no matrix balancing

    References
    ----------
    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)
       https://doi.org/10.1137/04061101X

    """
    if a.size == 0:
        return cp.zeros((0, 0), dtype=a.dtype)

    n = a.shape[0]

    # try reducing the norm
    mu = cp.diag(a).sum() / n
    A = a - cp.eye(n) * mu

    # scale factor
    nrmA = cp.linalg.norm(A, ord=1).item()

    scale = nrmA > th13
    if scale:
        s = int(math.ceil(math.log2(float(nrmA) / th13))) + 1
    else:
        s = 1

    A /= 2**s

    # compute [13/13] Pade approximant
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4

    E = cp.eye(A.shape[0])

    u1, u2, v1, v2 = _expm_inner(E, A, A2, A4, A6, cp.asarray(b))
    u = A @ (A6 @ u1 + u2)
    v = A6 @ v1 + v2

    r13 = cp.linalg.solve(-u + v, u + v)

    # squaring
    x = r13
    for _ in range(s):
        x = x @ x

    # undo preprocessing
    x *= np.exp(mu)

    return x


@cp.fuse
def _expm_inner(
    E: np.ndarray,
    A: np.ndarray,
    A2: np.ndarray,
    A4: np.ndarray,
    A6: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    u2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * E

    v1 = b[12] * A6 + b[10] * A4 + b[8] * A
    v2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * E
    return u1, u2, v1, v2
