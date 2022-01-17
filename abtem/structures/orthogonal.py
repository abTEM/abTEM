import numpy as np


def make_atoms_orthogonal(atoms, max_reps, tol=1e-6):
    k = np.arange(-max_reps, max_reps)
    l = np.arange(-max_reps, max_reps)
    m = np.arange(-max_reps, max_reps)

    a, b, c = atoms.cell
    ka = k[None] * a[:, None]
    lb = l[None] * b[:, None]
    mc = m[None] * c[:, None]

    vectors = np.abs((ka.T[:, None, None] +
                      lb.T[None, :, None] +
                      mc.T[None, None, :]))

    norm = np.linalg.norm(vectors, axis=-1)
    nonzero = norm > tol
    norm[nonzero == 0] = tol

    for i in range(3):
        angles = (vectors[..., i] / norm)
        optimal = angles.max() == angles
        optimal = np.where(optimal * nonzero)

        j = np.argmin(np.linalg.norm(vectors[optimal], axis=1))

        ki = k[optimal[0][j]]
        li = l[optimal[1][j]]
        mi = m[optimal[2][j]]
        print(ki, li, mi)