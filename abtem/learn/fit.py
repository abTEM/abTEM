import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize
from sklearn.neighbors import NearestNeighbors


# @numba.jit(nopython=True)
def _disk(radius):
    L = np.arange(-radius, radius + 1)
    return ((L.reshape((-1, 1)) ** 2 + L.reshape((1, -1)) ** 2).reshape((-1)) <= radius ** 2)


# @numba.jit(nopython=True)
def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


class SinglePeakModel:

    def __init__(self, r, num_parameters):
        self._r = r
        disk = _disk(r)
        self._u = ((np.arange(0, (2 * r + 1) ** 2) % (2 * r + 1)) - r)[disk]
        self._v = (np.repeat(np.arange(0, 2 * r + 1), 2 * r + 1) - r)[disk]
        self._num_parameters = num_parameters
        self._parameters = None

    @property
    def positions(self):
        return self._positions

    @property
    def r(self):
        return self._r

    @property
    def parameters(self):
        return self._parameters

    @property
    def num_parameters(self):
        return self._num_parameters

    def fit(self, image, positions):
        self._parameters = np.zeros((len(positions), self._num_parameters), dtype=np.float64)
        integer_positions = np.rint(positions).astype(np.int64)
        self._positions = positions.astype(np.float64)

        shape = image.shape
        image = image.astype(np.float64)
        image = image.ravel()

        for i in range(len(integer_positions)):
            position = integer_positions[i]
            ind = sub2ind(position[0] + self._u, position[1] + self._v, shape)
            params, displacement = self.fit_single(image[ind])
            self._parameters[i] = params
            self._positions[i] = position + displacement

    def fit_single(self, z):
        raise NotImplementedError()


class Polynomials(SinglePeakModel):

    def __init__(self, r):
        super().__init__(r, 6)
        self._X = (np.vstack(
            (np.ones_like(self._u), self._u, self._v, self._u ** 2, self._u * self._v, self._v ** 2))).T.astype(
            np.float64)

    def fit_single(self, z):
        params, residues, rank, singval = np.linalg.lstsq(self._X, z, rcond=None)
        M = np.array([[2 * params[3], params[4]], [params[4], 2 * params[5]]])
        displacement = np.linalg.solve(M, -params[1:3])
        return params, displacement


class Gaussians(SinglePeakModel):

    def __init__(self, r, elliptical=False):
        self._elliptical = elliptical
        if elliptical:
            num_parameters = 7
        else:
            num_parameters = 5
        super().__init__(r, num_parameters)
        self._X = (np.vstack(
            (np.ones_like(self._u), self._u, self._v, self._u ** 2, self._u * self._v, self._v ** 2))).T.astype(
            np.float64)

    @property
    def elliptical(self):
        return self._elliptical

    def fit_single(self, z):
        initial = np.zeros(self._num_parameters)
        initial[0] = (self._u * z).sum() / z.sum()
        initial[1] = (self._v * z).sum() / z.sum()
        initial[2] = z.min()
        initial[3] = z.max() - z.min()

        params, residues, rank, singval = np.linalg.lstsq(self._X, z, rcond=None)

        if self.elliptical:
            initial[4] = np.abs(params[3])
            initial[5] = 0
            initial[6] = np.abs(params[5])

            bounds = [(-self.r, -self.r, 0, 0, 0, -self.r, 0),
                      (self.r, self.r, z.max(), np.inf, self.r, self.r, self.r)]

            def fit_func(p):
                x0, y0, z0, A, a, b, c = p
                return A * np.exp(
                    -(a * (self._u - x0) ** 2 - 2 * b * (self._u - x0) * (self._v - y0) + c * (self._v - y0) ** 2)) - z
        else:
            initial[4] = (np.abs(params[3]) + np.abs(params[5])) / 2

            bounds = [(-self.r, -self.r, 0, 0, 0),
                      (self.r, self.r, z.max(), np.inf, self.r)]

            def fit_func(p):
                x0, y0, z0, A, a = p
                return z0 + A * np.exp(-a * ((self._u - x0) ** 2 + (self._v - y0) ** 2)) - z

        ls = optimize.least_squares(fit_func, initial, bounds=bounds)

        return ls.x, ls.x[:2]


class GaussianSuperposition:

    def __init__(self, r, n_neighbors=12, max_iter=100):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.r = r

    def get_image(self):
        r2 = torch.sum((self.pixel_positions - self.positions[self.closest]) ** 2, axis=-1)
        return torch.sum(self.heights[self.closest] * torch.exp(-self.widths[self.closest] * r2), axis=1)

    def fit(self, image, positions, initial_width):

        # gaussians = Gaussians(self.r)
        # gaussians.fit(image, positions)

        pixel_positions = np.rollaxis(np.indices(image.shape), 0, 3).reshape((-1, 2)).astype(np.float32)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(positions)
        _, closest = nbrs.kneighbors(pixel_positions)

        self.pixel_positions = torch.from_numpy(pixel_positions)[:, None]
        self.closest = torch.from_numpy(closest)
        image = torch.from_numpy(image.astype(np.float32).ravel())


        self.positions = torch.tensor(positions.astype(np.float32), requires_grad=True)
        self.heights = torch.tensor(np.ones(len(positions), dtype=np.float32), requires_grad=True)
        self.widths = torch.tensor(np.full(len(positions), initial_width, dtype=np.float32), requires_grad=True)

        loss_fn = F.mse_loss
        opt_1 = torch.optim.SGD([self.positions], lr=1)
        opt_2 = torch.optim.SGD([self.heights], lr=1)
        opt_3 = torch.optim.SGD([self.widths], lr=.0010)

        for epoch in range(self.max_iter):

            loss = loss_fn(self.get_image(), image)
            loss.backward()
            opt_1.step()
            opt_1.zero_grad()
            opt_2.step()
            opt_2.zero_grad()
            opt_3.step()
            opt_3.zero_grad()

            if (epoch % 10) == 0:
                print(epoch, loss.detach().numpy())
