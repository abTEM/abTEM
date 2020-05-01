import torch.nn as nn


class RadialFunction:

    def __init__(self, eta, rs):
        self.eta = eta
        self.rs = rs

    def __call__(self, rij):
        return -torch.exp(-self.eta * (rij - self.rs) ** 2)


class Test(nn.Module):

    def __init__(self, positions, radial_function):
        super().__init__()
        self.positions = torch.nn.Parameter(data=positions, requires_grad=True)
        self.radial_function = radial_function

    def forward(self):
        rij = torch.pdist(self.positions)
        # rij = rij[rij < 1]
        return torch.sum(self.radial_function(rij))

    energy = test()

    energy.backward()
    optimizer.step()
    optimizer.zero_grad()

import math

def calc_row_idx(k, n):
    return (torch.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1)).type(torch.int64)

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return (n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

positions = torch.tensor(points.positions)
angles = torch.tensor([0, 2*np.pi/3, 4*np.pi/3])
#positions = torch.stack((torch.cos(angles), -torch.sin(angles))).T


rij = torch.pdist(positions)

k = rij < 2

#plt.imshow((k[:,None] * k[None]).numpy())

#for i in tqdm(range(20000)):
condensed = torch.where(k)[0]#.#numpy()
z = torch.zeros((len(positions),)*2)
z = z.to(device)
idx = condensed_to_square(condensed, len(positions))
upper = idx[0] > idx[1]
z[idx[0],idx[1]] = 1
z[idx[1],idx[0]] = 1

idx

#Z = z[:,:,None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test = Test(positions, radial_function)
test.to(device)

optimizer = torch.optim.Adam(test.parameters(), lr=1e-2)

import torch.nn as nn


def calc_row_idx(k, n):
    return torch.ceil((- np.sqrt(-8 * k + 4 * n ** 2 - 4 * n - 7) + 2 * n - 1) / 2. - 1).type(torch.int64)


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    return (n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def square_to_condensed(i, j, n):
    return n * j - j * (j + 1) // 2 + i - 1 - j


def distances_and_angles(positions, cutoff):
    distances = torch.pdist(positions)
    mask = distances < cutoff

    adjacent = condensed_to_square(torch.where(mask)[0], len(positions))
    adjacency_matrix = torch.zeros((len(positions),) * 2)
    adjacency_matrix = adjacency_matrix.to(device)
    adjacency_matrix[adjacent[0], adjacent[1]] = 1
    adjacency_matrix[adjacent[1], adjacent[0]] = 1
    adjacency_matrix = adjacency_matrix[:, :, None] * adjacency_matrix[:, None, :]

    triplets = torch.stack(torch.where(adjacency_matrix)).T
    triplets = triplets[triplets[:, 1] < triplets[:, 2]]
    triplet_positions = positions[triplets]
    # triplet_distances
    print(triplets)
    AB = triplet_positions[:, 0, None, :] - triplet_positions[:, 1, None, :]
    AC = triplet_positions[:, 0, :, None] - triplet_positions[:, 2, :, None]
    angles = torch.acos(torch.bmm(AB, AC) / (torch.norm(AB, dim=2, keepdim=True) * torch.norm(AC, dim=1, keepdim=True)))

    print(torch.norm(AB, dim=2, keepdim=True)[0, 0], torch.norm(AC, dim=1, keepdim=True)[0, 0])
    print(square_to_condensed(triplets[:, 0], triplets[:, 0]))

    return distances[distances < cutoff], angles.squeeze()


angles = torch.tensor([0, 2 * np.pi / 4, 2 * 2 * np.pi / 4, 3 * 2 * np.pi / 4])
positions = torch.stack((torch.cos(angles), -torch.sin(angles))).T

# positions

distances_and_angles(positions, 2.5)

from scipy.ndimage import zoom
import skimage.morphology as morphology
from sklearn.neighbors import BallTree



def region_border(region, scale_factor=.25):
    region = zoom(region, scale_factor, order=1)
    region = region > .5
    dilated_region = morphology.binary_dilation(region, selem=morphology.disk(1))
    return np.array(np.where(dilated_region^region)).T / scale_factor

def points_close_to_other_points(points, other_points, distance):
    ball_tree = BallTree(points)
    close_points = ball_tree.query_radius(other_points, distance)
    close_points = np.array(close_points)
    return np.concatenate(close_points)