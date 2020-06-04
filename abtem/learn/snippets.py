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


def segment_to_points():
    segmentation = merge_dopants_into_contamination(segmentation)
    contamination = segmentation == 1


    not_contaminated = contamination[points[:, 0], points[:, 1]] == 0
    points = points[not_contaminated]
    labels = labels[not_contaminated]

    scale_factor = 16
    contamination = skimage.util.view_as_blocks(contamination, (scale_factor,) * 2).sum((-2, -1)) > (
            scale_factor ** 2 / 2)
    contamination = (np.array(np.where(contamination)).T) * 16 + 8

    points = np.vstack((points, contamination))
    labels = np.concatenate((labels, np.ones(len(contamination))))


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





column_names = ['include', 'num_missing', 'enclosed_area', 'center_x', 'center_y', 'num_enclosed', 'dopant',
                'stable', 'contamination', 'closed', 'outside_image']

# column_names += ['<4-sided', '4-sided', '5-sided', '6-sided', '7-sided', '8-sided', '9-sided', '>9-sided']

data = dict(zip(column_names, [] * len(column_names)))
self._table_source = models.ColumnDataSource(data)
columns = [models.TableColumn(field=name, title=name) for name in column_names]
self._data_table = models.DataTable(source=self._table_source, columns=columns, width=800)


class R2PartialConv(e2nn.R2Conv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(self.kernel_size)
        self.register_buffer('mask_kernel', torch.ones(1, 1, self.kernel_size, self.kernel_size))

        self._last_size = None
        self._mask_ratio = None

    def forward(self, x):

        if self._last_size != tuple(x.shape):
            self._last_size = tuple(x.shape)

            with torch.no_grad():
                mask = torch.ones(1, 1, x.shape[2], x.shape[3]).to(self.mask_kernel)

                self._mask_ratio = F.conv2d(mask, self.mask_kernel, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self._mask_ratio = self.kernel_size ** 2 / (self._mask_ratio + 1e-8)

        x = super().forward(x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.bias.shape[0], 1, 1)

            x.tensor = torch.mul(x.tensor - bias_view, self._mask_ratio) + bias_view
        else:
            x.tensor = torch.mul(x.tensor, self._mask_ratio)

        return x



def analyse_points(points, labels, shape):
    faces = stable_delaunay_faces(points, 2., 3 / .05)
    edges = faces_to_edges(faces)

    polygons = [points[face].astype(np.float) for face in faces]
    dual_points = np.array([polygon.mean(axis=0) for polygon in polygons])

    border_adjacent = labels == 1

    image_bounding_box = [[-20, 1044], [-20, 1044]]
    border_adjacent += points_in_bounding_box(points, image_bounding_box) == 0
    outer_faces = outer_faces_from_faces(faces)
    border_adjacent[flatten_list_of_lists(outer_faces)] = 1
    border_adjacent = np.where(border_adjacent)[0]
    border_adjacent_faces = select_faces_around_nodes(border_adjacent, faces)

    dual_degree = np.array([len(face) for face in faces])

    polygons = [polygon - dual_point for polygon, dual_point in zip(polygons, dual_points)]
    hexagon = regular_polygon(27, 6)
    rmsd = pairwise_rmsd([hexagon], polygons)[0]

    defect_faces = rmsd > 20
    image_bounding_box = [[-5, 1029], [-5, 1029]]
    labels[(labels == 2) & (points_in_bounding_box(points, image_bounding_box) == 0)] = 0

    defect_faces[select_faces_around_nodes(np.where((labels == 2))[0], faces)] = 1
    defect_faces[border_adjacent_faces] = 0

    dual_adjacency = order_adjacency_clockwise(dual_points, faces_to_dual_adjacency(faces))

    defect_faces_indices = np.where(defect_faces)[0]
    column_names = ['include', 'num_missing', 'enclosed_area', 'center_x', 'center_y', 'num_enclosed', 'dopant',
                    'stable', 'contamination', 'closed', 'outside_image', 'histogram', 'enclosing_path']
    data = dict(zip(column_names, [[] for i in range(len(column_names))]))

    paths = []
    if len(defect_faces_indices) == 0:
        return data, points[edges], dual_points, dual_degree, paths

    defects = [defect_faces_indices[component] for component in
               connected_components(subgraph_adjacency(defect_faces_indices, dual_adjacency))]

    for defect in defects:
        try:
            path = enclosing_path(dual_points, defect, dual_adjacency)
        except:
            print('No path found')
            continue

        polygon = dual_points[np.concatenate((path, [path[0]]))]

        inside_all = np.array([point_in_polygon(point, polygon) for point in points])
        if np.any(labels[inside_all] == 3):
            continue
        paths.append(polygon)
        reference_path = hexagonal_reference_path_from_path(path, dual_adjacency)
        density = 6 / (3 * np.sqrt(3) / 2)
        inside = np.array([point_in_polygon(point, polygon) for point in points[labels != 1]])
        outside = np.ceil(np.abs(min(0., np.min(polygon), np.min(np.array((1024, 1024)) - polygon))))
        center = np.mean(polygon, axis=0)

        is_contamination = np.any(labels[inside_all] == 1)

        data['num_missing'].append(int(np.round(polygon_area(reference_path) * density - np.sum(inside))))
        data['enclosed_area'].append(round(polygon_area(polygon), 3))

        data['center_x'].append(round(center[0], 3))
        data['center_y'].append(round(center[1], 3))
        data['num_enclosed'].append(np.sum(inside))

        data['contamination'].append(is_contamination)
        data['dopant'].append(np.any(labels[inside_all] == 2))
        data['stable'].append(not np.any(labels[inside_all] == 3))

        data['closed'].append(np.all(np.isclose(reference_path[0], reference_path[-1])))
        data['outside_image'].append(outside)

        data['include'].append(data['closed'][-1] * (data['outside_image'][-1] < 15) *
                               data['stable'][-1] * (not data['contamination'][-1]))

        data['histogram'].append(np.histogram(dual_degree[defect], np.arange(3, 12))[0])
        data['enclosing_path'].append(polygon)

    return data, points[edges], dual_points, dual_degree, paths

# self._dual_source.data = {'x': list(dual_points[:, 0]), 'y': list(dual_points[:, 1]),
#                           'labels': list(dual_degree)}
#
# self._table_source.data = data
#
# top = [polygon[:, 1].max() for polygon in polygons]
# bottom = [polygon[:, 1].min() for polygon in polygons]
# left = [polygon[:, 0].min() for polygon in polygons]
# right = [polygon[:, 0].max() for polygon in polygons]
# line_color = ['green' if include else 'red' for include in data['include']]
#
# self._rect_source.data = {'top': top, 'bottom': bottom, 'left': left, 'right': right, 'line_color': line_color}
#
#
# self._dual_source = models.ColumnDataSource(data=dict(x=[], y=[], labels=[]))
#         self._dual_model = models.Circle(x='x', y='y', fill_color=mapper, radius=8)
#         self._dual_glyph = viewer._figure.add_glyph(self._dual_source, glyph=self._dual_model)
#
#         self._rect_source = models.ColumnDataSource(data=dict(top=[], bottom=[], left=[], right=[], line_color=[]))
#         self._rect_model = models.Quad(top='top', bottom='bottom', left='left', right='right', line_color='line_color',
#                                        fill_color=None, line_width=3)
#         self._rect_glyph = viewer._figure.add_glyph(self._rect_source, glyph=self._rect_model)