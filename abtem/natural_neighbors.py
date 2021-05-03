import numpy as np
from scipy.spatial import cKDTree, ConvexHull, qhull


def triangle_area(pt1, pt2, pt3):
    a = pt1[0] * pt2[1] - pt2[0] * pt1[1] + pt2[0] * pt3[1] - pt3[0] * pt2[1] + pt3[0] * pt1[1] - pt1[0] * pt3[1]
    return abs(a) / 2


def circumcircle_radius(pt0, pt1, pt2):
    a = np.linalg.norm(pt0 - pt1)
    b = np.linalg.norm(pt1 - pt2)
    c = np.linalg.norm(pt2 - pt0)

    t_area = triangle_area(pt0, pt1, pt2)

    if t_area > 0:
        radius = (a * b * c) / (4 * t_area)
    else:
        radius = np.nan

    return radius


def circumcenter(pt0, pt1, pt2):
    a_x, a_y = pt0
    b_x, b_y = pt1
    c_x, c_y = pt2

    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    cb_x_diff = c_x - b_x
    ac_x_diff = a_x - c_x
    ba_x_diff = b_x - a_x

    d_div = (a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff)
    d_inv = 0.5 / d_div

    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y

    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv
    return cx, cy


def find_natural_neighbors(tri, grid_points):
    # Used for fast identification of points with a radius of another point
    tree = cKDTree(grid_points)

    # Mask for points that are outside the triangulation
    in_triangulation = tri.find_simplex(tree.data) >= 0

    triangle_info = []
    members = {key: [] for key in range(len(tree.data))}
    for i, indices in enumerate(tri.simplices):
        # Find the circumcircle (center and radius) for the triangle.
        triangle = tri.points[indices]
        cc = circumcenter(*triangle)
        r = circumcircle_radius(*triangle)
        triangle_info.append(cc)

        # Find all grid points within the circumcircle.
        for point in tree.query_ball_point(cc, r):
            # If this point is within the triangulation, add this triangle to its list of
            # natural neighbors
            #if in_triangulation[point]:
            members[point].append(i)

    return members, np.array(triangle_info)


def find_local_boundary(tri, triangles):
    r"""Find and return the outside edges of a collection of natural neighbor triangles.
    There is no guarantee that this boundary is convex, so ConvexHull is not
    sufficient in some situations.
    Parameters
    ----------
    tri: `scipy.spatial.Delaunay`
        A Delaunay Triangulation
    triangles: (N, ) array
        List of natural neighbor triangles.
    Returns
    -------
    edges: list
        List of vertex codes that form outer edges of
        a group of natural neighbor triangles.
    """
    edges = []

    for triangle in triangles:

        for i in range(3):

            pt1 = tri.simplices[triangle][i]
            pt2 = tri.simplices[triangle][(i + 1) % 3]

            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))

            elif (pt2, pt1) in edges:
                edges.remove((pt2, pt1))

            else:
                edges.append((pt1, pt2))

    return edges


def area(poly):
    r"""Find the area of a given polygon using the shoelace algorithm.
    Parameters
    ----------
    poly: (2, N) ndarray
        2-dimensional coordinates representing an ordered
        traversal around the edge a polygon.
    Returns
    -------
    area: float
    """
    a = 0.0
    n = len(poly)

    for i in range(n):
        a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]

    return abs(a) / 2.0


def order_edges(edges):
    r"""Return an ordered traversal of the edges of a two-dimensional polygon.
    Parameters
    ----------
    edges: (2, N) ndarray
        List of unordered line segments, where each
        line segment is represented by two unique
        vertex codes.
    Returns
    -------
    ordered_edges: (2, N) ndarray
    """
    edge = edges[0]
    edges = edges[1:]

    ordered_edges = [edge]

    num_max = len(edges)
    while len(edges) > 0 and num_max > 0:

        match = edge[1]

        for search_edge in edges:
            vertex = search_edge[0]
            if match == vertex:
                edge = search_edge
                edges.remove(edge)
                ordered_edges.append(search_edge)
                break
        num_max -= 1

    return ordered_edges


def natural_neighbor_weights(points, query_point, tri, neighbors, circumcenters):
    edges = find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    c1 = circumcenter(query_point, tri.points[p1], tri.points[p2])
    polygon = [c1]

    weights = np.zeros(len(points))

    for i in range(num_vertices):
        p3 = edge_vertices[(i + 2) % num_vertices]
        try:
            c2 = circumcenter(query_point, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(circumcenters[check_tri])

            pts = [polygon[i] for i in ConvexHull(polygon).vertices]

            cur_area = area(pts)
            weights[(tri.points[p2][0] == points[:, 0]) & (tri.points[p2][1] == points[:, 1])] += cur_area

        except (ZeroDivisionError, qhull.QhullError) as e:
            print(polygon, query_point)
            weights[:] = np.nan
            return weights

        polygon = [c2]
        p2 = p3

    weights = weights / weights.sum()

    return weights
