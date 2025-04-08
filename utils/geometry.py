import numpy as np
from numba import njit
from numba import prange
from numba.types import boolean

@njit
def is_inside_sm(polygon, point):
    """
    Returns bool. True if point inside polygon.
    Ensure polygon is closed (first and last points are the same).
    """
    if (polygon[0] != polygon[-1]).any():
        polygon = np.vstack((polygon, polygon[:1]))
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    return intersections & 1

@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    """Returns bool. True if point inside polygon. Parallel version."""
    ln = len(points)
    D = np.empty(ln, dtype=boolean)
    for i in prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D

def trajectory_inside_vertices_idxs(x, vertices, closed=False):
    """
    x: trajectory. x[0] = lat, x[1] = lon
    vertices: vertices of polygon. vertices[:, 0] = lon, vertices[:, 1] = lat
    """
    if not closed:
        vertices_closed = np.vstack([vertices, vertices[0]])
    else:
        vertices_closed = vertices.copy()
    inside = is_inside_sm_parallel(x[:2][::-1].T, vertices_closed)
    if inside.any():
        # take longest continuous sequence of True values
        inside_changes = np.diff(inside.astype(int))
        inside_start = np.where(inside_changes == 1)[0]
        inside_end = np.where(inside_changes == -1)[0]
        if inside[0]:
            inside_start = np.hstack((0, inside_start))
        if inside[-1] and inside_start.size > inside_end.size:
            inside_end = np.hstack((inside_end, inside.size-1))
        if inside_end.size == 0:
            inside_end = np.array([inside.size-1])
        else:
            inside_end += 1
        assert inside_start.size == inside_end.size, "inside_start and inside_end must have same size"
        inside_max_size = np.argmax(inside_end - inside_start)
        inside_start_max = inside_start[inside_max_size]
        inside_end_max = inside_end[inside_max_size]
        return inside_start_max, inside_end_max
    else:
        return None, None

def trajectory_inside_vertices(x, vertices, *other_xs, closed=False):
    """
    x: trajectory. x[0] = lat, x[1] = lon
    vertices: vertices of polygon. vertices[:, 0] = lon, vertices[:, 1] = lat
    """
    inside_start_max, inside_end_max = trajectory_inside_vertices_idxs(x, vertices, closed)
    other_xs = list(other_xs)
    if inside_start_max is not None:
        y = x[:, inside_start_max:inside_end_max]
        if other_xs:
            for i, other_x in enumerate(other_xs):
                other_xs[i] = other_x[inside_start_max:inside_end_max]
    else:
        y = np.nan
        if other_xs:
            for i, other_x in enumerate(other_xs):
                other_xs[i] = np.nan
    other_xs = tuple(other_xs)
    return (y, *other_xs)

def prune_trajectories_inside_vertices(trajectories, years, labels, vertices):
    """
    trajectories: trajectories. trajectories[0] = lat, trajectories[1] = lon
    years: year of trajectories
    labels: labels of trajectories
    """
    trajectories_inside = []
    years_inside = []
    labels_idxs = []
    for i, (trajectory, year) in enumerate(zip(trajectories, years)):
        trajectory_pruned, year_pruned = trajectory_inside_vertices(trajectory, vertices, year)
        if not np.isnan(trajectory_pruned).any():
            trajectories_inside.append(trajectory_pruned)
            years_inside.append(year_pruned)
            labels_idxs.append(i)
    labels_inside = labels.iloc[labels_idxs]
    return trajectories_inside, years_inside, labels_inside
