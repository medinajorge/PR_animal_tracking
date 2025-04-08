import math
import numpy as np
import pandas as pd
from numba import njit
import gc
from contourpy import contour_generator
from scipy.stats import norm, multivariate_normal as mvn
from scipy.stats.distributions import beta
from scipy.stats.mstats import hdquantiles
from scipy.spatial import ConvexHull, Delaunay
try:
    from triangle import triangulate
    from shapely.geometry import Polygon, MultiPolygon
except:
    pass
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
try:
    from alpha_shapes.alpha_shapes import Alpha_Shaper
    from alpha_shapes.boundary import get_boundaries, Boundary
    alpha_pkg = True
except:
    alpha_pkg = False
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)
import warnings
from copy import deepcopy
from phdu.plots.plotly_utils import get_figure
from phdu import geometry
import plotly.graph_objects as go
from . import custom_metrics
from .preprocessing import load, space

def optimize_bw(x_q, num_iter=50):
    """
    Optimize bandwidth using leave one out cross validation, maximizing the log likelihood.
    """
    std = x_q.std(ddof=1)
    bws = np.logspace(-1, 1, num_iter) * std # from 0.1*std to 10*std
    grid = GridSearchCV(estimator = KernelDensity(),
                        param_grid = {'bandwidth': bws},
                        cv=x_q.shape[0], # leave one out
                        )
    grid.fit(x_q[:, None])
    bw_scipy = grid.best_params_['bandwidth'] / std
    return bw_scipy

def hd_beta_weights(n, prob):
    """
    Store Beta weights for HD quantiles to accelerate computation.
    """
    w = np.empty((n, prob.size))
    v = np.arange(n+1, dtype=np.float64) / n
    betacdf = beta.cdf
    for (i, p) in enumerate(prob):
        _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
        w[:, i] = _w[1:] - _w[:-1]
    return w

def separate_non_intersecting_polygons(polygons):
    """
    Takes a list of polygons and returns a list where no two polygons intersect.
    The intersecting parts of each original polygon are added to only one component, so no area is lost.

    Parameters:
    polygons (list of numpy.ndarray): List of polygons as arrays of vertices.

    Returns:
    list of numpy.ndarray: List of non-intersecting polygons as arrays of vertices.
    """
    polygons = [Polygon(poly) for poly in polygons]
    non_intersecting_polygons = [polygons[0]]

    for poly in polygons[1:]:
        to_merge = []

        # Check each existing non-intersecting polygon
        for existing_poly in non_intersecting_polygons:
            # check if existing_poly and poly are the same
            if poly.equals(existing_poly):
                poly = None
                break
            elif poly.intersects(existing_poly):
                # Find the intersection and union parts
                intersection_area = poly.intersection(existing_poly)
                poly = poly.union(existing_poly).difference(intersection_area)
            else:
                # No intersection, keep in non-intersecting set
                to_merge.append(existing_poly)

        # Update the list of non-intersecting polygons
        if poly is not None:
            if isinstance(poly, MultiPolygon):
                to_merge += [p for p in poly.geoms if p.area > 0]
            elif poly.area > 0:
                to_merge.append(poly)
            non_intersecting_polygons += to_merge

    non_intersecting_polygons = [np.array(poly.exterior.coords) for poly in non_intersecting_polygons]

    return non_intersecting_polygons

def triangulate_polygon(vertices):
    """
    Triangulate a polygon given by its vertices using Delaunay triangulation.

    Args:
        vertices (list of tuples): Vertices of the polygon as [(x1, y1), (x2, y2), ...].

    Returns:
        list of np.ndarray: Each element is a 3x2 array representing a triangle with vertices.
    """
    # Create Delaunay triangulation
    try:
        delaunay = Delaunay(vertices)
    except:
        warnings.warn("Could not triangulate the polygon. Returning NaN.", RuntimeWarning)
        return np.nan

    # Filter triangles to keep only those within the polygon
    polygon = Polygon(vertices)
    triangles = []

    for simplex in delaunay.simplices:
        triangle = np.array([vertices[simplex[0]], vertices[simplex[1]], vertices[simplex[2]]])
        triangle_polygon = Polygon(triangle)
        # Add triangle if it's within the original polygon
        if polygon.contains(triangle_polygon.centroid):
            triangles.append(triangle)

    return triangles

def conformal_delaunay(poly, quality=False):
    """
    Generate a conformal Delaunay triangulation for a given polygon.

    Parameters:
    poly (ndarray): An array of polygon vertices. The last element must be the same as the first.
    quality (bool): If True, ensures quality mesh generation (angles > 20 degrees). Default is False.

    Returns:
    ndarray: An array of triangles, each represented by its vertex coordinates.
    """
    if poly.shape[0] < 4: # at least 4 points needed (in case the 1st and last points are the same)
        return np.nan
    else:
        polygon = dict(vertices=list(zip(*poly[:-1].T)),
                       segments=[(i, i+1) for i in range(len(poly)-2)] + [(len(poly)-2, 0)])
        # Opts: p (triangulates a Planar Straight Line Graph), q (quality mesh generation), D (conforming Delaunay)
        opts = 'pD'
        if quality:
            opts += 'q'
        info = triangulate(polygon, opts)
        v = info['vertices']
        triangle_idxs = info['triangles']
        triangles = v[triangle_idxs]
        return triangles

@njit
def triangle_area(triangle):
    """
    Calculate the area of a triangle using its vertex coordinates.

    Args:
        triangle (np.ndarray): 3x2 array of triangle vertices [(x1, y1), (x2, y2), (x3, y3)].

    Returns:
        float: Area of the triangle.
    """
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

@njit
def integrate_triangle(triangle, func):
    """
    Integrate a function over a triangle using 3-point Gaussian quadrature.

    Args:
        triangle (np.ndarray): 3x2 array of triangle vertices [(x1, y1), (x2, y2), (x3, y3)].
        func (function): Function to integrate, func(x, y).

    Returns:
        float: Approximate integral of func over the triangle.
    """
    # Triangle vertices
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]

    # 3-point Gaussian quadrature weights and points in barycentric coordinates
    weights = [1/3, 1/3, 1/3]
    points = [
        (1/6, 1/6, 2/3),
        (1/6, 2/3, 1/6),
        (2/3, 1/6, 1/6)
    ]

    # Compute area of the triangle
    area = triangle_area(triangle)

    # Perform quadrature
    integral = 0.0
    for w, (l1, l2, l3) in zip(weights, points):
        # Convert barycentric coordinates to Cartesian coordinates
        x = l1 * x1 + l2 * x2 + l3 * x3
        y = l1 * y1 + l2 * y2 + l3 * y3
        integral += w * func(x, y)

    # Multiply by the triangle's area
    return area * integral

@njit
def subdivide_triangle(triangle):
    """Divide a triangle into four subtriangles"""
    a, b, c = triangle
    ab = (a + b) / 2
    bc = (b + c) / 2
    ca = (c + a) / 2

    subtriangles = np.empty((4, 3, 2), dtype=np.float64)

    subtriangles[0, 0] = a
    subtriangles[0, 1] = ab
    subtriangles[0, 2] = ca

    subtriangles[1, 0] = ab
    subtriangles[1, 1] = b
    subtriangles[1, 2] = bc

    subtriangles[2, 0] = ca
    subtriangles[2, 1] = bc
    subtriangles[2, 2] = c

    subtriangles[3, 0] = ab
    subtriangles[3, 1] = bc
    subtriangles[3, 2] = ca
    return subtriangles

@njit
def adaptive_triangle_integration(triangle, func, target_error=1e-5, max_iter=10):
    I1 = integrate_triangle(triangle, func)
    subtriangles = subdivide_triangle(triangle)
    I2 = np.array([integrate_triangle(triangle, func) for triangle in subtriangles]).sum()
    abs_I2 = abs(I2)
    epsilon = 1e-10
    n = 0
    while abs((I2 - I1) / max(abs_I2, epsilon)) > target_error and abs_I2 > epsilon and n < max_iter:
        I1 = I2
        subtriangles_ph = np.empty((subtriangles.shape[0] * 4, 3, 2), dtype=np.float64)
        for i, sub_t in enumerate(subtriangles):
            subtriangles_ph[i*4:i*4+4] = subdivide_triangle(sub_t)
        subtriangles = subtriangles_ph
        I2 = np.array([integrate_triangle(triangle, func) for triangle in subtriangles]).sum()
        abs_I2 = abs(I2)
        n += 1
    return I2

@njit
def recursive_triangle_integration(triangle, func, target_error=1e-5, depth=0, max_depth=100):
    """
    Recursive adaptive integration over a triangle.

    Parameters:
        triangle: The triangle to integrate (3, 2 array).
        func: Function to integrate, accepting a point (x, y).
        target_error: Relative error tolerance.
        depth: Current recursion depth (default is 0).
        max_depth: Maximum recursion depth to prevent stack overflow.

    Returns:
        Estimated integral value for the triangle.
    """
    I1 = integrate_triangle(triangle, func)
    subtriangles = subdivide_triangle(triangle)
    I2 = 0.0
    for subtriangle in subtriangles:
        I2 += integrate_triangle(subtriangle, func)

    epsilon = 1e-10

    # Check convergence
    if abs((I2 - I1) / max(abs(I2), epsilon)) <= target_error or depth >= max_depth or abs(I2) < epsilon:
        return I2

    # Recursively compute integral for each subtriangle
    total_integral = 0.0
    for subtriangle in subtriangles:
        total_integral += recursive_triangle_integration(subtriangle, func, target_error, depth + 1, max_depth)

    return total_integral

def _integrate_polygon(vertices, func):
    """
    DEPRECATED: Use integrate_polygon instead.

    Numerically integrate a function over a polygonal domain using triangular decomposition.

    Args:
        vertices (list of tuples): Vertices of the polygon [(x1, y1), (x2, y2), ...].
        func (function): Function to integrate, func(x, y).

    Returns:
        float: Approximate integral of func over the polygon.
    """
    triangles = triangulate_polygon(vertices)
    if isinstance(triangles, float):
        return 0
    else:
        total_integral = np.array([integrate_triangle(triangle, func) for triangle in triangles]).sum()
        return total_integral

def integrate_polygon(vertices, func, target_error=1e-5, mode='adaptive'):
    """
    Numerically integrate a function over a polygonal domain using conformal Delaunay triangulation and adaptive integration.

    Args:
        vertices (list of tuples): Vertices of the polygon [(x1, y1), (x2, y2), ...].
        func (function): Function to integrate, func(x, y).
        target_error (float): Relative error tolerance for adaptive integration.
        mode (str): Integration mode, 'recursive' or 'adaptive'. Recursive mode is slower but more memory-efficient, and potentially more accurate.

    Returns:
        float: Approximate integral of func over the polygon.
    """
    triangles = conformal_delaunay(vertices)
    if isinstance(triangles, float):
        return 0
    else:
        computer = globals()[f'{mode}_triangle_integration']
        total_integral = np.array([computer(triangle, func, target_error) for triangle in triangles]).sum()
        return total_integral

@njit
def f_integrand(x, y): # multiply by 4 at the end to obtain the area
    return custom_metrics.solid_angle_y_integrand(y)

def area_polygon_on_Earth(vertices, **kwargs):
    """
    Vertices are in Mercator projection (x, y) and the area is computed on the Earth's surface, in km^2.
    """
    # Define the function to integrate
    return 4 * integrate_polygon(vertices, f_integrand, **kwargs)

def ensure_increasing_quantiles(qs):
    x_q, y_q = np.sort(qs, axis=-1)
    x_q = ensure_positive_differences(x_q)
    y_q = ensure_positive_differences(y_q)
    return x_q, y_q

def ensure_increasing_quantiles_1d(qs):
    x_q = np.sort(qs)
    x_q = ensure_positive_differences(x_q)
    return x_q

def ensure_positive_differences(x, eps=1e-6):
    dx = np.diff(x)
    idx = np.where(dx == 0)[0]
    if idx.size > 0:
        if idx.size == x.size -1: # all values are the same
            return x[0]
        else:
            idx += 1
            if idx[-1] == x.size - 1:
                x[-1] += eps
                if idx.size > 1:
                    idx = idx[:-1]
                else:
                    return x
            x[idx] = np.NaN
            x = pd.Series(x).interpolate(method='linear', limit_direction='both').values
    return x

@njit
def nb_isclose(a, b, rtol=1e-5, atol=1e-8):
    return np.abs(a - b) <= atol + rtol * np.abs(b)

@njit
def join_open_polygons(polygons):
    closed_p = []
    open_p = []
    for p in polygons:
        if p.shape[0] > 3 and nb_isclose(p[0], p[-1]).all(): # at least 4 if the 1st and last points are the same
            closed_p.append(p)
        else:
            open_p.append(p)
    return closed_p, open_p

@njit
def concatenate_lines(lines, path):
    """Concatenate lines in the order defined by the path."""
    length = np.array([lines[i].shape[0] for i in path])
    total_length = length.sum() + 1  # +1 to close the polygon
    poly = np.empty((total_length, 2), dtype=np.float64)
    idx = 0
    for i, l in zip(path, length):
        poly[idx:idx+l] = lines[i]
        idx += l
    # Closing the polygon by adding the first point again
    poly[idx] = poly[0]
    return poly

@njit
def create_polygon_from_lines(lines):
    """
    Joins lines to form a polygon, using the shortest path between the last point of line i and the first point of line j.

    The list of lines is not ordered and have different sizes.
    """
    # find the closest lines
    n = len(lines)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.linalg.norm(lines[i][-1] - lines[j][0])
    # find the shortest path
    path = []
    i = 0
    while len(path) < n:
        path.append(i)
        dist[:, i] = np.inf
        i = np.argmin(dist[i])
    # join the lines
    poly = concatenate_lines(lines, path)
    return poly

def is_p2_inside_p1(p1, p2):
    p1 = Polygon(p1)
    p2 = Polygon(p2)
    return p1.contains(p2)

def process_contours(contours):
    if len(contours) == 1:
        return contours, []

    closed_p, open_p = join_open_polygons(contours)
    if open_p:
        p = create_polygon_from_lines(open_p)
        if not Polygon(p).is_valid:
            # fall back to hull
            try:
                hull = ConvexHull(p)
                hull = p[hull.vertices]
                p = np.vstack((hull, hull[0]))
                warnings.warn("Could not create a polygon from the contour lines. Using the convex hull instead.", RuntimeWarning)
            except:
                p = np.empty((1)) # will be pruned in the next line
        if p.shape[0] > 3: # at least 4 if the 1st and last points are the same
            closed_p.append(p)
    areas = np.array([_polygon_area(*p.T) for p in closed_p])
    biggest = areas.argmax()
    big_p = closed_p.pop(biggest)

    exterior = [big_p]
    holes = []
    for p in closed_p:
        if is_p2_inside_p1(big_p, p):
            holes.append(p)
        else:
            exterior.append(p)
    return exterior, holes

def eval_pr(dist, y_real, c, method='hull', del_after_compute=False):
    out = {}
    cds = {}
    pbar = tqdm(range(len(dist)))
    for (animal, time_step), d in dist.items():
        y = y_real[animal, time_step]

        if isinstance(d, float):
            area = np.nan
            coverage = np.nan
            cds[(time_step, animal, c)] = np.nan
        elif isinstance(d, NonParametricBivariateDistribution):
            result = d.pr(method=method, confidence=c)
            if isinstance(result, float):
                area = np.nan
                coverage = np.nan
                cds[(time_step, animal, c)] = np.nan
            elif method == 'hull':
                polygon, area = result
                coverage = geometry.is_point_inside_polygon(y, polygon)
                cds[(time_step, animal, c)] = polygon
            elif method == 'alpha_shape' or method == 'contour':
                exterior, holes, area = result
                area = area.sum()
                coverage = any(geometry.is_point_inside_polygon(y, ex) for ex in exterior) and not any(geometry.is_point_inside_polygon(y, h) for h in holes)
                cds[(time_step, animal, c)] = exterior, holes
            if del_after_compute:
                del result
                gc.collect()

        elif isinstance(d, NonParametricMixedDistribution):
            pr = d.pr(confidence=c)
            x0, x1, y0, y1 = pr.ravel()
            x_r, y_r = y
            area = custom_metrics.area_integral_rectangle(x0, x1, y0, y1)
            coverage = ((x_r >= x0) & (x_r <= x1) & (y_r >= y0) & (y_r <= y1))
            cds[(time_step, animal, c)] = (x0, x1, y0, y1)
        else:
            raise RuntimeError(f"Unknown distribution type: {type(d)}")

        out[(time_step, animal, c, 'area')] = area
        out[(time_step, animal, c, 'coverage')] = coverage
        pbar.update()
    pbar.close()

    out = pd.Series(out)
    cds = pd.Series(cds)
    return out, cds

@njit
def expand_polygon(p, mpl):
    center = custom_metrics.nb_mean_axis_0(p)
    r = p - center
    p_expanded = r * (1 + mpl) + center
    return p_expanded

def expand_pr(cds_s, mpl, expand_exterior_first_only=True):
    cds = deepcopy(cds_s)
    if len(cds.index.levels) == 3:
        cds = cds.droplevel(2)
    cds_new = {}
    for (time_step, animal), c in cds.items():
        if isinstance(c, float):
            cds_new[(time_step, animal)] = np.nan
        elif isinstance(c, np.ndarray):  # hull
            c = expand_polygon(c.copy(), mpl)
            cds_new[(time_step, animal)] = c
        elif isinstance(c, tuple):  # alpha_shape, rectangle
            if len(c) == 2:  # alpha_shape, contours
                exterior, holes = deepcopy(c)
                if len(exterior) == 0:
                    cds_new[(time_step, animal)] = np.nan
                else:
                    if expand_exterior_first_only:
                        exterior[0] = expand_polygon(exterior[0], mpl)
                    else:
                        exterior = [expand_polygon(ex, mpl) for ex in exterior]
                        if len(exterior) > 1:
                            exterior = separate_non_intersecting_polygons(exterior)
                    cds_new[(time_step, animal)] = exterior, holes
            elif len(c) == 4:
                x0, x1, y0, y1 = deepcopy(c)
                rectangle = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]])
                rectangle = expand_polygon(rectangle, mpl)
                x0, y0 = rectangle[0]
                x1, y1 = rectangle[2]
                cds_new[(time_step, animal)] = (x0, x1, y0, y1)
            else:
                raise RuntimeError(f"Unknown tuple length: {len(c)}")
        else:
            raise RuntimeError(f"Unknown coordinate type: {type(c)}")

    cds_new = pd.Series(cds_new)
    cds_new.index.names = ['time_step', 'animal']
    return cds_new

def eval_pr_from_cds(cds, y_real):
    out = {}
    pbar = tqdm(range(len(cds)))
    for (time_step, animal), c in cds.items():
        y = y_real[animal, time_step]
        if isinstance(c, float):
            area = np.nan
            coverage = np.nan
        elif isinstance(c, np.ndarray):  # hull
            area = area_polygon_on_Earth(c, mode='adaptive')
            coverage = geometry.is_point_inside_polygon(y, c)
        elif isinstance(c, tuple):  # alpha_shape, rectangle
            if len(c) == 2:  # alpha_shape, contours
                exterior, holes = c
                area = np.array([area_polygon_on_Earth(ex, mode='recursive') for ex in exterior] + [-area_polygon_on_Earth(h, mode='recursive') for h in holes]).sum()
                coverage = any(geometry.is_point_inside_polygon(y, ex) for ex in exterior) and not any(geometry.is_point_inside_polygon(y, h) for h in holes)
            elif len(c) == 4:
                x0, x1, y0, y1 = c
                x_r, y_r = y
                area = custom_metrics.area_integral_rectangle(x0, x1, y0, y1)
                coverage = ((x_r >= x0) & (x_r <= x1) & (y_r >= y0) & (y_r <= y1))
            else:
                raise RuntimeError(f"Unknown tuple length: {len(c)}")
        else:
            raise RuntimeError(f"Unknown coordinate type: {type(c)}")

        out[(time_step, animal, 'area')] = area
        out[(time_step, animal, 'coverage')] = coverage
        pbar.update()
    pbar.close()

    df = pd.Series(out).unstack()
    df.index.names = ['time_step', 'animal']
    return df

class NonParametricPointDistribution:
    def __init__(self, x, y=None, dx=10, dy=10, **kwargs):
        assert isinstance(x, (float, int)), "x must be a scalar"
        assert (y is None) or isinstance(y, (float, int)), "y must be a scalar or None"
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def pp(self, **kwargs): # added for compatibility
        if self.y is None:
            return self.x
        else:
            return np.array([self.x, self.y])

    def pr(self, dx=None, dy=None, **kwargs):
        """
        Returns PR as a 2x2 array with shape (target, PI)
        """
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        else:
            return np.array([[self.x-dx, self.x+dx], [self.y-dy, self.y+dy]])

    def pi(self, dx=None, **kwargs): # kwargs added for compatibility
        return np.array([self.x-dx, self.x+dx])


def mode_roots(pdf):
    derivative = pdf.derivative()
    max_or_min = derivative.roots()
    pdf_values = pdf(max_or_min)
    mode = max_or_min[pdf_values.argmax()]
    return mode

class NonParametricUnivariateDistribution:
    """
    Estimate point predictions and prediction regions for a univariate distribution, taking as input the quantiles of the marginal distribution.
    """
    def __init__(self, x_q, q, n_sample=10000, n_grid=100, mode_margin=0.1, mode_weighted=False, mode_method='sample', density=None, **kwargs): # kwargs added for compatibility
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.density = density
        self.x_q = x_q
        self.q = q
        self.n_sample = n_sample
        self.n_grid = n_grid
        self.mode_margin = mode_margin
        self.mode_weighted = mode_weighted
        self.mode_method = mode_method

    # To be implemented in subclasses
    def get_uniform(self, n):
        raise NotImplementedError

    def f_inv(self, u):
        raise NotImplementedError

    def get_mode_root(self, **kwargs):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError
    #############################

    def initial_sample(self, sample_dist=True, sample_pdf=False):
        if sample_dist:
            self.get_sample()
        else:
            self.sample = None
            self.p_sample = None
            self.sample_sorted = None
        if sample_pdf:
            p, l = self.get_pdf()
            self.p = p
            self.l = l
        else:
            self.p = None
            self.l = None
        return

    def get_sample(self, n=None):
        """
        Samples from the empirical distribution using the inverse CDF method.

        Parameters:
        - n (int): Number of samples to generate. If None, uses self.n_sample.

        Returns:
        - samples (numpy array): Samples from the empirical distribution, with size `n`.
        """
        if n is None:
            n = self.n_sample
        u = self.get_uniform(n)
        sample = self.f_inv(u)
        self.sample = sample
        self.p_sample = self.pdf(self.sample)
        self.sample_sorted = self.sample[np.argsort(-self.p_sample)] # sort by probability
        return self.sample

    def get_pdf(self, n=None):
        if n is None:
            n = self.n_grid
        if self.density == 'pchip':
            u_min = self.q.min()
            u_max = self.q.max()
        else:
            u_min = self.cdf(self.xq_hd[0])[0]
            u_max = 1
        l = self.f_inv(np.linspace(u_min, u_max, n))
        p = self.pdf(l)
        return p, l

    def get_mode(self, n=None, mode_margin=None, mode_weighted=None, mode_method=None):
        if mode_margin is None:
            mode_margin = self.mode_margin
        if mode_weighted is None:
            mode_weighted = self.mode_weighted
        if mode_method is None:
            mode_method = self.mode_method

        if mode_method == 'roots':
            X_mode = self.get_mode_root()
        elif mode_method == 'grid':
            if n == self.n_grid and self.pdf is not None:
                pdf = self.pdf
                l = self.l
            else:
                pdf, l = self.get_pdf(n)

            i = np.where(pdf >= pdf.max()*(1-mode_margin))[0]

            if i.size > 1:
                if mode_weighted:
                    w = pdf[i]
                    X_mode = l[i].dot(w)/w.sum()
                else:
                    X_mode = l[i].mean()
            else:
                X_mode = l[i]
        elif mode_method == 'sample':
            if n != self.n_sample or self.sample is None:
                self.get_sample(n)
            sample = self.sample
            pdf_sample = self.p_sample
            i = np.where(pdf_sample >= pdf_sample.max()*(1-mode_margin))[0]
            if mode_weighted:
                w = pdf_sample[i]
                X_mode = sample[i].dot(w)/w.sum()
            else:
                X_mode = sample[i].mean()
        else:
            raise ValueError("mode_method must be 'grid' or 'sample'")
        return X_mode

    def weight_PR(self, alpha=0.1):
        if self.sample_sorted is None:
            self.get_sample()
        selected_idxs = int((1-alpha)*self.n_sample)
        return self.sample_sorted[:selected_idxs]

    def pi(self, alpha=0.1, **kwargs): # kwargs added for compatibility
        """
        Prediction interval
        """
        selected_sample = self.weight_PR(alpha)
        return selected_sample[[0, -1]]

    def pp(self, mode, **mode_kwargs):
        """
        Point prediction.
        """
        if mode == 'mean':
            return self.sample.mean()
        elif mode == 'median':
            if hasattr(self, 'exclude_me') and self.exclude_me:
                return self.x_q_me
            else:
                me_idx = np.where(self.q == 0.5)[0][0]
                return self.x_q[me_idx]
        elif mode == 'mode':
            return self.get_mode(**mode_kwargs)
        else:
            raise ValueError("mode must be 'mean', 'median' or 'mode'")

    def summary_pp(self, **mode_kwargs):
        pp = {}
        for mode in ['mean', 'median', 'mode']:
            pp[mode] = self.pp(mode, **mode_kwargs)
        return pp


class PCHIPUnivariate(NonParametricUnivariateDistribution):
    """
    Estimate point predictions and prediction regions for a univariate distribution, taking as input the quantiles of the marginal distribution.

    Uses piecewise cubic hermite interpolating polynomial to smooth the CDF and get the PDF.
    """
    def __init__(self, x_q, q, n_sample=10000, n_grid=100, sample_dist=True, sample_pdf=False, **kwargs): # kwargs added for compatibility
        super().__init__(x_q=x_q, q=q, n_sample=n_sample, n_grid=n_grid, **kwargs, density='pchip')
        self.cdf = PchipInterpolator(x_q, q)
        self.pdf = self.cdf.derivative()
        self.f_inv = PchipInterpolator(q, x_q)
        self.initial_sample(sample_dist, sample_pdf)

    def get_uniform(self, n):
        u = np.random.randint(self.q.min()*n, self.q.max()*n, n) / n
        return u

    def get_mode_root(self):
        return mode_roots(self.pdf)


class QRDEUnivariate(NonParametricUnivariateDistribution):
    """
    Quantile Respectful Density estimation of a univariate distribution.

    https://arxiv.org/abs/2404.03835
    """
    def __init__(self, x_q, q, exclude_me=False, dp=0.001, n_sample=int(1e5), n_grid=100, sample_dist=True, sample_pdf=False, interpolate=False, beta_weights=None, **kwargs):
        self.exclude_me = exclude_me
        if exclude_me:
            me_index = np.where(q == 0.5)[0][0]
            self.x_q_me = x_q[me_index]
            x_q = np.delete(x_q, me_index)
        super().__init__(x_q=x_q, q=q, n_sample=n_sample, n_grid=n_grid, **kwargs, density='qrde')

        # Perform QRDE
        self.dp = dp
        self.interpolate = interpolate
        self.beta_weights = beta_weights
        if beta_weights is not None:
            beta_n = x_q.size + 2
            prob_size = 1 + int(1/dp)
            assert beta_weights.shape == (beta_n, prob_size), f"beta_weights must have shape {(beta_n, prob_size)}"
        self.qrde()
        self.initial_sample(sample_dist, sample_pdf)

    @staticmethod
    @njit
    def hd_1d(data, prob, w):
        """
        Computes the HD quantiles for a 1D array.

        Attributes:
            - data: 1D array (SORTED)
            - prob: 1D array of probabilities (SORTED)
            - w: 2D array of  Beta weights (n, prob.size)
        """
        hd = np.dot(data, w)

        if prob[0] == 0:
            hd[0] = data[0]
        if prob[-1] == 1:
            hd[-1] = data[-1]
        return hd

    def qrde(self):
        """
        Perform quantile regression density estimation (QRDE).

        This method estimates the probability density function (PDF) and cumulative distribution function (CDF)
        based on the quantiles of the data. It uses the Harrell-Davis quantile estimator and ensures the PDF is restricted to the range of the data.

        Steps:
        1. Extend the quantile range by adding plausible minimum and maximum values.
        2. Estimate the quantiles using the Harrell-Davis quantile estimator.
        3. Handle masked values in the quantile estimation.
        4. Ignore the added tails and rescale the probability density.
        7. Calculate the PDF and normalize it.
        8. Calculate the CDF and normalize it.

        Raises:
            RuntimeError: If masked values are found in the quantile estimation.

        Attributes:
            x_q (numpy.ndarray): Array of quantiles.
            dp (float): Probability increment.
            xq_hd (numpy.ndarray): Array of Harrell-Davis quantiles.
            pdf (numpy.ndarray): Probability density function.
            cdf (numpy.ndarray): Cumulative distribution function.
        """
        # Initial estimation. Counter the tail errors of the HD estimator by adding plausible min and max, and then restricting the PDF to the range of the data
        # prob = np.arange(dp, 1, dp) for the original sample
        # prob_filled = np.hstack((0, prob, 1))
        dx = np.diff(self.x_q)
        # Interpolate quantiles at the edges (q=0 and q=1)
        p_minus = np.polyfit(self.q[:3], dx[:3], 1)
        dx_minus = np.polyval(p_minus, 0)
        if dx_minus < dx[0]:
            dx_minus = dx[:2].sum() # Ensure Probability decreases with distance
        x_minus = self.x_q[0] - dx_minus
        p_plus = np.polyfit(self.q[-3:], dx[-3:], 1)
        dx_plus = np.polyval(p_plus, 1)
        if dx_plus < dx[-1]:
            dx_plus = dx[-2:].sum() # Ensure Probability decreases with distance
        x_plus = self.x_q[-1] + dx_plus

        x_q_filled = np.hstack((x_minus, self.x_q, x_plus))
        prob_filled = np.arange(0, 1+self.dp, self.dp)
        if self.beta_weights is None:
            xq_hd_filled = hdquantiles(x_q_filled, prob_filled)
            if xq_hd_filled.mask:
                raise RuntimeError("Masked values in quantile estimation")
            else:
                xq_hd_filled = xq_hd_filled.data
        else:
            xq_hd_filled = self.hd_1d(x_q_filled, prob_filled, self.beta_weights)

        # ignore tails and rescale
        start = np.where(xq_hd_filled < self.x_q.min())[0][-1]
        finish = np.where(xq_hd_filled > self.x_q.max())[0][0]

        xq_hd = xq_hd_filled[start-1:finish+1]
        dq = np.diff(xq_hd)
        self.xq_hd_left = xq_hd[0]
        self.xq_hd_right = xq_hd_filled[finish+1]
        self.xq_hd = xq_hd[1:] #  p(xq_hd[0]) = 0
        self.pdf_bin = self.dp / dq
        self.pdf_bin /= (self.pdf_bin * dq).sum()
        # self.cdf_bin = np.hstack((0, np.cumsum(self.pdf_bin*dq)))
        self.cdf_bin = np.cumsum(self.pdf_bin*dq)
        self.cdf_bin /= self.cdf_bin[-1] # normalize

    @staticmethod
    def get_uniform(n):
        return np.random.rand(n)

    @staticmethod
    @njit
    def search_bin(x, x_bins, x_eval, interpolate=False, side='right', subtract=1):
        if side == 'right':
            indices = np.searchsorted(x_bins, x, side='right') - subtract
        elif side == 'left':
            indices = np.searchsorted(x_bins, x, side='left') - subtract
        else:
            raise ValueError("side must be 'left' or 'right'")
        if interpolate:
            x_left = x_bins[indices]
            x_right = x_bins[indices + 1]
            eval_left = x_eval[indices]
            eval_right = x_eval[indices + 1]
            return eval_left + (x - x_left) / (x_right - x_left) * (eval_right - eval_left)
        else:
            return x_eval[indices]

    def pdf(self, x, interpolate=None):
        """
        Computes the interpolated probability density for each value in `x` based on
        the empirical distribution defined by intervals [x_i, x_{i+1}] in `self.xq_hd`
        and corresponding densities in `self.pdf_bin`.

        Parameters:
        - x (numpy array): Array of sample points where densities need to be computed.

        Returns:
        - densities (numpy array): Interpolated density values for each sample in `x`.
        """
        if isinstance(x, float):
            x = np.array([x])
        if interpolate is None:
            interpolate = self.interpolate
        out = self.search_bin(x, self.xq_hd, self.pdf_bin, interpolate=interpolate)
        out[(x < self.xq_hd[0]) | (x > self.xq_hd[-1])] = 0
        return out

    def f_inv(self, u, interpolate=None):
        """
        Samples from the empirical distribution using the inverse CDF method.

        Parameters:
        - u (numpy array): Array of uniform random variables to generate samples.

        Returns:
        - samples (numpy array): Samples from the empirical distribution, with size `n`.
        """
        if isinstance(u, float):
            u = np.array([u])
        if interpolate is None:
            interpolate = self.interpolate
        out = self.search_bin(u, self.cdf_bin, self.xq_hd, interpolate=interpolate)
        out[u < self.cdf_bin[0]] = self.xq_hd_left
        return out

    def cdf(self, x, interpolate=None):
        """
        Computes the interpolated cumulative distribution for each value in `x` based on
        the empirical distribution defined by intervals [x_i, x_{i+1}] in `self.xq_hd`
        and corresponding CDF values in `self.cdf_bin`.

        Parameters:
        - x (numpy array): Array of sample points where CDF values need to be computed.

        Returns:
        - cdfs (numpy array): Interpolated CDF values for each sample in `x`.
        """
        if isinstance(x, float):
            x = np.array([x])
        if interpolate is None:
            interpolate = self.interpolate
        out = self.search_bin(x, self.xq_hd, self.cdf_bin, interpolate=interpolate)
        out[x < self.xq_hd[0]] = 0
        out[x >= self.xq_hd[-1]] = 1
        return out

    def get_mode_root(self):
        warnings.warn("Mode estimation from root is not implemented for QRDE. Falling back to mode estimation from sample with margin 0.", RuntimeWarning)
        return self.get_mode(n=self.n_sample, mode_margin=0, mode_weighted=False, mode_method='sample')


class NonParametricBivariateDistribution:
    """
    Estimate point predictions and prediction regions for a bivariate distribution, taking as input the quantiles of the marginal distributions.

    This method can handle correlations between the marginals using the rho parameter (Gaussian copula).

    Uses piecewise cubic hermite interpolating polynomial to smooth the CDFs and get the PDFs.
    """
    def __init__(self, x_q, y_q, q, cds='mercator', to_mercator=False, density='pchip', n_sample=int(1e5), n_grid=100, mode_margin=0.1, mode_weighted=False, mode_method='sample', sample_dist=True, sample_pdf=False, rho=0, sample_dn=0.05, rho_eps=0.8, rho_spread=0, max_rho_spread=None, **kwargs):
        self.cds = cds
        self.to_mercator = cds != 'mercator' and to_mercator
        if self.to_mercator:
            self.r0 = load.reference_point()
        self.density = density
        if density == 'qrde':
            self.univariate_gen = QRDEUnivariate
            self.sample_dn = 0
            self.q0 = 0
            self.qf = 1
        elif density == 'pchip':
            self.univariate_gen = PCHIPUnivariate
            self.sample_dn = sample_dn
            self.q0 = q.min()
            self.qf = q.max()
        else:
            raise ValueError("density must be 'qrde' or 'pchip'")

        self.x_q = x_q
        self.y_q = y_q
        self.q = q
        self.rho_eps = rho_eps
        self.rho_spread = rho_spread
        self.max_rho_spread = max_rho_spread
        # self.rho = np.clip(rho, -1+rho_eps, 1-rho_eps) # else it will raise LinalgError
        self.rho = rho
        self.n_sample = n_sample
        self.n_grid = n_grid
        self.mode_margin = mode_margin
        self.mode_weighted = mode_weighted
        self.mode_method = mode_method

        self.dist_x = self.univariate_gen(x_q, q, n_sample=n_sample, n_grid=n_grid, sample_dist=False, sample_pdf=False, **kwargs)
        self.dist_y = self.univariate_gen(y_q, q, n_sample=n_sample, n_grid=n_grid, sample_dist=False, sample_pdf=False, **kwargs)
        self.cdf_x = self.dist_x.cdf
        self.cdf_y = self.dist_y.cdf
        self.pdf_x = self.dist_x.pdf
        self.pdf_y = self.dist_y.pdf
        self.f_inv_x = self.dist_x.f_inv
        self.f_inv_y = self.dist_y.f_inv
        self.pad_x_left = self.dist_x.xq_hd_left
        self.pad_x_right = self.dist_x.xq_hd_right
        self.pad_y_left = self.dist_y.xq_hd_left
        self.pad_y_right = self.dist_y.xq_hd_right

        if self.apply_rho:
            self.G = norm()
            self.z0 = self.G.ppf(self.q0)
            self.zf = self.G.ppf(self.qf)

        if sample_dist:
            self.get_sample()
        else:
            self.sample = None
            self.p_sample = None
            self.sample_sorted = None

        if sample_pdf:
            pdf_xy, l_x, l_y = self.get_pdf()
            self.pdf = pdf_xy
            self.l_x = l_x
            self.l_y = l_y
        else:
            self.pdf = None
            self.l_x = None
            self.l_y = None

    @property
    def apply_rho(self):
        return self.rho != 0 and (self.max_rho_spread is None or self.rho_spread < self.max_rho_spread)

    @staticmethod
    def get_copula(rho):
        return mvn(mean=[0, 0], cov=[[1, rho], [rho, 1]], seed=0)

    def p_xy(self, x, y, rho_eps=None):
        if x.ndim == 2 and y.ndim == 2: # grid
            f = self.pdf_x(x[:, 0])[:, None] * self.pdf_y(y[0])[None]
        else:
            f = self.pdf_x(x)*self.pdf_y(y)
        if self.apply_rho:
            if rho_eps is None:
                rho_eps = self.rho_eps
            rho = np.clip(self.rho, -1+rho_eps, 1-rho_eps)
            copula = self.get_copula(rho)
            epsilon_u = 1e-5
            if x.ndim == 2 and y.ndim == 2: # grid
                u_x = self.cdf_x(x[:, 0]) # uniform
                u_x = np.clip(u_x, epsilon_u, 1-epsilon_u)
                z_x = self.G.ppf(u_x)[: ,None] # gaussian (normal)
                u_y = self.cdf_y(y[0])
                u_y = np.clip(u_y, epsilon_u, 1-epsilon_u)
                z_y = self.G.ppf(u_y)[None] # gaussian (normal)
                # u = np.array(np.meshgrid(u_x, u_y)).T.reshape(-1, 2)
                z = np.array(np.meshgrid(z_x, z_y)).reshape(2, -1).T
                num = copula.pdf(z).reshape(x.shape[0], y.shape[1])
                den = self.G.pdf(z_x)*self.G.pdf(z_y)
                f_copula = num / den
            else:
                u = np.vstack([self.cdf_x(x), self.cdf_y(y)]).T
                u = np.clip(u, epsilon_u, 1-epsilon_u)
                z = self.G.ppf(u)
                num = copula.pdf(z)
                den = self.G.pdf(z).prod(axis=1)
                f_copula = num / den
            f *= f_copula
        return f

    def copula_sample(self, n, dn=0.05, rho_eps=None):
        if rho_eps is None:
            rho_eps = self.rho_eps
        rho = np.clip(self.rho, -1+rho_eps, 1-rho_eps)
        copula = self.get_copula(rho)
        if dn > 0:
            z = copula.rvs(int((1+dn) * n))
            valid = ((z[:, 0] >= self.z0) & (z[:, 0] <= self.zf)
                     & (z[:, 1] >= self.z0) & (z[:, 1] <= self.zf))
            z = z[valid][:n]
            while len(z) < n:
                additional_z = copula.rvs(int((1+dn) * n))
                valid = ((additional_z[:, 0] >= self.z0) & (additional_z[:, 0] <= self.zf)
                         & (additional_z[:, 1] >= self.z0) & (additional_z[:, 1] <= self.zf))
                z = np.vstack([z, additional_z[valid]])[:n]
        else:
            z = copula.rvs(n)
        assert z.shape[0] == n, "Sample size mismatch"
        u = self.G.cdf(z)
        return u

    def get_sample(self, n=None, dn=None, rho_eps=None):
        if n is None:
            n = self.n_sample
        if self.apply_rho:
            if dn is None:
                dn = self.sample_dn
            u_x, u_y = self.copula_sample(n, dn, rho_eps).T
        else:
            u_x = self.dist_x.get_uniform(n)
            u_y = self.dist_y.get_uniform(n)
        X = self.f_inv_x(u_x)
        Y = self.f_inv_y(u_y)
        self.sample = np.vstack([X, Y]).T
        if self.to_mercator:
            self.sample = space.spherical_fixed_point_to_mercator(self.sample, self.r0)
        self.p_sample = self.p_xy(X, Y)
        self.sample_sorted = self.sample[np.argsort(-self.p_sample)] # sort by probability
        return self.sample, self.p_sample

    def get_pdf(self, n=None, rho_eps=None):
        if n is None:
            n = self.n_grid
        if self.density == 'pchip':
            u_min_x = self.q.min()
            u_min_y = u_min_x
            u_max = self.q.max()
        else:
            u_min_x = self.cdf_x(self.dist_x.xq_hd[0])[0]
            u_min_y = self.cdf_y(self.dist_y.xq_hd[0])[0]
            u_max = 1
        l_x = self.f_inv_x(np.linspace(u_min_x, u_max, n))
        l_y = self.f_inv_y(np.linspace(u_min_y, u_max, n))
        pdf_xy = self.p_xy(l_x[:, None], l_y[None, :], rho_eps)
        return pdf_xy, l_x, l_y

    def get_mode(self, n=None, mode_margin=None, mode_weighted=None, mode_method=None):
        if mode_margin is None:
            mode_margin = self.mode_margin
        if mode_weighted is None:
            mode_weighted = self.mode_weighted
        if mode_method is None:
            mode_method = self.mode_method

        if mode_method == 'roots': # and self.rho == 0: # for rho != 0, the roots are not the MLE
            if self.rho == 0:
                X_mode = self.dist_x.get_mode(mode_method='roots')
                Y_mode = self.dist_y.get_mode(mode_method='roots')
                if self.to_mercator:
                    X_mode, Y_mode = space.spherical_fixed_point_to_mercator(np.array([[X_mode, Y_mode]]), self.r0)[0]
            else:
                # fallback to sample
                warnings.warn("Mode estimation from roots is not implemented for rho != 0. Falling back to mode estimation from sample.", RuntimeWarning)
                return self.get_mode(n=n, mode_margin=0, mode_weighted=False, mode_method='sample')
        elif mode_method == 'grid':
            if n == self.n_grid and self.pdf is not None:
                pdf_xy = self.pdf
                l_x = self.l_x
                l_y = self.l_y
            else:
                pdf_xy, l_x, l_y = self.get_pdf(n)

            i, j = np.where(pdf_xy >= pdf_xy.max()*(1-mode_margin))

            if i.size > 1:
                if mode_weighted:
                    w = pdf_xy[i, j]
                    X_mode = l_x[i].dot(w)/w.sum()
                    Y_mode = l_y[j].dot(w)/w.sum()
                else:
                    X_mode = l_x[i].mean()
                    Y_mode = l_y[j].mean()
            else:
                X_mode = l_x[i[0]]
                Y_mode = l_y[j[0]]
            if self.to_mercator:
                X_mode, Y_mode = space.spherical_fixed_point_to_mercator(np.array([[X_mode, Y_mode]]), self.r0)[0]
        elif mode_method == 'sample':
            if n != self.n_sample or self.sample is None:
                self.get_sample(n)
            sample = self.sample
            pdf_sample = self.p_sample
            i = np.where(pdf_sample >= pdf_sample.max()*(1-mode_margin))[0]
            if mode_weighted:
                w = pdf_sample[i]
                X_mode, Y_mode = sample[i].T.dot(w)/w.sum()
            else:
                X_mode, Y_mode = sample[i].mean(axis=0)
        else:
            raise ValueError("mode_method must be 'grid' or 'sample'")
        return X_mode, Y_mode

    def weight_PR(self, alpha=0.1, **kwargs):
        if self.sample_sorted is None:
            self.get_sample(**kwargs)
        selected_idxs = int((1-alpha)*self.n_sample)
        return self.sample_sorted[:selected_idxs]

    def compute_hull(self, alpha=0.1, **kwargs):
        selected_sample = self.weight_PR(alpha, **kwargs)
        if selected_sample.shape[0] < 3:
            return np.nan
        else:
            try:
                hull = ConvexHull(selected_sample)
            except:
                return np.nan
            hull_polygon = selected_sample[hull.vertices]
            if (hull_polygon[0] != hull_polygon[-1]).all():
                hull_polygon = np.vstack([hull_polygon, hull_polygon[0]])
            return hull_polygon

    def compute_alpha_shape(self, alpha=0.1, **kwargs):
        selected_sample = self.weight_PR(alpha, **kwargs)
        shaper = Alpha_Shaper(selected_sample)
        _, alpha_shape = shaper.optimize()
        exterior = []
        holes = []
        for boundary in get_boundaries(alpha_shape):
            exterior.append(boundary.exterior)
            holes += boundary.holes
        return exterior, holes

    def compute_contour(self, alpha=0.1, n_grid=None, rho_eps=None):
        if self.pdf is None:
            if n_grid is None:
                n_grid = self.n_grid
            pdf, x, y = self.get_pdf(n=n_grid, rho_eps=rho_eps)
        else:
            pdf, x, y = self.pdf, self.l_x, self.l_y

        # pad all borders with zeros.
        pdf_pad = np.pad(pdf, 1, mode='constant')
        x_pad = np.hstack((self.pad_x_left, x, self.pad_x_right))
        y_pad = np.hstack((self.pad_y_left, y, self.pad_y_right))

        q = np.quantile(pdf, alpha)
        cont_gen = contour_generator(z=pdf_pad.T, x=x_pad, y=y_pad)
        contours = cont_gen.lines(q)
        return contours

    def pp(self, mode, **mode_kwargs):
        """
        Point prediction.
        """
        if mode == 'mean':
            return self.sample.mean(axis=0)
        elif mode == 'median':
            me_idx = np.where(self.q == 0.5)[0][0]
            out = np.array([self.x_q[me_idx], self.y_q[me_idx]])
            if self.to_mercator:
                out = space.spherical_fixed_point_to_mercator(out[None], self.r0)[0]
            return out
        elif mode == 'mode':
            return self.get_mode(**mode_kwargs)
        else:
            raise ValueError("mode must be 'mean', 'median' or 'mode'")

    def pr(self, method, confidence=0.9, n_grid=None, n_sample=None):
        alpha = 1 - confidence
        if method == 'hull':
            hull_vertices = self.compute_hull(alpha)
            if isinstance(hull_vertices, np.ndarray):
                area = area_polygon_on_Earth(hull_vertices, mode='adaptive') # for hull the adaptive does not raise memory error
                return hull_vertices, area
            else:
                return np.nan
        elif method == 'alpha_shape':
            exterior, holes = self.compute_alpha_shape(alpha)
            area = np.array([area_polygon_on_Earth(ext, mode='recursive') for ext in exterior]
                            + [-area_polygon_on_Earth(hole, mode='recursive') for hole in holes])
            return exterior, holes, area
        elif method == 'contour':
            contours = self.compute_contour(alpha, n_grid=n_grid)
            if contours:
                exterior, holes = process_contours(contours)
                if self.to_mercator:
                    exterior = [space.spherical_fixed_point_to_mercator(ext, self.r0) for ext in exterior]
                    holes = [space.spherical_fixed_point_to_mercator(hole, self.r0) for hole in holes]
                area_ext = np.array([area_polygon_on_Earth(ext, mode='recursive') for ext in exterior])
                area_holes = (-1) * np.array([area_polygon_on_Earth(h, mode='recursive') for h in holes])

                # check invalid areas
                idx_ext = np.where(area_ext == 0)[0]
                if idx_ext.size > 0:
                    exterior = [exterior[i] for i in range(len(exterior)) if i not in idx_ext]
                    area_ext = area_ext[area_ext != 0]
                idx_holes = np.where(area_holes == 0)[0]
                if idx_holes.size > 0:
                    holes = [holes[i] for i in range(len(holes)) if i not in idx_holes]
                    area_holes = area_holes[area_holes != 0]

                area = np.hstack([area_ext, area_holes])
                return exterior, holes, area
            else:
                return np.nan
        else:
            raise ValueError("method must be 'hull', 'alpha_shape' or 'contour'")

    def summary_pr(self, confidence=[0.5, 0.9, 0.95]):
        """
        Summarizes the prediction regions for the given confidence levels.

        Parameters:
        confidence (list of float): List of confidence levels for which to compute the prediction regions.

        Returns:
        dict: A pandas Series containing the vertices and areas of the prediction regions for each confidence level.
              The keys are tuples of the form (confidence, method, attribute), where:
              - confidence is the confidence level (e.g., 0.5, 0.9, 0.95)
              - method is the method used to compute the prediction region ('hull', 'alpha_shape' or 'contour')
              - attribute is the attribute of the prediction region ('vertices', 'area', 'exterior', 'holes')
        """
        PR = {}
        for c in confidence:
            hull_vertices, hull_area = self.pr('hull', c)
            PR[(c, 'hull', 'vertices')] = hull_vertices
            PR[(c, 'hull', 'area')] = hull_area

            for method in ['alpha_shape', 'contour']:
                exterior, holes, area = self.pr(method, c)
                PR[(c, method, 'exterior')] = exterior
                PR[(c, method, 'holes')] = holes
                PR[(c, method, 'area')] = area

        PR = pd.Series(PR)
        PR.index.names = ['confidence', 'method', 'attribute']
        return PR

    def summary_pp(self, **mode_kwargs):
        pp = {}
        for mode in ['mean', 'median', 'mode']:
            pp[mode] = self.pp(mode, **mode_kwargs)
        return pp


class NonParametricMixedDistribution:
    def __init__(self, x_q, y_q, q, dx=10, dy=10, density='qrde', **kwargs):
        self.x_q = x_q
        self.y_q = y_q
        self.q = q
        self.dx = dx
        self.dy = dy
        self.kwargs = kwargs
        self.density = density
        self.apply_rho = False
        if density == 'qrde':
            self.univariate_gen = QRDEUnivariate
        elif density == 'pchip':
            self.univariate_gen = PCHIPUnivariate
        else:
            raise ValueError("density must be 'qrde' or 'pchip'")

        if isinstance(x_q, (float, int)):
            self.x_dist = NonParametricPointDistribution(x_q, dx=dx)
            self.x_point = True
        else:
            self.x_dist = self.univariate_gen(x_q, q, **kwargs)
            self.x_point = False

        if isinstance(y_q, (float, int)):
            self.y_dist = NonParametricPointDistribution(y_q, dx=dy)
            self.y_point = True
        else:
            self.y_dist = self.univariate_gen(y_q, q, **kwargs)
            self.y_point = False

    def pp(self, **kwargs):
        x_pp = self.x_dist.pp(**kwargs)
        y_pp = self.y_dist.pp(**kwargs)
        return np.array([x_pp, y_pp])

    def pr(self, confidence=0.9, dx=None, dy=None, **kwargs): # kwargs added for compatibility
        alpha = 1 - confidence
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        x_pi = self.x_dist.pi(alpha=alpha, dx=dx)
        y_pi = self.y_dist.pi(alpha=alpha, dy=dy)

        return np.vstack([x_pi, y_pi])


class BivariateMixture:
    def __init__(self, means, covariances, weights, size=10000, seed=0):
        assert math.isclose(np.sum(weights), 1, rel_tol=1e-5, abs_tol=1e-8), "weights must sum to 1"
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.size = size
        self.seed = seed
        self.pdf = None
        self.sample = None
        self.probs = None
        self.region_data = None
        self.hull = None
        self.alpha_shape = None
        self.exterior = None
        self.holes = None
        self.test_sample = None

    def generate_pdf(self, sample=None, x=None):
        if sample is not None:
            x_min = sample.min(axis=0) * 1.1
            x_max = sample.max(axis=0) * 1.1
            x, y = np.mgrid[x_min[0]:x_max[0]:0.01, x_min[1]:x_max[1]:0.01]
            pos = np.dstack((x, y))
        elif x is not None:
            x, y = x
            pos = np.dstack((x, y))
        else:
            raise ValueError("either sample or x must be provided")

        pdf = np.zeros_like(x)
        for mean, covariance, weight in zip(self.means, self.covariances, self.weights):
            pdf += weight * multivariate_normal(mean, covariance).pdf(pos)
        self.pdf, self.x, self.y = pdf, x, y
        return pdf, x, y

    @staticmethod
    def _generate_sample(means, covariances, weights, size=10000, seed=0):
        np.random.seed(seed)
        sample = np.empty((size, 2))
        component_idxs = np.random.randint(0, len(weights), size=size)
        for idx in range(len(weights)):
            is_idx = component_idxs == idx
            n_samples = is_idx.sum()
            sample[is_idx] = np.random.multivariate_normal(means[idx], covariances[idx], n_samples)
        return sample

    def generate_sample(self, size=None, seed=None):
        if size is None:
            size = self.size
        if seed is None:
            seed = self.seed

        self.sample = self._generate_sample(self.means, self.covariances, self.weights, size=size, seed=seed)
        return self.sample

    def generate_test_sample(self, size=None, seed=None):
        if size is None:
            size = self.size
        if seed is None:
            seed = self.seed + 1

        self.test_sample = self._generate_sample(self.means, self.covariances, self.weights, size=size, seed=seed)
        return self.test_sample

    def evaluate_pdf(self, sample=None):
        if sample is None:
            if self.sample is None:
                sample = self.generate_sample()
            else:
                sample = self.sample

        probs = np.zeros((sample.shape[0]))
        for mean, covariance, weight in zip(self.means, self.covariances, self.weights):
            probs += weight * multivariate_normal(mean, covariance).pdf(sample)
        self.probs = probs
        return probs

    def generate_cr_points(self, alpha=0.1):
        if self.probs is None:
             self.evaluate_pdf()
        sorted_indices = np.argsort(-self.probs)
        sorted_sample = self.sample[sorted_indices]
        selected_data = int((1-alpha) * self.sample.shape[0])
        region_data = sorted_sample[:selected_data]
        self.region_data = region_data
        return region_data

    def compute_convex_hull(self):
        if self.region_data is None:
            self.generate_cr_points()
        self.hull = ConvexHull(self.region_data)
        self.hull_polygon = self.region_data[self.hull.vertices]
        return self.hull

    def compute_alpha_shape(self, alpha=None):
        if self.region_data is None:
            self.generate_cr_points()
        shaper = Alpha_Shaper(self.region_data)
        if alpha is None:
            _, alpha_shape = shaper.optimize()
        else:
            alpha_shape = shaper.get_alpha_shape(alpha)
        self.alpha_shape = alpha_shape
        return self.alpha_shape

    def extract_boundaries_and_holes(self):
        if self.alpha_shape is None:
            self.compute_alpha_shape()
        exterior = []
        holes = []
        for boundary in get_boundaries(self.alpha_shape):
            exterior.append(boundary.exterior)
            holes += boundary.holes
        self.exterior, self.holes = exterior, holes
        return exterior, holes

    def plot_alpha_shape(self, fig=None):
        if self.alpha_shape is None:
            self.compute_alpha_shape()
        if fig is None:
            fig = get_figure()

        for boundary in get_boundaries(self.alpha_shape):
            fig.add_traces(_plot_boundary(boundary))
        return fig

    def CR_plot(self, hull=True, alpha_shape=True, alpha=0.05):
        fig = get_figure(xaxis_title="WE distance", yaxis_title="SN distance", title=f"Confidence Region (𝛼 = {alpha})")
        fig.add_trace(go.Scatter(x=self.sample[:, 0], y=self.sample[:, 1], mode="markers", marker=dict(size=10, opacity=0.2, symbol='x'), showlegend=False))
        if hull:
                if self.hull is None:
                        self.compute_convex_hull()
                for simplex in self.hull.simplices:
                        fig.add_trace(go.Scatter(x=self.region_data[simplex, 0], y=self.region_data[simplex, 1], mode="lines", line=dict(color='black', width=4), showlegend=False))
        if alpha_shape:
                if self.exterior is None or self.holes is None:
                        self.extract_boundaries_and_holes()
                for ext in self.exterior:
                    fig.add_trace(go.Scatter(x=ext[:, 0], y=ext[:, 1], mode="lines", line=dict(color='red', width=4), showlegend=False))
                for hole in self.holes:
                    fig.add_trace(go.Scatter(x=hole[:, 0], y=hole[:, 1], mode="lines", line=dict(color='red', width=4), showlegend=False))
        return fig

    def eval_confidence_convex_hull(self, **kwargs):
        """
        Evaluate the confidence of the new sample based on the convex hull of the region data.
        This is, the fraction of the new sample that is within the convex hull.
        """
        if self.test_sample is None:
            self.generate_test_sample(**kwargs)
        return geometry.is_inside_polygon_parallel(self.test_sample, self.hull_polygon).mean()

    def eval_confidence_alpha_shape(self, **kwargs):
        """
        Evaluate the confidence of the new sample based on the alpha shape of the region data.
        This is, the fraction of the new sample that is within the alpha shape.
        """
        if self.test_sample is None:
            self.generate_test_sample(**kwargs)

        c = 0
        for ext in self.exterior:
            c += geometry.is_inside_polygon_parallel(self.test_sample, ext).mean()
        for hole in self.holes:
            c -= geometry.is_inside_polygon_parallel(self.test_sample, hole).mean()
        return c

    def convex_hull_area(self):
        if self.hull is None:
            self.compute_convex_hull()
        return _polygon_area(*self.hull_polygon.T)

    def alpha_shape_area(self):
        if self.exterior is None or self.holes is None:
            self.extract_boundaries_and_holes()
        area = 0
        for ext in self.exterior:
            area += _polygon_area(*ext.T)
        for hole in self.holes:
            area -= _polygon_area(*hole.T)
        return area

def _polygon_area(x, y):
    """
    Shoelace formula
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

if alpha_pkg:
    def _plot_boundary(boundary: Boundary):
        exterior = boundary.exterior
        holes = boundary.holes

        # Create the exterior polygon trace
        exterior_trace = go.Scatter(
            x=exterior[:, 0],
            y=exterior[:, 1],
            mode='lines+markers',
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.5)',
            line=dict(color='blue', width=0.8),
            showlegend=False,
        )

        # Create the hole polygons traces
        hole_traces = []
        for hole in holes:
            hole_trace = go.Scatter(
                x=hole[:, 0],
                y=hole[:, 1],
                mode='lines+markers',
                fill='toself',
                fillcolor='rgba(255, 255, 255, 0.5)',
                line=dict(color='blue', width=0.8),
                showlegend=False,
            )
            hole_traces.append(hole_trace)

        return [exterior_trace] + hole_traces
