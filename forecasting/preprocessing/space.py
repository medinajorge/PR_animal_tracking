import numpy as np
import pandas as pd
import math
from numba import njit
import gc
import calendar
from collections.abc import Iterable
from collections import defaultdict
import datetime
import random
from copy import deepcopy
import os
from pathlib import Path
import warnings
import sys
sys.stdout.flush()
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)

RootDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fullPath = lambda path: os.path.join(RootDir, path)
from ..params import *

##############################################################################################################################
"""                                                III. Trajectory shift                                                   """
##############################################################################################################################

def to_cartesian(lat, lon):
    """lat lon in rads"""
    if type(lat) == np.ndarray:
        r_cartesian = np.empty((lat.size, 3))
        r_cartesian[:, 0] = np.cos(lon)*np.cos(lat)
        r_cartesian[:, 1] = np.sin(lon)*np.cos(lat)
        r_cartesian[:, 2] = np.sin(lat)
    else:
        r_cartesian = np.empty((3))
        r_cartesian[0] = math.cos(lon)*math.cos(lat)
        r_cartesian[1] = math.sin(lon)*math.cos(lat)
        r_cartesian[2] = math.sin(lat)
    return r_cartesian

def to_spherical(r):
    if len(r.shape) > 1:
        lat = np.arctan2(r[:,2], np.sqrt(np.square(r[:,:2]).sum(axis=1)))
        lon = np.arctan2(r[:,1], r[:,0])
    else:
        lat = math.atan2(r[2], math.sqrt(np.square(r[:2]).sum()))
        lon = math.atan2(r[1], r[0])
    return lat, lon

def great_circle_distance(lat, lon, lat_f=None, lon_f=None):
    if lat_f is None:
        lat_f, lat_0 = lat[1:], lat[:-1]
        lon_f, lon_0 = lon[1:], lon[:-1]
    else:
        lat_0, lon_0 = lat, lon
    sigma = 2*np.arcsin(np.sqrt(np.sin(0.5*(lat_f-lat_0))**2 + np.cos(lat_f)*np.cos(lat_0)*np.sin(0.5*(lon_f - lon_0))**2))
    return sigma

def great_circle_distance_cartesian(r):
    r_spherical = to_spherical(r)
    return great_circle_distance(*r_spherical)

def spherical_velocity(x, dt=None):
    """
    Returns: velocity (if dt is provided) or distance in terms of the spherical unit vectors (r, theta, phi).
    Attributes:
        - x:  array containing latitude and longitude in the first 2 rows.
        - dt: array of time increments between points.
    """
    if x.shape[1] < 2:
        warnings.warn("x should have at least 2 observations to compute the velocity")
        return (np.ones(3) * np.NaN)[None]
    else:
        d_cartesian = log_map_vec(to_cartesian(*x[:2])) # tangent vectors with modulus equal to the distance traveled.
        A = conversion_matrix_vec(*x[:2])
        d_spherical = np.array([a.dot(d) for a, d in zip(A, d_cartesian)]) # convert to spherical coordinates
        d_spherical[:, 1] *= -1 # avoid reflection
        if dt is None:
            return d_spherical
        else:
            v = d_spherical / dt[:, None]
            return v

def conversion_matrix(lat, lon):
    """For a vector V:
    V = (Vr, Vlat, Vlon) = A (Vx, Vy, Vz)
    """
    sin_lat = math.sin(lat)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)

    e_r = np.array([cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                   ])
    e_lat = np.array([sin_lat * cos_lon,
                      sin_lat * sin_lon,
                      -cos_lat
    ])
    e_phi = np.array([-sin_lon,
                      cos_lon,
                      0
    ])
    A = np.vstack([e_r, e_lat, e_phi])
    return A

def conversion_matrix_vec(Lat, Lon):
    """
    Vectorized version of 'conversion_matrix'.
    Returns conversion matrices A_i stacked on axis 0.
    """
    lat = Lat[:, None]
    lon = Lon[:, None]
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    e_r = np.hstack([cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                   ])
    e_lat = np.hstack([sin_lat * cos_lon,
                      sin_lat * sin_lon,
                      -cos_lat
    ])
    e_phi = np.hstack([-sin_lon,
                      cos_lon,
                      np.zeros((lat.shape[0], 1))
    ])
    A = np.stack([e_r, e_lat, e_phi], axis=1)
    return A

def exponential_map(P, v):
    """Exponential map. From point P in Riemannian fold, returns the point Q resulting from the shift of P according to a vector v in the Euclidean tangent space."""
    if len(P.shape) > 1:
        v_norm = np.sqrt(np.square(v).sum(axis=1))[:, None]
        Q = P*np.cos(v_norm) + v*np.sin(v_norm)/v_norm
        not_displacing = v_norm.squeeze() == 0
        Q[not_displacing] = P[not_displacing]
    else:
        v_norm = math.sqrt(v.dot(v))
        if v_norm > 0:
            Q = P*math.cos(v_norm) + v*math.sin(v_norm)/v_norm
        else:
            Q = P
    return Q

def log_map(P, Q):
    """Logarithmic map. From two points P, Q in the manifold, determine the vector v in the tangent space of P, such that an exponential map Exp(P,v) takes point P to point Q
    in the manifold."""
    d = great_circle_distance_cartesian(np.vstack([P, Q]))
    if d > 0:
        u = Q - P.dot(Q)*P
        v = d * u / math.sqrt(u.dot(u))
    else:
        v = np.zeros((3))
    return v

def log_map_vec(r):
    """Logarithmic map. From two points P, Q in the manifold, determine the vector v in the tangent space of P, such that an exponential map Exp(P,v) takes point P to point Q
    in the manifold. v verifies ||v|| = d(P,Q).
    Returns v for each pair of points in the trajectory.
    """
    d = great_circle_distance_cartesian(r)[:, None]
    P = r[:-1]
    Q = r[1:]
    u = Q - (P*Q).sum(axis=1)[:,None]*P
    v = d * u / np.sqrt(np.square(u).sum(axis=1))[:, None]
    v[d.squeeze() == 0] = np.zeros((3))
    return v

def log_map_vec_fixed_point(r0, r):
    """
    r0: point on the sphere (lat, lon). Shape (2,)
    r: trajectory on the sphere (lat, lon). Shape (2, n)
    """
    if r0.ndim == 1:
        r0 = r0[:, None]
    d = great_circle_distance(*r0, *r)[:, None]
    Q = to_cartesian(*r)
    P = to_cartesian(*r0)
    u = Q - (P*Q).sum(axis=1)[:, None]*P
    v = d * u / np.sqrt(np.square(u).sum(axis=1))[:, None]
    A = conversion_matrix(*r0[:, 0]) # from cartesian to spherical
    v_spherical = A.dot(v.T)
    sn_we = v_spherical[-2:]
    return sn_we

def exp_map_vec_fixed_point(r0, v):
    """
    r0: point on the sphere (lat, lon)
    v: tangent vector on the sphere, in spherical coordinates. Shape (2, n)
    """
    if r0.ndim == 1:
        r0 = r0[:, None]
    v = np.vstack((np.zeros((1, v.shape[1])), v)) # (r, theta, phi)
    A = conversion_matrix(*r0)
    v_c = A.T.dot(v).T
    v_norm = np.sqrt(np.square(v_c).sum(axis=1))[None]
    P = to_cartesian(*r0).T
    Q = P*np.cos(v_norm) + v_c.T*np.sin(v_norm)/v_norm
    lat_lon = np.vstack(to_spherical(Q.T))
    return lat_lon

def transport_vector(P, Q, v):
    v_spherical = conversion_matrix(*to_spherical(P)).dot(v)
    v_Q = conversion_matrix(*to_spherical(Q)).T.dot(v_spherical)
    return v_Q

def sphere_translation_riemann(original_lat, original_lon, origin=(0,0)):
    """
    Translation of the latitude and longitude to the origin O (1,0,0).
    Returns: Shifted lat, lon coordinates.
    Both input and output are in rads (!).
    """

    original_trajectory = to_cartesian(original_lat, original_lon)
    tangent_vectors = log_map_vec(original_trajectory)
    P = to_cartesian(*origin)
    new_trajectory = np.empty((original_lat.size, 3))
    new_trajectory[0] = P
    for i, (P_original, v) in enumerate(zip(original_trajectory, tangent_vectors), start=1):
        v_shifted = transport_vector(P_original, P, v)
        P = exponential_map(P, v_shifted)
        new_trajectory[i] = P

    lats, lons = to_spherical(new_trajectory)
    return lats, lons

def straight_path(x0, xf, num_steps):
    """
    x0: initial point (X, Y) in mercator coordinates
    xf: final point (X, Y) in mercator coordinates
    num_steps: number of steps to interpolate between x0 and xf. The first returned point is not x0, and the last is not xf. Returns only the intermediate points.
    """
    lat, lon = mercator_inv(*x0)
    lat_f, lon_f = mercator_inv(*xf)
    P = to_cartesian(lat, lon)
    Q = to_cartesian(lat_f, lon_f)
    v = log_map(P, Q)
    v_i = v / (num_steps + 1)
    Qs = np.empty((num_steps, 3))
    for i in range(num_steps):
        Qs[i] = exponential_map(P, v_i * (i+1))
    lat_Q, lon_Q = to_spherical(Qs)
    X_Q = np.vstack(mercator(lat_Q, lon_Q)).T # (num_steps, 2)
    return X_Q

def shift_trajectories(weather=False, split_by="day", groupby="ID", pad_day_rate=None, savingDir=fullPath('utils/data')):
    """Displaces each trajectory to (lat,lon) = (0,0) and saves the result as a lzma file."""
    weather_str = "_weather" if weather else ""
    equal_spacing_str = "" if pad_day_rate is None else f'_equally-spaced-local-dr{pad_day_rate}'

    X = file_management.load_lzma('utils/data/trajectories{}_split-by-{}_groupby-{}_default{}.lzma'.format(weather_str, split_by, groupby, equal_spacing_str))
    X_shifted = []
    for x in tqdm(X):
        lat, lon = sphere_translation_riemann(*(x[:2]*np.pi/180))
        lat *= 180/np.pi
        lon *= 180/np.pi
        x_total = np.vstack((lat, lon, x[2:]))
        X_shifted.append(x_total)

    print("Saving")
    file_management.save_lzma(X_shifted, 'trajectories{}_split-by-{}_groupby-{}_to-origin{}'.format(weather_str, split_by, groupby, equal_spacing_str), savingDir)
    return

@njit
def mercator_joint(lat, lon, R=R_earth):
    """
    Assumes latitude and longitude are in radians and returns x, y coordinates in km
    Output shape: (2, n)
    """
    return np.vstack(mercator(lat, lon, R))

@njit
def mercator(lat, lon, R=R_earth):
    """Assumes latitude and longitude are in radians and returns x, y coordinates in km"""
    x = R * lon
    y = R * np.log(np.tan(np.pi / 4 + lat / 2))
    return x, y

@njit
def mercator_inv(x, y, R=R_earth):
    """Returns latitude and longitude in radians"""
    lat = 2 * np.arctan(np.exp(y / R)) - np.pi / 2
    lon = x / R
    return lat, lon

def spherical_fixed_point_to_mercator(sample, r0):
    lat_lon = exp_map_vec_fixed_point(r0, sample.T)
    xy = mercator_joint(*lat_lon)
    return xy.T # (n, 2)
