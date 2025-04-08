import numpy as np
from numba import njit

@njit
def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Multiply by radius of Earth to get distance in km.
    Assumes lat1, lon1, lat2, lon2 are all in radians.
    """
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return d

@njit
def great_circle_distance_trajectory_min(x1, x2):
    """
    Minimum great circle distance between two trajectories.
    """
    n1 = x1.shape[1]
    n2 = x2.shape[1]
    d = np.empty((n1, n2))
    for i in range(n1):
        d[i] = great_circle_distance(x1[0, i], x1[1, i], x2[0], x2[1])
    return d.min()

@njit
def great_circle_distance_trajectory_mean(x1, x2):
    """
    Average great circle distance between two trajectories.
    """
    n1 = x1.shape[1]
    n2 = x2.shape[1]
    d = np.empty((n1, n2))
    for i in range(n1):
        d[i] = great_circle_distance(x1[0, i], x1[1, i], x2[0], x2[1])
    return d.mean()

@njit
def great_circle_distance_by_time_step(x):
    """
    Great circle distance between consecutive points in a trajectory.
    x is a 2 x n array of latitudes and longitudes in radians.
    """
    n = x.shape[1]
    d = np.empty(n-1)
    for i in range(n-1):
        d[i] = great_circle_distance(x[0, i], x[1, i], x[0, i+1], x[1, i+1])
    return d

@njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Only valid for 2D arrays.
    """
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def nb_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@njit
def nb_median(array, axis):
  return np_apply_along_axis(np.median, axis, array)

@njit
def avg_of_func(func, X, label, label_categories):
    """
    Average of a function applied to each group in X.
    """
    result = np.empty((len(label_categories), X.shape[1]))
    for i, l in enumerate(label_categories):
        result[i] = func(X[label == l], axis=0)
    return nb_mean(result, axis=0)

@njit
def avg_of_medians(X, label, label_categories):
    return avg_of_func(nb_median, X, label, label_categories)

@njit
def avg_of_means(X, label, label_categories):
    return avg_of_func(nb_mean, X, label, label_categories)
