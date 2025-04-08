import numpy as np
import pandas as pd
import math
from numba import njit
from pvlib import solarposition
import gc
import calendar
import warnings
from collections.abc import Iterable
from collections import defaultdict
import scipy.stats as ss
from scipy.stats import norm, levy, wrapcauchy
from scipy.special import erf, erfinv, erfc, erfcinv
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.interpolate import interp1d
import datetime
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.cluster import DBSCAN
from copy import deepcopy
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys
import warnings
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

try:
    import xgboost as xgb
except:
    pass
from tidypath import storage
from phdu import savedata, np_utils, pd_utils, SavedataSkippedComputation
from . import geometry, nb_funcs
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    num_cores = 0
    tf.config.threading.set_inter_op_parallelism_threads(2) #num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.set_soft_device_placement(True)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
except:
    pass

RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fullPath = lambda path: os.path.join(RootDir, path)


##############################################################################################################################
"""                                                    I. Parameters                                                       """
##############################################################################################################################

interesting_cols = ["ID", "COMMON_NAME", "Taxa", "Class", "SEX", "DATABASE", "TAG", "NumberOfSatellites", "Length", "Mean_latitude", "Mean_longitude", "Mean_year"]
updated_cols = ['Cluster ID', 'Cluster ID confidence', 'Cluster ID confidence interval', 'Animals in dataset', 'Animals in dataset interval', 'Length interval', "Mean year interval"]
secondary_cols = ["Order", "Family", "SatelliteProg", "TAG_TYPE", "ResidualError", "Stage", "AGE", "BODY_LENGTH"]
totally_unimportant_cols = ["Colour"]
all_cols = interesting_cols + secondary_cols + totally_unimportant_cols + updated_cols
#NOTE: ID != UniqueAnimal_ID, but equally identify the animals I think

discarded_weather_cols = ["2 metre temperature",
                          '100 metre U wind component', '100 metre V wind component', 'Neutral wind at 10 m u-component', 'Neutral wind at 10 m v-component',
                          'Surface pressure'
                         ]

weather_col_selection = ["Bathymetry", 'Sea ice area fraction', 'Sea surface temperature', 'Surface net solar radiation', 'Surface net thermal radiation',
                         'Mean sea level pressure', 'Significant height of combined wind waves and swell']

weather_cols = dict(temperature = ['Sea ice area fraction', 'Sea surface temperature',
                                   'Surface net solar radiation', 'Surface net thermal radiation'
                                  ],
                    wind = ['10 metre U wind component', '10 metre V wind component','K index',
                            'Mean sea level pressure', "Total precipitation"
                           ] ,
                    waves = ['Mean wave period', 'Significant height of combined wind waves and swell',
                             'Mean wave direction_x', 'Mean wave direction_y'],
                    bathymetry = ["Bathymetry"]
)
weather_cols_idxs = {}
pointer = 0
for k, v in weather_cols.items():
    list_len = len(v)
    weather_cols_idxs[k] = np.arange(pointer, pointer + list_len)
    pointer += list_len

weather_cols.update(dict(all=[col for value in weather_cols.values() for col in value]))
weather_cols_idxs["all"] = np.arange(len(weather_cols["all"]))

weather_cols_v2 = {}
weather_cols_v2['all-depth'] = ['votemper_10m', 'votemper_97m', 'votemper_1046m',
                                'vomecrtn_1m', 'vomecrtn_97m', 'vomecrtn_1516m',
                                'vozocrte_1m', 'vozocrte_97m', 'vozocrte_1516m',
                                'vosaline_1m', 'vosaline_10m', 'vosaline_97m',
                                '10 metre U wind component', '10 metre V wind component',
                                'Mean sea level pressure', 'Sea surface temperature',
                                'Mean wave period',
                                'Significant height of combined wind waves and swell',
                                'Total precipitation', 'K index', 'Sea ice area fraction',
                                'Surface net short-wave (solar) radiation',
                                'Surface net long-wave (thermal) radiation',
                                'Near IR albedo for diffuse radiation',
                                'Near IR albedo for direct radiation',
                                'Downward UV radiation at the surface', 'Evaporation', 'Geopotential',
                                'Mean wave direction_x', 'Mean wave direction_y', 'coast-d',
                                'bathymetry']
weather_cols_v2['all'] = [col for col in weather_cols_v2['all-depth'] if not col.startswith("vo") or (not col.startswith('votemper') and '1m' in col)]
weather_cols_v2[None] = []
weather_cols_v2['pruned'] = ['Sea surface temperature', # pruned using multi-collinearity
                             'votemper_1046m',
                             'vomecrtn_1m',
                             'vomecrtn_97m',
                             'vomecrtn_1516m',
                             'vozocrte_1m',
                             'vozocrte_97m',
                             'vozocrte_1516m',
                             'vosaline_1m',
                             '10 metre U wind component',
                             '10 metre V wind component',
                             'Mean sea level pressure',
                             'Mean wave period',
                             'Significant height of combined wind waves and swell',
                             'Total precipitation',
                             'K index',
                             'Sea ice area fraction',
                             'Surface net short-wave (solar) radiation',
                             'Surface net long-wave (thermal) radiation',
                             'Near IR albedo for diffuse radiation',
                             'Near IR albedo for direct radiation',
                             'Evaporation',
                             'Geopotential',
                             'Mean wave direction_x',
                             'Mean wave direction_y',
                             'coast-d',
                             'bathymetry']

weather_cols_v2['mrmr+collinear'] = ['votemper_1046m', # TODO: to be use with delete_features=['sin t', 'cos t']
                                     'vosaline_1m',
                                     'vozocrte_1516m',
                                     'vomecrtn_1516m',
                                     'vozocrte_97m',
                                     'vomecrtn_97m',
                                     'Geopotential',
                                     'Sea surface temperature',
                                     'vomecrtn_1m',
                                     'vozocrte_1m',
                                     'Mean wave period',
                                     'Significant height of combined wind waves and swell',
                                     'bathymetry',
                                     'Mean wave direction_x',
                                     'Mean wave direction_y',
                                     'coast-d']

weather_cols_v2['vif'] = ['Sea surface temperature',
                          'votemper_1046m',
                          'vomecrtn_1m',
                          'vomecrtn_97m',
                          'vomecrtn_1516m',
                          'vozocrte_1m',
                          'vozocrte_97m',
                          'vozocrte_1516m',
                          '10 metre U wind component',
                          '10 metre V wind component',
                          'Total precipitation',
                          'K index',
                          'Sea ice area fraction',
                          'Surface net short-wave (solar) radiation',
                          'Evaporation',
                          'Geopotential',
                          'Mean wave direction_x',
                          'Mean wave direction_y',
                          'bathymetry']

weather_cols_v2['mrmr+vif'] = ['Sea surface temperature', # TODO: to be use with delete_features=['sin t', 'cos t']
                               'votemper_1046m',
                               'vomecrtn_1m',
                               'vomecrtn_97m',
                               'vomecrtn_1516m',
                               'vozocrte_1m',
                               'vozocrte_97m',
                               'vozocrte_1516m',
                               'Geopotential',
                               'Mean wave direction_x',
                               'Mean wave direction_y',
                               'bathymetry']

weather_cols_v2['mrmrloop+vif'] = ['Sea surface temperature', # TODO: to be use with delete_features=['sin t', 'cos t']
                                   'votemper_1046m',
                                   'vomecrtn_1516m',
                                   'vozocrte_97m',
                                   'vozocrte_1516m',
                                   'Geopotential',
                                   ]

weather_cols_v2['mrmr'] = ['votemper_1046m', # TODO: delete 'sin t' only
                           'Sea surface temperature',
                           'vosaline_97m',
                           'votemper_97m',
                           'vozocrte_1516m',
                           'vomecrtn_1516m',
                           'votemper_10m',
                           'vosaline_10m',
                           'vosaline_1m',
                           'Geopotential',
                           'vozocrte_97m',
                           'vomecrtn_97m',
                           'Significant height of combined wind waves and swell',
                           'Mean wave period',
                           'Mean wave direction_x',
                           'vomecrtn_1m',
                           'vozocrte_1m']

spatiotemporal_cols = ['x', 'y', 'z', 'sin t', 'cos t']
spatiotemporal_cols_raw = ['lat', 'lon', 't']


##############################################################################################################################
"""                                                    II. Main                                                       """
##############################################################################################################################

def get_leap_year(y):
    """Returns bool array. True if time data belongs to a leap year"""
    year_change = np.argwhere(y[:-1] != y[1:])[:,0]
    is_leap = np.empty((y.size), dtype=np.bool)
    year_change_edges = np.hstack([0, year_change, y.size])
    for start, end in zip(year_change_edges[:-1], year_change_edges[1:]):
        is_leap[start:end] = calendar.isleap(y[start])
    return is_leap

@njit
def vectorized_is_leap(years):
    """
    Create a boolean mask where True indicates leap years
    """
    is_leap_mask = np.logical_or(np.logical_and(years % 4 == 0, years % 100 != 0), years % 400 == 0)
    return is_leap_mask

def rescale_dt(dt, is_leap_year):
    """Rescaling dt has to take into account wether there is a leap year."""
    is_leap = is_leap_year[:dt.size]
    leap_group = [is_leap, ~is_leap]
    end_of_year = [366, 365]
    for leap, end_of_year in zip(leap_group, end_of_year):
        dt[leap & (dt < -end_of_year)] %= (end_of_year + 1)
        dt[leap & (dt < 0)] %= end_of_year
    return dt

def replace_dt_zeros(delta_t, by="mean", threshold=1e-8):
    """
    by: "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
    """
    zero_dt_bool = delta_t < threshold
    zero_dt = np.argwhere(zero_dt_bool)[:,0]
    num_zeros = zero_dt.size
    default_dt = 0.11 # median dt of the whole dataset

    if num_zeros > 0:
        idxs = set(range(delta_t.size))
        if by == "closest":
            valid_idxs = idxs - set(zero_dt)
            candidates = np.empty((num_zeros))
            start = 0
            end = num_zeros
            if zero_dt[0] == 0:
                candidate_idxs = valid_idxs - set(range(2))
                if len(candidate_idxs) > 0:
                    candidates[0] = delta_t[min(candidate_idxs)]
                else:
                    candidates[0] = default_dt
                start = 1
            if zero_dt[-1] == (delta_t.size - 1) and delta_t.size > 3:
                candidate_idxs = valid_idxs - set(range(num_zeros - 3, num_zeros))
                if len(candidate_idxs) > 0:
                    candidates[-1] = delta_t[max(candidate_idxs)]
                else:
                    candidates[-1] = default_dt
                end -= 1
            for i, idx in enumerate(zero_dt[slice(start, end)], start=start):
                candidates_upper = valid_idxs - set(range(idx))
                candidates_lower = valid_idxs - set(range(idx, num_zeros))

                if len(candidates_upper) > 0 and len(candidates_lower) > 0:
                    closest_upper = min(candidates_upper) if len(candidates_upper) > 0 else max(candidates_lower)
                    closest_lower =  max(candidates_lower) if len(candidates_lower) > 0 else min(candidates_upper)
                    candidates[i] = delta_t[[closest_upper, closest_lower]].mean()
                elif len(candidates_upper) > 0:
                    candidates[i] = delta_t[min(candidates_upper)]
                elif len(candidates_lower) > 0:
                    candidates[i] = delta_t[max(candidates_lower)]
                else:
                    candidates[i] = default_dt

            delta_t[zero_dt] = candidates

        elif by == "mean":
            non_zeros = delta_t.size - num_zeros
            if non_zeros > 0:
                delta_t[zero_dt_bool] = delta_t[~zero_dt_bool].mean()
            else:
                delta_t[zero_dt] = default_dt # IDEA: Declare as NANS in the preprocessing step and try to infer the value using imputation.

    return delta_t

def compute_dt(t, year, replace_zero_by="mean"):
    if t.size < 2:
        return np.array([0])
    else:
        dt = (t[1:] - t[:-1])
        is_leap = get_leap_year(year)
        new_year_expected = np.argwhere(t[1:] < t[:-1])[:,0] # +1 for idx in t array. As it is, for idx in dt array
        new_year = np.argwhere(year[:-1] != year[1:])[:,0]
        new_year_unexpected = set(new_year) - set(new_year_expected)

        dt = rescale_dt(dt, is_leap)
        dt = replace_dt_zeros(dt, by=replace_zero_by)
        for idx in new_year_unexpected:
            dt[idx] += (366 if is_leap[idx-1] else 365)
        return dt

def split_trajectory_chunks(X, year, meta, max_dt=7):
    """
    Split trajectories into separate chunks when the time difference between consecutive points is greater than max_dt.
    """
    print(f"Splitting trajectories into chunks when the time difference between consecutive points is greater than {max_dt} days.")
    meta = meta.assign(trajectory_section=np.nan).set_index("ID")
    if isinstance(X, list):
        X = pd.Series(X, index=meta.index)
    if isinstance(year, list):
        year = pd.Series(year, index=meta.index)

    def update_label(l, section):
        l.loc['ID'] = l.name
        l.loc['trajectory_section'] = section
        return l

    X_prunned = []
    year_prunned = []
    labels_prunned = []
    pbar = tqdm(range(len(X)))
    for label, x in X.items():
        y = year[label]
        dt = compute_dt(x[2], y)
        splits = np.where(dt > max_dt)[0] + 1
        if splits.size > 0:
            for section, x_split in enumerate(np.split(x, splits, axis=1), start=1):
                X_prunned.append(x_split)
                labels_prunned.append(update_label(meta.loc[label], section))
            for year_split in np.split(y, splits):
                year_prunned.append(year_split)
        else:
            X_prunned.append(x)
            labels_prunned.append(update_label(meta.loc[label], np.nan))
            year_prunned.append(y)
        pbar.update()

    labels_prunned = pd.concat(labels_prunned, axis=1).T
    return X_prunned, year_prunned, labels_prunned

def get_prunning_function(column=None, colvalue=None, label="COMMON_NAME", NaN_value=None, min_animals=5, minlen=5, min_days=0, mode="all", mapping=None, func=None,
                          func_args='meta',
                          vertices=[],
                          **func_kwargs):
    """
    Returns function for prunning the dataset.
    Chooses those rows such that the column "column" has value "colvalue" and sets the classification target on the label "label".

    NaN_value: character for recognizing missing values.
    min_animals: minimum number of animals per species.
    minlen: minimum length of the trajectory.
    min_days: minimum number of days in the trajectory.
    func: function to apply to the metadata.
    vertices: iterable of vertices to prune the data. Only the longest part of the trajectory inside the polygon will be kept.
    """
    def prunning_function(X, Year, y):
        nonlocal colvalue, NaN_value
        print(f"Prunning trajectories:\nmin_observations: {minlen}\nmin_animals: {min_animals}\nmin_days: {min_days}")

        # Handle NaN values
        if NaN_value is not None:
            if not isinstance(NaN_value, (list, tuple, np.ndarray)):
                NaN_value = [NaN_value]
            for nan_v in NaN_value:
                y[label].replace(nan_v, np.nan, inplace=True)

        # Apply custom function if provided
        if func is not None:
            if func_args == 'meta':
                y = func(y, **func_kwargs)
            elif func_args == 'X':
                X = func(X, **func_kwargs)
            elif func_args == 'both':
                X, y = func(X, y, **func_kwargs)
            elif func_args == 'all':
                X, y, Year = func(X, y, Year, **func_kwargs)
            else:
                raise ValueError("func_args must be 'meta', 'X', 'both' or 'all'")

        # Apply column-based filter
        if colvalue is None:
            good_measurements = (~y[label].isna()) & (y[label] != NaN_value)
        else:
            if isinstance(colvalue, str):
                colvalue = [colvalue]
            good_measurements = pd.concat([y[column] == val for val in colvalue], axis=1).any(axis=1) & ((y[label] != NaN_value) if NaN_value is not None else ~y[label].isna())

        # Apply length and days filter
        xlens = np.array([x.shape[1] for x in X])
        good_measurements = good_measurements & (xlens > minlen) & (y[f"Days in trajectory ({mode})"] > min_days)

        # Prune based on good measurements
        X = [x for x, t in zip(X, good_measurements) if t]
        Year = [year for year, t in zip(Year, good_measurements) if t]
        labels = y[good_measurements]

        # Apply mapping if provided
        if mapping is not None:
            labels[f"{label}-original"] = labels[label].values
            if callable(mapping):
                labels.loc[:, label].replace(mapping(labels[label]), inplace=True)
            elif isinstance(mapping, dict):
                labels.loc[:, label].replace(mapping, inplace=True)
            else:
                raise TypeError("'mapping' must be a function, dict or None.")

        # Apply vertices filter if provided
        if len(vertices) > 0:
            X, Year, labels = geometry.prune_trajectories_inside_vertices(X, Year, labels, vertices)

        # Apply min_animals filter as the final step
        animals_per_cat = defaultdict(int)
        animals_per_cat.update(labels[label].value_counts(dropna=False).to_dict())
        sufficient_animals = np.array([np.nan if pd.isna(cat) else animals_per_cat[cat] >= min_animals for cat in labels[label].values])
        labels = labels[sufficient_animals]
        X = [x for x, t in zip(X, sufficient_animals) if t]
        Year = [year for year, t in zip(Year, sufficient_animals) if t]

        y = labels[label].values

        return X, Year, y, labels, label

    return prunning_function

@savedata
def load_data_v2(max_dt=7, meta_filename='metadata.csv', data_filename='dataset.csv'):
    """
    Loads the data from v2.

    Parameters:
        - weather: if None, only the trajectory data is loaded. Otherwise, the weather data is also loaded.
        - pad_day_rate: number of measurements per day when interpolating for obtaining homogeneous sampling rate.
        - max_dt: maximum time difference between consecutive points in a trajectory. If greater, the trajectory is split into chunks.
    """
    meta = pd.read_csv(fullPath(f"data/{meta_filename}"))
    colmap = {c: c for c in meta.columns}
    colmap['Species'] = 'COMMON_NAME'
    meta.columns = meta.columns.map(colmap)
    if 'COMMON_NAME' not in meta.columns:
        meta['COMMON_NAME'] = "" # assign dummy

    df = pd.read_csv(fullPath(f"data/{data_filename}"))
    cols = df.columns
    required_cols = ['DATE_TIME', 'LATITUDE', 'LONGITUDE', 'ID']
    assert all(c in cols for c in required_cols), "Missing columns: {}".format(set(required_cols) - set(cols))
    covariates = [c for c in cols if c not in required_cols]
    output_cols = ['LATITUDE', 'LONGITUDE', 'day'] + covariates

    df['date'] = pd.to_datetime(df['DATE_TIME'])
    df['day'] = (df.date.dt.dayofyear + df.date.dt.hour / 24 + df.date.dt.minute / (24 * 60) + df.date.dt.second / (24 * 60 * 60)) - 1
    df['year'] = df.date.dt.year
    X = [s[output_cols].values.T for _, s in df.groupby("ID")]

    year = [s.year.values for _, s in df.groupby("ID")]
    if max_dt > 0:
        X, year, meta = split_trajectory_chunks(X, year, meta, max_dt=max_dt)

    meta['Days in trajectory (all)'] = [temporal_extension(x) for x in X]

    meta['observations'] = [x.shape[1] for x in X]
    return X, year, meta, covariates

def temporal_extension(x):
    t = x[2] # days
    dt = np.diff(t) % 365 # assumes it is not a leap year
    return dt.sum()

def undersample_trajectories(df, year, dt_threshold=1/24):
    print(f"Undersampling trajectories. Keeping observations separated by at least {dt_threshold} days")

    is_year_series = isinstance(year, pd.Series)
    if not is_year_series:
        year = pd.Series(year, index=df.index)
    T = df.apply(lambda x: x[2])
    T_year = pd_utils.tuple_wise(T.to_frame(), year.to_frame())
    DT = T_year.applymap(lambda x: compute_dt(*x))

    @njit
    def find_indices(dt, dt_threshold):
        """
        Find the indices of the trajectory that are separated by at least dt_threshold.
        """
        dt_cumsum = np.cumsum(dt)
        idxs = [0] # always include the first index
        last_index = 0

        for i, time in enumerate(dt_cumsum):
            # Check if the time since the last recorded index exceeds dt_threshold
            if time - (dt_cumsum[last_index - 1] if last_index > 0 else 0) >= dt_threshold:
                # Adjust the index to align with the t array
                idx = i + 1
                idxs.append(idx)
                last_index = i + 1

        return np.array(idxs)

    idxs_undersampling = DT[0].apply(find_indices, dt_threshold=dt_threshold)
    df_undersampling = pd_utils.tuple_wise(df.to_frame(), idxs_undersampling.to_frame())[0]
    df_undersampled = df_undersampling.apply(lambda x: x[0][:, x[1]])
    year_undersampling = pd_utils.tuple_wise(year.to_frame(), idxs_undersampling.to_frame())[0]
    year_undersampled = year_undersampling.apply(lambda x: x[0][x[1]])

    if not is_year_series:
        year_undersampled = list(year_undersampled.values)
    return df_undersampled, year_undersampled

def load_all_data(weather=None, return_labels=True, v2=True, expand_df=False, max_dt=0,
                  minlen=50, min_animals=10, min_days=5, min_dt=0,
                  **kwargs):
    """
    If species_train is provided, the data is inverted for those species that are in the opposite hemisphere.
    expand_df: if True, the Series containing ID -> trajectory is expanded in a single DataFrame.

    pad_day_rate: number of measurements per day when interpolating for obtaining homogeneous sampling rate.
    """
    X, year, labels, covariates = load_data_v2(max_dt=max_dt, **kwargs)
    labels = labels.set_index(["COMMON_NAME", "ID"])
    df = pd.Series(X, index=labels.index)

    if min_dt > 0:
        df, year = undersample_trajectories(df, year, dt_threshold=min_dt)

    if (minlen > 0 or min_animals > 0 or min_days > 0):
        X, year, _, labels, _2 = get_prunning_function(minlen=minlen, min_animals=min_animals, min_days=min_days, label="COMMON_NAME")(list(df.values), year, labels.reset_index())
        labels = labels.set_index(["COMMON_NAME", "ID"])
        df = pd.Series(X, index=labels.index)

    if expand_df:
        df.index = df.index.droplevel(0)
        X = np.concatenate(tuple(df.values), axis=1).T
        L = df.apply(lambda x: x.shape[1])
        ID = np.repeat(df.index, L)
        columns = ['lat', 'lon', 'day'] + covariates
        df = pd.DataFrame(X, index=ID, columns=columns)

    if return_labels:
        return df, labels, year, covariates
    else:
        return df

def make_periodic(z, year, added_dt=False, to_origin=None, velocity='norm', replace_zero_by="mean", diff=False, diff_idxs=None, add_absolute_z=False, add_hour_angle=True):
    """
    Attributes:

        - z:                     (lat, lon, t) vector with shape (3, length)

        - year:                  array of year values. Used to compute the days per year and dt values.

        - added_dt:              Bool. If true, returns the vector except for the last point, where dt is undefined. Mainly thought for the equal-time case.

        - to_origin:             "time":             Shift initial time to 1 Jan.

        - velocity:              None:               Does not add velocity.
                                 "arch-segment":     Velocities in the SN-WE components.
                                 "x-y-z":            Velocities as the derivatives w.r.t. time of x, y, z.

        - replace_zero_by:       "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
        - diff:                  Bool. If True, returns the difference in each magnitude (except for the velocity)

        - diff_idxs:             List of indexes of the variables to be differentiated. If None, all variables are differentiated.

    Returns:

        periodic_x := (x, y, z, sin t, cos t, sin h, cos h, {weather vars},  {dt}, {velocity_vars})

    Considerations:
    {x, y, z} = {cos(theta) cos(phi), cos(theta) sin(phi), sin(theta)}
    theta = lat (not the polar angle)
    """
    x = z.copy()
    year_arr = np.array(year)
    is_leap = vectorized_is_leap(year_arr)
    if to_origin in ["time", "all"]:
        x[2] = (x[2] - x[2,0])
        x[2] = rescale_dt(x[2], is_leap)

    t_angle = (2*np.pi) * x[2] # maybe / 366 for leap
    t_angle[is_leap] /= 366
    t_angle[~is_leap] /= 365
    theta = (np.pi/180) * x[0]
    phi = (np.pi/180) * x[1]
    cos_theta = np.cos(theta)

    periodic_x = np.empty((6 if add_absolute_z else 5, x.shape[1]))
    periodic_x[0] = cos_theta * np.cos(phi)
    periodic_x[1] = cos_theta * np.sin(phi)
    periodic_x[2] = np.sin(theta)
    periodic_x[3] = np.sin(t_angle)
    periodic_x[4] = np.cos(t_angle)

    if add_absolute_z:
        periodic_x[7] = np.abs(periodic_x[2])

    if x.shape[0] > 3: # there are weather variables or/and dt
        periodic_x = np.vstack([periodic_x, x[3:]])

    if added_dt:
        if diff_idxs is None:
            periodic_x = np.diff(periodic_x, axis=1)
        elif len(diff_idxs) > 0:
            no_diff_idxs = np.array([i for i in np.arange(periodic_x.shape[0]) if i not in diff_idxs])
            periodic_x = np.vstack([periodic_x[no_diff_idxs, :-1],
                                    np.diff(periodic_x[diff_idxs], axis=1) if diff else periodic_x[diff_idxs, :-1]
                                   ])
        else:
            periodic_x = periodic_x[:,:-1]
    return periodic_x, year[:periodic_x.shape[1]]

def shift_to_origin(Z, x0=None):
    """
    Shifts the trajectory to the origin: (lat, lon) -> (0, 0)

    Parameters
    Z: np.array of shape (M>3, N).
       M is the number of features, N is the number of observations
       Z[:3] is assumed to be (x, y, z)
    """
    X = Z[:3].copy()
    if x0 is None:
        x0 = X[:, 0]

    # We need to calculate the rotation matrix that aligns x0 with the x-axis
    # This is a two-step rotation: First, rotate x0 around the z-axis so it lies in the xz-plane
    # Then, rotate it around the y-axis so it aligns with the x-axis

    # Step 1: Rotate around the z-axis
    theta_z = -np.arctan2(x0[1], x0[0])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    xz_plane = R_z @ x0

    # Step 2: Rotate around the y-axis to align with the x-axis
    theta_y = np.arctan2(xz_plane[2], xz_plane[0])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # Combined rotation matrix
    R = R_y @ R_z

    # Apply the rotation to the entire trajectory
    X1 = (R @ X)
    return np.vstack([X1, Z[3:]])

def trajectory_differences(x):
    """
    x: trajectory. Shape (3+k, N). 3 for (x,y,z), k for additional features, N for number of points

    Returns: Spatial differences between consecutive points. Shape (3+k, N-1)
    """
    return np.hstack((shift_to_origin(x[:, i:i+2])[:, 1:] for i in range(x.shape[1] - 1)))

def undo_periodic(periodic_z, year, has_time=False):
    """
    (x, y, z, sin t, cos t)   -->   (lat, lon, t)
    (x, y, z, sin t, cos t, v_SN, V_WE)   -->   (lat, lon, t, v_SN, V_WE)
    """
    periodic_x = periodic_z.copy()
    is_leap = get_leap_year(year)
    num_cds = 3 if has_time else 2
    x = np.empty((num_cds, periodic_x.shape[1]))
    x[0] = (180/np.pi) * np.arctan2(periodic_x[2], np.sqrt(periodic_x[0]**2 + periodic_x[1]**2))
    x[1] = (180/np.pi) * np.arctan2(periodic_x[1], periodic_x[0])

    if has_time: # Time data is present
        for partition, num_days in zip([is_leap, ~is_leap], [366, 365]):
            x[2][partition] = ((num_days/ (2*np.pi)) * np.arctan2(periodic_x[3][partition], periodic_x[4][partition])) % num_days # not adding one. ([0, 365]) undo_periodic(periodic) != identity. Maybe 365 -> 366 for leap
    x = np.vstack((x, periodic_x[(5 if has_time else 3):]))
    return x

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

def to_spherical(r, degrees=False):
    if len(r.shape) > 1:
        lat = np.arctan2(r[:,2], np.sqrt(np.square(r[:,:2]).sum(axis=1)))
        lon = np.arctan2(r[:,1], r[:,0])
    else:
        lat = math.atan2(r[2], math.sqrt(np.square(r[:2]).sum()))
        lon = math.atan2(r[1], r[0])
    if degrees:
        lat *= 180/np.pi
        lon *= 180/np.pi
    return lat, lon

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

def spherical_velocity(x, dt=None):
    """
    Returns: velocity (if dt is provided) or distance in terms of the spherical unit vectors.
    Attributes:
        - x:  array containing latitude and longitude in the first 2 rows.
        - dt: array of time increments between points.
    """
    d_cartesian = log_map_vec(to_cartesian(*x[:2]))
    A = conversion_matrix_vec(*x[:2])
    d_spherical = np.array([a.dot(d) for a, d in zip(A, d_cartesian)])
    if dt is None:
        return d_spherical
    else:
        v = d_spherical / dt[:, None]
        return v

def _get_hour_angle(time_index, lat, lon):
    """
    Returns the hour angle in radians
    """
    solar_position = solarposition.get_solarposition(time_index, lat, lon, method='nrel_numpy')
    hour_angle = solarposition.hour_angle(time_index, lon, solar_position.equation_of_time.values)
    return hour_angle * np.pi / 180

def hour_angle_from_trajectory(x, year, is_leap=None):
    """
    Returns the hour angle in radians
    """
    if is_leap is None:
        is_leap = vectorized_is_leap(year)
    lat, lon, day = x[:3]
    day %= (365 + is_leap.astype(int))
    T0 = pd.Timestamp(year=year[0], month=1, day=1, tz='UTC')
    time_index = T0 + pd.TimedeltaIndex(day, unit='D')
    return _get_hour_angle(time_index, lat, lon)

def average_periodic(angles, period=2*np.pi):
    """
    Compute the average of periodic variables considering the discontinuity.

    Parameters:
    angles (numpy array): The array of periodic variable.
    period (float): The period of the variable. Default is 2*pi for radians.

    Returns:
    float: The average of the periodic variable.
    """
    # Convert angles to unit circle representation
    if math.isclose(period, 2*np.pi):
        x = np.cos(angles)
        y = np.sin(angles)
        avg_angle = np.arctan2(y.mean(), x.mean())
    else:
        x = np.cos(angles * 2 * np.pi / period)
        y = np.sin(angles * 2 * np.pi / period)
        avg_angle = np.arctan2(y.mean(), x.mean())
        avg_angle *= period / (2 * np.pi) # rescale
    return avg_angle

@savedata
def get_dt(**kwargs):
    df, _, Year = load_all_data(**kwargs)
    T = df.apply(lambda x: x[2]).to_frame(name='time')
    T['Year'] = Year
    dt = T.apply(lambda x: compute_dt(x['time'], x['Year']), axis=1)
    dt.index = dt.index.droplevel(0)
    return dt
