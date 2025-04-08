import numpy as np
import pandas as pd
import math
import torch
from numba import njit
from scipy import stats as ss
from collections import defaultdict
try:
    from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer, TorchNormalizer, NaNLabelEncoder
    from pytorch_forecasting import TimeSeriesDataSet
except:
    print("pytorch-forecasting not installed")
from sklearn.model_selection import train_test_split
import warnings
from pathlib import Path
import os as _os
import sys as _sys
_sys.stdout.flush()
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)
RootDir = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(RootDir)
_sys.path.append(RootDir)

from phdu import savedata, bootstrap, pd_utils
import phdu.stats.corr as phdu_corr
from tidypath import storage
import utils as clf_utils
from utils import params as clf_params
from . import space
from ..dataset import ImputationDataset, ForecastingDataset, test_val_start

@savedata
def reference_point(weather='all', species='Southern elephant seal'):
    df, *_ = clf_utils.preprocessing.load_all_data(v2=True, weather=weather, return_labels=True,
                                                             minlen=5, min_days=0, min_animals=0, min_dt=0)
    df = df.loc[species]

    LAT, LON = np.hstack(df.values)[:2]
    LAT *= np.pi / 180
    LON *= np.pi / 180
    r0 = np.array([LAT.mean(), LON.mean()])
    return r0

def load_data(weather='all', species='Southern elephant seal', sampling_freq=6, cds='mercator',
              velocity='norm', add_hour_angle=True, add_dt=True, delta=10,
              meta_filename='metadata.csv', data_filename='dataset.csv',
              static_features=[],
              animotum=False,
              ):
    """
    Load and preprocess animal mobility data.

    Parameters:
    weather (str, optional): Type of weather data to include. Default is 'all'.
    species (str, optional): Species of animal to load data for. Default is 'Southern elephant seal'.
    sampling_freq (int, optional): Frequency in hours at which an animal's location is recorded. Default is 6.
    cds (str, optional): Coordinate system to use. 'mercator' for Mercator projection, 'spherical' for spherical coordinates. Default is 'mercator'.
    velocity (str, optional): Type of velocity calculation to use. 'norm' for absolute value of speed, 'arch-segment' for arch-segment velocity. Default is 'norm'.
    add_hour_angle (bool, optional): Whether to add hour angle to the data. Default is True.
    add_dt (bool, optional): Whether to add time difference between consecutive data points. Default is True.
    delta (int, optional): Number of previous data points to consider for mean calculation. Default is 10.

    Returns:
    data (DataFrame): Preprocessed data.
    features (list): List of feature names.
    center (np.array): Center coordinates.
    cds_cols (list): List of coordinate system column names.
    """
    args = locals()
    def compute_result():
        df, labels, year, covariate_names = clf_utils.preprocessing.load_all_data(v2=True, weather=weather, return_labels=True,
                                                                                  minlen=5, min_days=0, min_animals=0, min_dt=0,
                                                                                  meta_filename=meta_filename, data_filename=data_filename)
        year = pd.Series(year, index=df.index)
        df = df.loc[species]
        labels = labels.loc[species]
        year = year.loc[species]

        YEAR = np.hstack(year.values)
        LAT, LON = np.hstack(df.values)[:2]
        LAT *= np.pi / 180
        LON *= np.pi / 180
        if animotum:
            cds_data = dict(LAT=LAT, LON=LON)
            dt_data = dict()
            velocity_data = dict()
            H_angle_data = dict()
            cds_predict = dict()
        else:
            cds_predict = {}
            cds_cols = []
            if cds == 'spherical': # SN and WE projections in the tangent plane.
                center = np.array([LAT.mean(), LON.mean()])
                center = center[:, None]

                df_distances = df.apply(lambda x: space.spherical_velocity(np.hstack((center, x[:2]*np.pi/180)))[:, -2:].cumsum(axis=0))

                SN, WE = np.vstack(df_distances.values).T
                cds_data = dict(SN=SN, WE=WE)
            elif cds == 'spherical-ref-point': # SN and WE projections in the tangent plane with respect to the reference point.
                center = reference_point(weather=weather, species=species)
                df_v = df.apply(lambda x: space.log_map_vec_fixed_point(center, x[:2]*np.pi/180)[-2:])
                SN, WE = np.hstack(df_v.values)
                cds_data = dict(SN=SN, WE=WE)
            elif cds == 'spherical-dx': # prediction = increments in latitude and longitude for the trajectory shifted to (lat.lon)=(0,0).
                df_year = pd_utils.tuple_wise(df, year)
                XYZ = df_year.apply(lambda x: clf_utils.preprocessing.make_periodic(*x, velocity=None, add_hour_angle=False))
                x0 = np.array([1,0,0])[:, None]
                def get_spherical_dx(x):
                    x, year = x
                    dxyz = np.hstack((x0, clf_utils.preprocessing.trajectory_differences(x)[:3]))
                    dx = clf_utils.preprocessing.undo_periodic(dxyz, year, has_time=False) # differences in lat lon
                    return dx
                DX = XYZ.apply(get_spherical_dx)
                DLAT, DLON = np.hstack(DX.values)
                cds_predict = dict(DLAT=DLAT, DLON=DLON)
                cds_data = dict(LAT=LAT, LON=LON)
                center = np.zeros((2))
                cds_cols = ['x', 'y', 'z']
            elif cds == 'mercator':
                X, Y = space.mercator(LAT, LON)
                center = np.array([X.mean(), Y.mean()])
                cds_data = dict(X=X, Y=Y)
            else:
                raise ValueError("Invalid coordinate system.")
            if not cds_cols:
                cds_cols = [*cds_data.keys()]
            cds_predict_cols = [*cds_predict.keys()]
            print("Coordinates: {}".format(cds_cols))

            def _fill_with_mean(x):
                if x.ndim == 1:
                    x = np.atleast_2d(x)
                if delta is None:
                    return np.hstack((x, x.mean(axis=1)[:, None])).squeeze()
                else:
                    return np.hstack((x, x[:, -delta:].mean(axis=-1)[:, None])).squeeze()

            if add_dt or velocity is not None:
                df_year = pd_utils.tuple_wise(df, year)
                DT = df_year.apply(lambda x: clf_utils.preprocessing.compute_dt(x[0][2], x[1]))
            if add_dt:
                print("Adding dt")
                dt_data = dict(dt=np.hstack(DT.apply(_fill_with_mean).values))
            else:
                dt_data = dict()

            if velocity is not None:
                print(f"Adding velocity: '{velocity}'")
                df_dt = pd_utils.tuple_wise(df, DT)
                if velocity == 'norm':
                    V = df_dt.apply(lambda x: clf_utils.preprocessing.nb_funcs.great_circle_distance_by_time_step((np.pi/180) * x[0][:2]) / x[1])
                    velocity_data = dict(V=np.hstack(V.apply(_fill_with_mean).values))
                elif velocity == 'arch-segment':
                    V = df_dt.apply(lambda x: clf_utils.preprocessing.spherical_velocity((np.pi/180) * x[0][:2], x[1]).T[-2:])
                    V = V.apply(_fill_with_mean)
                    V_SN = np.hstack(V.apply(lambda x: x[0]).values)
                    V_WE = np.hstack(V.apply(lambda x: x[1]).values)
                    velocity_data = dict(V_SN=V_SN, V_WE=V_WE)
                else:
                    raise ValueError(f"Invalid velocity type: {velocity}")
            else:
                velocity_data = dict()

            if add_hour_angle:
                print("Adding hour angle")
                Is_leap = year.apply(clf_utils.preprocessing.vectorized_is_leap)
                df_year_is_leap = pd_utils.tuple_wise(df, year, Is_leap)
                H_angle = df_year_is_leap.apply(lambda x: clf_utils.preprocessing.hour_angle_from_trajectory(x[0], x[1], is_leap=x[2]))
                sin_H = np.hstack(H_angle.apply(lambda x: np.sin(x)).values)
                cos_H = np.hstack(H_angle.apply(lambda x: np.cos(x)).values)
                H_angle_data = dict(sin_H=sin_H, cos_H=cos_H)
            else:
                H_angle_data = dict()

        T = np.hstack(df.apply(lambda x: x[2]).values)
        covariates = np.hstack(df.apply(lambda x: x[3:]).values)
        ID = np.hstack(df.groupby(df.index).apply(lambda S: [S.index[0]] * S.iloc[0].shape[1]).values)
        covariates_data = {feature: val for feature, val in zip(covariate_names, covariates)}
        df_melt = pd.DataFrame({**dict(T=T, YEAR=YEAR, ID=ID),
                                **cds_data,
                                **cds_predict,
                                **covariates_data,
                                **dt_data,
                                **velocity_data,
                                **H_angle_data})

        T_intervals = pd.cut(df_melt["T"], bins=365*int(24/sampling_freq))
        T_idx = T_intervals.cat.codes
        T_idx_max = 365*int(24/sampling_freq)
        T_idx_year_adjusted = T_idx + (df_melt.YEAR - df_melt.YEAR.min()) * T_idx_max
        T_idx_year_adjusted -= T_idx_year_adjusted.min()
        T_idx_year_adjusted = T_idx_year_adjusted.astype(np.int32)
        df_melt["T bins"] = T_intervals
        df_melt["T idx"] = T_idx_year_adjusted

        if animotum:
            print("Using full data for animotum (not averages over time bins). Bining will be used during prediction to allow for comparison with TFT")
            data = df_melt
            data.LAT *= 180/np.pi
            data.LON *= 180/np.pi
            time_delta = data['T'] * pd.Timedelta(1, 'D')
            jan_1 = data['YEAR'].astype(int).apply(lambda y: pd.Timestamp(f"{y}-01-01"))
            data['date'] = jan_1 + time_delta
            data = data.rename(columns={'LAT': 'lat',
                                        'LON': 'lon',
                                        'T idx': 'time_idx'})
            data = data[['ID', 'lat', 'lon', 'date', 'time_idx']]
            if 'lc' in labels.columns:
                data['lc'] = labels.loc[data.ID.values, 'lc'].values
            return data
        else:
            # take the mean values for each bin
            print("Averaging over time bins")
            if 'LON' in cds_data:
                print("Taking into account periodicity for longitude")
                order = [c for c in df_melt.columns if c not in ['ID', 'T idx', 'T bins']]
                def average_over_time_bins(x):
                    lon = x['LON']
                    if lon.max() > 2 and lon.min() < -2:
                        out = x.drop(columns=['LON']).mean()
                        out['LON'] = clf_utils.preprocessing.average_periodic(lon)
                    else:
                        out = x.mean()
                    return out[order]
                data = df_melt.groupby(["ID", "T idx"]).apply(average_over_time_bins).dropna().reset_index()
                X, Y, Z = clf_utils.preprocessing.to_cartesian(data.LAT.values, data.LON.values).T
                data = data.drop(columns=['LAT', 'LON'])
                # insert cds cols after YEAR
                data = pd.concat([data.loc[:, :'YEAR'], pd.DataFrame(dict(x=X, y=Y, z=Z)), data.loc[:, 'DLAT':]], axis=1)
            else:
                # data = df_melt.groupby(["ID", "T idx"]).mean().dropna().reset_index()
                data = df_melt.drop(columns=["T bins"]).groupby(["ID", "T idx"]).mean().dropna().reset_index()

            data['T idx'] = data['T idx'].astype(np.int32)
            extra_features = [*dt_data.keys(), *velocity_data.keys(), *H_angle_data.keys()]
            data.columns = (['ID', 'time_idx'] + ['t', 'YEAR']
                            + cds_cols + cds_predict_cols
                            + covariate_names
                            + extra_features)
            features = cds_cols + covariate_names + extra_features
            if cds_predict:
                features_predict = cds_predict_cols
            else:
                features_predict = cds_cols

            # Categoricals (time varying)
            data['month'] = pd.Categorical((data.t / 31).astype(int).astype(str))
            remap = {11: 'winter', 0: 'winter', 1: 'winter',
                     2: 'spring', 3: 'spring', 4: 'spring',
                     5: 'summer', 6: 'summer', 7: 'summer',
                     8: 'autumn', 9: 'autumn', 10: 'autumn'}
            data['season'] = pd.Categorical(data.month.astype(int).map(remap))
            # data['day'] = pd.Categorical(data.t.astype(int).astype(str))

            # Numericals known (time)
            data['cos t'] = np.cos(2 * np.pi * data.t / 366) # 366 for leap years. The difference for normal years is negligible.
            data['sin t'] = np.sin(2 * np.pi * data.t / 366)

            if static_features:
                labels_data = labels.loc[data.ID]
                for f in static_features:
                    data[f] = labels_data[f].values
            return data, features, features_predict, center

    if cds == 'spherical-dx':
        # store output only if cds is spherical-dx (takes ~10min to compute)
        @savedata
        def _load_data(**kwargs):
            return compute_result()
        return _load_data(**args)
    else:
        return compute_result()

def preprocess_animotum(species="Southern elephant seal", sampling_freq=6, split=[80, 10, 10], **kwargs):
    data = load_data(species=species, sampling_freq=sampling_freq, **kwargs,
                     animotum=True)
    # drop duplicates of time_idx for each ID
    t_idx = np.hstack(data.groupby('ID').apply(lambda x: x['time_idx'].unique()).values)

    split_endpoints = np.cumsum(split)
    training_cutoff, val_cutoff, _ = np.percentile(t_idx, split_endpoints).astype(np.int32) # test_cutoff = T_idx.max()

    data_train = data[lambda x: x.time_idx <= training_cutoff]
    data_val = data[lambda x: (x.time_idx > training_cutoff) & (x.time_idx <= val_cutoff)]
    data_test = data[lambda x: x.time_idx > val_cutoff]

    savedir = _os.path.join(RootDir, f"data/forecasting_models/{species}")
    Path(savedir).mkdir(parents=True, exist_ok=True)
    data_train.to_csv(_os.path.join(savedir, 'train.csv'), index=False)
    data_val.to_csv(_os.path.join(savedir, 'val.csv'), index=False)
    data_test.to_csv(_os.path.join(savedir, 'test.csv'), index=False)

    return data_train, data_val, data_test


def subsample_data(df, delete_prob=0.8, seed=0):
    """
    This function subsamples a pandas DataFrame based on a given deletion probability.
    It groups the DataFrame by 'ID', calculates the mode of 'YEAR' for each group,
    and then performs a stratified train-test split on the IDs based on the calculated mode.
    The function then selects the rows of the DataFrame that correspond to the remaining IDs
    after the split.

    Parameters:
    df (pandas.DataFrame): The DataFrame to subsample.
    delete_prob (float, optional): The proportion of data to delete. Default is 0.8.
    seed (int, optional): The seed for the random number generator. Default is 0.

    Returns:
    pandas.DataFrame: The subsampled DataFrame.
    """
    IDs = []
    year = []
    for ID, df_ID in df.groupby('ID'):
        IDs.append(ID)
        # append most frecuent year (mode)
        year.append(int(df_ID['YEAR'].mode().values[0]))
    IDs = np.array(IDs)
    year = np.array(year)
    try:
        idxs_remain, _ = train_test_split(np.arange(len(IDs)), test_size=delete_prob, stratify=year, random_state=seed)
    except ValueError: # number of test samples is less than the number of classes
        success = False
        while not success:
            # delete IDs associated to the oldest year
            oldest_year = year.min()
            valid = year > oldest_year
            if valid.any():
                year = year[valid]
                IDs = IDs[valid]
                try:
                    idxs_remain, _ = train_test_split(np.arange(len(IDs)), test_size=delete_prob, stratify=year, random_state=seed)
                    success = True
                except ValueError:
                    continue
            else:
                # split without stratifying
                idxs_remain, _ = train_test_split(np.arange(len(IDs)), test_size=0.65, random_state=seed) # n=5
                success = True
    idxs_remain = np.sort(idxs_remain)
    idxs_remain = IDs[idxs_remain]
    df = df.query('ID in @idxs_remain').reset_index(drop=True)
    return df

@njit
def rolling_rho(x, semiwidth):
    """
    Rolling correlation function.
    """
    n = x.shape[0]
    if n < 2*semiwidth:
        return np.zeros((n))
    else:
        out = np.empty((n))
        for i in range(semiwidth, n-semiwidth):
            z = x[i-semiwidth:i+semiwidth+1]
            out[i] = np.corrcoef(z.T)[0, 1]
        out[:semiwidth] = out[semiwidth]
        out[-semiwidth:] = out[-semiwidth-1]
        return out

def load_dataset(data=None, features=None, features_target=None, center=None,
                 sampling_freq=6, max_train_days=28, max_pred_days=7,
                 static_categoricals=['Sex'],
                 static_reals=['Weight', 'Length'],
                 known_reals=['sin t', 'cos t'],
                 known_categoricals=['month', 'season'],
                 split = None,
                 group_ids=['ID'],
                 add_target_scales=False,
                 add_relative_time_idx=True,
                 ID=None,
                 criteria=None,
                 min_obs_test = 1,
                 min_obs_val = 1,
                 min_obs_train = 5,
                 randomize_length=True,
                 task = 'forecasting',
                 eval_mode = None,
                 fix_training_prediction_length = False,
                 max_future_gap = 1,
                 max_decoder_gap = 0,
                 min_future_obs = None,
                 expand_encoder_until_future_length = True,
                 store_missing_idxs = False,
                 predict_shift = None,
                 reverse_future = False,
                 verbose = 1,
                 delete_prob = None,
                 delete_seed = 0,
                 subsample_partition = 'train',
                 target = 'cds',
                 rho_days_semi_window = 4,
                 **load_kwargs):
    """
    Attributes:
    sampling_freq: Sampling frequency in hours. This is the frequency at which an animal's location is recorded.
    max_train_days: Maximum number of days the model uses to make predictions.
    max_pred_days: Maximum number of days the model predicts.
    split: List of floats that sum to 100. The first element is the fraction of the data used for training, the second is the fraction used for validation, and the third is the fraction used for testing.
           The split is done by time, so the model is evaluated in the most recent data.
    group_ids: List of strings that are associated to each separate time series ('ID' in our dataset). It would let the model know with sub-sequences belong to the same animal.
               However, it will not be able to use this information for the prediction of a new animal.
               Since the test set includes different animals due to the temporal split, it may not make sense to use this information.
    static_categoricals: Categorical attributes that do not change with time.
    static_reals: Numeric attributes that do not change with time. #TODO: ask an expert if it is okay to consider the weight and length as static variables.
    known_categoricals: == time_varing_known_categoricals: Categorical attributes that change with time and are known.
                Month can be thought as a number, but since it is cylical it is better to treat it as a categorical variable.
    add_target_scales: Whether to add scales for the target. This is useful if the target has a large range and scales are important for training.
    add_relative_time_idx: Whether to add a relative time index as feature. This is the number of time steps since the last observed time step.
    ID: Integer or string that specifies the trajectory to use. If None, all trajectories are used.
    criteria: Criteria to use to filter out trajectories. 'days' for the number of days, 'observations' for the number of observations, 'observations-by-set' for the number of observations in each set.
    min_obs_test: Minimum number of observations for the test set.
    min_obs_val: Minimum number of observations for the validation set.
    min_obs_train: Minimum number of observations for the training set.
    randomize_length: Whether to randomize the length of the sequences (training only).
    task: 'forecasting' or 'imputation'.
    eval_mode: None, 'last', 'non-overlapping', 'encoder-overlap'. 'last' evaluates the model on the last max_pred_days days of the time series. 'non-overlapping' evaluates the model on non-overlapping sequences of max_pred_days days. 'encoder-overlap' evaluates the model on overlapping sequences of max_pred_days days.
    fix_training_prediction_length: Whether to fix the prediction length for training.
    max_future_gap: Maximum gap between the imputation window and the last observed past location.
    max_decoder_gap: Maximum gap between the imputation window and the first observed future location.
    min_future_obs: Minimum number of future observations for the imputation task (min encoder length for the future part)
    expand_encoder_until_future_length: Whether to expand the encoder until the future length.
    store_missing_idxs: Whether to store the missing indices.
    """
    assert all(isinstance(param, list) for param in [static_categoricals, static_reals, known_reals, known_categoricals, group_ids]), "The parameters must be lists."

    static_features = static_categoricals + static_reals
    if data is None:
        data, features, features_target, center = load_data(sampling_freq=sampling_freq, static_features=static_features, **load_kwargs)
    else:
        data = data.copy()

    if static_features:
        has_nans = data[static_features].isna().values.any()
        if has_nans:
            warnings.warn("NaNs found in static features. Filling with the average values.")
            data[static_features] = data[static_features].fillna(data[static_features].mean())

    if target == 'rho':
        print("Setting target to 'rho'")
        semiwidth = rho_days_semi_window * 24 // sampling_freq
        rho_args = ['X', 'Y']
        rho = data.groupby('ID').apply(lambda x: rolling_rho(x[rho_args].values, semiwidth))
        rho = np.hstack(rho.values)
        data['rho'] = rho
        features_target = 'rho'
        # target_normalizer = TorchNormalizer(method='identity', center=False, transformation=dict(forward=torch.nn.Tanh(), reverse=torch.atanh)) # rho is between -1 and 1
        # target_normalizer = None # identity
        # target_normalizer = EncoderNormalizer(transformation=dict(forward=lambda x: x, reverse=torch.nn.Tanh()))
        # target_normalizer = TorchNormalizer(transformation=dict(forward=torch.nn.Identity(), reverse=torch.nn.Tanh()))
        target_normalizer = EncoderNormalizer()
        # Do not use rho as a feature, since it contains information about the future (it is calculated using a centered rolling window).
    elif target == 'cds':
        target_normalizer = MultiNormalizer([EncoderNormalizer(), EncoderNormalizer()])
    else:
        raise NotImplementedError(f"Target {target} not implemented. Choose from 'rho' or 'cds'.")


    max_encoder_length = max_train_days * 24 // sampling_freq
    max_prediction_length = max_pred_days * 24 // sampling_freq
    min_prediction_length = 1
    if split is None:
        split = [80, 10, 10]

    assert math.isclose(sum(split), 100), "The elements of split must sum to 100."
    split_endpoints = np.cumsum(split)
    training_cutoff, val_cutoff, _ = np.percentile(data['time_idx'], split_endpoints).astype(np.int32) # test_cutoff = T_idx.max()

    if ID is not None: # single trajectory for train, test, val
        data_test = data[lambda x: x.time_idx > val_cutoff]
        dataset_splits = None
        obs_per_day = 24 / sampling_freq
        if task == 'forecasting':
            min_days = 2 * max_pred_days + max_train_days # at least 2*max_pred_days (val+test) and max_train_days for training and maximize input length for val/test.
            if criteria is None:
                criteria = 'observations-by-set'
        else:
            # train: 2*max_train_days + max_pred_days
            # val: overlaps max_train_days + max_pred_days
            # test: overlaps max_train_days + max_pred_days + max_train_days in the future
            min_days = 3*max_train_days + 3*max_pred_days
            if criteria is None:
                criteria = 'fixed-decoder-length'
        if criteria == 'observations-by-set':
            min_obs = pd.Series(dict(train=min_obs_train, val=min_obs_val, test=min_obs_test))
            null_out = pd.Series(dict(training_cutoff=np.nan, val_cutoff=np.nan, days=np.nan))
            imputation_max_length = 2*max_encoder_length + max_prediction_length
            def prune_by_obs(S):
                t = S.time_idx.values
                t -= t[0]
                if task == 'imputation':
                    test_start = t[(t - (t[-1] - imputation_max_length)) >= 0][0] + max_encoder_length - 1
                    t_pruned = t[t <= t[-1] - max_prediction_length]
                    if t_pruned.size == 0:
                        val_start = -1
                    else:
                        val_start = t_pruned[(t_pruned - (t_pruned[-1] - imputation_max_length - 1)) >= 0][0] + max_encoder_length - 1
                else:
                    test_start = t[-1] - max_pred_days*obs_per_day - 1
                    val_start = test_start - max_pred_days*obs_per_day - 1
                if val_start < 0 or test_start == val_start:
                    return null_out
                else:
                    num_obs = pd.cut(t, bins=[-1, val_start, test_start, t[-1]], labels=['train', 'val', 'test']).value_counts()
                    valid = (num_obs >= min_obs).all()
                    if valid:
                        return pd.Series(dict(training_cutoff=val_start, val_cutoff=test_start, days=t[-1] / obs_per_day))
                    else:
                        return null_out
            dataset_splits = data_test.groupby('ID').apply(prune_by_obs)
            dataset_splits = dataset_splits[dataset_splits.days >= min_days]
            dataset_splits = dataset_splits.drop(columns='days')
            dataset_splits = dataset_splits.dropna().astype(int)
            valid_IDs = dataset_splits.index.values
        elif criteria == 'fixed-decoder-length':
            dataset_splits = data_test.groupby('ID').apply(test_val_start, max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length)
            dataset_splits = pd.DataFrame(np.vstack(dataset_splits.values),
                         index=dataset_splits.index,
                         columns=['train_end', 'val_start', 'val_end', 'test_start'])
            dataset_splits = dataset_splits.dropna().astype(int)
            valid_IDs = dataset_splits.index.values
        else:
            if criteria == 'days':
                def total_day_span(S):
                    return S.time_idx.iloc[[0, -1]].diff().iloc[-1] / obs_per_day
                tracking_meta = data_test.groupby('ID').apply(total_day_span).sort_values()
            elif criteria == 'observations':
                tracking_meta = data_test.value_counts('ID') / obs_per_day
            else:
                raise ValueError(f"criteria {criteria} not recognized. Choose from 'days', 'observations' or 'observations-by-set'.")
            valid_IDs = tracking_meta[lambda x: x >= min_days].index.values

        if isinstance(ID, int):
            ID = valid_IDs[ID]
        elif isinstance(ID, str):
            assert ID in valid_IDs, "ID not found in valid_IDs."
        else:
            raise ValueError("ID should be an integer or a string.")

        if verbose:
            print(f"Using single trajectory with ID: {ID}")

        data = data_test.query(f"ID == '{ID}'")
        data['time_idx'] -= data['time_idx'].iloc[0]
        if dataset_splits is None:
            training_cutoff, val_cutoff, _ = np.percentile(data['time_idx'], split_endpoints).astype(np.int32)
        else:
            if criteria == 'fixed-decoder-length':
                training_cutoff, val_start, val_end, test_start = dataset_splits.loc[ID].values
            else:
                training_cutoff, val_cutoff = dataset_splits.loc[ID].values
        if task == 'forecasting':
            data_val = data[lambda x: (x.time_idx > training_cutoff - max_encoder_length) & (x.time_idx <= val_cutoff)] # overlap with training only in the input
            data_test = data[lambda x: x.time_idx > val_cutoff - max_encoder_length] # overlap with val (and possibly train) only in the input
        else:
            if criteria == 'fixed-decoder-length':
                data_val = data[lambda x: (x.time_idx >= val_start) & (x.time_idx <= val_end)]
                data_test = data[lambda x: x.time_idx >= test_start]
            else:
                data_val = data[lambda x: (x.time_idx > training_cutoff - max_encoder_length) & (x.time_idx <= val_cutoff + max_encoder_length)] # overlap with training only in the input. overlap with test but not in the prediction window (future of val = past of test)
                data_test = data[lambda x: x.time_idx > val_cutoff - max_encoder_length] # overlap with val (and possibly train) only in the input
    else:
        data_val = data[lambda x: (x.time_idx > training_cutoff) & (x.time_idx <= val_cutoff)]
        data_test = data[lambda x: x.time_idx > val_cutoff]

    data_train = data[lambda x: x.time_idx <= training_cutoff]

    if delete_prob is not None:
        assert ID is None, "ID should be None when using delete_prob."
        if subsample_partition == 'train':
            data_train = subsample_data(data_train, delete_prob=delete_prob, seed=delete_seed)
        elif subsample_partition == 'all':
            data_train = subsample_data(data_train, delete_prob=delete_prob, seed=delete_seed)
            data_val = subsample_data(data_val, delete_prob=delete_prob, seed=delete_seed)
            data_test = subsample_data(data_test, delete_prob=delete_prob, seed=delete_seed)
        else:
            raise ValueError(f"Invalid subsample_partition: {subsample_partition}")

    for partition, data_partition in dict(train=data_train, val=data_val, test=data_test).items():
        N_ID = data_partition['ID'].nunique()
        print(f"{partition} set: {len(data_partition)} observations, {N_ID} animals")

    if task == 'forecasting':
        dataset_kwargs = dict(store_missing_idxs=store_missing_idxs)
        dataset_predict_kwargs = {}
        DatasetGenerator = ForecastingDataset
        if eval_mode is None:
            predict_mode = 'default'
        elif eval_mode == 'last':
            predict_mode = 'original'
        elif isinstance(eval_mode, str):
            predict_mode = eval_mode
        else:
            raise ValueError("eval_mode should be None, 'last', 'non-overlapping' or 'encoder-overlap'.")
        if verbose:
            print(f"Forecasting task: predict_mode={predict_mode}")
    elif task == 'imputation':
        dataset_kwargs = dict(fix_training_prediction_length=fix_training_prediction_length, expand_encoder_until_future_length=expand_encoder_until_future_length,
                              store_missing_idxs=store_missing_idxs, reverse_future=reverse_future,
                              max_future_gap=max_future_gap, max_decoder_gap=max_decoder_gap)
        if min_future_obs is None and ID is not None:
            min_future_obs = 1
        dataset_predict_kwargs = dict(min_future_obs=min_future_obs)
        DatasetGenerator = ImputationDataset
        if eval_mode is None:
            predict_mode = 'non-overlapping'
        else:
            predict_mode = eval_mode
        if verbose:
            print(f"Imputation task: predict_mode={predict_mode}")
            if store_missing_idxs:
                print("Setting zero attention to encoder and future missing values.")
    else:
        raise ValueError(f"Task {task} not recognized. Choose from 'forecasting' or 'imputation'.")

    categorical_encoders = {k: NaNLabelEncoder(add_nan=True) for k in static_categoricals + known_categoricals + group_ids}

    training = DatasetGenerator(
        data_train,
        time_idx="time_idx",
        target=features_target,
        group_ids=group_ids,
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals, # ["time_idx"], #["time_idx", "price_regular", "discount_in_percent"],
        #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=features,
        target_normalizer=target_normalizer,
        categorical_encoders=categorical_encoders,
        #target_normalizer=GroupNormalizer(
        #    groups=["ID"], transformation="softplus"
        #),  # use softplus and normalize by group
        add_relative_time_idx=add_relative_time_idx,
        add_target_scales=add_target_scales,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        randomize_length=randomize_length,
        # scalers=StandardScaler() by default
        **dataset_kwargs
    )

    # group_ids_hidden = f'__group_id__{group_ids[0]}'
    # if group_ids[0] in training.categorical_encoders:
    #     training.categorical_encoders[group_ids[0]].add_nan = True
    # else:
    #     training.categorical_encoders[group_ids_hidden].add_nan = True

    # for feature in known_categoricals + static_categoricals: # group_ids to nan because animals in the val and test sets are not in the training set.
    #     training.categorical_encoders[feature].add_nan = True


    dataset_predict_kwargs.update(predict=predict_mode, stop_randomization=True, predict_shift=predict_shift)

    training_predict = DatasetGenerator.from_dataset(training,
                                                     data_train,
                                                     **dataset_predict_kwargs)

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series.
    validation = DatasetGenerator.from_dataset(training,
                                               data_val,
                                               **dataset_predict_kwargs)

    test = DatasetGenerator.from_dataset(training,
                                         data_test,
                                         **dataset_predict_kwargs)
    if verbose:
        print(f"Data loaded for max_train_days={max_train_days}:\nTraining set: {len(training)}, Validation set: {len(validation)}, Test set: {len(test)}")

    return training, validation, test, training_predict, center

def dataloaders(batch_size=128, val_batch_mpl=10, **kwargs):
    training, validation, test, training_predict, _ = load_dataset(**kwargs)
    # create dataloaders for model
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=0)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=0)
    training_predict_dataloader = training_predict.to_dataloader(train=False, batch_size=batch_size * val_batch_mpl, num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader, training_predict_dataloader

def _format(value, CI, ndigits=1):
    value_str = str(round(value, ndigits))
    CI_str = str(CI.squeeze().round(ndigits).tolist())
    return f"{value_str} ({CI_str})"

def _get_dataset_specs(df, dt, name=None):
    num_trajectories = df.shape[0]
    observations = df.apply(lambda x: x.shape[1]).values
    @njit
    def nb_mean(x):
        return x.mean()

    observations_mean = observations.mean()
    CI_observations = bootstrap.CI_bca(observations.astype(float), nb_mean, R=int(1e5))
    CI_observations = np.round(CI_observations).astype(int)

    days_in_trajectory = dt.apply(np.sum).values
    days_mean = days_in_trajectory.mean()
    CI_days = bootstrap.CI_bca(days_in_trajectory.astype(float), nb_mean, R=int(1e5))
    CI_days = np.round(CI_days).astype(int)

    sampling_step = dt.apply(np.mean).values
    # To hours
    sampling_step *= 24
    sampling_step_mean = sampling_step.mean()
    CI_sampling_step = bootstrap.CI_bca(sampling_step, nb_mean, R=int(1e5))

    S = pd.Series({'Trajectories': str(num_trajectories),
                   'Observations': _format(observations_mean, CI_observations),
                   'Days': _format(days_mean, CI_days),
                   'Sampling period (h)': _format(sampling_step_mean, CI_sampling_step)})
    if name is not None:
        S.name = name
    return S

@savedata
def dataset_raw_stats(species='Southern elephant seal'):
    df = clf_utils.preprocessing.load_all_data(v2=True, return_labels=False)
    df = df.loc[species]
    dt = clf_utils.preprocessing.get_dt(v2=True).loc[df.index]

    return _get_dataset_specs(df, dt, name='Original')

@savedata
def dataset_binned_stats(species='Southern elephant seal', sampling_freq=6, **kwargs):
    data, *_ = load_data(species=species, sampling_freq=sampling_freq, **kwargs)
    df = data.groupby("ID").apply(lambda x: np.vstack(x[['t', 'YEAR']].values).T)
    dt = df.apply(lambda x: clf_utils.preprocessing.compute_dt(x[0], x[1]))

    return _get_dataset_specs(df, dt, name=f'Binned (dt={sampling_freq}h)')

def dataset_stat_comparison(species='Southern elephant seal', sampling_freq=6):
    df_raw = dataset_raw_stats(species=species)
    df_binned = dataset_binned_stats(species=species, sampling_freq=sampling_freq)
    df = pd.concat([df_raw, df_binned], axis=1)
    return pd_utils.latex_table(df, index=True)

def feature_corr(ns_to_nan=True, method='spearman', weather='all-depth', alpha=0.05, **loading_kwargs):
    """
    Computes the correlation between the features.
    """
    @savedata
    def _feature_corr(method='spearman', alpha=0.05, **loading_kwargs):
        data, features, *_ = load_data(**loading_kwargs)
        df = data[features]
        c, p = phdu_corr.corr_pruned(df, method=method, alpha=alpha, ns_to_nan=False)
        return c, p
    c, p = _feature_corr(method=method, alpha=alpha, **loading_kwargs, weather=weather)
    if ns_to_nan:
        c[p > alpha] = np.nan
    return c, p

@savedata
def avg_features(task='forecasting', partition='test', batch_size=128, **kwargs):
    """
    Returns dataframe with the following feature properties:
    - Average and standard deviation of numerical features.
    - Correlation between numerical features.
    - Most frequent value of categorical features.
    """
    _, validation, test, training_predict, _2 = load_dataset(task=task, **kwargs)
    if partition == 'train':
        data = training_predict
    elif partition == 'val':
        data = validation
    elif partition == 'test':
        data = test
    else:
        raise ValueError("Invalid partition")
    dataloader = data.to_dataloader(batch_size=batch_size*10000, train=False, num_workers=0)
    x = next(iter(dataloader))[0]
    if task == 'forecasting':
        numerical_features = data.reals
        categorical_features = data.categoricals
    elif task == 'imputation':
        numerical_features = data.reals_with_future
        categorical_features = data.categoricals_with_future
    else:
        raise ValueError("Invalid task")

    # numeric features
    numerical_data = x['encoder_cont'].numpy()
    avg_features = numerical_data.mean(axis=1)
    avg_features = pd.DataFrame(avg_features, columns=[f'{f}_loc' for f in numerical_features])
    std_features = numerical_data.std(axis=1)
    std_features = pd.DataFrame(std_features, columns=[f'{f}_scale' for f in numerical_features])

    # Forecasting: target data will have avg 0 since it uses the target normalizer. Take the values prior to normalization.
    # Imputation: target data will have avg and std different from 0 and 1, but is still normalized instance by instance.
    # Reconstruct target and compute avg and std

    stack_targets = lambda t: torch.stack(t, axis=0).numpy()
    def get_target_data(future=False):
        if future:
            cds = [f'future_{f}' for f in data.target]
        else:
            cds = data.target
        cds_idx = [numerical_features.index(f) for f in cds]
        target_cds = x['encoder_cont'][..., cds_idx]
        loc, scale = stack_targets(x['target_scale']).T[:, :, None]
        target_cds = target_cds*scale + loc
        target_cds = target_cds.numpy()
        target_cds = target_cds.transpose(2, 0, 1)
        target_data = pd.DataFrame(np.hstack((target_cds.mean(axis=-1).T,
                                              target_cds.std(axis=-1).T)),
                                   columns=[f'{f}_{s}' for s in ['loc', 'scale'] for f in cds])
        return target_data
    target_data = get_target_data()
    if task == 'imputation':
        future_target_data = get_target_data(future=True)
        target_data = pd.concat([target_data, future_target_data], axis=1)

    print("Computing correlations")
    corr = defaultdict(list)
    for animal in tqdm(range(numerical_data.shape[0])):
        animal_data = numerical_data[animal]
        for i, f1 in enumerate(numerical_features[:-1]):
            x1 = animal_data[:, i]
            for j, f2 in enumerate(numerical_features[i+1:], start=i+1):
                x2 = animal_data[:, j]
                for method in ['spearman', 'pearson']:
                    corr[f'{method}_{f1}-{f2}'].append(getattr(ss, f'{method}r')(x1, x2)[0])
    corr = pd.DataFrame(corr)
    nan_cols = corr.isna().all()
    corr = corr.loc[:, ~nan_cols]

    numericals = pd.concat([avg_features.drop(columns=target_data.columns.intersection(avg_features.columns)),
                            std_features.drop(columns=target_data.columns.intersection(std_features.columns)),
                            target_data,
                            corr], axis=1)

    # categoricals
    categorical_values = x['encoder_cat'].mode(axis=1).values # most frequent value
    categoricals = dict()
    cat_encoders = data.categorical_encoders
    for c, v in zip (categorical_features, categorical_values.T):
        categoricals[c] = cat_encoders[c.replace("future_", "")].inverse_transform(v)
    categoricals = pd.DataFrame(categoricals)
    return numericals, categoricals
