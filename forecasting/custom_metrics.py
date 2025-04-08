import numpy as np
import pandas as pd
import math
from copy import deepcopy
import torch
from pytorch_forecasting.metrics import MultiHorizonMetric
from torch.nn.utils import rnn
from pytorch_forecasting.utils import unpack_sequence, unsqueeze_like, create_mask, masked_op
from numba import njit
import warnings
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)
from phdu import bootstrap, integration, np_utils, pd_utils
from . import params
from .preprocessing import space
try:
    from phdu.stats.rtopy._helper import load_R_pkg, r, ro
    load_R_pkg("pracma")
except:
    warnings.warn("R not available. Some functions may not work.")


def beta_interval_1D(alpha):
    """
    Returns the confidence level for a 1D interval in order to obtain a 2D confidence region with confidence level `alpha`.

    Derivation:
    (1 - beta)**2 = 1 - alpha
    beta = 1 - sqrt(1 - alpha)
    """
    beta = 1 - np.sqrt(1 - alpha)
    return beta

def exact_quantiles(alpha=[0.5, 0.1, 0.05]):
    alpha = np.array(alpha)
    beta = beta_interval_1D(alpha)
    quantiles = np.hstack([(beta/2)[::-1],
                           0.5, # median (point estimate)
                           1 - beta/2
                           ])
    return quantiles

def time_step_to_days(t, sampling_freq=6):
    return (t+1) / (24 / sampling_freq)

@njit
def nb_mean(x):
    return x.mean()

@njit
def nb_median(x):
    return np.median(x)

@njit
def np_apply_along_axis_not_nans(func1d, axis, arr):
    """
    Only valid for 2D arrays.
    """
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i][~np.isnan(arr[:, i])])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :][~np.isnan(arr[i, :])])
    return result

@njit
def nb_mean_not_nans(array):
    """
    averages first across rows excluding nans, then across columns.
    """
    x = np_apply_along_axis_not_nans(np.mean, 0, array)
    return x[~np.isnan(x)].mean()

@njit
def nb_mean_not_nans_across_seeds(array):
    """
    averages first across axis 3, then across rows excluding nans, then across columns.
    """
    x = np.sum(array, axis=2) / array.shape[2]
    x = np_apply_along_axis_not_nans(np.mean, 0, x)
    return x[~np.isnan(x)].mean()

class MultiHorizonMetricMulti(MultiHorizonMetric):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def update(self, y_pred, target):
        """
        Update method of metric that handles masking of values for joint target

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack weight
        if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target[0].device)

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * unsqueeze_like(weight, losses)
        self._update_losses_and_lengths(losses, lengths)

class MAEMulti(MultiHorizonMetricMulti):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs().mean()``
    """
    def loss(self, y_pred, target):
        loss = (y_pred - target).abs().mean(axis=-1)
        return loss

class RMSEMulti(MultiHorizonMetricMulti):
    """
    Root mean squared error.

    Defined as ``torch.sqrt((y_pred - target).pow(2).mean())``
    """
    def loss(self, y_pred, target):
        loss = torch.sqrt((y_pred - target).pow(2).mean(axis=-1))
        return loss

def rae(x, target=0.5):
    """
    Relative absolute error.
    To be used for comparing empirical coverage against target coverage.
    """
    return (target - x) / target

@njit
def nb_rae(x, target=0.5):
    return (target - x) / target

def arae(x, target=0.5):
    """
    Absolute relative absolute error.
    To be used for comparing empirical coverage against target coverage.
    """
    return np.abs(rae(x, target))

@njit
def nb_arae(x, target=0.5):
    return np.abs(nb_rae(x, target))

# TODO: check arae for alpha instead of confidence

def aggregate_metric_CI(df, metric=None, target=0.5, boot='percentile', **kwargs):
    X = df.values.astype(float)
    if np.isnan(X).any():
        avg_computer = nb_mean_not_nans
    else:
        avg_computer = nb_mean
    if metric == 'rae':
        @njit
        def statistic(x):
            return nb_rae(avg_computer(x), target=target)
    elif metric == 'arae':
        @njit
        def statistic(x):
            return nb_arae(avg_computer(x), target=target)
    elif metric is None:
        statistic = avg_computer
    else:
        raise ValueError(f"metric '{metric}' not recognized.")
    sample_stat = statistic(X)
    computer = getattr(bootstrap, f'CI_{boot}')
    CI = computer(X, statistic, **kwargs)
    return sample_stat, CI

def aggregate_CI_summary(func, force_CI_expansion=False, CI_expansion=True, use_tqdm=False, task='forecasting', ci_func=True, **kwargs):
    """
    CI_expansion: whether to allow expanding the CI based on validation data. If true, selects the best result between expanding and not expanding.
    """
    if CI_expansion:
        if force_CI_expansion:
            model_specs = params.model_specs_force_CI_expansion()
        else:
            model_specs = params.model_specs()
    else:
        model_specs = params.model_specs_no_CI_expansion()
    out = []
    iterator = tqdm(model_specs.items()) if use_tqdm else model_specs.items()
    for model, specs in iterator:
        if model in ['TFT_dist', 'TFT', 'TFT_single', 'Naive']:
            specs.update(params.TFT_specs[task])
            if model == 'Naive' and task == 'imputation':
                extra_specs = params.baseline_specs_imputation.copy()
                if not ci_func:
                    del extra_specs['naive_pred_lengths']
                specs.update(extra_specs)
            if model == 'TFT_dist' and not ci_func:
                specs['dist_mode'] = 'median'
        sample_stat, CI = func(**specs, task=task, **kwargs)
        out.append(pd.Series(dict(CI=CI, sample_stat=sample_stat), name=model))
    out = pd.concat(out, axis=1).T
    return out

@njit
def solid_angle_y_integrand(y):
    return np.exp(2*y / params.R_earth) / ((1 + np.exp(2*y / params.R_earth))**2)

def area_integral_rectangle(x0, x1, y0, y1):
    """
    x0, x1: min and max X mercator projection coordinates
    y0, y1: min and max Y mercator projection coordinates

    Returns:
    --------
    The area of the region defined by the coordinates in km^2.
    """
    return (-2)*params.R_earth * (x1 - x0) * (1/(1+np.exp(2*y1/params.R_earth)) - 1/(1+np.exp(2*y0/params.R_earth)))

@njit
def solid_angle_integrand_ellipse(y, semilength_1, semilength_2, c_y):
    """
    y: y-coordinate of the mercator projection
    semilength_1: major axis semilength
    semilength_2: minor axis semilength
    c_y: y-coordinate of the ellipse center
    """
    term_ellipse = np.sqrt(np.clip(1 - ((y-c_y)/semilength_2)**2, 0, None)) # numerical problems near zero (can lead to small negative numbers
    return ((8*semilength_1 / params.R_earth**2)
            * term_ellipse
            * solid_angle_y_integrand(y)
            )

def net_movement_x(x):
    return np.diff(x[0][[0, -1]])[0]

def net_movement_y(x):
    return np.diff(x[1][[0, -1]])[0]

def abs_distance_x(x):
    return np.abs(np.diff(x[0])).sum()

def abs_distance_y(x):
    return np.abs(np.diff(x[1])).sum()

def directionality_angle(z):
    """
    Orientation of the displacements relative to NS-WE plane.
    """
    x, y = z
    beta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    return beta

def compute_hurst(beta, min_size=10):
    if beta.size < min_size:
        return np.nan
    else:
        try:
            ro.globalenv["x"] = beta
            r("out = hurstexp(x, d=50)")
            out = r("out$Hrs")[0]
            if math.isnan(out):
                warnings.warn("Hurst exponent corrected is NaN. Returning without correction.")
                out = r("out$Hs")[0]
            out = np.clip(out, 0, 1)
            return out
        except:
            return np.nan

def directionality_hurst(z):
    beta = directionality_angle(z)
    return compute_hurst(beta)

def relative_angle(z, nan='fill'):
    """
    Relative angle between consecutive displacements.
    """
    dX = np.diff(z, axis=1).T
    beta = np.arccos(np.sum(dX[:-1] * dX[1:], axis=1) / (np.linalg.norm(dX[:-1], axis=1) * np.linalg.norm(dX[1:], axis=1)))
    if nan == 'fill':
        beta[np.isnan(beta)] = 0
    elif nan == 'drop':
        beta = beta[~np.isnan(beta)]
    return beta

def relative_angle_Hurst(z, nan='fill'):
    beta = relative_angle(z, nan=nan)
    return compute_hurst(beta)

def average_speed(x, sampling_freq=6):
    lat, lon = space.mercator_inv(*x)
    d = space.great_circle_distance(lat, lon)
    return np.nanmean(d) * params.R_earth / sampling_freq

def detect_missing_values(x):
    # check if previous element is equal to next element
    equal_consecutive = np.all(np.diff(x, axis=1) == 0, axis=0)
    equal_consecutive = np.hstack((False, equal_consecutive)) # first element has no previous element
    return equal_consecutive

def is_missing_1_axis(x):
    return np.hstack((False, np.diff(x) == 0))

def percentage_repeated_values(x):
    equal_consecutive = detect_missing_values(x)
    return equal_consecutive.mean()

def num_repeated_values_just_before_end(x):
    """
    Number of consecutive missing values just before the end
    """
    equal_consecutive = detect_missing_values(x)
    if not equal_consecutive[-1]:
        return 0
    else:
        return np_utils.idxs_condition(equal_consecutive[::-1], lambda x: x == False)[0]

def missing_info(results, partition='test', var_type='encoder', reverse=True):
    """
    reverse: bool, default=True.
        If var_type == 'future', the missing values before end correspond to the missing values closest to the imputation window.
    """
    missing = pd.Series(list(results[f'out_{partition}'][f'{var_type}_missing'].numpy()))
    if var_type == 'future' and reverse:
        missing = missing.apply(lambda x: x[::-1])
    def num_missing_values_just_before_end(x):
        """
        Number of consecutive missing values just before the end
        """
        if not x[-1]:
            return 0
        else:
            return np_utils.idxs_condition(x[::-1], lambda x: x == False)[0]
    num_before_end = missing.apply(num_missing_values_just_before_end)
    ptg_missing = missing.apply(np.mean)
    missing_data = pd.concat([num_before_end, ptg_missing], axis=1)
    if var_type == 'encoder':
        cols = ['num_missing_before_end', 'ptg_missing']
    elif var_type == 'future':
        cols = ['future_num_missing_from_start', 'future_ptg_missing']
    else:
        raise ValueError(f"var_type '{var_type}' not recognized. Available options are 'encoder' and 'future'.")
    missing_data.columns = cols
    return missing_data

def encoder_length(x):
    return x.shape[1]

def stack_targets(x):
    return torch.stack(x, axis=0).numpy()

def baseline_prediction_future(results, x=None, partition=None, cds='mercator'):
    """
    Returns the first observed value of the future variables.
    """
    if x is None:
        assert partition in ['train', 'val', 'test'], "partition should be provided if x is None."
        x = results[f'x_{partition}']
    targets = params.cds_to_cols[cds]
    encoder_variables = results['encoder_variables']
    future_cds = [encoder_variables.index(f'future_{target}') for target in targets]
    X_future = x['encoder_cont'][..., future_cds].numpy() # shape (N, time_step, target)
    loc, scale = stack_targets(x['target_scale']).T # shape (loc-scale, N, target)
    encoder_target_future = (X_future * scale[:, None]) + loc[:, None]
    encoder_target_future = encoder_target_future.transpose(2, 0, 1) # shape (target, N, time_step)
    baseline_prediction_future = encoder_target_future[:, :, 0]
    return baseline_prediction_future

def compute_future_encoder(results, x=None, partition=None, cds='mercator'):
    """
    Returns:
        1. Future values of the target variables passed to the encoder
        2. Encoder lengths of the future variables for each animal.
    """
    if x is None:
        assert partition in ['train', 'val', 'test'], "partition should be provided if x is None."
        x = results[f'x_{partition}']
    targets = params.cds_to_cols[cds]
    stack_targets = lambda t: torch.stack(t, axis=0).numpy()
    encoder_targets = stack_targets(x[f"encoder_target"])
    num_targets, N, max_encoder_length = encoder_targets.shape
    max_future_length = x['future_lengths'].max().item()
    encoder_variables = results['encoder_variables']
    future_cds = [encoder_variables.index(f'future_{target}') for target in targets]
    # cds = [encoder_variables.index(target) for target in targets]
    X_future = x['encoder_cont'][..., future_cds].numpy() # shape (N, time_step, target)
    loc, scale = stack_targets(x['target_scale']).T # shape (loc-scale, N, target)
    encoder_target_future = (X_future * scale[:, None]) + loc[:, None]
    encoder_target_future = encoder_target_future.transpose(2, 0, 1) # shape (target, N, time_step)
    encoder_length_future = x['future_lengths'].numpy()
    return encoder_target_future, encoder_length_future

trj_properties_funcs = [net_movement_x, net_movement_y, abs_distance_x, abs_distance_y, directionality_hurst, relative_angle_Hurst, average_speed,
                        percentage_repeated_values, num_repeated_values_just_before_end, encoder_length]

def stack_last_observation(x):
    """
    To be used in decoder_missing_values
    """
    real, base = x
    return np.hstack((base[:, -1][:, None], real))

def is_missing(x):
    """
    To be used in decoder_missing_values
    """
    equal_consecutive = np.all(np.diff(x, axis=1) == 0, axis=0)
    return equal_consecutive

def decoder_missing_values(baseline, y_real):
    """
    x = results['x_{partition}']
    """
    df = pd_utils.tuple_wise(y_real, baseline)
    y_real_plus_last_loc = df.apply(stack_last_observation)
    missing = y_real_plus_last_loc.apply(is_missing)
    return missing

def eval_coverage(x):
    """
    Evaluate the coverage of the CI.
    input: row containing the CI and the real value.
    """
    CI, y = x
    if isinstance(CI, float):
        return np.nan
    else:
        return (CI[0] <= y) & (CI[1] >= y)

def max_distance(days):
    max_speed = 10 # km / h
    max_distance_1_day = max_speed * 24
    # max_distance_1_day = 100
    return max_distance_1_day * days

def max_daily_distance(days):
    return 100 * days

@np.vectorize
def circle_intersection_area(R, r, d):
    """
    https://mathworld.wolfram.com/Circle-CircleIntersection.html
    """
    if d >= R + r:
        return 0  # No intersection
    if d <= abs(R - r):
        return np.pi * min(R, r)**2  # One circle inside the other

    return ( R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
            + r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
            + 0.5 * np.sqrt((-d + R + r) * (d + R - r) * (d - R + r) * (d + R + r)) )

def maximal_area_imputation(baseline_pred, sampling_freq=6):
    """
    Area associated to a southern elephant seal moving at maximum speed for a given number of days, taking into account that the location is known at t0 and t1=t0+prediction_days+sampling_freq/24.

    The maximal area is computed as the area of the intersection of two circles with
     r1 = radius equal to the distance covered by the seal in the given number of days
     r2 = same as r1, but in reverse order (starting from the last location)
      d = distance between the two locations.
    """
    lat_0, lon_0 = space.mercator_inv(*baseline_pred[:, 0])
    lat_f, lon_f = space.mercator_inv(*baseline_pred[:, -1])
    d = space.great_circle_distance(lat_0, lon_0, lat_f, lon_f) * params.R_earth

    prediction_length = baseline_pred.shape[1]
    daily_step = sampling_freq / 24
    prediction_days = prediction_length * daily_step
    days = np.arange(daily_step, prediction_days + daily_step, daily_step)
    r1 = max_distance(days)
    r2 = r1[::-1] # same distance but in reverse order
    return circle_intersection_area(r1, r2, d)

def maximal_area_forecasting(days):
    """
    Area associated to a southern elephant seal moving at maximum speed for a given number of days.

    Northern elephant seal: vmax=16km/h, dmax=60miles/day
    Source: https://elephantseal.org/an-elephant-seal-deep-dive/
    In reality they can move around 60 miles (100km) per day: https://elephantseal.org/fun-facts/

    Southern elephant seal: vmax=10km/h  migrate 33800km in 365 days = 92.6km/day
    Source: https://oceanwide-expeditions.com/to-do/wildlife/elephant-seal

    The maximal area is computed as the area of a circle with radius equal to the distance covered by the seal in the given number of days.
    """
    return np.pi * max_distance(days)**2

def maximal_area(*, task, baseline=None, sampling_freq=6, days=None):
    if task == 'imputation':
        return baseline.apply(maximal_area_imputation, sampling_freq=sampling_freq)
    elif task == 'forecasting':
        return maximal_area_forecasting(days)
    else:
        raise ValueError(f"task '{task}' not recognized. Available tasks are 'imputation' and 'forecasting'.")

@njit
def nb_cumsum_axis1_except_last(x):
    """
    Compute the cumulative sum along axis 1 for a 2D numpy array.
    """
    rows, cols = x.shape
    cols -= 1 # delete for full cumsum

    x_cumsum = np.zeros((rows, cols))
    for i in range(rows):
        x_cumsum[i, 0] = x[i, 0]
        for j in range(1, cols):
            x_cumsum[i, j] = x_cumsum[i, j - 1] + x[i, j]
    return x_cumsum

@njit
def displacement_bootstrap_sample_imputation(real, baseline, R=int(1e4)):
    x0 = baseline[:,:1]
    xf = baseline[:,-1:]
    X = np.hstack((x0, real, xf))
    dx = X[:, 1:] - X[:, :-1]
    num_time_steps = real.shape[1]
    num_dxs = num_time_steps + 1
    possible_idxs = np.arange(num_dxs)
    np.random.seed(0)
    def intermediate_locations():
        idxs = np.random.choice(possible_idxs, size=num_dxs, replace=False)
        dx_i = dx[:, idxs]
        xi = x0 + nb_cumsum_axis1_except_last(dx_i) # last location will always be xf
        return xi

    boot_sample = np.empty((R, *real.shape))
    for i in range(R):
        boot_sample[i] = intermediate_locations()
    return boot_sample

def alpha_1D(alpha):
    """
    Returns the confidence level for a 1D interval to obtain a 2D confidence region with confidence level `alpha`, constructed by stacking the PIs of each dimension.
    """
    return 1 - np.sqrt(1 - alpha)

def compute_reference_area_imputation(z, alpha=0.05, R=int(1e4)):
    """
    Area of an imputation model that knows the displacements delta_i between x0 and xf, but not the intermediate locations.

    In the imputation region: x_i = x_0 + sum_{j=0}^{i} delta_j

    The reference area is computed as the (1-alpha) confidence PR, constructed by stacking the uncertainties (PIs) of the intermediate positions.
    """
    real, baseline = z
    num_time_steps = real.shape[1]
    boot_sample = displacement_bootstrap_sample_imputation(real, baseline, R)
    alpha_i = alpha_1D(alpha)
    CI = bootstrap._compute_CI_percentile(boot_sample, alpha=alpha_i, alternative='two-sided', to_ptg=True) # shape (time, cds, 2)

    A_ref = np.empty((num_time_steps))
    for i, c_i in enumerate(CI):
        x0_i, x1_i, y0_i, y1_i = c_i.flatten()
        A_ref[i] = area_integral_rectangle(x0_i, x1_i, y0_i, y1_i)
    return A_ref

def displacement_bootstrap_sample_forecasting(real, baseline, R=int(1e4)):
    x0 = baseline[:,:1]
    X = np.hstack((x0, real))
    dx = X[:, 1:] - X[:, :-1]
    boot_sample = bootstrap.resample_nb_X(dx.T, R=R)
    boot_sample = boot_sample.transpose(0, 2, 1)
    boot_sample = boot_sample.cumsum(axis=-1)
    boot_sample += x0[None] # dx -> trj
    return boot_sample

def compute_reference_area_forecasting(z, alpha=0.05, R=int(1e4)):
    """
    Estimate uncertainty from the displacements at the target time interval.

    The reference area is computed as the (1-alpha) confidence PR, constructed by stacking the uncertainties (PIs) of the intermediate positions.
    """
    real, baseline = z
    num_time_steps = real.shape[1]
    boot_sample = displacement_bootstrap_sample_forecasting(real, baseline, R=R)
    alpha_i = alpha_1D(alpha)
    CI = bootstrap._compute_CI_percentile(boot_sample, alpha=alpha_i, alternative='two-sided', to_ptg=True) # shape (time, cds, 2)
    A_ref = np.empty((num_time_steps))
    for i, c_i in enumerate(CI):
        x0_i, x1_i, y0_i, y1_i = c_i.flatten()
        A_ref[i] = area_integral_rectangle(x0_i, x1_i, y0_i, y1_i)
    return A_ref

def _compute_reference_area_forecasting(z, method='cumulative-by-direction', double=False):
    """
    DEPRECATED. Use compute_reference_area_forecasting instead.

    Area of a forecaster that predicts the last observed location and makes a perfect prediction region always including future locations inside, with minimal area.

    method='cumulative-by-direction' (defaul behavior). We have defined the reference area as the area predicted by a forecaster that knows the direction of the movement, and sets the maximum distance traveled in each coordinate as the exact distance observed in the trajectory. When multiple time steps are concatenated, the net result for each coordinate is the accumulation of uncertainties in the direction of the movement.

    method='cumulative', double=False: if True, the area is computed as the cumulative sum of the distance between consecutive points. This ensures the area increases with time.

    method = 'raw': the area can decrease at times where the animal moves back.

    method='cumulative', double=True: area of a forecaster that knows the distance (dx, dy) > 0,  between consecutive locations (x,y), but not the direction. Such forecaster would have to account for the two possible locations (x+dx, y+dy) and (x-dx, y-dy) to ensure the future location is always inside the PR.
    Furthermore, for the second location it would still not know the direction of the movement, but also the starting point. Hence, the uncertainty accumulates the uncertainty of previous time steps:
    A_{ref}(t_i) = (sum_{j=0}^{i} dx_j) * (sum_{j=0}^{i} dy_j)
    """
    real, baseline = z
    x, y = real
    x_base, y_base = baseline
    if method == 'cumulative-by-direction':
        def cum_by_dir(u, u_base):
            u_plus_base = np.hstack((u_base[0], u))
            du = np.diff(u_plus_base)
            positive = du > 0
            du_positive = du.copy()
            du_positive[~positive] = 0
            du_negative = du.copy()
            du_negative[positive] = 0
            u0 = u_base[0] + du_negative.cumsum()
            u1 = u_base[0] + du_positive.cumsum()
            return u0, u1
        x0, x1 = cum_by_dir(x, x_base)
        y0, y1 = cum_by_dir(y, y_base)
    else:
        if method == 'cumulative':
            def to_cum(u):
                du = np.hstack((0, np.diff(u)))
                direction = np.sign(du.sum())
                return u[0] + direction * np.abs(du).cumsum()
            x = to_cum(x)
            y = to_cum(y)
        elif method != 'raw':
            raise ValueError(f"method {method} not recognized. Available methods are 'cumulative-by-direction', 'cumulative', 'raw'.")
        if double:
            x_rev = x_base - np.diff(np.hstack((x_base[0], x))).cumsum()
            y_rev = y_base - np.diff(np.hstack((y_base[0], y))).cumsum()
            x0 = np.fmin(x, x_rev)
            x1 = np.fmax(x, x_rev)
            y0 = np.fmin(y, y_rev)
            y1 = np.fmax(y, y_rev)
        else:
            x0 = np.fmin(x, x_base)
            x1 = np.fmax(x, x_base)
            y0 = np.fmin(y, y_base)
            y1 = np.fmax(y, y_base)

    A = np.empty((x.size))
    for i, (x0_, x1_, y0_, y1_) in enumerate(zip(x0, x1, y0, y1)):
        if x0_ == x1_ and y0_ == y1_:
            A[i] = 0
        else:
            A[i] = area_integral_rectangle(x0_, x1_, y0_, y1_)
    return A

def compute_reference_area(z,*, task, alpha=0.05, R=int(1e4), method='cumulative-by-direction', double=False):
    if task == 'imputation':
        return compute_reference_area_imputation(z, alpha=alpha, R=R)
    elif task == 'forecasting':
        # return compute_reference_area_forecasting(z, method=method, double=double) # DEPRECATED
        return compute_reference_area_forecasting(z, alpha=alpha, R=R)
    else:
        raise ValueError(f"task '{task}' not recognized. Available tasks are 'imputation' and 'forecasting'.")

def area_quality(A, A_ref, A_max):
    """
    Quality of a forecaster with area A, reference area A_ref, and maximal area A_max.

    The reference area is the area of a forecaster that predicts the last observed location and makes a perfect prediction region always including future locations inside, with minimal area.

    The maximal area is the area associated to a southern elephant seal moving at maximum speed for a given number of days.
    """
    A = np.clip(A, A_ref, A_max)
    with warnings.catch_warnings(): # ignore division by zero & invalid value in logs
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c = np.log(A/A_ref) / np.log(A_max/A_ref)
    Q = 1 - c
    return Q

def coverage_quality_symmetric(alpha, alpha_target=0.05):
    """
    Ideally h would satisfy three properties:
    1. Penalize underestimation and overestimation equally:
        h(b*alpha) = h(b^{-1} * alpha) >=  h(alpha) ~ |log(alpha/alpha_target)|
    2. Be ranged from 0 (worst) to 1 (best). h(alpha_target)=1, h(min_alpha)=0, h(max_alpha)=0
    3. Having a cutoff value for low alpha, such that the maximum underestimation is limited by the maximum overestimation.

    The function:
    h(x) = |log(x/alpha_target) / log(alpha_target)|
    verifies the first two properties. To satisfy the third one, we can set a cutoff value, such that h(cutoff) = 1.
    the maximum overestimation is alpha=1:
    h(1) = |log(1/alpha_target) / log(alpha_target)| = 1
    the maximum underestimation is alpha = alpha_target^2:
    h(alpha_target^2) = |log(alpha_target^2/alpha_target) / log(alpha_target)| = 1
    """
    alpha = np.array(alpha)
    c = np.abs(np.log(alpha/alpha_target) / np.log(alpha_target))
    c[alpha < alpha_target**2] = 1
    Q = 1 - c
    return Q

def coverage_quality(alpha, alpha_target=0.05):
    """
    Same as coverage_quality_symmetric, but only penalizes overestimation in alpha. This is because the target is that the animal lies within the PR with probability greater or equal to 1 - alpha_target.
    If achieves better coverage, it is not penalized. It will be penalized in the 'area_quality' function, as a too conservative PR may use more area than needed.
    """
    alpha = np.array(alpha)
    with warnings.catch_warnings(): # ignore division by zero & invalid value in logs
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c = np.abs(np.log(alpha/alpha_target) / np.log(alpha_target))
    Q = 1 - c
    Q[alpha < alpha_target] = 1
    return Q

@njit
def coverage_quality_single(alpha, alpha_target=0.05):
    """
    Same as coverage_quality_symmetric, but only penalizes overestimation in alpha. This is because the target is that the animal lies within the PR with probability greater or equal to 1 - alpha_target.
    If achieves better coverage, it is not penalized. It will be penalized in the 'area_quality' function, as a too conservative PR may use more area than needed.
    """
    if alpha < alpha_target:
        return 1
    else:
        c = np.abs(np.log(alpha/alpha_target) / np.log(alpha_target))
        Q = 1 - c
        return Q

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
def nb_mean_axis_0(array):
  return np_apply_along_axis(np.mean, 0, array)

@njit
def nb_nanmean_axis_0(x):
    return np_apply_along_axis(np.nanmean, 0, x)

@njit
def normalized_avg(x):
    att_by_time_step = nb_nanmean_axis_0(x)
    # replace nans by zeros
    att_by_time_step[np.isnan(att_by_time_step)] = 0
    att_by_time_step /= att_by_time_step.sum()
    return att_by_time_step

@njit
def normalized_avg_cumsum(x):
    return normalized_avg(x).cumsum()

def CI_mean_by_col(df):
    out = {}
    for feature in df.columns:
        x = df[feature].values
        out[(feature, 'sample_stat')] = nb_mean(x)
        out[(feature, 'CI')] = bootstrap.CI_percentile(x, nb_mean, R=int(1e4))[0]
    out = pd.Series(out).unstack()
    return out

def get_compute_Qs(alpha_target):
    @njit
    def compute_Qs(x_t):
        alpha, Q_area = nb_mean_axis_0(x_t)
        Q_alpha = coverage_quality_single(alpha, alpha_target)
        Q = Q_alpha * Q_area
        return np.array([Q_alpha, Q_area, Q], dtype=np.float64)
    return compute_Qs

def join_encoder_future_features(df, future=False):
    prefix = 'future_' if future else ''
    add_prefix = lambda features: [f"{prefix}{f}" for f in features]
    are_features_in_df = lambda features: all([f in df.columns for f in features])

    if are_features_in_df(add_prefix(['X', 'Y', 'Z'])):
        cds = 'spherical-dx'
    elif are_features_in_df(add_prefix(['SN', 'WE'])):
        cds = 'spherical'
    elif are_features_in_df(add_prefix(['X', 'Y'])):
        cds = 'mercator'
    else:
        raise ValueError("Features not recognized.")

    remaps = dict(Location=add_prefix(params.cds_to_cols[cds]),
                  Time=add_prefix(['cos t', 'sin t', 'season', 'month']),
                  Hour_angle=add_prefix(['sin_H', 'cos_H']),
                  Wind=add_prefix(['Wind (U)', 'Wind (V)']),
                  Wave_direction=add_prefix(['Wave direction (x)', 'Wave direction (y)']),
                  Marine_current=add_prefix(['Marine current (meridional)', 'Marine current (zonal)']),
                  IR_albedo=add_prefix(['IR albedo (diffuse)', 'IR albedo (direct)']),
                  )
    for group, fs in remaps.items():
        df[group] = df[fs].sum(axis=1)
        df = df.drop(columns=fs)
    # take out prefix
    df.columns = [f.replace(prefix, '') for f in df.columns]
    return df

def join_encoder_features(df):
    return join_encoder_future_features(df, future=False)

def join_future_features(df):
    return join_encoder_future_features(df, future=True)

def join_decoder_features(df):
    time_cols = ['cos t', 'sin t', 'season', 'month']
    df['Time'] = df[time_cols].sum(axis=1)
    df = df.drop(columns=time_cols)
    return df

def preprocess_attention(*, results, partition='test', delete_missing=True, join_features=True):
    """
    This function processes the attention weights and variables from the output of a transformer model.

    Parameters:
    out (dict): The output dictionary from a transformer model. It should contain keys for 'decoder_attention',
                'encoder_lengths', 'encoder_variables', 'encoder_lengths', 'decoder_variables', 'decoder_lengths',
                and 'static_variables'.
    x (dict): The input dictionary to the transformer model. It should contain keys for 'encoder_target' and 'decoder_target'.
    delete_missing (bool): If True, replace missing values in the attention weights and variables with NaN. Default is True.
    join_features (bool): If True, join the feature groups (e.g. 'Wind (U)', 'Wind (V)' -> 'Wind'). Default is True.

    Returns:
    tuple: A tuple containing the processed attention weights and variables. The tuple contains four elements:
           - attention (torch.Tensor): The processed attention weights.
           - encoder (pd.Series): The processed encoder variables.
           - decoder (pd.Series): The processed decoder variables.
           - future (pd.Series): The processed future variables if task is 'imputation'. Else None.
           - static (torch.Tensor): The static variables from the output of the transformer model.
    """
    out = deepcopy(results[f'out_{partition}'])
    x = results[f'x_{partition}']
    if 'decoder_missing' in x:
        out['decoder_missing'] = x['decoder_missing']
    has_future = 'future_variables' in out

    def replace_by_nans(z):
        x, l = z
        if isinstance(l, int):
            x[l:] = np.nan
        elif isinstance(l, np.ndarray) and l.dtype == bool:
            x[l] = np.nan
        else:
            raise ValueError(f"l should be an integer or boolean array. Got {type(l)}")
        return x

    def preprocess_vars(var_type):
        # NOTE: for the imputation task, out[f"{var_type}_sparse_weights_rev"] is the equivalent for the LSTM pass backwards in time (future -> decoder -> encoder). The results are equal. (must reverse time using torch.flip(variables, [1]))
        variables = out[f"{var_type}_variables"].squeeze(-2).clone()
        S = pd.Series(list(variables.numpy())) # index = animal
        l = pd.Series(out[f"{var_type}_lengths"])
        S_l = pd_utils.tuple_wise(S, l)
        S = S_l.apply(replace_by_nans)

        if f"{var_type}_missing" in out:
            is_missing = pd.Series(list(out[f"{var_type}_missing"].numpy()))
        else:
            stack_targets = lambda t: torch.stack(t, axis=0).numpy()
            target = stack_targets(x[f'{var_type}_target'])
            target = np.moveaxis(target, 0, 1)
            is_missing = pd.Series(list(target)).apply(detect_missing_values)

        if delete_missing:
            S_missing = pd_utils.tuple_wise(S, is_missing)
            S = S_missing.apply(replace_by_nans)
        S_avg = S.apply(np.nanmean, axis=0) # average across time steps
        if has_future and var_type != 'decoder':
            variables = results[f'{var_type}_only_variables']
        else:
            variables = results[f'{var_type}_variables']
        df = pd.DataFrame(np.vstack(S_avg.values), columns=variables)
        df.index.name = 'animal'
        df.columns.name = 'feature'
        if join_features:
            df = globals()[f'join_{var_type}_features'](df)
        return df, is_missing

    encoder, encoder_missing = preprocess_vars('encoder')
    decoder, decoder_missing = preprocess_vars('decoder')
    if has_future:
        future, future_missing = preprocess_vars('future')
    else:
        future, future_missing = None, None
    static = pd.DataFrame(out['static_variables'].squeeze(), columns=results['static_variables'])

    def preprocess_attention():
        decoder_attention = out["decoder_attention"].clone()
        decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
        decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")# set attention to nan where not valid (index > decoder_length)
        if delete_missing:
            decoder_missing_mask = torch.tensor(np.stack(decoder_missing, axis=0))
            decoder_attention[decoder_missing_mask[..., None, None].expand_as(decoder_attention)] = float("nan")

# roll encoder attention (so start last encoder value is on the right)
# this is because the last value on the right is the closest to the prediction window. So the encoder is filled from right (t0) to left (-encoder_length).
        encoder_attention = out["encoder_attention"].clone()
        encoder_mask = create_mask(out["encoder_attention"].size(3), out["encoder_lengths"])
        encoder_attention[encoder_mask[:, None, None].expand_as(encoder_attention)] = float("nan")
        if delete_missing:
            encoder_missing_mask = torch.tensor(np.stack(encoder_missing, axis=0))
            encoder_attention[encoder_missing_mask[:, None, None].expand_as(encoder_attention)] = float("nan")
        shifts = encoder_attention.size(3) - out["encoder_lengths"]
        new_index = (
            torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                encoder_attention
            )
            - shifts[:, None, None, None]
        ) % encoder_attention.size(3)
        encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)

        attns = [encoder_attention, decoder_attention]
        if has_future:
            future_attention = out["future_attention"].clone()
            future_mask = create_mask(out["future_attention"].size(3), out["future_lengths"])
            future_attention[future_mask[:, None, None].expand_as(future_attention)] = float("nan")
            if delete_missing:
                future_missing_mask = torch.tensor(np.stack(future_missing, axis=0))
                future_attention[future_missing_mask[:, None, None].expand_as(future_attention)] = float("nan")
            attns.append(future_attention)

        attention = torch.concat(attns, dim=-1)
        attention[attention < 1e-5] = float("nan")
# attention shape: (N, decoder_length, num_heads, encoder_length + decoder_length), i.e. (num_instances, time, num_heads, time_to_attend)
        attention = masked_op( # average over attention heads
            attention,
            op="mean",
            dim=-2,
        )
        max_decoder_length = out['decoder_lengths'].max().item()
        if has_future:
            attention_by_time_step = pd.Series([attention[:,i] for i in range(max_decoder_length)]) # keep prediction attention and attention on observed timesteps
        else:
            max_encoder_length = out['encoder_lengths'].max().item()
            attention_by_time_step = pd.Series([attention[:,i, :max_encoder_length+i:] for i in range(max_decoder_length)]) # keep prediction attention and attention on observed timesteps
        attention_by_time_step = attention_by_time_step.apply(lambda attention: attention / masked_op(attention, dim=1, op="sum").unsqueeze(-1)) # renormalize (sum 1 across observed timesteps for each trajectory)
        return attention_by_time_step

    attention_by_time_step = preprocess_attention()

    return attention_by_time_step, encoder, decoder, future, static

@njit
def nb_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out

def _aggregate_CI_across_time_steps(Zs, statistic, labels, agg_func=None, R=int(1e5), alpha=0.05, delete_nans_before_resampling=True):
    """
    DEPRECATED: Use aggregate_CI_across_time_steps instead. Error: resampling should be at the trajectory level.

    Xs: list of numpy arrays containing the samples of the target statistic across time steps.
    statistic: function that computes the target statistic for one time step.
    labels: list of labels for the target statistic.
    R: number of bootstrap samples.
    alpha: significance level for the CI.

    Returns: DataFrame with the sample statistic and the CI for each label.
    """
    if isinstance(labels, str):
        labels = [labels]
    Xs = deepcopy(Zs)
    ndims = Xs[0].ndim
    if delete_nans_before_resampling:
        if ndims == 2:
            Xs = [x[~np.any(np.isnan(x), axis=1)] for x in Xs]
        elif ndims == 1:
            Xs = [x[~np.isnan(x)] for x in Xs]
        else:
            if any(np.isnan(x).any() for x in Xs):
                warnings.warn("Nans detected in the data. For ndim > 2 nan values have to be deleted manually.")
        compute_stat = statistic
    else:
        if ndims == 2:
            @njit
            def statistic_not_nans(x_t):
                x_t_pruned = x_t[~nb_any_axis1(np.isnan(x_t))]
                return statistic(x_t_pruned)
        elif ndims == 1:
            @njit
            def statistic_not_nans(x_t):
                x_t_pruned = x_t[~np.isnan(x_t)]
                return statistic(x_t_pruned)
        else:
            raise NotImplementedError("Only 1D and 2D arrays are supported.")
        compute_stat = statistic_not_nans

    if agg_func is None:
        agg_func = lambda x: x.mean(axis=0)

    sample_stat = agg_func(np.array([compute_stat(x_t) for x_t in Xs]))

    if isinstance(sample_stat, float):
        output_len = 1
    else:
        output_len = sample_stat.size


    def bootstrap_distribution(Xs):
        """
        Does not resample across time steps, since the target statistic is the average across the T prediction days (time steps are not exchangeable).
        """
        n_X = [len(x) for x in Xs]
        idxs_resampling_X = [np.random.randint(low=0, high=n, size=R*n).reshape(R, n) for n in n_X]

        out = np.empty((R+1, output_len))
        for i in tqdm(range(R)):
            stat_r = np.array([compute_stat(x[idxs_resampling_X[k][i]]) for k, x in enumerate(Xs)])
            out[i] = agg_func(stat_r)
        out[R] = sample_stat
        return out

    boot_sample = bootstrap_distribution(Xs)
    CI = bootstrap._compute_CI_percentile(boot_sample, alpha=alpha, alternative='two-sided')
    out = pd.concat([pd.Series(sample_stat, index=labels, name='sample_stat'),
                     pd.Series(list(CI), index=labels, name='CI')
                     ], axis=1)
    return out

def aggregate_CI_across_time_steps(df, statistic, labels, R=int(1e5), alpha=0.05, CI_method='bca'):
    """
    Ensure statistic handles NaN values!

    df: DataFrame with input data as columns and index levels: ['time_step', 'animal', optional: 'confidence'].
    statistic: function that computes the aggregate value across time steps and animals.
    """
    unique_values_per_index = [np.unique(df.index.get_level_values(level)) for level in df.index.names]
    full_index = pd.MultiIndex.from_product(unique_values_per_index, names=df.index.names)
    df_reindexed = df.copy().reindex(full_index, fill_value=np.nan)
    if 'confidence' in df.index.names:
        num_confidences = df.index.levels[-1].size
        X = np.stack(df_reindexed.groupby('time_step').apply(lambda x: x.values.reshape(-1, num_confidences, 2)), axis=1) # (N, T, C, F)
    else:
        X = np.stack(df_reindexed.groupby('time_step').apply(lambda x: x.values), axis=1) # (N, T, F)
    X = X.squeeze() # remove singleton dimension

    sample_stat = statistic(X)
    CI_computer = getattr(bootstrap, f'CI_{CI_method}')
    CI = CI_computer(X, statistic, R=R, alpha=alpha, alternative='two-sided')
    CI = np.atleast_2d(CI)
    out = pd.concat([pd.Series(sample_stat, index=labels, name='sample_stat'),
                     pd.Series(list(CI), index=labels, name='CI')
                     ], axis=1)
    return out

def join_results_by_delete_seed(func, delete_prob=0.45, delete_seeds=range(5), **kwargs):
    out = []
    for seed in tqdm(delete_seeds):
        df = func(delete_prob=delete_prob, delete_seed=seed, **kwargs)
        if not isinstance(df.index, pd.MultiIndex):
            df = df.stack().swaplevel().to_frame()
        unique_values_per_index = [np.unique(df.index.get_level_values(level)) for level in df.index.names]
        full_index = pd.MultiIndex.from_product(unique_values_per_index, names=df.index.names)
        df_reindexed = df.copy().reindex(full_index, fill_value=np.nan)
        if 'confidence' in df.index.names:
            num_confidences = df.index.levels[-1].size
            X = np.stack(df_reindexed.groupby('time_step').apply(lambda x: x.values.reshape(-1, num_confidences, 2)), axis=1) # (N, T, C, F)
        else:
            X = np.stack(df_reindexed.groupby('time_step').apply(lambda x: x.values), axis=1) # (N, T, F)
        X = X.squeeze()
        out.append(X)
    return np.stack(out, axis=-1)

def delete_seed_aggregate_CI(*, n, func, statistic, labels=None, delete_seeds=range(5), R=int(1e5), **kwargs):
    delete_prob = params.n_to_delete_prob[n]
    X = join_results_by_delete_seed(func, delete_prob=delete_prob, delete_seeds=delete_seeds, **kwargs)
    sample_stat = statistic(X)
    CI = bootstrap.CI_bca(X, statistic, R=R)
    if labels is None:
        index = [n]
    else:
        index = pd.MultiIndex.from_product([[n], labels], names=['n', 'target'])
    CI = np.atleast_2d(CI)
    out = pd.concat([pd.Series(sample_stat, index=index, name='sample_stat'),
                     pd.Series(list(CI), index=index, name='CI')
                     ], axis=1)
    return out

def merge_quantile_results(results_list):
    """
    Merges the results of multiple quantile results for single trajectories (ID) into a single dictionary.
    """
    print(f"Merging quantile results for {len(results_list)} IDs...")
    results_single_ID = results_list[0]
    is_imputation = 'future_only_variables' in results_single_ID

    get_max_length = lambda part: max(results_single_ID[f'out_{partition}'][f'{part}_lengths'].item() for partition in ['val', 'test'])

    max_encoder_length = get_max_length('encoder')
    if is_imputation:
        max_encoder_length = max(max_encoder_length, get_max_length('future'))
    max_prediction_length = get_max_length('decoder')
    num_encoder_vars = len(results_single_ID['encoder_variables'])
    num_decoder_vars = len(results_single_ID['decoder_variables'])
    num_static_vars = len(results_single_ID['static_variables'])

    def fill_gap(value, max_length, num_vars):
        """
        Fills the gap between the length of value and max_length by repeating the last observation. To be used to extend all encoder/decoder variables to the same length.
        """
        dims = np.array(value.shape)
        is_var_dim = dims == num_vars
        if dims.size == 2 and is_var_dim.any():
            return value
        elif not (dims == max_length).any():
            valid_dims = dims[~is_var_dim]
            closest = np.argmin(np.abs(valid_dims- max_length))
            dim_length = valid_dims[closest]
            dim_diff = max_length - dim_length
            # find dimension corresponding to the chosen valid dimension
            dim_idx = np.where(~is_var_dim)[0][closest]
            # repeat last observation until max_length
            last_observation = torch.index_select(value, dim_idx, torch.tensor(dim_length - 1))
            value = torch.cat([value] + [last_observation]*dim_diff, dim=dim_idx)
        return value

    def extend_variables(key, value):
        if 'lengths' in key:
            return value
        elif 'encoder' in key or 'future' in key:
            value = fill_gap(value, max_encoder_length, num_encoder_vars)
        elif 'decoder' in key:
            value = fill_gap(value, max_prediction_length, num_decoder_vars)
        return value

    as_tensor = lambda x: x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)

    def merge_dicts(dict_list, avoid_keys=['histogram', 'state_dict']):
        dict_list = deepcopy(dict_list)
        merged_dict = {}

        # Iterate through each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if not any(k in key for k in avoid_keys):
                    if key not in merged_dict:
                        if isinstance(value, torch.Tensor):
                            value = torch.atleast_2d(value)
                            value = extend_variables(key, value)
                        elif isinstance(value, np.ndarray):
                            value = np.atleast_2d(value)
                        elif isinstance(value, list):
                            if len(value) == 2:
                                value = [extend_variables(key, torch.atleast_2d(as_tensor(v))) for v in value]
                        merged_dict[key] = value
                    else:
                        if isinstance(value, list):
                            if len(value) == 2:
                                for i, v in enumerate(value):
                                    v = extend_variables(key, v)
                                    merged_dict[key][i] = torch.cat((merged_dict[key][i], v), dim=0)
                            else:
                                merged_dict[key].extend(value)
                        elif isinstance(value, torch.Tensor):
                            # atleast 2D
                            value = torch.atleast_2d(value)
                            value = extend_variables(key, value)
                            concat_dim = np.where(np.array(value.shape) == 1)[0][0]
                            merged_dict[key] = torch.cat((merged_dict[key], value), dim=concat_dim)
                        elif isinstance(value, np.ndarray):
                            # atleast 2D
                            value = np.atleast_2d(value)
                            concat_dim = np.where(np.array(value.shape) == 1)[0][0]
                            merged_dict[key] = np.concatenate((merged_dict[key], value), axis=concat_dim)
                        elif isinstance(value, dict):
                            merged_dict[key] = merge_dicts([merged_dict[key], value])
                        else:
                            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")

        return merged_dict

    def post_process_merged_dict(merged_dict):
        # Avoid repetition of variable names
        var_type_to_num = dict(encoder=num_encoder_vars, decoder=num_decoder_vars, static=num_static_vars)
        if is_imputation:
            var_type_to_num['future_only'] = num_encoder_vars
        for var_type, num_vars in var_type_to_num.items():
            merged_dict[f'{var_type}_variables'] = merged_dict[f'{var_type}_variables'][:num_vars]

        # Delete extra dim for 'encoder/decoder lengths'
        keys = ['encoder_lengths', 'decoder_lengths']
        if is_imputation:
            keys.append('future_lengths')
        def squeeze(d):
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and k in keys:
                    d[k] = v.squeeze()
                elif isinstance(v, dict):
                    squeeze(v)
            return
        squeeze(merged_dict)
        return merged_dict

    results = merge_dicts(results_list)
    results = post_process_merged_dict(results)
    return results

def training_step_cpu(self, batch, batch_idx):
    """
    Train on batch.
    """
    x, y = batch
    log, out = self.step(x, y, batch_idx)  # Assuming 'step' is defined in your model
    return log

def training_step_gpu(self, batch, batch_idx):
    """
    Train on batch.
    """
    x, (y, weight) = batch
    # Move inputs and targets to GPU with non_blocking=True
    def move_input_to_gpu(v):
        if isinstance(v, list):
            return [vv.cuda(non_blocking=True) for vv in v]
        elif isinstance(v, torch.Tensor):
            return v.cuda(non_blocking=True)
        else:
            return v

    x = {k: move_input_to_gpu(v) for k, v in x.items()}
    y = [yi.cuda(non_blocking=True) for yi in y]
    if weight is not None:
        weight = move_input_to_gpu(weight)
    log, out = self.step(x, (y, weight), batch_idx)
    return log

def extract_future_cds(results, x=None, partition=None):
    if x is None:
        assert partition is not None, "Either x or partition must be provided"
        x = results[f'x_{partition}']
    features = results['encoder_variables']
    future_cds = [features.index(f'future_{f}') for f in ['X', 'Y']]# correct way: dataset.reals_with_future.index('future_X'), however indices for future_X,Y coincide between results['encoder_variables'] and dataset.reals_with_future
    x_cont = x['encoder_cont']
    Xf, Yf = x_cont[..., future_cds].transpose(0, 2).transpose(1, 2)
    def rescale(Z, i):
        avg, sigma = x['target_scale'][i].T
        return Z*sigma[:, None] + avg[:, None]
    Xf = rescale(Xf, 0).numpy()
    Yf = rescale(Yf, 1).numpy()
    cds_future = np.stack((Xf, Yf), axis=0)
    return cds_future

def gap_distance(results, partition):
    """
    Great circle distance between the last observed location and the first future location.
    """
    x = results[f'x_{partition}']

    future, _ = compute_future_encoder(results, x=x, partition=partition)
    future_0 = future[..., 0]
    lat_f, lon_f = space.mercator_inv(*future_0.copy())

    encoder = stack_targets(x['encoder_target'])
    encoder_f = encoder[..., -1]
    lat, lon = space.mercator_inv(*encoder_f.copy())

    d = space.great_circle_distance(lat, lon, lat_f, lon_f)
    d = pd.Series(d, name='gap distance')
    return d

def gap_of_avg_locations(feature_avg):
    """
    Great circle distance between the average of past locations and the average of future locations.
    """
    XY = feature_avg[['X_loc', 'Y_loc']].values
    XY_f = feature_avg[['future_X_loc', 'future_Y_loc']].values
    lat, lon = space.mercator_inv(*XY.T.copy())
    lat_f, lon_f = space.mercator_inv(*XY_f.T.copy())
    d = space.great_circle_distance(lat, lon, lat_f, lon_f)
    d = pd.Series(d, name='gap distance (avg locations)')
    return d

def expand_sequences(x):
    y = np.stack(x.apply(lambda x: x.T), axis=0) # (n, time_steps, targets)
    x = {}
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            x[(i, j)] = y[i, j]
    x = pd.Series(x)
    return x
