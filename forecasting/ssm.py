import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from pathlib import Path
from copy import deepcopy
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
    from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)

import optuna
from tidypath.fmt import dict_to_id
from phdu import geometry, bootstrap, pd_utils, integration
from phdu import savedata, savefig
from phdu.plots.plotly_utils import CI_plot, CI_ss_plot, get_figure, plot_confidence_bands
from phdu.plots.base import color_std, plotly_default_colors
from .preprocessing import space
from . import params, custom_metrics

func_dir = Path(__file__).resolve().parent
root_dir = func_dir.parent


def load_ssm_data(model='rw', species='Southern elephant seal', partition='test', task='forecasting', sampling_freq=6, max_prediction_length=28,
                  chunk_as_animal=True,
                  end_time_step_margin=1, recompute_mercator=False):
    """
    recompute_mercator: bool NOT IMPLEMENTED
        If True, recompute the mercator coordinates from the lat,lon coordinates. This is because there is a discrepancy between the Y coordinates of aniMotum and the formula used. There is no discrepancy in X.

    chunk_as_animal: bool
        If True, each chunk is considered as a different animal, otherwise, the ID is used.
    """
    pdir = os.path.join(root_dir, f"data/forecasting_models/{species}/{model}")
    filename = f"test_test-partition-{partition}_{task}_pruned.csv"
    def _load():
        df = pd.read_csv(os.path.join(pdir, filename))
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
        return df
    if model == 'mp':
        try:
            df = _load()
        except:
            warnings.warn("Falling back to individual fits for model 'mp'.")
            df = load_individual(task=task, partition=partition, model='mp', species=species)
    else:
        df = _load() # allow errors to be raised

    df_real = pd.read_csv(os.path.join(root_dir,
                                       f"data/forecasting_models/{species}/pf_only/{filename}"))
    df_real = pd.DataFrame(df_real.reset_index().values[:, :-2],
                           columns = df_real.columns[:-1]) # delete geometry column containing vector of length 2
    df_real['date'] = pd.to_datetime(df_real['date']).dt.tz_localize('UTC')
    assert df_real['keep'].all(), 'Check preprocessed data (keep column)'

    if recompute_mercator:
        x, y = space.mercator(*(df[['lat', 'lon']] * np.pi/180).values.T)
        df['x'] = x
        df['y'] = y
        x_real, y_real = space.mercator(*(df_real[['lat', 'lon']].astype(float) * np.pi/180).values.T)
        df_real['x'] = x_real
        df_real['y'] = y_real

    df = df.set_index(['id', 'date'])
    df_real = df_real.set_index(['id', 'date'])
    df_real = df_real.rename(columns = dict(x='x_real', y='y_real'))
    cols = ['x_real', 'y_real', 'emf.x', 'emf.y']
    if 'lc' in df_real.columns:
        cols.append('lc')
    common_idx = df.index.intersection(df_real.index)
    df = pd.concat([df, df_real.loc[common_idx, cols]], axis=1).reset_index()
    # Set time step
    time_unit = pd.Timedelta(hours=sampling_freq)
    df = df.groupby('id', group_keys=False).apply(set_time_step, max_prediction_length=max_prediction_length, time_unit=time_unit)

    # add baseline (last observed loc)
    df_base = pd.read_csv(os.path.join(root_dir,
                                       f"data/forecasting_models/{species}/baseline/{filename}"))
    df_base = pd.DataFrame(df_base.reset_index().values[:, :-2],
                           columns = df_base.columns[:-1]) # delete geometry column containing vector of length 2
    df_base['date'] = pd.to_datetime(df_base['date']).dt.tz_localize('UTC')
    if task == 'imputation':
        first_half = np.arange(max_prediction_length // 2)
        last_half = np.arange(max_prediction_length // 2, max_prediction_length)
        def imputation_baseline(df_ID_chunk):
            """
            end_dt_margin: time steps allowed to be missing at the end of the trajectory
            """
            row_first = df_ID_chunk.iloc[0]
            row_last = df_ID_chunk.iloc[-1]

            # baseline for initial time: last observed location
            base = df_base.query('id == @row_first.id')
            # dates occurring 'sampling_freq' hours prior to the row.date
            close_dates = base['date'].apply(lambda x: x < row_first.date and ((x + pd.Timedelta(hours=sampling_freq+1)) > row_first.date))
            if close_dates.any():
                dt_0 = 0
                date_base_0 = base[close_dates]['date'].iloc[0]
            else:
                real_index = np.where(base.date == row_first.date)[0][0]
                date_base_0 = base.date.iloc[real_index - 1] # closest
                close_dates = base.date == date_base_0
                dt_0 = int(-1 + (row_first.date - date_base_0) / time_unit)
            x_base_0, y_base_0 = base[close_dates][['x', 'y']].mean()

            # baseline for final time: first future observed location
            close_dates = base['date'].apply(lambda x: x > row_last.date and ((x - pd.Timedelta(hours=sampling_freq+1)) < row_last.date))
            if close_dates.any():
                dt_f = 0
                date_base_f = base[close_dates]['date'].iloc[0]
            else:
                real_index = np.where(base.date == row_last.date)[0][0]
                date_base_f = base.date.iloc[real_index + 1] # closest
                close_dates = base.date == date_base_f
                dt_f = int(-1 + (date_base_f - row_last.date) / time_unit)
            x_base_f, y_base_f = base[close_dates][['x', 'y']].mean()

            df_ID_chunk.time_step += dt_0
            df_ID_chunk['dt_0'] = dt_0
            df_ID_chunk['dt_f'] = dt_f
            is_first_half = df_ID_chunk.time_step.isin(first_half)
            is_second_half = df_ID_chunk.time_step.isin(last_half)
            df_ID_chunk.loc[is_first_half, 'x_base'] = x_base_0
            df_ID_chunk.loc[is_first_half, 'y_base'] = y_base_0
            df_ID_chunk.loc[is_second_half, 'x_base'] = x_base_f
            df_ID_chunk.loc[is_second_half, 'y_base'] = y_base_f
            return df_ID_chunk
        df = df.groupby(['id', 'chunk_num']).apply(imputation_baseline)
        df = df[df.dt_f <= end_time_step_margin]
    else:
        if recompute_mercator:
            raise NotImplementedError
            # def baseline_pred(row):
            #     if row.time_step > 0: # later forward fill
            #         return np.nan
            #     else:
            #         df_ID = df_base.query('ID == @row.id')
            #         # dates occurring 'sampling_freq' hours prior to the row.date
            #         close_dates = df_ID['date'].apply(lambda x: x < row.date and ((x + pd.Timedelta(hours=sampling_freq)) > row.date))
            #         lat, lon = df_ID[close_dates][['lat', 'lon']].mean() * np.pi / 180
            #         x, y = space.mercator(lat, lon)
            #         return pd.Series(dict(x_base=x, y_base=y), name=row.name)
        else:
            def baseline_pred(df_real_ID):
                df_real_ID = df_real_ID.reset_index()
                row = df_real_ID.iloc[0]

                df_ID = df_base.query('id == @row.id').sort_values('date')
                # dates occurring 'sampling_freq' hours prior to the row.date
                close_dates = df_ID['date'].apply(lambda x: x < row.date and ((x + pd.Timedelta(hours=sampling_freq+1)) > row.date))
                if close_dates.any():
                    dt = 0
                    date_base = df_ID[close_dates]['date'].iloc[0]
                else:
                    real_index = np.where(df_ID.date == row.date)[0][0]
                    date_base = df_ID.date.iloc[real_index - 1]
                    close_dates = df_ID.date == date_base
                    dt = -1 + (row.date - date_base) / time_unit
                x, y = df_ID[close_dates][['x', 'y']].mean()
                out = pd.Series(dict(x_base=x, y_base=y, date_base=date_base, dt=dt)).to_frame().T
                out = pd.concat([out]*df_real_ID.shape[0], axis=0)
                return out
            # def baseline_pred(row): # DEPRECATED (failed when missing time_step 0)
            #     if row.time_step > 0: # later forward fill
            #         return np.nan
            #     else:
            #         df_ID = df_base.query('id == @row.id').sort_values('date')
            #         # dates occurring 'sampling_freq' hours prior to the row.date
            #         close_dates = df_ID['date'].apply(lambda x: x < row.date and ((x + pd.Timedelta(hours=sampling_freq+1)) > row.date))
            #         if close_dates.any():
            #             dt = 0
            #             date_base = df_ID[close_dates]['date'].values[0]
            #         else:
            #             real_index = np.where(df_ID.date == row.date)[0][0]
            #             date_base = df_ID.date.iloc[real_index - 1] # closest
            #             dt = -1 + (row.date - date_base) / time_unit
            #         x, y = df_ID[close_dates][['x', 'y']].mean()
            #         return pd.Series(dict(x_base=x, y_base=y, date_base=date_base, dt=dt), name=row.name)
            # def baseline_pred(row):
            #     if row.time_step > 0: # later forward fill
            #         return np.nan
            #     else:
            #         df_ID = df_base.query('id == @row.id')
            #         # dates occurring 'sampling_freq' hours prior to the row.date
            #         close_dates = df_ID['date'].apply(lambda x: x < row.date and ((x + pd.Timedelta(hours=sampling_freq+1)) > row.date))
            #         if not close_dates.any():
            #             # interpolate between real date and last observed location
            #             real = df_ID.date == row.date
            #             real_index = np.where(real)[0][0]
            #             close_dates = real.copy()
            #             close_dates.iloc[real_index - 1] = True
            #         date_base = df_ID[close_dates]['date'].values[0]
            #         x, y = df_ID[close_dates][['x', 'y']].mean()
            #         return pd.Series(dict(x_base=x, y_base=y, date_base=date_base), name=row.name)
        # baseline = df.apply(baseline_pred, axis=1) # ROW approach DEPRECATED (failed when missing time_step 0)
        # baseline = baseline.fillna(method='ffill')
        baseline = df.groupby('id').apply(baseline_pred)
        baseline = baseline.reset_index(drop=True)
        df = pd.concat([df, baseline], axis=1)
        df['time_step_prev'] = df.time_step.copy()
        df['time_step'] += df['dt'].astype(int)

    if task == 'imputation' and chunk_as_animal:
        df['id'] += '__' + df['chunk_num'].astype(str)
    df = df.groupby('id', group_keys=False).apply(lambda x: x[x.time_step < max_prediction_length]).reset_index(drop=True)
    # id to category
    df['id'] = df['id'].astype('category')
    df['animal'] = df['id'].cat.codes
    # numerical cols
    numerical = ['x', 'y', 'x_real', 'y_real', 'x_base', 'y_base']
    df[numerical] = df[numerical].astype(float)
    # Ensure no duplicates
    df = df.drop_duplicates()
    return df

def set_time_step(S, *, time_unit, max_prediction_length=28):
    S = S.sort_values('date')
    timesteps = S.date.diff().fillna(pd.Timedelta(0))
    if max_prediction_length is None:
        max_t = timesteps.max()
        threshold = max_t / 2
    else:
        threshold = pd.Timedelta(max_prediction_length/2, 'D')
    date_breaks = timesteps > threshold
    timesteps /= time_unit
    timesteps = timesteps.values

    sequence_idxs = np.unique(np.hstack((0, np.where(date_breaks)[0], len(S))))
    start = sequence_idxs[:-1]
    end = sequence_idxs[1:]
    dts = []
    chunk_num = []
    for i, (start_i, end_i) in enumerate(zip(start, end)):
        dt = timesteps[start_i:end_i]
        dt[0] = 0
        dts.append(dt.cumsum())
        chunk_num.append(np.repeat(i, len(dt)))

    dts = np.hstack(dts)
    chunk_num = np.hstack(chunk_num)
    if max_prediction_length is not None:
        dts[dts >= max_prediction_length] = np.nan

    S['time_step'] = dts
    S['chunk_num'] = chunk_num
    S = S.dropna(subset=['time_step'])
    S['time_step'] = S['time_step'].astype(int)
    return S

def load_individual(task='forecasting', partition='test', model='mp', species='Southern elephant seal', pred_only=True):
    pdir = f"data/forecasting_models/{species}/{model}/single_ID"
    df = []
    for file in os.listdir(pdir):
        if task in file and partition in file:
            df.append(pd.read_csv(os.path.join(pdir, file)))
    df = pd.concat(df, axis=0, ignore_index=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
    if pred_only:
        filename = f"test_test-partition-{partition}_{task}_pruned.csv"
        pdir_rw = os.path.join(root_dir, f"data/forecasting_models/{species}/rw")
        rw = pd.read_csv(os.path.join(pdir_rw, filename))
        rw['date'] = pd.to_datetime(rw['date']).dt.tz_localize('UTC')
        valid = rw.set_index(['id', 'date']).index
        df = df.set_index(['id', 'date'])
        df = df.loc[valid.intersection(df.index)].reset_index()
    return df

def confidence_ellipse_from_se(se_x, se_y, alpha=0.05):
    """
    Determine the semiaxes for a confidence ellipse based on the standard errors of `x` and `y`.

    Parameters
    ----------
    se_x : float or array-like
        Standard error of `x`.
    se_y : float or array-like
        Standard error of `y`.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    Lengths of the semiaxes of the confidence ellipse.
    """
    from scipy.stats import chi2

    df = 2
    chi2_val = chi2.ppf(1 - alpha, df)
    multiplyer = np.sqrt(chi2_val)
    a = se_x * multiplyer
    b = se_y * multiplyer
    return a, b

def confidence_interval_from_se(se, alpha=0.05):
    """
    Determine the confidence interval based on the standard error.

    Parameters
    ----------
    se : float or array-like
        Standard error.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    Lengths of the lower and upper bounds of the confidence interval (interval semi-widths).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    return se * z

def is_point_within_ellipse(x, y, a, b, c_x, c_y):
    """
    Determine if a point (x, y) is within an ellipse with semiaxes a and b.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
    y : float
        y-coordinate of the point.
    a : float
        Semiaxis a.
    b : float
        Semiaxis b.
    c_x : float
        x-coordinate of the ellipse center.
    c_y : float
        y-coordinate of the ellipse center.

    Returns
    -------
    bool
        True if the point is within the ellipse, False otherwise.
    """
    return ((x - c_x) / a)**2 + ((y - c_y) / b)**2 <= 1

def ellipse_area(a, b):
    return np.pi * a * b


def eval_coverage(row, alpha=0.05):
    a, b = confidence_ellipse_from_se(row['x.se'], row['y.se'], alpha=alpha)
    return is_point_within_ellipse(row['x_real'], row['y_real'], a, b, row['x'], row['y'])

def eval_area(row, alpha=0.05):
    # TODO: replace it by area in km
    a, b = confidence_ellipse_from_se(row['x.se'], row['y.se'], alpha=alpha)
    return ellipse_area(a, b)

def area_ellipse_km(row, alpha=0.05, error=1e-5, **kwargs):
    c_y = row['y']
    a, b = confidence_ellipse_from_se(row['x.se'], row['y.se'], alpha=alpha)
    y0 = c_y - b
    y1 = c_y + b
    solid_angle = integration.definite_integral(custom_metrics.solid_angle_integrand_ellipse,
                                                y0, y1, # integration limits
                                                a, b, c_y, # params for the integrand
                                                target_error=error, **kwargs)
    area = solid_angle * params.R_earth**2
    return area


def eval_Q_area(df,*, task, ref_method='cumulative-by-direction', ref_double=False, ref_alpha=0.05, ref_R=int(1e4)):
    # y_real = df.groupby('id').apply(lambda s: s[['x_real', 'y_real']].values.T)
    y_base = df.groupby('id').apply(lambda s: s[['x_base', 'y_base']].values.T)
    days = custom_metrics.time_step_to_days(df['time_step'].values)
    A_max = custom_metrics.maximal_area(task=task, baseline=y_base, days=days)
    if isinstance(A_max, pd.Series):
        A_max = np.hstack(A_max)

    # y_real_base = pd_utils.tuple_wise(y_real, y_base)
    # A_ref = y_real_base.apply(custom_metrics.compute_reference_area, task=task, method=ref_method, double=ref_double, alpha=ref_alpha, R=ref_R)
    # df['area_ref'] = np.hstack(A_ref)

    t = np.arange(df.time_step.max() + 1)
    def extract_filled_trajectory(s, col_type='real'):
        cols = [f'x_{col_type}', f'y_{col_type}']
        s = s[cols + ['time_step']]
        s = s.set_index('time_step').reindex(t)
        # fill missing values with last observed
        s = s.fillna(method='ffill')
        s = s.fillna(method='bfill')
        return s[cols].values.T

    def get_missing_positions(s):
        time = s[['time_step', 'x_real']]
        time = time.set_index('time_step').reindex(t)
        return time['x_real'].isna().values

    y_real_filled = df.groupby('id').apply(extract_filled_trajectory, col_type='real')
    y_base_filled = df.groupby('id').apply(extract_filled_trajectory, col_type='base')
    y_real_base_filled = pd_utils.tuple_wise(y_real_filled, y_base_filled)
    missing = df.groupby('id').apply(get_missing_positions)

    df['area_max'] = A_max
    for c in params.confidences:
        alpha = 1 - c
        A = y_real_base_filled.apply(custom_metrics.compute_reference_area, task=task, alpha=alpha)
        A = pd_utils.tuple_wise(A, missing).apply(lambda x: x[0][~x[1]])
        df[f'area_ref_{c}'] = np.hstack(A.values)
        area_data = df[[f'area_{c}', f'area_ref_{c}', 'area_max']].values.astype(float).T
        df[f'Q_area_{c}'] = custom_metrics.area_quality(*area_data)
    return df

def optimize_se_mpl(df, alpha=0.05, optimize_var=False, num_trials=200, min_mpl=0.01, max_mpl=2, separate_mpl=False):
    confidence = 1 - alpha

    hp_space = {'x.se': (min_mpl, max_mpl),
                'y.se': (min_mpl, max_mpl)}

    def aux(trial):
        df_i = df.copy()
        if separate_mpl:
            for col in ['x.se', 'y.se']:
                df_i[col] *= trial.suggest_uniform(col, *hp_space[col])
        else:
            mpl = trial.suggest_uniform('mpl', *hp_space['x.se'])
            df_i[['x.se', 'y.se']] *= mpl
        df_i[f'coverage_{confidence}'] = df_i.apply(eval_coverage, alpha=alpha, axis=1)
        error = df_i.groupby('time_step')[f'coverage_{confidence}'].mean() - confidence
        rmse = np.sqrt((error**2).mean())
        return error, rmse

    if optimize_var:
        def objective(trial):
            error, rmse = aux(trial)
            error_var = error.var()
            return rmse, error_var
        study = optuna.create_study(directions=['minimize', 'minimize'])
    else:
        def objective(trial):
            return aux(trial)[1]
        study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)
    return study

@savedata
def se_mpl_val(task='forecasting', optimize_var=False, num_trials=500, separate_mpl=False, model='rw', by_lc=False, recompute_mercator=False, chunk_as_animal=True, **kwargs):
    df = load_ssm_data(partition='val', task=task, model=model, recompute_mercator=recompute_mercator, chunk_as_animal=chunk_as_animal, **kwargs)
    min_mpl = 0.1
    model_to_mpl = {'forecasting': dict(rw=2.5, crw=1, mp=1.5), # crw tends to overestimate the standard errors
                    'imputation': dict(rw=1, crw=1, mp=1), # all models tend to overestimate the standard errors
                    }
    if recompute_mercator:
        max_mpl = 80
        min_mpl = 1
    else:
        max_mpl = model_to_mpl[task][model]

    se_mpl = {}
    has_lc = 'lc' in df.columns
    if by_lc and has_lc:
        for lc, df_lc in tqdm(df.groupby('lc')):
            for confidence in params.confidences:
                alpha = 1 - confidence
                study = optimize_se_mpl(df_lc, alpha=alpha, optimize_var=optimize_var, num_trials=num_trials, separate_mpl=separate_mpl, max_mpl=max_mpl, min_mpl=min_mpl)
                best = study.best_trials[0]
                if separate_mpl:
                    se_mpl[(lc, confidence, 'x.se')] = best.params['x.se']
                    se_mpl[(lc, confidence, 'y.se')] = best.params['y.se']
                else:
                    se_mpl[(lc, confidence, 'mpl')] = best.params['mpl']
                se_mpl[(lc, confidence, 'rmse')] = best.values[0]
                if optimize_var:
                    se_mpl[(lc, confidence, 'var')] = best.values[1]
        se_mpl = pd.Series(se_mpl).unstack()
        se_mpl.index.names = ['lc', 'confidence']
    else:
        for confidence in params.confidences:
            alpha = 1 - confidence
            study = optimize_se_mpl(df.copy(), alpha=alpha, optimize_var=optimize_var, num_trials=num_trials, separate_mpl=separate_mpl, max_mpl=max_mpl)
            best = study.best_trials[0]
            if separate_mpl:
                se_mpl[(confidence, 'x.se')] = best.params['x.se']
                se_mpl[(confidence, 'y.se')] = best.params['y.se']
            else:
                se_mpl[(confidence, 'mpl')] = best.params['mpl']
            se_mpl[(confidence, 'rmse')] = best.values[0]
            if optimize_var:
                se_mpl[(confidence, 'var')] = best.values[1]
        se_mpl = pd.Series(se_mpl).unstack()
        se_mpl.index.name = 'confidence'
    return se_mpl

def se_mpl_val_summary(**kwargs):
    """
    Return the RMSE for all combinations of parameters for adjusting CI_expansion.
    """
    rmse = {}
    for by_lc in [True, False]:
        for optimize_var in [True, False]:
            for model in params.ssm_models:
                for separate_mpl in [True, False]:
                    df = se_mpl_val(model=model, optimize_var=optimize_var, separate_mpl=separate_mpl, by_lc=by_lc, **kwargs)
                    rmse[(model, by_lc, separate_mpl, optimize_var)] = df['rmse'].mean()
    rmse = pd.Series(rmse).unstack(0)
    rmse.index.names = ['by_lc', 'separate_mpl', 'optimize_var']
    return rmse

def eval_ssm(task='forecasting', model='rw', se_val_fit='best', area_method='km', partition='test', by_lc=False, optimize_var=False, separate_mpl=False, recompute_mercator=False,
             ref_method='cumulative-by-direction', ref_double=False, ref_alpha=0.05, ref_R=int(1e4),
             **kwargs):
    """
    Evaluate the SSM model in terms of coverage and area

    area_method: str
        - 'km': area in km^2
        - None: unreliable (product of mercator coordinates)
    """
    area_kwargs = dict(ref_method=ref_method, ref_double=ref_double, ref_alpha=ref_alpha, ref_R=ref_R) # DEPRECATED
    if se_val_fit == 'best':
        df_val_fit = eval_ssm(task=task, model=model, area_method=area_method, se_val_fit=True, partition=partition, recompute_mercator=recompute_mercator, **area_kwargs, **kwargs)
        df_no_fit = eval_ssm(task=task, model=model, area_method=area_method, se_val_fit=False, partition=partition, recompute_mercator=recompute_mercator, **area_kwargs, **kwargs)
        def eval_model(df):
            coverage_cols = [col for col in df if col.startswith('coverage')]
            df[coverage_cols] = df[coverage_cols].astype(int).astype(float)
            pred = df[coverage_cols].mean()
            target = np.array([float(i.split("_")[1]) for i in pred.index])
            error = (pred - target).abs().mean()
            return error
        error_val_fit = eval_model(df_val_fit)
        error_no_fit = eval_model(df_no_fit)
        if error_val_fit < error_no_fit:
            print(f"Best {model} model: se_val_fit=True")
            return df_val_fit
        else:
            print(f"Best {model} model: se_val_fit=False")
            return df_no_fit
    else:
        if area_method == 'km':
            area_computer = area_ellipse_km
        elif area_method is None:
            area_computer = eval_area
        else:
            raise ValueError(f"area_method={area_method} not recognized.")
        df = load_ssm_data(task=task, model=model, partition=partition, recompute_mercator=recompute_mercator, **kwargs)
        has_lc = 'lc' in df.columns
        if se_val_fit:
            val_fit = se_mpl_val(task=task, model=model, optimize_var=optimize_var, separate_mpl=separate_mpl, by_lc=by_lc, recompute_mercator=recompute_mercator)
            for confidence in params.confidences:
                df_c = df.copy()
                if separate_mpl:
                    if by_lc and has_lc:
                        lc_to_mpl_x = val_fit.query("confidence == @confidence")['x.se']
                        lc_to_mpl_y = val_fit.query("confidence == @confidence")['y.se']
                        df_c['x.se'] *= df_c['lc'].map(lc_to_mpl_x)
                        df_c['y.se'] *= df_c['lc'].map(lc_to_mpl_y)
                    else:
                        df_c['x.se'] *= val_fit.loc[confidence, 'x.se']
                        df_c['y.se'] *= val_fit.loc[confidence, 'y.se']
                else:
                    if by_lc and has_lc:
                        lc_to_mpl = val_fit.query("confidence == @confidence")['mpl'].droplevel(1)
                        df_c['mpl'] = df_c['c'].map(lc_to_mpl)
                    else:
                        df_c['mpl'] = val_fit.loc[confidence, 'mpl']
                    df_c['x.se'] *= df_c['mpl']
                    df_c['y.se'] *= df_c['mpl']
                alpha = 1 - confidence
                df[f'coverage_{confidence}'] = df_c.apply(eval_coverage, alpha=alpha, axis=1)
                df[f'area_{confidence}'] = df_c.apply(area_computer, alpha=alpha, axis=1)
        else:
            for confidence in params.confidences:
                alpha = 1 - confidence
                df[f'coverage_{confidence}'] = df.apply(eval_coverage, alpha=alpha, axis=1)
                df[f'area_{confidence}'] = df.apply(eval_area, alpha=alpha, axis=1)
        # add area quality
        df = eval_Q_area(df, task=task)#, **area_kwargs)
        return df

def reformat_CI_dataframe(df):
    # Reset the index to include 'timestep' and 'confidence' as columns
    df_reset = df.reset_index()

    # Rename 'sample_stat' to 'mean'
    df_reset = df_reset.rename(columns={'sample_stat': 'mean'})

    # Melt the DataFrame to make 'CI' and 'mean' values in a single column with corresponding stat type
    df_melted = df_reset.melt(id_vars=['time_step', 'confidence'], value_vars=['CI', 'mean'], var_name='stat', value_name='value')

    # Pivot the DataFrame to get 'confidence' and 'stat' in columns
    df_pivoted = df_melted.pivot_table(index='time_step', columns=['confidence', 'stat'], values='value')

    # Flatten the multi-index columns
    df_pivoted.columns = [f"{col[0]}_{col[1]}" for col in df_pivoted.columns]

    # Reset the column names to the desired format
    def process_col(col):
        confidence, stat = col.split('_')
        return (float(confidence), stat)
    df_pivoted.columns = pd.MultiIndex.from_tuples([process_col(col) for col in df_pivoted.columns], names=['confidence', 'stat'])

    # make index start at 0
    df_pivoted = df_pivoted.reset_index(drop=True)

    return df_pivoted

def reformat_df(df, magnitudes, to_float=True):
    """
    From a df with columns of the form {magnitude}_{confidence}, return a df with index 'animal', 'time_step', 'confidence', and the corresponding magnitude values.
    """
    results = {}
    for magnitude in magnitudes:
        for confidence in params.confidences:
            for i, value in enumerate(df[f'{magnitude}_{confidence}'].values):
                results[(confidence, magnitude, i)] = value

    df_ref = pd.Series(results).unstack(1).droplevel(-1)
    df_ref.index.name = 'confidence'
    num_confidences = len(params.confidences)
    df_ref['animal'] = np.hstack([df['animal'].values]*num_confidences)
    df_ref['time_step'] = np.hstack([df['time_step'].values]*num_confidences)
    df_ref = df_ref.reset_index().set_index(['time_step', 'animal', 'confidence'])
    if to_float:
        df_ref = df_ref.astype(float)
    df_ref = df_ref.sort_index()
    return df_ref

@savedata
def eval_ssm_CI(task='forecasting', magnitude='coverage', model='rw', se_val_fit="best", area_method='km', chunk_as_animal=True, **loading_kwargs):
    df = eval_ssm(task=task, model=model, se_val_fit=se_val_fit, area_method=area_method, chunk_as_animal=chunk_as_animal, **loading_kwargs)

    @njit
    def nb_mean(x):
        return x.mean()

    results = {}
    for t, df_t in tqdm(df.groupby('time_step')):
        for confidence in [0.5, 0.9, 0.95]:
            x = df_t[f'{magnitude}_{confidence}'].values.astype(float)
            results[(t, confidence, 'sample_stat')] = nb_mean(x)
            results[(t, confidence, 'CI')] = bootstrap.CI_bca(x, nb_mean, alpha=0.05, R=int(1e4))
    results = pd.Series(results).unstack()
    results.columns.name = magnitude
    results.index.names = ['time_step', 'confidence']
    results = reformat_CI_dataframe(results)
    return results

def point_prediction_errors(metric='great_circle_distance', **kwargs):
    df = load_ssm_data(**kwargs)
    if metric == 'great_circle_distance':
        df[['lat_rad', 'lon_rad']] = np.vstack(space.mercator_inv(*df[['x', 'y']].values.T)).T
        df[['lat_real', 'lon_real']] = np.vstack(space.mercator_inv(*df[['x_real', 'y_real']].values.T.astype(float))).T
        df[metric] = space.great_circle_distance(*df[['lat_rad', 'lon_rad', 'lat_real', 'lon_real']].values.T)
        df[metric] *= params.R_earth
    elif metric == 'MAE':
        df['x.MAE'] = (df['x'] - df['x_real']).abs()
        df['y.MAE'] = (df['y'] - df['y_real']).abs()
        df['MAE'] = df[['x.MAE', 'y.MAE']].mean(axis=1)
    elif metric == 'RMSE':
        raise NotImplementedError
    else:
        raise ValueError(f"metric={metric} not recognized.")
    data = df.groupby('time_step')[metric].apply(np.array).to_frame()
    data = pd_utils.expand_sequences(data).T.loc[metric]
    data.index.name = 'animal'
    return data
