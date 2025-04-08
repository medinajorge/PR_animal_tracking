import numpy as np
import pandas as pd
import math
import torch
from pytorch_forecasting.utils import create_mask
import optuna
from numba import njit
from scipy.interpolate import interp1d
try:
    from shapely.geometry import Polygon
except:
    pass
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import plotly.graph_objects as go
import re
from collections import defaultdict
from typing import Dict, List
import contextlib
from PIL import Image
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

from tidypath.fmt import dict_to_id
from phdu import SavedataSkippedComputation, geometry, bootstrap, pd_utils
from phdu import savedata, savefig
from phdu.plots.plotly_utils import CI_plot, CI_ss_plot, get_figure, plot_confidence_bands, mod_logaxes_expfmt, get_subplots, mod_range, fig_base_layout
from phdu.plots.base import color_std, plotly_default_colors
from phdu.script_fmt import parse_optuna_output
from pytorch_forecasting.metrics import QuantileLoss
from . import model, distribution, params, ssm, custom_metrics
from .custom_metrics import nb_mean, nb_median, time_step_to_days, rae, arae, aggregate_metric_CI, aggregate_CI_summary
from .preprocessing import space, load

class OptimalHyperparameters:
    """
    Summarizes the optimal hyperparameters for the model given by 'mode'.
    """
    def __init__(self, seeds=range(6), verbose=0, include_stdin=True, **kwargs):
        if 'quantiles' in kwargs and kwargs['quantiles'] == 'exact':
            del kwargs['quantiles']
            warnings.warn("Exact quantiles is the default. Ignoring 'quantiles' argument.")
        self.seeds = seeds
        self.kwargs = kwargs
        self.include_stdin = include_stdin
        self.optim_params, self.unloaded_seeds = self.optimal_hyperparameters_summary(verbose=verbose)

    def optimal_hyperparameters_summary(self, verbose=0):
        optim_params = []
        unloaded_seeds = []
        for seed in self.seeds:
            hp_seed = optimal_hyperparameters(seed=seed, **self.kwargs, skip_seed_computation=True)
            if isinstance(hp_seed, SavedataSkippedComputation):
                unloaded_seeds.append(seed)
            else:
                optim_params.append(hp_seed.trials_dataframe().assign(seed=seed))
        if verbose:
            print(f"HP seeds not loaded: {unloaded_seeds}")

        if optim_params:
            optim_params = pd.concat(optim_params)
        else:
            optim_params = pd.DataFrame()

        if self.include_stdin:
            mid_rmse = self.kwargs.get('mid_rmse', False)
            if not mid_rmse:
                store_missing_idxs = self.kwargs.get('store_missing_idxs', False)
                quantiles = self.kwargs.get('quantiles', 'exact')
                s_q = self.kwargs.get('s_q', 1)
                task = self.kwargs.get('task', 'forecasting')
                cds = self.kwargs.get('cds', 'mercator')
                monotonic_q = self.kwargs.get('monotonic_q', False)
                add_z = self.kwargs.get('add_z', None)
                stdin_kwargs = dict(task=task, store_missing_idxs=store_missing_idxs, quantiles=quantiles, s_q=s_q, cds=cds, monotonic_q=monotonic_q, add_z=add_z)
                stdin_params = optimal_hp_from_stdin(**stdin_kwargs, skip_computation=True)
                if isinstance(stdin_params, SavedataSkippedComputation):
                    if verbose:
                        print("stdin params not loaded.")
                else:
                    print(f"Stdin params loaded for:\n{stdin_kwargs}")
                    if not 'params_dropout' in stdin_params.columns:
                        stdin_params['params_dropout'] = np.nan
                    stdin_params = stdin_params.assign(params_store_missing_idxs=store_missing_idxs) # Not stored but program was run with that argument
                    optim_params = pd.concat([optim_params, stdin_params], axis=0, ignore_index=True)
            else:
                print("stdin params not loaded. Only available for quantiles='exact' and mid_rmse=False.")
        optim_params = optim_params.set_index('seed')
        optim_params = optim_params.sort_values('value', ascending=True)
        return optim_params, unloaded_seeds

    def best(self, idxs=range(5)):
        if isinstance(idxs, int):
            idxs = [idxs]
        result = []
        for idx in idxs:
            S = self.optim_params.sort_values('value').iloc[idx]
            value = S.value
            params = {k.split("params_")[1]: v for k, v in S.items() if k.startswith("params_")}
            result.append((value, params))
        return result

def optimal_hyperparameters(timeout=17.0, prune_memory_error=False, extend=False, skip_seed_computation=False,
                            mode='quantile', seed=0, weather='all', species='Southern elephant seal', cds='mercator', task='forecasting', imputation_old=False, **kwargs):
    """
    R = iteration (optuna study)
    """
    args = {**locals(), **kwargs}
    del args['kwargs']
    del args['skip_seed_computation']
    del args['extend']

    @savedata('all-timeout-prune_memory_error-study')
    def _optimal_hyperparameters(study=None, timeout=17.0, prune_memory_error=False, R=0,
                                mode='quantile', seed=0, weather='all', species='Southern elephant seal', cds='mercator', task='forecasting', imputation_old=False, **kwargs):
        """
        Returns the optimal hyperparameters for the model given by 'mode'.
        """
        timeout *= 3600 # convert to seconds
        if mode == 'distribution':
            forecaster = model.DistributionForecaster
        elif mode == 'quantile':
            forecaster = model.QuantileForecaster
        else:
            raise ValueError(f"Unknown mode: {mode}. Available modes: 'distribution', 'quantile'.")
        model_keys = ['cumulative', 'decoder_missing_zero_loss', 'mid_rmse', 'mid_weight', 'quantiles', 's_q', 'monotonic_q']
        model_kwargs = {}
        for k in model_keys:
            if k in kwargs:
                model_kwargs[k] = kwargs.pop(k)

        target = kwargs.pop('target', 'cds')

        tft = forecaster(load_data=False, seed=seed, cds=cds, task=task, deprecated=imputation_old, target=target, **model_kwargs)
        study = tft.optimize_hyperparameters(study=study, timeout=timeout, prune_memory_error=prune_memory_error, target=target,
                                             weather=weather, species=species, cds=cds, task=task, **kwargs)
        return study

    # Automatically load the latest error
    R = 0
    study = _optimal_hyperparameters(**args, R=R, skip_computation=True)
    if isinstance(study, SavedataSkippedComputation):
        return _optimal_hyperparameters(**args, R=R, skip_computation=skip_seed_computation)
    else:
        # Find latest study
        while not isinstance(study, SavedataSkippedComputation):
            R += 1
            study = _optimal_hyperparameters(**args, R=R, skip_computation=True)
        R -= 1 # last found study
        study = _optimal_hyperparameters(**args, study=study, R=R)
        if extend:
            print(f"Existing study found for R={R}. Extending it.")
            return _optimal_hyperparameters(**args, study=study, R=R+1)
        else:
            print(f"Existing study found for R={R}")
            return study

@savedata(ext='csv', index=False, load_opts_default_save=False)
def optimal_hp_from_stdin(task='forecasting', store_missing_idxs=True, quantiles='exact', s_q=1, cds='mercator', monotonic_q=False, add_z=None):
    task_str = task.replace('ing', '') + '_optim'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(root_dir, 'nuredduna_scripts/stdin/opt_hp')
    store_missing_str = 'M-true' if store_missing_idxs else 'M-false'
    opt_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('err') and task_str in f and store_missing_str in f]
    quantile_str = 'q-all'
    s_q_str = f'Q-{s_q}'
    if quantiles == 'all':
        opt_files = [f for f in opt_files if quantile_str in f and s_q_str in f]
    else:
        opt_files = [f for f in opt_files if quantile_str not in f]
    if monotonic_q:
        monotonic_q_str = 'm-true'
        opt_files = [f for f in opt_files if monotonic_q_str in f]
    if cds != 'mercator':
        cds_str = f'c-{cds}'
        opt_files = [f for f in opt_files if cds_str in f]

    if add_z is not None:
        add_z_str = 'z-true' if add_z else 'z-false'
        opt_files = [f for f in opt_files if add_z_str in f]

    df = pd.concat([parse_optuna_output(f).assign(seed=i) for i, f in enumerate(opt_files)], axis=0)
    df = df.sort_values('value', ascending=True)
    df = df.assign(computed_from='stdin')
    return df

def train_and_store_distribution(epochs=100, limit_train_batches=0.25, gradient_clip_val=0.8, weather='all', species='Southern elephant seal', num_mixtures=1, **model_kwargs):
    model_kwargs_full = dict(weather=weather, species=species, num_mixtures=num_mixtures, **model_kwargs)
    train_kwargs = dict(epochs=epochs, limit_train_batches=limit_train_batches, gradient_clip_val=gradient_clip_val)
    label_kwargs = {**model_kwargs_full, **train_kwargs}

    # check if model has already been trained
    try:
        output = store_distribution_parameters(**label_kwargs)
    except:
        warnings.warn(f"Model has not been trained yet. Training now.")
        tft = model.DistributionForecaster(**model_kwargs_full)
        tft.train(**train_kwargs)
        output = store_distribution_parameters(tft=tft, **label_kwargs)

    # store also hull intervals
    hull_PR = hull_prediction(**label_kwargs)
    return output, hull_PR, label_kwargs

def _get_label_kwargs(params_idx=0, cds='mercator', **kwargs):
    model_specs = params.distribution_best_params[cds][params_idx][1].copy()
    num_mixtures = model_specs.pop('num_mixtures')
    gradient_clip_val = model_specs.pop('gradient_clip_val')

    args = dict(model_specs=model_specs, num_mixtures=num_mixtures, gradient_clip_val=gradient_clip_val)
    args.update(kwargs)
    if 'model_specs' in kwargs and not kwargs['model_specs']:
        args.pop('model_specs')
    if cds == 'spherical':
        args['cds'] = 'spherical'
    *_, label_kwargs = train_and_store_distribution(**args)
    return label_kwargs

def get_hp(params_idx=0, task='forecasting', cds='mercator', mode='quantile', verbose=0, **kwargs):
    assert isinstance(params_idx, int), "params_idx should be an integer."
    allowed_kwargs = getattr(params, f'{task}_extra_kwargs')
    op_hp_specs = {k: kwargs[k] for k in set(kwargs).intersection(allowed_kwargs)}
    if 'quantiles' in op_hp_specs and op_hp_specs['quantiles'] == 'exact':
        del op_hp_specs['quantiles']
    optq = None
    try:
        optq = OptimalHyperparameters(task=task, cds=cds, mode=mode, verbose=verbose, **op_hp_specs)
    except:
        print(f"Could not load optimal hyperparameters for specs: {op_hp_specs}. Trying excluding {params.imputation_extra_kwargs_exclude}.")
        op_hp_specs = {k: v for k, v in op_hp_specs.items() if k not in params.imputation_extra_kwargs_exclude}
        try:
            optq = OptimalHyperparameters(task=task, cds=cds, mode=mode, verbose=verbose, **op_hp_specs)
        except:
            print("Could not load optimal hyperparameters. Trying loading from params file.")
            hp = getattr(params, f'{mode}_best_params')[task][cds][params_idx][1].copy()
    if optq is not None:
        hp = optq.best(params_idx)[0][1].copy()

    for p in ['lstm_layers', 'max_train_days']:
        if p in hp:
            if math.isnan(hp[p]):
                del hp[p]
            else:
                hp[p] = int(hp[p])
    for p in ['store_missing_idxs']:
        if p in hp and math.isnan(hp[p]):
            del hp[p]
    if math.isnan(hp['dropout']):
        hp['dropout'] = 0.1 # default value
    return hp

def quantile_results(params_idx=0, ID=None, cds='mercator', verbose=0, eval_mode=None, task='forecasting',
                     mod_hp={},
                     delete_prob=None,
                     delete_seed=0,
                     return_path=False,
                     **kwargs):
    if ID == 'all':
        return quantile_results_multiple_IDs(cds=cds, verbose=verbose, task=task, **kwargs)
    elif eval_mode is not None:
        assert ID is None, "ID should not be passed when eval_mode is passed."
        return quantile_results_eval_mode(params_idx=params_idx, cds=cds, eval_mode=eval_mode, task=task, **kwargs)
    else:
        if ID is not None: # perform it AFTER training the model in all the dataset, to know what params are the best overall
            if params_idx == 'best':
                allowed_kwargs = getattr(params, f'{task}_extra_kwargs')
                extra_kwargs = {k: kwargs[k] for k in set(kwargs).intersection(allowed_kwargs)}
                params_idx, _ = quantile_best_model(ID=ID, cds=cds, task=task, **extra_kwargs)
            kwargs['ID'] = ID
            kwargs['epochs'] = 5000 # ~ 7 hours
            kwargs['limit_train_batches'] = None
            pretrained = kwargs.get('pretrained', False)
            if pretrained:
                kwargs['patience'] = 80 # ~ 10 min
            else:
                kwargs['patience'] = 40 # ~ 5 min

        if delete_prob is not None:
            if params_idx == 'best':
                # Use best parameters for the original model
                allowed_kwargs = getattr(params, f'{task}_extra_kwargs')
                extra_kwargs = {k: kwargs[k] for k in set(kwargs).intersection(allowed_kwargs)}
                params_idx, _ = quantile_best_model(cds=cds, task=task, **extra_kwargs)
            kwargs['epochs'] = 5000 # let patience decide

        if params_idx is not None: # getting specs from params_idx
            hp_kwargs = kwargs.copy()
            if mod_hp:
                print(f"Modifying hyperparameters with: {mod_hp}")
            hp_kwargs.update(mod_hp)
            for k in getattr(params, f'{task}_extra_kwargs'):
                if k in hp_kwargs and not hp_kwargs[k] and k != 'add_z':
                    del hp_kwargs[k]
                    print(f"Excluding {k} from hyperparameters input (Excluding False).")
            # EXclude hp_kwargs with value 'delete'
            hp_kwargs = {k: v for k, v in hp_kwargs.items() if not isinstance(v, str) or v != 'delete'}
            model_specs = get_hp(params_idx=params_idx, task=task, cds=cds, mode='quantile', verbose=verbose, **hp_kwargs)
            if verbose:
                print(f"Model specs:\n{model_specs}")
            other_specs = dict(gradient_clip_val=model_specs.pop('gradient_clip_val'))
            data_params = ['max_train_days']
            for p in data_params:
                if p in model_specs:
                    other_specs[p] = model_specs.pop(p)
            if 'store_missing_idxs' in model_specs:
                del model_specs['store_missing_idxs'] # it is passed as a kwarg
        else: # passing directly the model_specs
            model_specs = kwargs.pop('model_specs', {})
            if verbose:
                print(f"Model specs:\n{model_specs}")
            other_specs = dict(gradient_clip_val=kwargs.pop('gradient_clip_val', 0.8))

        if task == 'imputation': # else not needed
            kwargs['task'] = 'imputation'
        if delete_prob is not None:
            kwargs['delete_prob'] = delete_prob
            kwargs['delete_seed'] = delete_seed

        if verbose:
            print(f"Input model specs:\n{model_specs}")
            print(f"Input other specs:\n{other_specs}")
            print(f"Input kwargs:\n{kwargs}")

        overwrite = kwargs.get('overwrite', False)
        if overwrite or kwargs.get('skip_computation', False):
            results = store_quantile_parameters(model_specs=model_specs, cds=cds, return_path=return_path, **other_specs, **kwargs)
            if isinstance(results, SavedataSkippedComputation) and cds == 'mercator':
                if verbose:
                    print("Computation not found. Trying again without passing the 'cds' argument.")
                results = store_quantile_parameters(model_specs=model_specs, **other_specs, **kwargs)
        else:
            if 'skip_computation' in kwargs:
                del kwargs['skip_computation']
            if cds == 'mercator':
                results = store_quantile_parameters(model_specs=model_specs, cds=cds, return_path=return_path, **other_specs, **kwargs, skip_computation=True)
                if isinstance(results, SavedataSkippedComputation):
                    if verbose:
                        print("Computation not found passing the 'cds' argument. Trying again without passing the 'cds' argument.")
                    results = store_quantile_parameters(model_specs=model_specs, return_path=return_path, **other_specs, **kwargs)
            else:
                results = store_quantile_parameters(model_specs=model_specs, cds=cds, return_path=return_path, **other_specs, **kwargs)
        return results

def quantile_results_multiple_IDs(raise_not_computed=False, task='forecasting', criteria=None, params_idx='best', cds='mercator', **kwargs):
    """
    Joins the results of multiple IDs in a single dictionary with the same format as the results for the dataset (the output of `quantile_results`).
    """
    if 'pretrained' in kwargs and not kwargs['pretrained']:
        del kwargs['pretrained']
        warnings.warn("pretrained=False not needed for multiple IDs. Ignoring.")
    # compute best params only once
    if params_idx == 'best':
        allowed_kwargs = getattr(params, f'{task}_extra_kwargs')
        extra_kwargs = {k: kwargs[k] for k in set(kwargs).intersection(allowed_kwargs)}
        params_idx, _ = quantile_best_model(task=task, cds=cds, **extra_kwargs)
    results = []
    not_computed = []
    if criteria is None:
        criteria = 'obs_by_set' if task == 'forecasting' else 'fixed_length'
    valid_IDs = getattr(params, f'valid_IDs_by_{criteria}')[task]
    for ID in valid_IDs:
        results_ID = quantile_results(ID=ID, task=task, cds=cds, params_idx=params_idx, **kwargs, skip_computation=True)
        if isinstance(results_ID, dict):
            results_ID['ID'] = [ID]
            results.append(results_ID)
        else:
            not_computed.append(ID)
    if not_computed and raise_not_computed:
        raise RuntimeError(f"Results not computed for the following IDs: {not_computed}. To skip this error, set raise_not_computed=False.")
    return custom_metrics.merge_quantile_results(results)

@savedata
def quantile_results_train(params_idx=0,
                           quantiles = 'exact',
                           cds = 'mercator',
                           **kwargs):

    results = quantile_results(params_idx=params_idx, cds=cds, quantiles=quantiles, **kwargs)
    pretrained_model = results['state_dict']

    task = kwargs.pop('task', 'forecasting')
    # build tft
    model_specs = get_hp(params_idx=params_idx, task=task, cds=cds, mode='quantile', **kwargs)
    del model_specs['gradient_clip_val']
    other_specs = dict()
    data_params = ['max_train_days']
    for p in data_params:
        if p in model_specs:
            other_specs[p] = model_specs.pop(p)
    if 'store_missing_idxs' in model_specs:
        del model_specs['store_missing_idxs'] # it is passed as a kwarg
    tft = model.QuantileForecaster(quantiles='exact', cds=cds, model_specs=model_specs, task=task, **other_specs, **kwargs)
    tft.model.load_state_dict(pretrained_model)

    results_train = tft.get_results(partition='train_predict')
    return results_train

@savedata
def quantile_results_eval_mode(params_idx=0, quantiles='exact', cds='mercator', task='forecasting', eval_mode='non-overlapping', **kwargs):
    print(f"Computing results for task: {task} with eval_mode: {eval_mode}")
    ID = kwargs.get('ID', None)
    if ID is not None:
        raise NotImplementedError("ID not implemented for eval_mode.")

    if task == 'imputation':
        task_kwargs = dict(task='imputation')
    else:
        task_kwargs = {}
    results = quantile_results(params_idx=params_idx, cds=cds, quantiles=quantiles, **kwargs, **task_kwargs)
    pretrained_model = results['state_dict']

    # build tft
    model_specs = params.quantile_best_params[task][cds][params_idx][1].copy()
    del model_specs['gradient_clip_val']

    tft = model.QuantileForecaster(quantiles=quantiles, cds=cds, model_specs=model_specs, task=task, eval_mode=eval_mode)
    tft.model.load_state_dict(pretrained_model)
    if task == 'imputation' and eval_mode == 'all':
        results = tft.get_results('test') # this will be used only for visualization
    else:
        results = {**tft.get_results('train_predict'), **tft.get_results('val'), **tft.get_results('test')}
    return results

def distribution_results(params_idx=0, **kwargs):
    label_kwargs = _get_label_kwargs(params_idx=params_idx, **kwargs)
    output = store_distribution_parameters(**label_kwargs)
    return output

def _expand_attention(x, encoder_length, max_encoder_length, output_constructor=torch.tensor):
    """
    Expands attention vector for comparing attentions of different sizes.

    Example:
    >>> x = np.array([1, 2, 3])
    >>> expand_attention(x, encoder_length=3, max_encoder_length=5)
    array([1. , 1.5, 2. , 2.5, 3. ])

    the expanded attention vector has the same structure as the original one
    """
    if encoder_length == max_encoder_length:
        return x
    else:
        x_original = x[-encoder_length:]
        original = np.linspace(0, 1, len(x_original))
        expanded = np.linspace(0, 1, max_encoder_length)
        f = interp1d(original, x_original, kind='linear')
        X = f(expanded)
        return output_constructor(X)

@savedata
def attention_CIs(pred_horizon=0, params_idx='best', partition='test', delete_missing=True, join_features=True, **kwargs):
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(**kwargs)
    results = quantile_results(params_idx=params_idx, **kwargs)
    attention_by_time_step, encoder, decoder, future, static = custom_metrics.preprocess_attention(results=results, partition=partition, delete_missing=delete_missing, join_features=join_features)
    att = attention_by_time_step[pred_horizon].numpy() # (N, T)

    def att_CI(func):
        sample_stat = func(att)
        CI = bootstrap.CI_percentile(att, func, R=int(1e4)) # does not work with bca
        df = pd.DataFrame(dict(sample_stat=sample_stat, CI=list(CI)))
        return df

    df_att = att_CI(custom_metrics.normalized_avg)
    df_att_cum = att_CI(custom_metrics.normalized_avg_cumsum)

    def CI_mean_by_col(df):
        out = {}
        for feature in df.columns:
            x = df[feature].values
            out[(feature, 'sample_stat')] = nb_mean(x)
            out[(feature, 'CI')] = bootstrap.CI_bca(x, nb_mean, R=int(1e4))[0]
        out = pd.Series(out).unstack()
        return out
    df_encoder = CI_mean_by_col(encoder)
    df_decoder = CI_mean_by_col(decoder)
    df_static = CI_mean_by_col(static)
    if future is None:
        df_future = None
    else:
        df_future = CI_mean_by_col(future)

    out = dict(att=df_att, att_cum=df_att_cum, encoder=df_encoder, decoder=df_decoder, static=df_static, future=df_future)
    return out

@savedata
def _attention_CIs(R=int(1e5), partition='test', params_idx='best', boot='bca', **kwargs):
    """
    The order of the attention components is
        -max_encoder_length     ->     (attention.size(0) - max_encoder_length).

    Explanation:
        attention.size(0) in [0, max_encoder_length]. For shorter signals the attention is padded with zeros.
        NOTE: attention in the code above has been summed accross batches. see pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.interpret_output.

        In pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.plot_interpretation:
            fig, ax = plt.subplots()
            attention = interpretation["attention"].detach().cpu()
            attention = attention / attention.sum(-1).unsqueeze(-1)
            ax.plot(
                np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
            )
            ax.set_xlabel("Time index")
            ax.set_ylabel("Attention")

    NOTE: To compare attentions of different encoder lengths, we expand the attention vector to the max_encoder_length, preserving the stucture.
    """
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(**kwargs)
    elif not isinstance(params_idx, int):
        raise ValueError(f"params_idx should be an integer or 'best'. Got {params_idx}.")

    results = quantile_results(params_idx=params_idx, **kwargs)
    encoder_lengths = results[f'x_{partition}']['encoder_lengths'].numpy()
    max_encoder_length = encoder_lengths.max()
    interpretation = results[f'interpretation_raw_{partition}']
    variables = ['attention', 'static_variables', 'encoder_variables', 'decoder_variables']
    components = dict(attention=np.arange(-max_encoder_length, 0),
                      static_variables=results['static_variables'],
                      encoder_variables=results['encoder_variables'],
                      decoder_variables=results['decoder_variables'])
    # expand the attention:
    interpretation['attention'] = torch.stack([_expand_attention(x, l, max_encoder_length) for x, l in zip(interpretation['attention'], encoder_lengths)], axis=0)

    CIs = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    pbar = tqdm(range(np.sum([len(v) for v in components.values()])))
    for v in variables:
        attention = interpretation[v].T.numpy() # (time, batch)
        for component, att in zip(components[v], attention):
            CIs[(v, component, 'mean')] = nb_mean(att)
            CIs[(v, component, 'CI')] = computer(att, nb_mean, R=R)
            pbar.update(1)
    CIs = pd.Series(CIs)

    # Computing the cumulative attention CI
    att = interpretation['attention'].numpy()
    def nb_cumsum(x):
        return np.mean(x.cumsum(axis=1), axis=0)
    att_cum_CI = bootstrap.CI_bca(att, nb_cumsum, R=R, use_numba=False)
    att_cum_mean = nb_cumsum(att)
    df_att_cum = pd.DataFrame(dict(mean=att_cum_mean, CI=list(att_cum_CI)))
    return CIs, df_att_cum

def load_quantile_results(params_idx=0,
                          quantiles = 'exact',
                          task = 'forecasting',
                          cds = 'mercator',
                          partition = 'test',
                          area_method = 'km',
                          mpl_val = False,
                          delete_missing = True,
                          eval_mode = None,
                          conservative = False,
                          iterative = False,
                          individual_mpl_val = False, # gives worse results
                          ID_criteria = None,
                          **kwargs):

    quantile_method = deepcopy(quantiles)
    quantiles = model.get_quantiles(quantiles, s_q=kwargs.get('s_q', 1))

    if partition == 'train' and eval_mode is None:
        results = quantile_results_train(params_idx=params_idx, quantiles=quantile_method, cds=cds, task=task, **kwargs)
    else:
        results = quantile_results(params_idx=params_idx, cds=cds, quantiles=quantile_method, eval_mode=eval_mode, task=task, **kwargs)
    x = results[f'x_{partition}']
    y_quantiles = results[f'y_pred_quantiles_{partition}']
    stack_targets = lambda t: torch.stack(t, axis=0).numpy()
    decoder_targets = stack_targets(x[f"decoder_target"])
    decoder_lengths = x[f"decoder_lengths"].numpy()
    targets = params.cds_to_cols[cds]
    if mpl_val:
        ID = kwargs.get('ID', None)
        if ID == 'all' and individual_mpl_val:
            print("Using individual CI expansion for each ID.")
            del kwargs['ID']
            if ID_criteria is None:
                ID_criteria = 'obs_by_set' if task == 'forecasting' else 'fixed_length'
            valid_IDs = getattr(params, f'valid_IDs_by_{ID_criteria}')[task]
            specs = dict(task=task, params_idx=params_idx, quantiles=quantile_method, cds=cds, partition=partition, area_method=area_method, mpl_val=True, delete_missing=delete_missing, conservative=conservative, **kwargs)
            y_quantiles = np.concatenate([load_quantile_results(**specs, ID=ID)[2] for ID in valid_IDs], axis=1)
        else:
            mpls = TFT_mpl_val(params_idx=params_idx, task=task, cds=cds, quantiles=quantile_method, delete_missing=delete_missing,
                               conservative=conservative, iterative=iterative, **kwargs)
            y_quantiles = expand_TFT_quantiles(y_quantiles, mpls)
    return results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method


@savedata
def quantile_coverages(params_idx=0,
                       quantiles = 'exact',
                       cds = 'mercator',
                       partition = 'test',
                       area_method = 'km',
                       mpl_val = 'best',
                       delete_missing = True,
                       chunk_as_animal = True,
                       **kwargs):
    """
    Returns coverage for each:
        - quantile.
        - interval (+ length)
        - region (+ area)

    area_method: str
        Method to compute the area of the region. Default is 'km' (square kilometers).
        If None, computes the area in the same units as the data, i.e. it will have the units of (coord1 * coord2).

    for each target, animal and time step.
    """
    params = dict(params_idx=params_idx, quantiles=quantiles, cds=cds, partition=partition, area_method=area_method, delete_missing=delete_missing)
    if mpl_val == 'best':
        c_no_mpl = quantile_coverages(**params, mpl_val=False, **kwargs)
        c_mpl = quantile_coverages(**params, mpl_val=True, **kwargs)
        def rmse_eval(c):
            coverage_region = c[2]['coverage'].unstack()
            avg_cov = coverage_region.mean()
            error = avg_cov - avg_cov.index
            return np.sqrt((error**2).mean())

        if rmse_eval(c_mpl) < rmse_eval(c_no_mpl):
            print("Best TFT model uses CI expansion")
            return c_mpl
        else:
            print("Best TFT model does not use CI expansion")
            return c_no_mpl
    else:
        results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(**params, mpl_val=mpl_val, **kwargs)

        coverage_quantile = {}
        coverage_interval = {}
        coverage_region = {}
        for i, q in enumerate(quantiles):
            for j, target in enumerate(targets):
                c_q = (decoder_targets[j] <= y_quantiles[j, :, :, i])
                for animal, c_qk in enumerate(c_q):
                    pred_length = decoder_lengths[animal]
                    for time_step, c_qk_t in enumerate(c_qk):
                        if time_step < pred_length:
                            coverage_quantile[(target, time_step, animal, q, 'coverage')] = c_qk_t
                        else:
                            coverage_quantile[(target, time_step, animal, q, 'coverage')] = np.nan

            if i < len(quantiles) // 2:
                # Coverage each target
                alpha_each_target = 1 - (quantiles[-i-1] - q)
                c_each_target = np.round(1 - alpha_each_target, decimals=3)
                lengths = y_quantiles[:, :, :, -i-1] - y_quantiles[:, :, :, i]
                cov_interval = (
                    (decoder_targets >= y_quantiles[:, :, :, i])
                    & (decoder_targets <= y_quantiles[:, :, :, -i-1])
                )
                for j, target in enumerate(targets):
                    for animal, (l_k, cov_k) in enumerate(zip(lengths[j], cov_interval[j])):
                        pred_length = decoder_lengths[animal]
                        for time_step, (l, cov) in enumerate(zip(l_k, cov_k)):
                            if time_step < pred_length:
                                coverage_interval[(target, time_step, animal, c_each_target, 'length')] = l
                                coverage_interval[(target, time_step, animal, c_each_target, 'coverage')] = cov
                            else:
                                coverage_interval[(target, time_step, animal, c_each_target, 'length')] = np.nan
                                coverage_interval[(target, time_step, animal, c_each_target, 'coverage')] = np.nan

                # Coverage region
                if quantile_method == 'bonferroni':
                    alpha_joint = alpha_each_target * 2
                elif quantile_method == 'exact':
                    alpha_joint = 1 - (1 - alpha_each_target)**2
                else:
                    alpha_joint = 1 - (1 - alpha_each_target)**2
                c = np.round(1 - alpha_joint, decimals=3)
                c_joint = cov_interval.all(axis=0)
                if area_method == 'km':
                    x0, y0 = y_quantiles[:, :, :, i]
                    x1, y1 = y_quantiles[:, :, :, -i-1]
                    area = custom_metrics.area_integral_rectangle(x0, x1, y0, y1)
                elif area_method is None:
                    area = lengths.prod(axis=0)
                else:
                    raise ValueError(f"Unknown area_method: {area_method}. Available methods: ['km', None].")

                for animal, (c_jointk, area_k) in enumerate(zip(c_joint, area)):
                    pred_length = decoder_lengths[animal]
                    for time_step, (c_joint_kt, area_kt) in enumerate(zip(c_jointk, area_k)):
                        if time_step < pred_length:
                            coverage_region[(time_step, animal, c, 'coverage')] = c_joint_kt
                            coverage_region[(time_step, animal, c, 'area')] = area_kt
                        else:
                            coverage_region[(time_step, animal, c, 'coverage')] = np.nan
                            coverage_region[(time_step, animal, c, 'area')] = np.nan

        names = ['target', 'time_step', 'animal', 'quantile']
        names_c = names[:-1] + ['confidence']

        coverage_quantile = pd.Series(coverage_quantile).astype(int).unstack(-1)
        coverage_quantile.index.set_names(names, inplace=True)
        coverage_interval = pd.Series(coverage_interval).unstack(-1)
        coverage_interval['coverage'] = coverage_interval['coverage'].astype(int)
        coverage_interval.index.set_names(names_c, inplace=True)
        coverage_region = pd.Series(coverage_region).unstack(-1)
        coverage_region['coverage'] = coverage_region['coverage'].astype(int)
        coverage_region.index.set_names(names_c[1:], inplace=True)
        if delete_missing:
            params_missing = params.copy()
            del params_missing['delete_missing']
            missing = get_missing_values(**params_missing, repeat_by_conf=True, **kwargs)
            coverage_region[missing] = np.nan
        coverage_quantile = coverage_quantile.astype(np.float64)
        coverage_interval = coverage_interval.astype(np.float64)
        coverage_region = coverage_region.astype(np.float64)
        if not chunk_as_animal:
            chunk_to_animal = {i: g for i, g in enumerate(x['groups'].numpy()[:,0])}
            def remap_index(df):
                names = df.index.names
                df = df.reset_index()
                df['animal'] = df['animal'].map(chunk_to_animal)
                df = df.set_index(names)
                return df
            coverage_quantile = remap_index(coverage_quantile)
            coverage_interval = remap_index(coverage_interval)
            coverage_region = remap_index(coverage_region)

        return coverage_quantile, coverage_interval, coverage_region

def get_missing_values(x=None,
                       params_idx=2,
                       quantiles = 'exact',
                       cds = 'mercator',
                       partition = 'test',
                       area_method = 'km',
                       repeat_by_conf = False,
                       y_real = None,
                       baseline = None,
                       task = 'forecasting',
                       force_store_missing_idxs = True,
                       **kwargs):
    if 'store_missing_idxs' not in kwargs and force_store_missing_idxs:
        kwargs['store_missing_idxs'] = True

    if x is None:
        results, x, y_quantiles, *_ = load_quantile_results(task=task, params_idx=params_idx, quantiles=quantiles, cds=cds, partition=partition, area_method=area_method, mpl_val=False, **kwargs)

    lengths = x['decoder_lengths']
    length_mask = create_mask(lengths.max(), lengths) # take into account missing due to decoder length
    if 'decoder_missing' in x:
        missing = (x['decoder_missing'] | length_mask).numpy() # missing intermediate values
    elif task == 'forecasting':
        warnings.warn(f"decoder_missing not found in x_{partition}. Using custom_metrics.decoder_missing_values.")
        if baseline is None:
            baseline, _, y_real = get_predictions(results=results, x=x, cds=cds, task=task)
        missing = custom_metrics.decoder_missing_values(baseline, y_real)
        missing = np.vstack(missing.values) | length_mask.numpy()
    else:
        raise ValueError(f"decoder_missing not found in x_{partition} and task is not 'forecasting'.")
    missing = pd.Series(list(missing))

    if repeat_by_conf:
        num_confidences = y_quantiles.shape[-1] // 2
        missing = np.hstack(missing.values)
        missing = np.repeat(missing, num_confidences)
    return missing

@savedata
def area_coverage_CI(task='forecasting', params_idx=0, area_method='km', delete_missing=True, boot='bca', **kwargs):
    quantiles = kwargs.get('quantiles', 'exact')
    if quantiles == 'exact':
        cov_region = quantile_coverages(task=task, params_idx=params_idx, area_method=area_method, delete_missing=delete_missing, **kwargs)[2]
    elif quantiles == 'all':
        del kwargs['quantiles']
        params_idx = kwargs.get('params_idx', 'best')
        if params_idx == 'best':
            density = kwargs.get('density', 'qrde')
            rho = kwargs.get('rho', True)
            mpl_val = kwargs.get('mpl_val', True)
            rho = rho and mpl_val
            specs = dist_best_model_specs(mpl_val=mpl_val, task=task, density=density, rho=rho)
            kwargs.update(specs)
            if not mpl_val:
                del kwargs['rho']
        else:
            kwargs['params_idx'] = params_idx
        cov_region = dist_pr(task=task, delete_missing=delete_missing, **kwargs)
    else:
        raise ValueError(f"Unknown quantiles: {quantiles}. Available options: ['exact', 'all'].")

    CIs = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    for col in ['coverage', 'area']:
        for (time_step, confidence), df in cov_region[col].groupby(['time_step', 'confidence']):
            X = df.dropna().values.astype(np.float64)
            CIs[(col, time_step, confidence, 'mean')] = nb_mean(X)
            CIs[(col, time_step, confidence, 'CI')] = computer(X, nb_mean)
    CIs = pd.Series(CIs)
    coverage = CIs.loc['coverage'].unstack([1, 2])
    area = CIs.loc['area'].unstack([1, 2])
    return coverage, area

@savedata
def ref_area_CI(boot='bca', delete_missing=True, **kwargs):
    df = compute_quality_df(delete_missing=delete_missing, **kwargs)
    if delete_missing:
        df = df[df.area.notna()]
    result = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    for col in tqdm(['area_max', 'area_ref']):
        for (time_step, confidence), df_col in df[col].groupby(['time_step', 'confidence']):
            X = df_col.dropna().values.astype(np.float64)
            result[(col, time_step, confidence, 'mean')] = nb_mean(X)
            result[(col, time_step, confidence, 'CI')] = computer(X, nb_mean)
    result = pd.Series(result)
    area_max = result.loc['area_max'].unstack([1, 2])
    area_ref = result.loc['area_ref'].unstack([1, 2])
    return area_max, area_ref

@savedata
def area_coverage_CI_baseline(task='forecasting', area_method='km', delete_missing=True, boot='bca', **kwargs):
    _ = kwargs.pop('params_idx', 0) # do not use params_idx, the results are the same.
    area = quantile_baseline_area(task=task, area_method=area_method, delete_missing=delete_missing, **kwargs)
    coverage = baseline_coverage_area(task=task, delete_missing=delete_missing, **kwargs).stack('confidence', dropna=False)
    magnitude = dict(area=area, coverage=coverage)
    CIs = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    for label, df_magnitude in magnitude.items():
        for (time_step, confidence), df in df_magnitude.groupby(['time_step', 'confidence']):
            X = df.dropna().values.astype(np.float64)
            CIs[(label, time_step, confidence, 'mean')] = nb_mean(X)
            CIs[(label, time_step, confidence, 'CI')] = computer(X, nb_mean)
    CIs = pd.Series(CIs)
    coverage = CIs.loc['coverage'].unstack([1, 2])
    area = CIs.loc['area'].unstack([1, 2])
    return coverage, area

@savefig
def _region_coverage_plot(params_idx=0, **kwargs):
    """
    First version. Not incluying the baseline.
    """
    coverage, _ = area_coverage_CI(params_idx=params_idx, **kwargs)
    x = time_step_to_days(coverage.index.values)
    fig = get_figure(xaxis_title='Days', yaxis_title='Coverage',
                     xaxis_range=[0, x.max()], yaxis_range=[0, 1])
    colors = plotly_default_colors()
    for (confidence, color) in zip(coverage.columns.levels[0], colors):
        df = coverage[confidence]
        y = df['mean'].values
        CI = np.clip(np.vstack(df['CI'].values), 0, 1)
        plot_confidence_bands(fig=fig, x=x, y=y, CI=CI, color=color, opacity=0.2, label=str(confidence))
        fig.add_shape(type="line", x0=x.min(), y0=confidence, x1=x.max(), y1=confidence, line=dict(color=color, dash='dash', width=5))
    return fig

@savefig
def region_coverage_plot(params_idx=0, add_baseline=True, **kwargs):
    coverage, _ = area_coverage_CI(params_idx=params_idx, **kwargs)
    confidences = coverage.columns.levels[0]
    colors = plotly_default_colors(confidences.size)
    tickvals = np.hstack((0, confidences))
    ticktext = ['0'] + [f'<span style="color:{color}; font-weight:bold">{confidence}</span>' for confidence, color in zip(confidences, colors)]
    x = time_step_to_days(coverage.index.values)
    fig = get_figure(xaxis_title='Days', yaxis_title='Coverage',
                     xaxis_range=[0, x.max()], yaxis_range=[0, 1],
                     yaxis_tickvals=tickvals,
                     yaxis_ticktext=ticktext,
                     )
    def _plot(coverage, dash, add_reference=True):
        line_specs=dict(dash=dash)
        for (confidence, color) in zip(confidences, colors):
            df = coverage[confidence]
            y = df['mean'].values
            CI = np.clip(np.vstack(df['CI'].values), 0, 1)
            plot_confidence_bands(fig=fig, x=x, y=y, CI=CI, color=color, opacity=0.2, label=None, line_specs=line_specs)
            if add_reference:
                fig.add_shape(type="line", x0=x.min(), y0=confidence, x1=x.max(), y1=confidence, line=dict(color='black', dash='dash', width=4))
        return

    _plot(coverage, 'solid', add_reference=True)

    if add_baseline:
        baseline_coverage, _ = area_coverage_CI_baseline(params_idx=params_idx, **kwargs)
        _plot(baseline_coverage, 'dot', add_reference=False)
    return fig

@savefig
def region_area_plot(params_idx=0, add_baseline=True, **kwargs):
    _, area = area_coverage_CI(params_idx=params_idx, **kwargs)
    confidences = area.columns.levels[0]
    x = time_step_to_days(area.index.values)
    fig = get_figure(xaxis_title='Days', yaxis_title='Area [km²]',
                     xaxis_range=[0, x.max()],
                     yaxis_type='log', yaxis_exponentformat='power', yaxis_dtick=1)
    colors = plotly_default_colors(confidences.size)
    def _plot(area, dash, add_label=True):
        line_specs=dict(dash=dash)
        for (confidence, color) in zip(confidences, colors):
            df = area[confidence]
            y = df['mean'].values
            CI = np.clip(np.vstack(df['CI'].values), 0, None)
            plot_confidence_bands(fig=fig, x=x, y=y, CI=CI, color=color, opacity=0.2, label=str(confidence) if add_label else None, line_specs=line_specs)
    _plot(area, 'solid', add_label=True)
    if add_baseline:
        _, area = area_coverage_CI_baseline(params_idx=params_idx, **kwargs)
        _plot(area, 'dot', add_label=False)
    return fig

def get_predictions(results=None, partition='test', x=None, task='forecasting', cds='mercator', naive_pred='last-obs'):
    """
    Returns:
        - baseline_pred: predictions of the baseline model (naive forecast, last encoder value).
        - y_preds: predictions of the model. (only if 'results' is passed).
        - y_real: real values.
    """
    if x is None:
        if results is None:
            raise ValueError("Either results or x should be passed.")
        x = results[f'x_{partition}']
    encoder_targets = custom_metrics.stack_targets(x[f"encoder_target"])
    encoder_targets = np.moveaxis(encoder_targets, 0, 1)
    decoder_targets = custom_metrics.stack_targets(x[f"decoder_target"])
    decoder_targets = np.moveaxis(decoder_targets, 0, 1)
    encoder_lengths = x["encoder_lengths"]
    decoder_lengths = x["decoder_lengths"]
    max_decoder_length = decoder_targets.shape[-1]

    if task == 'imputation':
        baseline_pred_future = custom_metrics.baseline_prediction_future(results, x=x, cds=cds)
    baseline_pred = []
    for animal, (encoder, l_enc, l_dec) in enumerate(zip(encoder_targets, encoder_lengths, decoder_lengths)):
        base = np.repeat(encoder[:, l_enc-1][:, None], l_dec, axis=1)
        if task == 'imputation':
            if naive_pred == 'last-obs':
                base[:, max_decoder_length//2:] = baseline_pred_future[:, animal][:, None]
            elif naive_pred == 'line':
                for target in range(base.shape[0]):
                    base[target] = np.linspace(base[target, 0], baseline_pred_future[target, animal], l_dec+2)[1:-1] # exclude first and last
            elif naive_pred == 'line-sphere':
                x0 = base[:, 0]
                xf = baseline_pred_future[:, animal]
                x_naive = space.straight_path(x0, xf, l_dec.item()) # (time_steps, target)
                base = x_naive.T
            else:
                raise ValueError(f"naive_pred {naive_pred} not recognized.")
        baseline_pred.append(base)

    if results is None:
        y_preds = None
    else:
        y_preds = results[f'y_pred_{partition}']
        y_preds = np.moveaxis(y_preds, 0, 1)
        y_preds = [y_pred[:, :l] for y_pred, l in zip(y_preds, decoder_lengths)]
    y_real = [decoder_target[:, :l] for decoder_target, l in zip(decoder_targets, decoder_lengths)]
    return pd.Series(baseline_pred), pd.Series(y_preds), pd.Series(y_real)

def quantile_baseline_lengths(task='forecasting', params_idx=0, cds='mercator', fair=False, CI_method='bca', to_width=False, partition='test', CI_mpl='best', delete_missing=True, naive_pred='last-obs', naive_pred_lengths='last-obs', **kwargs):
    """
    naive_pred: str. Only used for imputation task.
        - 'last-obs': the prediction of the baseline model is the closest observed value. This is, the last encoder point for the first half of the imputation window, and the first future point for the second half.
        - 'line': the prediction of the baseline model is a line in the (X,Y) plane between the last encoder point and the first future point.
        - 'line-sphere': the prediction of the baseline model is the closest line in the Earth surface between the last encoder point and the first future point.
    Returns:
        - baseline_lengths: the lengths of the prediction intervals.
        - baseline_pred: the average length of the prediction intervals (not the prediction of the baseline model).
    """
    # TODO: adapt this to imputation task (variable decoder lengths)
    def _quantile_baseline_lengths_fair(partition='test', **kwargs):
        coverage_interval = quantile_coverages(**kwargs)[1]
        targets = coverage_interval.index.levels[0].values
        empirical_coverage = coverage_interval.groupby(['target', 'confidence']).mean()
        c_1 = empirical_coverage.loc[targets[0]].values.squeeze()
        c_2 = empirical_coverage.loc[targets[1]].values.squeeze()
        return _quantile_baseline_lengths(c_1=c_1, c_2=c_2, partition=partition, **kwargs)

    @savedata
    def _quantile_baseline_lengths(params_idx=0, CI_method='bca',
                                   c_1=None, c_2=None,
                                   partition='test',
                                   quantiles='exact',
                                   cds = 'mercator',
                                   delete_missing = True,
                                   task = 'forecasting',
                                   fill_mode = 'last_obs',
                                   naive_pred = 'last-obs',
                                  **kwargs):

        if CI_method == 'bca':
            CI_computer = bootstrap.CI_bca
        elif CI_method == 'percentile':
            CI_computer = bootstrap.CI_percentile
        else:
            raise ValueError(f"CI_method {CI_method} not recognized.")

        def default_confidence():
            if isinstance(quantiles, str):
                q = params.default_quantiles[quantiles]
            elif not isinstance(quantiles, (list, np.ndarray)):
                raise ValueError(f"quantiles should be a list or numpy array. Got {type(quantiles)}. Alternatively, it can be a string in ['exact', 'bonferroni'].")
            else:
                q = deepcopy(quantiles)
            q = np.array(q)
            q_mid = len(q) // 2
            confidence_levels = np.round([q[-i- 1] - q[i] for i in range(q_mid)], decimals=3)[::-1]
            return confidence_levels

        if c_1 is None:
            c_1 = default_confidence()
        if c_2 is None:
            c_2 = default_confidence()

        confidence_levels = [c_1, c_2]

        results = quantile_results(params_idx=params_idx, cds=cds, quantiles=quantiles, task=task, **kwargs)
        x = results[f'x_{partition}']
        stack_targets = lambda t: torch.stack(t, axis=0).numpy()
        encoder_targets = stack_targets(x[f"encoder_target"])
        decoder_targets = stack_targets(x[f"decoder_target"])
        max_prediction_length = decoder_targets.shape[-1]
        encoder_lengths = x[f"encoder_lengths"].numpy()
        targets = params.cds_to_cols[cds]
        if task == 'imputation':
            encoder_target_future, encoder_length_future = custom_metrics.compute_future_encoder(results, x=x, cds=cds)
            encoder_target_future_reversed = encoder_target_future[:, :, ::-1] # from future to past
            encoder_missing = x['encoder_missing'].numpy()
            future_missing = x['future_missing'].numpy()

        def fill_with_nans(target, day, animal):
            target_pred[(target, day-1, animal, 'mean')] = np.nan
            for c in confidence_levels:
                target_widths[(target, day-1, animal, c)] = np.nan
            return

        if fill_mode == 'nan':
            fill = fill_with_nans
        elif fill_mode == 'last_obs':
            def fill(target, day, animal):
                day_last_obs = day - 2
                try:
                    mean_last_obs = target_pred[(target, day_last_obs, animal, 'mean')]
                except KeyError:
                    fill_with_nans(target, day, animal)
                    return
                while math.isnan(mean_last_obs):
                    day_last_obs -= 1
                    try:
                        mean_last_obs = target_pred[(target, day_last_obs, animal, 'mean')]
                    except KeyError:
                        fill_with_nans(target, day, animal)
                        return
                target_pred[(target, day-1, animal, 'mean')] = mean_last_obs
                for c in confidence_levels:
                    target_widths[(target, day-1, animal, c)] = target_widths[(target, day_last_obs, animal, c)]
                return
        else:
            raise ValueError(f"fill {fill} not recognized.")

        target_widths = {}
        target_pred = {}

        for i, (target, confidence_levels) in enumerate(zip(targets, confidence_levels)):
            encoder = encoder_targets[i]
            if task == 'imputation':
                encoder_future = encoder_target_future_reversed[i]
            if task == 'imputation' and naive_pred.startswith('line'):
                # constant uncertainty. The movement is taking into account by the point prediction.
                for animal in range(decoder_targets.shape[1]):
                    length_past = encoder_lengths[animal]
                    length_future = encoder_length_future[animal]
                    encoder_animal = encoder[animal][:length_past]
                    future_animal = encoder_future[animal][:length_future]
                    if delete_missing:
                        missing_past = encoder_missing[animal][:length_past]
                        missing_future = future_missing[animal][:length_future]
                        encoder_animal[missing_past] = np.nan
                        future_animal[missing_future] = np.nan
                    distance_traveled_past = encoder_animal[1:] - encoder_animal[:-1]
                    distance_traveled_future = future_animal[1:] - future_animal[:-1]
                    distance_traveled = np.hstack([distance_traveled_past, distance_traveled_future])
                    distance_traveled = distance_traveled[~np.isnan(distance_traveled)]

                    if distance_traveled.size > 1:
                        pred = nb_mean(distance_traveled)
                        for day in range(1, max_prediction_length+1):
                            target_pred[(target, day-1, animal, 'mean')] = pred

                        for c in confidence_levels:
                            alpha = 1-c
                            ci = CI_computer(distance_traveled, nb_mean, alpha=alpha, alternative='two-sided', R=int(1e4), n_min=1)
                            for day in range(1, max_prediction_length+1):
                                target_widths[(target, day-1, animal, c)] = ci
                    else:
                        for day in range(1, max_prediction_length+1):
                            fill(target, day, animal)
            else:
                for day in tqdm(range(1, max_prediction_length+1)):
                    is_first_half = day-1 < max_prediction_length//2
                    day_reciprocal = max_prediction_length+1 - day
                    for animal in range(decoder_targets.shape[1]):
                        data_length = encoder_lengths[animal]
                        if task == 'forecasting' or is_first_half:
                            enough_data = day-1 < data_length
                        else:
                            data_length_future = encoder_length_future[animal]
                            enough_data = day_reciprocal-1 < data_length_future

                        if enough_data:
                            if task == 'forecasting' or is_first_half:
                                encoder_animal = encoder[animal][:data_length]
                                if delete_missing:
                                    if task == 'forecasting':
                                        missing = custom_metrics.is_missing_1_axis(encoder_animal)
                                    else:
                                        missing = encoder_missing[animal][:data_length]
                                    encoder_animal[missing] = np.nan
                                distance_traveled = encoder_animal[day:] - encoder_animal[:-day] # error in naive forecast
                            elif task == 'imputation': # last half of the imputation window
                                encoder_animal = encoder_future[animal][-data_length_future:]
                                if delete_missing:
                                    missing = future_missing[animal][-data_length_future:]
                                    encoder_animal[missing] = np.nan
                                distance_traveled = encoder_animal[day_reciprocal:] - encoder_animal[:-day_reciprocal] # uncertainty decreases as approaches observed future

                            distance_traveled = distance_traveled[~np.isnan(distance_traveled)]
                            if distance_traveled.size > 1:
                                target_pred[(target, day-1, animal, 'mean')] = nb_mean(distance_traveled)
                                for c in confidence_levels:
                                    alpha = 1-c
                                    target_widths[(target, day-1, animal, c)] = CI_computer(distance_traveled, nb_mean, alpha=alpha,
                                                                                            alternative='two-sided', R=int(1e4), n_min=1)
                            elif distance_traveled.size == 1:
                                target_pred[(target, day-1, animal, 'mean')] = distance_traveled[0]
                                for c in confidence_levels:
                                    target_widths[(target, day-1, animal, c)] = np.array([distance_traveled[0], distance_traveled[0]])[None]
                            else:
                                fill(target, day, animal)
                        else:
                            fill(target, day, animal)

        target_widths = pd.Series(target_widths)
        target_widths = target_widths.apply(np.squeeze)
        target_pred = pd.Series(target_pred)
        return target_widths, target_pred

    if fair:
        if CI_mpl:
            raise ValueError("CI_mpl should be False when fair=True.")
        baseline_lengths, baseline_pred = _quantile_baseline_lengths_fair(CI_method=CI_method, task=task, params_idx=params_idx, cds=cds, partition=partition, delete_missing=delete_missing, naive_pred=naive_pred_lengths, **kwargs)
    else:
        baseline_lengths, baseline_pred = _quantile_baseline_lengths(CI_method=CI_method, task=task, params_idx=params_idx, cds=cds, partition=partition, delete_missing=delete_missing, naive_pred=naive_pred_lengths, **kwargs)

    baseline_lengths.index.set_names(['target', 'time_step', 'animal', 'confidence'], inplace=True)
    baseline_lengths.name = 'length'
    baseline_pred.index.set_names(['target', 'time_step', 'animal', 'statistic'], inplace=True)
    baseline_pred.name = 'value'

    if delete_missing:
        missing = get_missing_values(params_idx=params_idx, cds=cds, partition=partition, repeat_by_conf=True, task=task, **kwargs)
        # pred
        index_order = baseline_pred.index.names
        baseline_pred = baseline_pred.unstack(0)
        num_confs = baseline_lengths.index.levels[-1].size
        baseline_pred[missing[::num_confs]] = np.nan
        baseline_pred = baseline_pred.stack(0, dropna=False)
        baseline_pred = baseline_pred.reorder_levels(index_order) # sort by index order
        baseline_pred = baseline_pred.sort_index()
        # lengths
        index_order = baseline_lengths.index.names
        baseline_lengths = baseline_lengths.unstack(0)
        baseline_lengths[missing] = np.nan
        baseline_lengths = baseline_lengths.stack(0, dropna=False)
        baseline_lengths = baseline_lengths.reorder_levels(index_order)
        baseline_lengths = baseline_lengths.sort_index()

    if (isinstance(CI_mpl, bool) and CI_mpl) or (CI_mpl == 'best' and eval_best_baseline(task=task, cds=cds, CI_method=CI_method, delete_missing=delete_missing, **kwargs)):
        baseline_lengths = baseline_lengths.unstack(-1)
        CI_mpl = CI_mpl_val(task=task, cds=cds, params_idx=params_idx, delete_missing=delete_missing, naive_pred=naive_pred, naive_pred_lengths=naive_pred_lengths, **kwargs)['mpl']
        for (confidence, target), mpl in CI_mpl.items():
            baseline_lengths.loc[target, confidence] = baseline_lengths.loc[target, confidence].apply(apply_mpl, mpl=mpl).values
        baseline_lengths = baseline_lengths.stack(dropna=False)

    if to_width:
        baseline_lengths = baseline_lengths.apply(lambda x: x[1] - x[0] if isinstance(x, np.ndarray) else x)

    return baseline_lengths, baseline_pred

def eval_best_baseline(cds='mercator', **kwargs):
    """
    Returns true if the best baseline model uses CI expansion.
    """
    coverages_fit = baseline_coverage(cds=cds, CI_mpl=True, **kwargs)
    coverages = baseline_coverage(cds=cds, CI_mpl=False, **kwargs)
    def micro_avg(coverages):
        """
        Compute the micro-average of the coverage.
        Average is performed in this order:
        animal -> target -> time_step -> output
        """
        return coverages.groupby(['target', 'time_step']).mean().groupby('time_step').mean().mean()

    def rmse(coverages):
        return np.sqrt((coverages - coverages.index).pow(2).mean())
    avg_fit = micro_avg(coverages_fit)
    avg = micro_avg(coverages)
    fit_is_best = rmse(avg_fit) < rmse(avg)
    if fit_is_best:
        print("Best naive model uses CI expansion")
    else:
        print("Best naive model does not use CI expansion")
    return fit_is_best

def quantile_baseline_area(quantiles='exact', area_method='km', **kwargs):
    if area_method == 'km':
        baseline_lengths, _ = quantile_baseline_lengths(to_width=False, quantiles=quantiles, **kwargs)
        _, y_target = preprocess_baseline_coverage(quantiles=quantiles, **kwargs)
        df = baseline_lengths.unstack()
        df += y_target['naive'].values[:, None]
        df = df.stack(dropna=False).unstack(0)

        def compute_area(row):
            if row.isna().all():
                return np.nan
            else:
                x0, x1 = row.X
                y0, y1 = row.Y
                area = custom_metrics.area_integral_rectangle(x0, x1, y0, y1)
                return area
        baseline_area = df.apply(compute_area, axis=1)
    elif area_method is None:
        baseline_width, _ = quantile_baseline_lengths(to_width=True, quantiles=quantiles, **kwargs)
        baseline_area = baseline_width.groupby(['time_step', 'animal', 'confidence']).prod()
    else:
        raise ValueError(f"Unknown area_method: {area_method}. Available methods: ['km', None].")
    baseline_area.name = 'area'
    baseline_area = baseline_area.unstack('confidence')
    confidence_1D = baseline_area.columns.values
    alpha_1D = 1 - confidence_1D
    if quantiles == 'exact':
        alpha_2D = 1 - (1 - alpha_1D)**2

    else:
        alpha_2D = 2 * alpha_1D
    confidence_2D = np.round(1 - alpha_2D, 2)
    baseline_area.columns = confidence_2D
    baseline_area.columns.name = 'confidence'
    return baseline_area.stack('confidence', dropna=False)

def preprocess_baseline_coverage(partition='test', cds='mercator', CI_method='bca', CI_mpl=False, delete_missing=True, task='forecasting', naive_pred='last-obs', naive_pred_lengths='last-obs', **kwargs):
    """
    naive_pred: str. Used only for task='imputation'.
        'last-obs': impute the closest observed value. This is, x0 (last encoder location) for the first half of the imputation window and the xf (the first future location) for the second half.
        'line': impute from x0 to xf using a straight line in the (X, Y) plane.
        'line-sphere': impute from x0 to xf using the shortest path in the Earth surface.
    """
    baseline_width, _ = quantile_baseline_lengths(partition=partition, cds=cds, CI_mpl=CI_mpl, CI_method=CI_method, delete_missing=delete_missing, task=task, naive_pred=naive_pred, naive_pred_lengths=naive_pred_lengths, **kwargs)
    baseline_width = baseline_width.unstack(-1)
    results = quantile_results(cds=cds, task=task, **kwargs)
    x = results[f'x_{partition}']
    encoder_target = custom_metrics.stack_targets(x["encoder_target"])
    encoder_lengths = x["encoder_lengths"].numpy()
    targets = params.cds_to_cols[cds]

    naive_forecast = np.stack([encoder_target[:, i, encoder_lengths[i]-1] for i in range(encoder_target.shape[1])], axis=1)
    y_real = custom_metrics.stack_targets(x["decoder_target"])
    max_decoder_length = y_real.shape[-1]
    y_target = {}
    if task == 'imputation':
        naive_forecast_future = custom_metrics.baseline_prediction_future(results, x=x, cds=cds)
        if naive_pred == 'line':
            # impute from x0 to xf using a straight line in the (X, Y) plane
            for i, target in enumerate(targets):
                for animal in range(y_real.shape[1]):
                    x0 = naive_forecast[i, animal]
                    xf = naive_forecast_future[i, animal]
                    x_naive = np.linspace(x0, xf, max_decoder_length + 2)[1:-1] # exclude x0 and xf
                    for time_step in range(y_real.shape[2]):
                        y_target[(target, time_step, animal, "naive")] = x_naive[time_step]
                        y_target[(target, time_step, animal, "real")] = y_real[i, animal, time_step]
        elif naive_pred == 'line-sphere':
            # impute from x0 to xf using the shortest path in the Earth surface.
            for animal in range(y_real.shape[1]):
                x0 = naive_forecast[:, animal]
                xf = naive_forecast_future[:, animal]
                x_naive = space.straight_path(x0, xf, max_decoder_length)
                for i, target in enumerate(targets):
                    for time_step in range(y_real.shape[2]):
                        y_target[(target, time_step, animal, "naive")] = x_naive[time_step, i]
                        y_target[(target, time_step, animal, "real")] = y_real[i, animal, time_step]
        elif naive_pred == 'last-obs':
            for i, target in enumerate(targets):
                for animal in range(y_real.shape[1]):
                    for time_step in range(y_real.shape[2]):
                        y_target[(target, time_step, animal, "real")] = y_real[i, animal, time_step]
                        if time_step < max_decoder_length//2:
                            y_target[(target, time_step, animal, "naive")] = naive_forecast[i, animal]
                        else:
                            y_target[(target, time_step, animal, "naive")] = naive_forecast_future[i, animal]
        else:
            raise ValueError(f"naive_pred {naive_pred} not recognized.")
    else: # forecasting
        for i, target in enumerate(targets):
            for animal in range(y_real.shape[1]):
                for time_step in range(y_real.shape[2]):
                    y_target[(target, time_step, animal, "real")] = y_real[i, animal, time_step]
                    y_target[(target, time_step, animal, "naive")] = naive_forecast[i, animal]

    y_target = pd.Series(y_target).unstack(-1)
    y_target.index.set_names(['target', 'time_step', 'animal'], inplace=True)
    return baseline_width, y_target

def apply_mpl(x, mpl):
    """
    Increase the width of the prediction interval by a factor mpl.
    """
    if isinstance(x, float): # nan
        return np.nan
    else:
        y = x.copy()
        y[0] -= abs(x[0])*mpl
        y[1] += abs(x[1])*mpl
        return y

def optimize_CI_mpl(df, num_trials=200, max_mpl=100):
    """
    Optimize the mpl parameter for the CI expected coverage.
    """
    c = df.columns[0]
    hp_space = (1, max_mpl)

    def _eval_coverage(df):
        df_target = pd_utils.tuple_wise(df[c], df['target'], check_columns=False)
        coverages = df_target.apply(custom_metrics.eval_coverage).astype(float)
        return coverages

    def objective(trial):
        df_i = df.copy()
        mpl = trial.suggest_loguniform('mpl', *hp_space)
        df_i[c] = df_i[c].apply(apply_mpl, mpl=mpl) + df_i['pred'].values

        df_i['coverage'] = _eval_coverage(df_i)
        error = df_i.groupby('time_step')['coverage'].mean() - c
        rmse = np.sqrt((error**2).mean())
        return rmse


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)
    return study

@savedata
def CI_mpl_val(num_trials=1000, max_mpl=None, params_idx=0, cds='mercator', delete_missing=True, task='forecasting', quantiles='exact', naive_pred='last-obs', naive_pred_lengths='last-obs', **kwargs):
    """
    Optimize the mpl parameter for the CI expected coverage.

    Returns DataFrame with:
        - mpl: optimal mpl parameter.
        - rmse: RMSE of the expected coverage.
    """
    if max_mpl is None:
        if task == 'forecasting':
            max_mpl = 150
        elif task == 'imputation':
            max_mpl = 100 # much less uncertainty
        else:
            raise ValueError(f"task {task} not recognized.")

    baseline_width, y_target = preprocess_baseline_coverage(partition='val', CI_mpl=False, task=task, params_idx=params_idx, cds=cds, quantiles=quantiles, naive_pred=naive_pred, naive_pred_lengths=naive_pred_lengths, delete_missing=delete_missing, **kwargs)
    y_real = y_target['real'].values

    pbar = tqdm(total=baseline_width.columns.size * 2)
    CI_mpl = {}
    for confidence, df_c in baseline_width.groupby(level=0, axis=1):
        df_c['target'] = y_real
        df_c['pred'] = y_target['naive'].values
        df_c = df_c.unstack(0).swaplevel(0,1, axis=1)
        for target in df_c.columns.levels[0]:
            df_t = df_c[target].copy()
            study = optimize_CI_mpl(df_t, num_trials=num_trials, max_mpl=max_mpl)
            CI_mpl[(confidence, target, 'mpl')] = study.best_params['mpl']
            CI_mpl[(confidence, target, 'rmse')] = study.best_value
            pbar.update(1)
    CI_mpl = pd.Series(CI_mpl).unstack(-1)
    CI_mpl.index.names = ['confidence', 'target']
    return CI_mpl

def baseline_coverage(partition='test', cds='mercator', CI_mpl='best', **kwargs):
    # TODO: adapt this to imputation task (variable decoder lengths)
    if CI_mpl == 'best':
        if eval_best_baseline(cds=cds, **kwargs):
            return baseline_coverage(partition=partition, cds=cds, CI_mpl=True, **kwargs)
        else:
            return baseline_coverage(partition=partition, cds=cds, CI_mpl=False, **kwargs)
    else:
        baseline_width, y_target = preprocess_baseline_coverage(partition=partition, cds=cds, CI_mpl=CI_mpl, **kwargs)
        baseline_CI = baseline_width + y_target['naive'].values[:,None]
        y_real = y_target['real'].values
        num_confidences = baseline_CI.columns.size
        y_real_df = pd.DataFrame(np.hstack([y_real[:, None]]*num_confidences),
                                 columns=baseline_CI.columns, index=baseline_CI.index)
        CI_y_real = pd_utils.tuple_wise(baseline_CI, y_real_df)
        coverages = CI_y_real.applymap(custom_metrics.eval_coverage).astype(float)
        return coverages

def baseline_coverage_area(cds='mercator', quantiles='exact', **kwargs):
    coverage = baseline_coverage(cds=cds, quantiles=quantiles, **kwargs)
    confidence_1D = coverage.columns.values
    alpha_1D = 1 - confidence_1D
    if quantiles == 'exact':
        alpha_2D = 1 - (1 - alpha_1D)**2
    else:
        alpha_2D = 2 * alpha_1D
    confidence_2D = np.round(1 - alpha_2D, 2)
    targets = params.cds_to_cols[cds]

    coverage_1 = coverage.loc[targets[0]]
    coverage_2 = coverage.loc[targets[1]]
    coverage_area = coverage_1 * coverage_2
    coverage_area.columns = confidence_2D
    coverage_area.columns.name = 'confidence'
    return coverage_area

def point_prediction_errors(metric='great_circle_distance', partition='test', baseline=False, delete_missing=True, eval_mode=None, cds='mercator', task='forecasting', naive_pred='last-obs',
                            n_sample=None, n_grid=None, mode_margin=0.1, mode_weighted=False, dist_mode='median', mode_method='sample', rho_eps='best', rho=False, density='qrde',
                            **kwargs):
    """
    Must pass s_q when quantiles='all'
    """
    args = locals()
    args.update(kwargs)
    del args['kwargs']

    @savedata
    def _point_prediction_errors(metric='great_circle_distance', partition='test', baseline=False, delete_missing=True, eval_mode=None, cds='mercator', task='forecasting', naive_pred='last-obs',
                                 n_sample=None, n_grid=None, mode_margin=0.1, mode_weighted=False, dist_mode='median', mode_method='sample', rho_eps='best', rho=False, density='qrde',
                                 **kwargs):
        if partition == 'train' and eval_mode is None:
            results = quantile_results_train(cds=cds, task=task, **kwargs)
        else:
            results = quantile_results(eval_mode=eval_mode, cds=cds, task=task, **kwargs)
        x = results[f'x_{partition}']
        baseline_pred, y_pred, y_real = get_predictions(results=results, partition=partition, cds=cds, task=task, naive_pred=naive_pred)
        quantiles = kwargs.get('quantiles', 'exact')
        if not baseline and quantiles == 'all':
            if dist_mode != 'median': # the median prediction is correctly computed
                load_kwargs = kwargs.copy()
                load_kwargs['cds'] = cds
                del load_kwargs['quantiles']
                if n_sample is None:
                    n_sample = int(1e5)
                if n_grid is None:
                    n_grid = 1000
                if dist_mode == 'mode' and not (mode_method == 'roots' or mode_margin == 0) and (mode_margin == 'best' or mode_weighted == 'best'):
                    pp_kwargs = load_kwargs.copy()
                    if density != 'pchip':
                        pp_kwargs['density'] = density
                    if cds == 'mercator':
                        del pp_kwargs['cds']
                    best_hp = optimize_mode_pp(task=task, delete_missing=True, mode_method=mode_method, rho=rho, **pp_kwargs).iloc[0]
                    print(f"Best hyperparameters for mode: {best_hp}")
                    mode_margin = best_hp['mode_margin']
                    mode_weighted = best_hp['mode_weighted']
                load_kwargs['partition'] = partition
                s_q = load_kwargs.pop('s_q', 1)
                if rho and dist_mode == 'mode':
                    rho_kwargs = dict(rho=rho, rho_eps=rho_eps)
                else:
                    rho_kwargs = {}
                y_pred = dist_pp(task=task, s_q=s_q, load_kwargs=load_kwargs, mode_margin=mode_margin, mode_weighted=mode_weighted, mode=dist_mode, mode_method=mode_method,
                                 n_sample=n_sample, n_grid=n_grid, density=density, **rho_kwargs)
                y_pred = y_pred.apply(lambda x: np.array(x).astype(np.float32))
                # stack across time steps
                y_pred = y_pred.groupby(level=0).apply(lambda x: np.vstack(x).T) # xi shape (2, time_steps)
            if cds != 'mercator':
                r0 = load.reference_point()
                spherical_to_mercator = lambda y: space.spherical_fixed_point_to_mercator(y.T, r0).T
                baseline_pred = baseline_pred.apply(spherical_to_mercator)
                y_real = y_real.apply(spherical_to_mercator)
                if dist_mode == 'median':
                    y_pred = y_pred.apply(spherical_to_mercator)

        if metric == 'great_circle_distance':
            to_lat_lon = lambda S: S.apply(lambda x: np.vstack(space.mercator_inv(*x))) # in radians
            pred = to_lat_lon(baseline_pred if baseline else y_pred)
            y_real = to_lat_lon(y_real)
            df = pd_utils.vstack_wise(pred, y_real)
            out = df.apply(lambda x: space.great_circle_distance(*x))
            out *= params.R_earth
        else:
            if metric == 'MAE':
                func = lambda y, y_hat : np.abs(y - y_hat).mean(axis=0)
            elif metric == 'RMSE':
                func = lambda y, y_hat : np.sqrt(((y - y_hat)**2).mean(axis=0))
            else:
                raise ValueError(f"metric {metric} not recognized")

            if baseline:
                df = pd_utils.tuple_wise(baseline_pred, y_real)
            else:
                df = pd_utils.tuple_wise(y_pred, y_real)
            out = df.apply(lambda x: func(*x))

        if delete_missing:
            missing = get_missing_values(x=x, baseline=baseline_pred, y_real=y_real, cds=cds, partition=partition, task=task, **kwargs)
            out_missing = pd_utils.tuple_wise(out, missing)
            def replace_by_nans(x):
                z, missing = x
                z[missing] = np.nan
                return z
            out = out_missing.apply(replace_by_nans)

        out = pd_utils.expand_sequences(out.to_frame())[0]
        out.index.name = 'animal'
        out.columns.name = 'time_step'
        return out

    dive_data = args.get('dive_data', False)
    if dive_data:
        # Dive data uses only specific args
        if task == 'forecasting':
            kws = {'weather': 'all', 'epochs': 200, 'cds': 'mercator', 'overwrite': False, 'patience': 20, 'mod_hp': {}, 'store_missing_idxs': True, 'quantiles': 'all', 's_q': 3, 'dive_data': True, 'limit_train_batches': 400}
        elif task == 'imputation':
            kws = {'weather': 'all', 'epochs': 200, 'cds': 'mercator', 'overwrite': False, 'task': 'imputation', 'mod_hp': {}, 'patience': 20, 'expand_encoder_until_future_length': True, 'store_missing_idxs': True, 'predict_shift': 128, 'reverse_future': True, 'quantiles': 'all', 's_q': 3, 'dive_data': True, 'limit_train_batches': 400}
        else:
            raise ValueError(f"Task {task} not recognized")
        args.update(kws)
    return _point_prediction_errors(**args, save=dive_data)


@savedata
def point_prediction_error_CI(R=int(1e5), task='forecasting', metric='great_circle_distance', boot='bca', baseline=False, ssm_model=None, params_idx='best', **kwargs):
    if ssm_model is None:
        quantiles = kwargs.get('quantiles', 'exact')
        if params_idx == 'best':
            if quantiles == 'exact':
                params_idx, _ = quantile_best_model(task=task, **kwargs)
            elif quantiles == 'all':
                density = kwargs.get('density', 'qrde')
                specs = dist_best_model_specs(task=task, density=density, mpl_val=True, rho=True)
                params_idx = specs.pop('params_idx')
                del specs['rho']
                del specs['method']
                kwargs.update(specs)
        df = point_prediction_errors(metric=metric, baseline=baseline, params_idx=params_idx, task=task, **kwargs)
    else:
        df = ssm.point_prediction_errors(metric=metric, model=ssm_model, task=task, **kwargs)
    error = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    for time_step in tqdm(df.columns):
        x = df[time_step].dropna().values
        error[(time_step, 'sample_stat')] = nb_mean(x)
        error[(time_step, 'CI')] = computer(x, nb_mean, R=R)
    error = pd.Series(error).unstack()
    return error

def point_prediction_summary_across_time(task='forecasting'):
    out = []
    for model, specs in params.model_specs().items():
        if model in ['TFT_dist', 'TFT', 'TFT_single', 'Naive']:
            specs.update(params.TFT_specs[task])
            if model == 'Naive' and task == 'imputation':
                extra_specs = params.baseline_specs_imputation.copy()
                del extra_specs['naive_pred_lengths']
                specs.update(extra_specs)
            elif model == 'TFT_dist':
                specs['dist_mode'] = 'median'
        df = point_prediction_error_CI(task=task, **specs)
        df['CI'] = df['CI'].apply(np.squeeze)
        df = pd_utils.format_CI_results(df).to_frame(model).T
        out.append(df)
    out = pd.concat(out, axis=0)
    out.columns = time_step_to_days(out.columns)
    return out

def best_models(df):
    df = pd_utils.highlight_best(df, 'lower')
    top_best = df.applymap(lambda x: 'underline' in x).mean(axis=1)
    undisputed_best = df.applymap(lambda x: 'textbf' in x).mean(axis=1)
    top_best.name = 'indistinguishable_best'
    undisputed_best.name = 'undisputed_best'
    best = pd.concat([top_best, undisputed_best], axis=1)
    return best

def point_prediction_best_models_across_time():
    """
    Percentage of time steps where the model is the best.
    """
    out = []
    for task in ['forecasting', 'imputation']:
        df = point_prediction_summary_across_time(task)
        best = best_models(df)
        best.columns = pd.MultiIndex.from_product(([task], best.columns))
        out.append(best)
    out = pd.concat(out, axis=1)
    return out

@savedata
def point_prediction_aggregate_CI(task='forecasting', R=int(1e5), metric='great_circle_distance', baseline=False, ssm_model=None, params_idx='best', CI_method='bca', **kwargs):
    if ssm_model is None:
        quantiles = kwargs.get('quantiles', 'exact')
        if params_idx == 'best':
            if quantiles == 'exact':
                params_idx, _ = quantile_best_model(task=task, **kwargs)
            elif quantiles == 'all':
                specs = dist_best_model_specs(task=task, mpl_val=True)
                if 'method' in specs:
                    del specs['method']
                params_idx = specs.pop('params_idx')
                kwargs.update(specs)
            else:
                raise ValueError(f"quantiles {quantiles} not recognized.")
        df = point_prediction_errors(task=task, metric=metric, baseline=baseline, params_idx=params_idx, **kwargs)
    else:
        df = ssm.point_prediction_errors(task=task, metric=metric, model=ssm_model.lower(), **kwargs)
        df.columns.name = 'time_step'
    df_input = df.stack().swaplevel().to_frame('distance_error')
    out = custom_metrics.aggregate_CI_across_time_steps(df_input, custom_metrics.nb_mean_not_nans, labels=['distance'], R=R, CI_method=CI_method)

    sample_stat = out['sample_stat'].iloc[0]
    CI = out['CI'].iloc[0]
    return sample_stat, CI

@savedata
def point_prediction_aggregate_CI_delete_seeds(task='forecasting', R=int(1e5), params_idx='best', n=200, seeds=range(5), **kwargs):
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)
    return custom_metrics.delete_seed_aggregate_CI(n=n, func=point_prediction_errors, statistic=custom_metrics.nb_mean_not_nans_across_seeds, delete_seeds=seeds, R=R,
                                                   params_idx=params_idx, task=task, **kwargs)

def point_prediction_aggregate_CI_delete_seeds_summary(task, params_idx='best'):
    kwargs = params.TFT_specs[task]
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)
    out = []
    for n in tqdm(params.n_to_delete_prob.keys()):
        df = point_prediction_aggregate_CI_delete_seeds(task=task, n=n, params_idx=params_idx, **kwargs)
        out.append(df)
    # Add results for n=1 and n_max
    sample_stat, CI = point_prediction_aggregate_CI(task=task, **kwargs, ID='all')
    results_1 = pd.Series(dict(sample_stat=sample_stat, CI=list(CI)), name=1).to_frame().T

    data = quantile_results_train(task=task, **kwargs, params_idx=params_idx)
    n_train_max = np.unique(data['x_train']['groups'].numpy()).size
    sample_stat_max, CI_max = point_prediction_aggregate_CI(task=task, **kwargs)
    results_n = pd.Series(dict(sample_stat=sample_stat_max, CI=list(CI_max)), name=n_train_max).to_frame().T

    df = pd.concat([results_n] + out + [results_1], axis=0)
    return df

@savefig('all-yaxis_type-xaxis_type-ylim')
def point_prediction_across_n(task='forecasting', params_idx='best', ssm_model=None, yaxis_type=None, xaxis_type='log', ylim=[0, 240]):
    df = point_prediction_aggregate_CI_delete_seeds_summary(task, params_idx=params_idx)
    if ssm_model is None:
        ssm_model = 'mp' if task == 'forecasting' else 'rw'
    sample_stat_ssm, CI_ssm = point_prediction_aggregate_CI(task=task, ssm_model=ssm_model, baseline=False)
    CI_ssm = np.atleast_2d(CI_ssm)
    n_min = df.index[-1]
    n_max = df.index[0]
    ssm_n_min = pd.concat([pd.Series(sample_stat_ssm, index=[n_min], name='sample_stat'),
                         pd.Series(list(CI_ssm), index=[n_min], name='CI')
                         ], axis=1)
    ssm_n_max = ssm_n_min.copy()
    ssm_n_max.index = [n_max]
    df_ssm = pd.concat([ssm_n_min, ssm_n_max])

    fig = plot_confidence_bands(df=df, label='TFT', xaxis_title='Dataset size', yaxis_title='Distance error [km]', title_text=task.capitalize())
    plot_confidence_bands(fig=fig, df=df_ssm, label='SSM', color='black')
    fig.update_layout(xaxis_type=xaxis_type, yaxis_type=yaxis_type, yaxis_range=ylim)
    return fig

@savefig('all-ylim')
def point_prediction_error_plot(task='forecasting', metric='great_circle_distance', add_single=True, add_bivariate=True, ylim=None, **kwargs):
    kwargs['metric'] = metric
    colors = plotly_default_colors()
    df_tft = point_prediction_error_CI(baseline=False, task=task, **kwargs)
    x = time_step_to_days(df_tft.index.values)
    if metric == 'great_circle_distance':
        yaxis_title = 'Great circle distance [km]'
    else:
        yaxis_title = metric

    fig = get_figure(xaxis_title='Days', yaxis_title=yaxis_title)
    def _plot(df, dash, label, color=None):
        if color is None:
            color = next(colors)
        df.index = x
        plot_confidence_bands(fig=fig,
                              df=df,
                              label=label,
                              color=color,
                              line_specs=dict(dash=dash),
                              )
        return

    _plot(df_tft, 'solid', 'TFT')

    if add_bivariate:
        dist_kws = kwargs.copy()
        dist_kws.update(params.model_specs()['TFT_dist'])
        dist_kws['quantiles'] = 'all'
        dist_kws['max_train_days'] = 4
        dist_kws['dist_mode'] = 'median'
        del dist_kws['baseline']
        del dist_kws['ssm_model']
        df_2d = point_prediction_error_CI(task=task, **dist_kws)
        _plot(df_2d, 'dash', 'TFT[B]', color='gold')

    if add_single:
        df_single = point_prediction_error_CI(baseline=False, task=task, **kwargs, ID='all')
        _plot(df_single, 'dash', 'TFT[s]', color='black')
    for ssm_model in ['rw', 'crw', 'mp']:
        _plot(point_prediction_error_CI(task=task, ssm_model=ssm_model), 'dash', ssm_model.upper())
    if task == 'imputation':
        baseline_kwargs = params.baseline_specs_imputation.copy()
        del baseline_kwargs['naive_pred_lengths']
    else:
        baseline_kwargs = {}
    _plot(point_prediction_error_CI(baseline=True, task=task, **kwargs, **baseline_kwargs), 'dash', 'Naive')
    if ylim is not None:
        fig.update_layout(yaxis_range=ylim)
    return fig

@savedata
def eval_point_prediction(params_idx=0, metric='RMSE', out='quantile', partition='test', cds='mercator', boot='bca', **kwargs):
    if out == 'quantile':
        loader = quantile_results
    elif out == 'distribution':
        loader = distribution_results
    else:
        raise ValueError(f"out {out} not recognized")

    results = loader(params_idx=params_idx, cds=cds, **kwargs)

    if metric == 'MAE':
        func = lambda y, y_hat : np.abs(y - y_hat)
    elif metric == 'RMSE':
        func = lambda y, y_hat : np.sqrt(((y - y_hat)**2).mean(axis=0))
    else:
        raise ValueError(f"metric {metric} not recognized")

    x = results[f'x_{partition}']
    y_hats = results[f'y_pred_{partition}']
    y_baseline = np.moveaxis(results['y_baseline'], -1, 0) # TODO: change this
    encoder_targets = custom_metrics.stack_targets(x[f"encoder_target"])
    decoder_targets = custom_metrics.stack_targets(x[f"decoder_target"])
    max_prediction_length = decoder_targets.shape[-1]
    targets = params.cds_to_cols[cds]

    metric_base = func(decoder_targets, y_baseline)
    metric_model = func(decoder_targets, y_hats)
    metric_relative = (metric_model - metric_base) / (metric_base + 1e-8)

    computer = getattr(bootstrap, f'CI_{boot}')
    CIs = {}
    if metric_relative.ndim > 2:
        sample_stat = np.median(metric_relative, axis=1)
        for metric_rel_target, target in zip(metric_relative, targets):
            for time_step in range(max_prediction_length):
                CIs[(target, time_step)] = computer(metric_rel_target[:, time_step], nb_median, R=int(1e4))
        CIs = pd.Series(CIs).unstack(0)
    else:
        sample_stat = np.median(metric_relative, axis=0)
        for time_step in range(max_prediction_length):
            CIs[time_step] = computer(metric_relative[:, time_step], nb_median, R=int(1e4))
        CIs = pd.Series(CIs)
    days = np.arange(1, max_prediction_length+1) / 4 # time_step = 6h. 4 time steps = 1 day
    return days, sample_stat, CIs

@savefig('all-width')
def MAE_plot(params_idx=0, width=0.3, cds='mercator', **kwargs):
    days, sample_stat, CIs = eval_point_prediction(params_idx=params_idx, cds=cds, metric='MAE', **kwargs)
    targets = CIs.columns
    max_prediction_length = days.size
    fig = CI_plot(days, sample_stat[0], np.vstack(CIs[targets[0]]), label=targets[0], x_title='Days', y_title='Relative MAE', width=width)
    fig = CI_plot(days, sample_stat[1], np.vstack(CIs[targets[1]]), label=targets[1], color=color_std('#1f77b4'),fig = fig, width=width)
    fig.update_layout(xaxis_tickvals=np.arange(-1, max_prediction_length + 4, 4), xaxis_ticktext=[str(i) for i in range(8)])
    fig.add_hline(0, line=dict(color='black', dash='dash'))
    return fig

@savefig('all-width')
def RMSE_plot(params_idx=0, width=0.3, **kwargs):
    days, sample_stat, CIs = eval_point_prediction(params_idx=params_idx, metric='RMSE', **kwargs)
    max_prediction_length = days.size
    df = pd.DataFrame({'days': days, 'sample stat': sample_stat, 'CI': CIs}).set_index('days')
    df['lb'] = df['CI'].apply(lambda x: x[0,0])
    df['ub'] = df['CI'].apply(lambda x: x[0,1])
    fig = CI_ss_plot(df, width=width, x_title='Days', y_title='Relative RMSE')
    fig.layout.xaxis['tickvals'] = np.arange(-1, max_prediction_length+4, 4)
    fig.layout.xaxis['ticktext'] = ['0'] + [s.replace('.0', '') for s in fig.layout.xaxis['ticktext'][3::4]]
    return fig

@savedata
def target_scales(weather='all', species='Southern elephant seal', **model_kwargs):
    assert 'batch_size' not in model_kwargs, "batch_size should not be specified."
    tft = model.QuantileForecaster(weather=weather, species=species, **model_kwargs,
                                   batch_size=1)
    train_dataloader = tft.train_dataloader
    val_dataloader = tft.val_dataloader

    # compute target scales
    train_scales = torch.empty((len(train_dataloader), 2))
    for i, (x, _) in enumerate(tqdm(train_dataloader)):
        target_scales = x["target_scale"]
        t1, t2 = target_scales
        train_scales[i] = torch.stack([t1[0, 1], t2[0, 1]]) # scale1, scale2

    val_scales = torch.empty((len(val_dataloader), 2))
    for i, (x, _) in enumerate(tqdm(val_dataloader)):
        target_scales = x["target_scale"]
        t1, t2 = target_scales
        val_scales[i] = torch.stack([t1[0, 1], t2[0, 1]])

    train_scales = pd.DataFrame(train_scales.numpy(), columns=["scale1", "scale2"])
    val_scales = pd.DataFrame(val_scales.numpy(), columns=["scale1", "scale2"])
    return train_scales, val_scales

@savedata('all-tft')
def store_distribution_parameters(tft=None, **labels):
    def get_results(partition):
        dataloader = getattr(tft, f"{partition}_dataloader")
        raw_predictions, x = tft.model.predict(dataloader, mode="raw", return_x=True)
        distribution_params_raw = raw_predictions['prediction']
        distribution_params = tft.model.loss.map_x_to_distribution(distribution_params_raw)
        weights = distribution_params.mixture_distribution.probs.numpy()
        means = distribution_params.component_distribution.mean.numpy()
        scale_trils = distribution_params.component_distribution.scale_tril
        covariances = torch.matmul(scale_trils, scale_trils.transpose(-1, -2)).numpy()
        interpretation = tft.model.interpret_output(raw_predictions, reduction="sum")
        interpretation_raw = tft.model.interpret_output(raw_predictions, reduction="none")
        results = dict(x = x,
                       weights=weights,
                       means=means, # also prediction
                       scale_trils=scale_trils.numpy(),
                       covariances=covariances,
                       interpretation=interpretation,
                       interpretation_raw=interpretation_raw,
                       )
        results = {f'{k}_{partition}': v for k, v in results.items()}
        return results

    results = {**get_results('val'), **get_results('test')}
    results['static_variables'] = tft.model.static_variables
    results['encoder_variables'] = tft.model.encoder_variables
    results['decoder_variables'] = tft.model.decoder_variables

    return results

@savedata('all-patience')
def store_quantile_parameters(quantiles='exact', epochs=200, limit_train_batches=0.25, patience=40, gradient_clip_val=0.8, weather='all', species='Southern elephant seal', store_train=False, **model_kwargs):
    pretrained = model_kwargs.pop('pretrained', False)
    train_kwargs = dict(epochs=epochs, limit_train_batches=limit_train_batches, gradient_clip_val=gradient_clip_val, patience=patience)
    tft = model.QuantileForecaster(weather=weather, species=species, quantiles=quantiles, **model_kwargs)
    if pretrained:
        print("Loading pre-trained model ...\n")
        ID = model_kwargs.pop('ID', None)
        assert ID is not None, "ID should be specified when pretrained is True."
        results_full_dataset = quantile_results(weather=weather, species=species, quantiles=quantiles,
                                                params_idx=None, gradient_clip_val=gradient_clip_val, **model_kwargs) # passing params explicitely instead of params_idx
        pretrained_model = results_full_dataset['state_dict']
        tft_full_dataset = model.QuantileForecaster(weather=weather, species=species, quantiles=quantiles, **model_kwargs) # same params, all the dataset
        tft_full_dataset.model.load_state_dict(pretrained_model)
        tft.model = deepcopy(tft_full_dataset.model)
        model_kwargs['ID'] = ID
        print("Loaded pre-trained model.")
    tft.train(**train_kwargs)
    # store predictions
    print("Storing val and test predictions...")
    results = {**tft.get_results('val'), **tft.get_results('test')}
    ID = model_kwargs.get('ID', None)
    if ID is None: # if trained in all trajectories save model weights
        results['state_dict'] = tft.model.state_dict()

    if store_train:
        print("Storing train predictions...")
        results = {**results, **tft.get_results('train_predict')}
        # NOTE: Previous version is too demanding in memory
        # # store also training predictions to use it as input data for the distribution forecaster
        # out_raw_train, x_train = tft.model.predict(tft.train_dataloader, mode="raw", return_x=True)
        # y_pred_raw_train = out_raw_train['prediction']
        # y_pred_train = stack_targets(tft.model.to_prediction(out_raw_train))
        # y_pred_quantiles_train = stack_targets(tft.model.to_quantiles(out_raw_train))
        # results['x_train'] = x_train
        # results['y_pred_raw_train'] = y_pred_raw_train
        # results['y_pred_train'] = y_pred_train
        # results['y_pred_quantiles_train'] = y_pred_quantiles_train

    return results

@savedata
def hull_prediction(confidences=[0.5, 0.9, 0.95], partition='test', **labels):
    output = store_distribution_parameters(**labels)
    weights = output[f'weights_{partition}']
    means = output[f'means_{partition}']
    covariances = output[f'covariances_{partition}']
    num_trajectories, num_timesteps, *_ = means.shape
    x = output[f'x_{partition}']
    stack_targets = lambda t: torch.stack(t, axis=-1).numpy()
    y = stack_targets(x['decoder_target'])
    decoder_lengths = x['decoder_lengths'].numpy()

    results = {}
    for i in tqdm(range(num_trajectories)):
        for j in range(decoder_lengths[i]):
            dist_ij = distribution.BivariateMixture(means=means[i,j], covariances=covariances[i,j], weights=weights[i,j], size=10000)
            for c in confidences:
                dist_ij.generate_cr_points(alpha=1-c)
                # point results
                results[(i,j, f"y_pred-c", c)] = dist_ij.region_data.mean(axis=0)
                # distribution results (convex hull prediction interval)
                dist_ij.compute_convex_hull()
                results[(i,j, "hull-polygon", c)] = dist_ij.hull_polygon
                results[(i,j, "hull-area", c)] = dist_ij.convex_hull_area()
                results[(i,j, "hull-accuracy", c)] = int(geometry.is_point_inside_polygon(y[i,j], dist_ij.hull_polygon))
    return pd.Series(results)

@savedata
def alpha_shape_prediction(confidences=[0.5, 0.9, 0.95], partition='test', **labels):
    output = store_distribution_parameters(**labels)
    weights = output[f'weights_{partition}']
    means = output[f'means_{partition}']
    covariances = output[f'covariances_{partition}']
    num_trajectories, num_timesteps, *_ = means.shape
    x = output[f'x_{partition}']
    stack_targets = lambda t: torch.stack(t, axis=-1).numpy()
    y = stack_targets(x['decoder_target'])
    decoder_lengths = x['decoder_lengths'].numpy()

    results = {}
    pbar = tqdm(range(num_trajectories * num_timesteps))
    for i in range(num_trajectories):
        for j in range(decoder_lengths[i]):
            dist_ij = distribution.BivariateMixture(means=means[i,j], covariances=covariances[i,j], weights=weights[i,j], size=10000)
            for c in confidences:
                dist_ij.generate_cr_points(alpha=1-c)
                # alpha shape prediction
                dist_ij.compute_alpha_shape()
                exterior, holes = dist_ij.extract_boundaries_and_holes()
                results[(i,j, "alpha-shape-polygon", c)] = exterior
                results[(i,j, "alpha-shape-polygon-holes", c)] = holes
                results[(i,j, "alpha-shape-area", c)] = dist_ij.alpha_shape_area()
                inside_hole = any(geometry.is_point_inside_polygon(y[i,j], hole) for hole in holes)
                if inside_hole:
                    results[(i,j, "alpha-shape-accuracy", c)] = 0
                else:
                    inside_exterior = any(geometry.is_point_inside_polygon(y[i,j], ext) for ext in exterior)
                    results[(i,j, "alpha-shape-accuracy", c)] = int(inside_exterior)
            pbar.update(1)

    return pd.Series(results)


def quantile_prediction_2D(
    results: Dict = {},
    idx: int = 0,
    show_future_observed: bool = True,
    ax=None,
    day_pred=None,
    legend: bool = False,
    feature_names: List = ["SN", "WE"],
    partition: str = 'test',
    **kwargs
) -> plt.Figure:
    """
    Plot prediction of prediction vs actuals

    Args:
        idx: index of prediction to plot
        show_future_observed: if to show actuals for future. Defaults to True.
        ax: matplotlib axes to plot on

    Returns:
        matplotlib figure
    """
    if not results:
        results = quantile_results(**kwargs)

    x = results[f'x_{partition}']
    encoder_targets = custom_metrics.stack_targets(x["encoder_target"])
    decoder_targets = custom_metrics.stack_targets(x["decoder_target"])
    y_raws = results[f'y_pred_raw_{partition}']
    y_hats = results[f'y_pred_{partition}']
    y_quantiles = results[f'y_pred_quantiles_{partition}']

    # for each target, plot
    figs = []

    max_encoder_length = x["encoder_lengths"].max()
    def get_specs(target_idx):
        y_raw, y_hat, y_quantile, encoder_target, decoder_target = (y_raws[target_idx], y_hats[target_idx], y_quantiles[target_idx],
                                                                    encoder_targets[target_idx], decoder_targets[target_idx])
        y_all = np.concatenate([encoder_target[idx], decoder_target[idx]])
        y = np.concatenate(
            (
                y_all[: x["encoder_lengths"][idx]],
                y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
            ),
        )
        # move predictions to cpu
        y_hat = y_hat[idx, : x["decoder_lengths"][idx]]
        y_quantile = y_quantile[idx, : x["decoder_lengths"][idx]]
        y_raw = y_raw[idx, : x["decoder_lengths"][idx]]

        yrange = [
            min(y.min(), y_quantile.min()),
            max(y.max(), y_quantile.max())
        ]
        offset = 0.05 * (yrange[1] - yrange[0])
        yrange = [yrange[0] - offset, yrange[1] + offset]
        if day_pred is not None:
            y_quantile = y_quantile[[day_pred]]

        return yrange, y, y_hat, y_quantile, y_raw

    xlims, y1, y1_hat, y1_quantile, y1_raw = get_specs(0)
    ylims, y2, y2_hat, y2_quantile, y2_raw = get_specs(1)
    n_pred = y1_hat.shape[0]
    if day_pred is None:
        n_pred_shown = n_pred
    else:
        n_pred_shown = day_pred
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]
    plotter = ax.plot
    # plot observed history
    plotter(y1[:-n_pred], y2[:-n_pred], 'o-', label="observed", c=obs_color, ms=2)

    # plot observed prediction
    if show_future_observed:
        if n_pred_shown < n_pred:
            plotter(y1[-n_pred:-n_pred+n_pred_shown], y2[-n_pred:-n_pred+n_pred_shown], 'ko-', label=None, ms=5)
        else:
            plotter(y1[-n_pred:], y2[-n_pred:], 'ko-', label=None, ms=5)

    # plot prediction
    if day_pred is None:
        plotter(y1_hat, y2_hat, 'o-', label="predicted", c=pred_color, ms=5)
    else:
        plotter(y1_hat[:n_pred_shown-1], y2_hat[:n_pred_shown-1], 'o-', label=None, c='orange', ms=3)
        plotter([y1_hat[n_pred_shown]], [y2_hat[n_pred_shown]], 'o-', label="predicted", c=pred_color, ms=5)
    # plot predicted quantiles
    half_quantile = y1_quantile.shape[1] // 2
    plotter(y1_quantile[:, half_quantile], y2_quantile[:, half_quantile], c=pred_color, alpha=0.15)
    for i in range(half_quantile):
        x1, x2, y1, y2 = y1_quantile[:,i], y1_quantile[:, -i-1], y2_quantile[:,i], y2_quantile[:, -i-1]
        X = np.hstack([[x1[i], x1[i], x2[i], x2[i]] for i in range(x1.shape[0])])
        Y = np.hstack([[y1[i], y2[i], y2[i], y1[i]] for i in range(x1.shape[0])])
        ax.fill(X, Y, pred_color, alpha=0.15)
        #ax.fill_between(y1_hat, y2_quantile[:, i], y2_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
        #ax.fill_betweenx(y2_hat, y1_quantile[:, i], y1_quantile[:, -i - 1], alpha=0.15, fc=pred_color)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    if legend:
        fig.legend(loc='lower right')
    figs.append(fig)
    return fig

def quantile_prediction_2D_plotly(
    results: Dict = {},
    idx: int = 0,
    show_future_observed: bool = True,
    time_step: int = None,
    time_step_as_title: bool = True,
    legend: bool = False,
    feature_names: List = ["SN", "WE"],
    partition: str = 'test',
    **kwargs
) -> go.Figure:
    """
    Plot prediction of prediction vs actuals using Plotly

    Args:
        idx: index of prediction to plot
        show_future_observed: if to show actuals for future. Defaults to True.

    Returns:
        Plotly figure
    """
    if not results:
        results = quantile_results(**kwargs)

    x = results[f'x_{partition}']
    encoder_targets = custom_metrics.stack_targets(x["encoder_target"])
    decoder_targets = custom_metrics.stack_targets(x["decoder_target"])
    y_raws = results[f'y_pred_raw_{partition}']
    y_hats = results[f'y_pred_{partition}']
    y_quantiles = results[f'y_pred_quantiles_{partition}']
    max_encoder_length = x["encoder_lengths"].max()
    def get_specs(target_idx):
        y_raw, y_hat, y_quantile, encoder_target, decoder_target = (y_raws[target_idx], y_hats[target_idx], y_quantiles[target_idx],
                                                                    encoder_targets[target_idx], decoder_targets[target_idx])
        y_all = np.concatenate([encoder_target[idx], decoder_target[idx]])
        y = np.concatenate(
            (
                y_all[: x["encoder_lengths"][idx]],
                y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
            ),
        )
        # move predictions to cpu
        y_hat = y_hat[idx, : x["decoder_lengths"][idx]]
        y_quantile = y_quantile[idx, : x["decoder_lengths"][idx]]
        y_raw = y_raw[idx, : x["decoder_lengths"][idx]]

        yrange = [
            min(y.min(), y_quantile.min()),
            max(y.max(), y_quantile.max())
        ]
        offset = 0.05 * (yrange[1] - yrange[0])
        yrange = [yrange[0] - offset, yrange[1] + offset]
        if time_step is not None and time_step > 0:
            y_quantile = y_quantile[[time_step]]

        return yrange, y, y_hat, y_quantile, y_raw


    xlims, y1, y1_hat, y1_quantile, _ = get_specs(0)
    ylims, y2, y2_hat, y2_quantile, _ = get_specs(1)
    n_pred = y1_hat.shape[0]
    if time_step is None:
        n_pred_shown = n_pred
    else:
        n_pred_shown = time_step

    # Create a Plotly figure
    fig = get_figure(xaxis_title=feature_names[0], yaxis_title=feature_names[1],
                     xaxis_range=xlims, yaxis_range=ylims, showlegend=legend)
    if time_step_as_title:
        days = round(time_step_to_days(n_pred_shown), 2)
        fig.update_layout(title=dict(text=f"Day: {days}", x=0.5, xanchor='center', yanchor='top', y=0.95))

    prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
    obs_color = next(prop_cycle)["color"]
    pred_color = next(prop_cycle)["color"]

    if n_pred_shown >= 0:
        # Plot observed history
        fig.add_trace(go.Scatter(x=y1[:-n_pred], y=y2[:-n_pred], mode='lines+markers', name='Observed', marker=dict(size=8, color=obs_color)))

        if n_pred_shown > 0:
            # Plot observed prediction
            if show_future_observed:
                fig.add_trace(go.Scatter(x=y1[-n_pred:-n_pred+n_pred_shown], y=y2[-n_pred:-n_pred+n_pred_shown], mode='lines+markers', showlegend=False, marker=dict(color='black', size=10)))
            # Plot prediction
            fig.add_trace(go.Scatter(x=y1_hat[:n_pred_shown], y=y2_hat[:n_pred_shown], mode='lines+markers', showlegend=False, marker=dict(size=8, color=pred_color)))

        fig.add_trace(go.Scatter(x=[y1[-n_pred+n_pred_shown]], y=[y2[-n_pred+n_pred_shown]], mode='markers', name='Real', marker=dict(color='black', size=16)))

        fig.add_trace(go.Scatter(x=[y1_hat[n_pred_shown]], y=[y2_hat[n_pred_shown]], mode='markers', name='Predicted', marker=dict(size=16, color=pred_color)))

        # Plot predicted quantiles
        half_quantile = y1_quantile.shape[1] // 2
        for i in range(half_quantile):
            x1, x2, y1, y2 = y1_quantile[:,i], y1_quantile[:, -i-1], y2_quantile[:,i], y2_quantile[:, -i-1]
            fig.add_shape(type="rect", xref="x", yref="y",
                x0=x1[0], y0=y1[0], x1=x2[0], y1=y2[0],
                line=dict(color='rgba(255,255,255,0)', width=0),
                fillcolor=color_std(pred_color, opacity=0.15),
                          )
    else:
        # Plot only observed up to n_pred_shown in [-encoder_length, 0]
        end = max_encoder_length + n_pred_shown + 1
        fig.add_trace(go.Scatter(x=y1[:end], y=y2[:end], mode='lines+markers', name='Observed', marker=dict(size=8, color=obs_color)))

    return fig

def quantile_prediction_2D_gif(plotter='plotly', idx=0, results=None, n_pred=None, delete_components=True, partition='test', gif_observed=True, cds='mercator', **kwargs):
    if results is None:
        results = quantile_results(cds=cds, **kwargs)
    if n_pred is None:
        n_pred = results[f'y_pred_{partition}'].shape[-1]
    max_encoder_length = results[f'x_{partition}']['encoder_lengths'][0].numpy().max()
    if gif_observed:
        if plotter != 'plotly':
            warnings.warn(f"plotter {plotter} not supported for gif_observed=True. Using plotly instead.")
            plotter = 'plotly'
        day_preds = np.arange(-max_encoder_length, n_pred, dtype=int)
    else:
        day_preds = np.arange(0, n_pred, dtype=int)
    feature_names = params.cds_to_cols[cds]

    fig_dir = 'figs/forecasting/plot_prediction_2D/'
    gif_dir = os.path.join(fig_dir, "gif")
    component_dir = os.path.join(fig_dir, "components")
    Path(gif_dir).mkdir(exist_ok=True, parents=True)
    Path(component_dir).mkdir(exist_ok=True)

    fig_paths = []
    for day_pred in day_preds:
        fig_path = os.path.join(component_dir, f"{cds}_{plotter}_idx-{idx}_day_pred-{day_pred}.png")
        fig_paths.append(fig_path)
        if plotter == 'plotly':
            fig = quantile_prediction_2D_plotly(results, idx=idx, time_step=day_pred, feature_names=feature_names)
            fig.write_image(fig_path)
        elif plotter == 'matplotlib':
            fig = quantile_prediction_2D(results, idx=idx, day_pred=day_pred, feature_names=feature_names)
            fig.savefig(fig_path)
            plt.close(fig)
        else:
            raise ValueError(f"plotter {plotter} not recognized")

    gif_path = os.path.join(gif_dir, f"{cds}_{plotter}_idx-{idx}_n-pred-{n_pred}.gif")
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in fig_paths)
        # extract  first image from iterator
        img = next(imgs)
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=gif_path, format='GIF', append_images=imgs,
                 save_all=True, duration=1000, loop=0)
    if delete_components:
        for f in fig_paths:
            os.remove(f)
    return

def quantile_prediction_2D_movie(plotter='plotly', idx=0, results=None, n_pred=None, delete_components=True, partition='test', gif_observed=True,
                                 encoder_fps=10, decoder_fps=0.85, transition_fps=0.5, overwrite_components=False,
                                 cds='mercator', **kwargs):
    from moviepy.editor import concatenate_videoclips, ImageSequenceClip

    if results is None:
        results = quantile_results(cds=cds, **kwargs)
    if n_pred is None:
        n_pred = results[f'y_pred_{partition}'].shape[-1]
    max_encoder_length = results[f'x_{partition}']['encoder_lengths'].numpy()[idx]
    if gif_observed:
        if plotter != 'plotly':
            warnings.warn(f"plotter {plotter} not supported for gif_observed=True. Using plotly instead.")
            plotter = 'plotly'
        day_preds = np.arange(-max_encoder_length, n_pred, dtype=int)
    else:
        day_preds = np.arange(0, n_pred, dtype=int)
    feature_names = params.cds_to_cols[cds]

    fig_dir = 'figs/forecasting/plot_prediction_2D/'
    movie_dir = os.path.join(fig_dir, "movie")
    component_dir = os.path.join(fig_dir, "components")
    Path(movie_dir).mkdir(exist_ok=True, parents=True)
    Path(component_dir).mkdir(exist_ok=True)
    specs_str = dict_to_id(kwargs)


    print("Creating components...")
    fig_paths = []
    for day_pred in tqdm(day_preds):
        fig_path = os.path.join(component_dir, f"{cds}_{plotter}_{specs_str}_idx-{idx}_day_pred-{day_pred}.png")
        fig_paths.append(fig_path)
        if os.path.exists(fig_path) and not overwrite_components:
            continue
        else:
            if plotter == 'plotly':
                fig = quantile_prediction_2D_plotly(results, idx=idx, time_step=day_pred,
                                                    feature_names=feature_names, **kwargs)
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                fig.write_image(fig_path)
            elif plotter == 'matplotlib':
                fig = quantile_prediction_2D(results, idx=idx, day_pred=day_pred,
                                             feature_names=feature_names, **kwargs)
                fig.savefig(fig_path)
                plt.close(fig)
            else:

                raise ValueError(f"plotter {plotter} not recognized")
     # Split the paths into two sets
    encoder_fig_paths = fig_paths[:max_encoder_length-1]
    transition_fig_paths = fig_paths[max_encoder_length-1:max_encoder_length]
    pred_fig_paths = fig_paths[max_encoder_length:]

    # Create two clips with different fps
    encoder_clip = ImageSequenceClip(encoder_fig_paths, fps=encoder_fps)
    transition_clip = ImageSequenceClip(transition_fig_paths, fps=transition_fps)
    pred_clip = ImageSequenceClip(pred_fig_paths, fps=decoder_fps)

    # Concatenate the clips
    final_clip = concatenate_videoclips([encoder_clip, transition_clip, pred_clip])

    # Save the video
    movie_path = os.path.join(movie_dir, f"{cds}_{plotter}_{specs_str}_idx-{idx}_n-pred-{n_pred}.mp4")
    final_clip.write_videofile(movie_path, codec='libx264')
    if delete_components:
        for f in fig_paths:
            os.remove(f)
    return

@savefig('all-as_weekly-decoder_lims-day_offset-y_max')
def attention_plot(task='forecasting', add_cumulative=True, params_idx='best', ver=2, as_weekly=False, decoder_lims=True, day_offset=0.5, y_max=None, **kwargs):
    if ver == 1:
        interpretation, att_cum = _attention_CIs(task=task, params_idx=params_idx, **kwargs)
        att_cum = att_cum.applymap(lambda x: np.clip(x, 0, 1))
        attention = interpretation['attention'].unstack(-1)
    elif ver == 2:
        colmap = {'sample_stat': 'mean'}
        interpretation = attention_CIs(task=task, params_idx=params_idx, **kwargs)
        attention = interpretation['att'].rename(columns=colmap)
        att_cum = interpretation['att_cum'].rename(columns=colmap)
    else:
        raise ValueError(f"ver {ver} not recognized.")

    # get training days
    results = quantile_results(params_idx=0, task=task, **kwargs)
    x = results['x_test']
    train_days = time_step_to_days(x['max_encoder_length'][0])

    days = time_step_to_days(attention.index.values)
    days -= train_days # encoder -> days into the past

    CI = np.vstack(attention.CI.values)
    mean = attention['mean'].values
    max_attention = max(CI.max(), mean.max())

    color, color_cum = plotly_default_colors(2)
    if y_max is None:
        y_max = max_attention*1.05
    fig = get_figure(xaxis_title='Time (days)', yaxis_title='Attention', xaxis_range=[days[0]-day_offset, days[-1]+day_offset], yaxis_range=[0, y_max])
    plot_confidence_bands(fig=fig, x=days, y=mean, CI=CI, color=color, opacity=0.3)
    if add_cumulative:
        # plot cumulative in secondary y-axis. The ticks are in red in the rigth.
        cumulative = np.hstack([0, att_cum['mean'].values])
        CI_cum = np.vstack((np.zeros((1,2)), *att_cum.CI.values))
        days_cum = np.hstack([days[0], days])
        tickvals = [0, 0.25, 0.5, 0.75, 1]
        ticktext = [f'<span style="color: {color_cum};">{v}</span>' for v in tickvals]

        plot_confidence_bands(fig=fig, x=days_cum, y=cumulative, CI=CI_cum, color=color_cum, opacity=0.2, yaxis='y2')
        fig.update_layout(yaxis2=dict(anchor="x", overlaying="y", side="right", showgrid=False,
                                      tickvals=tickvals, ticktext=ticktext, tickfont_size=32))

    if task == 'imputation' and decoder_lims:
        dx = days[1] - days[0]
        predict_days = time_step_to_days(x['max_decoder_length'][0]) - dx # time_step_to_days automatically adds 1 step.
        decoder_end = predict_days - dx
        # add vertical lines at x=0 and x=decoder_end
        for pos in [0, decoder_end]:
            fig.add_shape(type='line', x0=pos, x1=pos, y0=0, y1=y_max, line=dict(color='black', width=6, dash='dot'))

    if as_weekly:
        tickvals = np.arange(days[0], 7, 7, dtype=int)
    else:
        tickvals = np.arange(days[0], days[-1]+1, 1, dtype=int)
    ticktext = [str(t) for t in tickvals]
    fig.update_layout(xaxis=dict(tickvals=tickvals, ticktext=ticktext))
    return fig

@savefig('all-max_display-bar_halfwidth-offset-text_pos-title-xlims')
def feature_importance_plot(ver=2, max_display=10, bar_halfwidth=0.4, offset=0.3, text_pos=-0.05, xlims=None, var_type='encoder', params_idx='best', title=True, **attention_kwargs):
    """
    var_type: 'encoder', 'decoder', 'static'
    """
    if ver == 1:
        interpretation, _ = _attention_CIs(params_idx=params_idx, **attention_kwargs)
        df = interpretation[f'{var_type}_variables'].unstack(-1).sort_values('mean', ascending=True)
        df['CI'] = df['CI'].apply(lambda x: x[0])
    elif ver == 2:
        interpretation = attention_CIs(params_idx=params_idx, **attention_kwargs)
        df = interpretation[var_type].rename(columns={'sample_stat': 'mean'})
        df = df.sort_values('mean', ascending=True)
    df = df.iloc[-max_display:]
    cds = attention_kwargs.get('cds', 'mercator')
    cds_cols = params.cds_to_cols[cds]
    features = [f.capitalize().replace("_", " ") if f not in cds_cols else f for f in df.index]
    task = attention_kwargs.get('task', 'forecasting')
    if task == 'forecasting':
        feature_map = params.feature_map
    elif task == 'imputation':
        feature_map = params.feature_map_imputation
    else:
        raise ValueError(f"task {task} not recognized.")
    features = [feature_map.get(f, f) for f in features]
    max_value = df['mean'].max()
    xmax = max(max_value*1.1, 0.25)

    fig = get_figure(xaxis_title="Feature importance (⟨Attention⟩)", xaxis_range=[-offset, xmax])
    fig.update_layout(yaxis=dict(visible=False),
                      xaxis=dict(showline=True, linecolor='black', linewidth=2.4,
                                 tickvals=np.arange(0, xmax, 0.25)))
    if title:
        fig.update_layout(title=dict(text=f"{var_type.capitalize()} variables", x=0.5, xanchor='center', yanchor='top', y=0.95))
    if xlims is not None:
        fig.update_layout(xaxis=dict(range=list(xlims),
                                     tickvals=np.arange(0, xlims[1], 0.25)))
    # first plot feature names in the center
    fig.add_trace(go.Scatter(x=text_pos * np.ones((max_display)), y=np.arange(max_display), text=features, mode='text', textposition='middle left', textfont=dict(color='black', size=30), showlegend=False))
    color = plotly_default_colors(1)[0]
    for i, val in enumerate(df['mean']):
        fig.add_shape(type='rect',
                      x0=0,
                      x1=val,
                      y0=i-bar_halfwidth,
                      y1=i+bar_halfwidth,
                      fillcolor=color
                      )
    # add CI
    def plot_errorbar(fig, x, y, color='black', lw=1.5, error_halfwidth=0.07):
        fig.add_shape(type='line',
                      x0=x,
                      x1=x,
                      y0=y-error_halfwidth,
                      y1=y+error_halfwidth,
                      line=dict(color=color, width=lw)
                      )
        return
    for i, ci in enumerate(df['CI']):
        fig.add_shape(type='line',
                      x0=ci[0],
                      x1=ci[1],
                      y0=i,
                      y1=i,
                      line=dict(color='black', width=2)
                      )
        plot_errorbar(fig, ci[0], i)
        plot_errorbar(fig, ci[1], i)
    return fig

def model_comp_plot(task='forecasting', params_idx='best', add_baseline=True, magnitude='area', area_method='km', metric='rae', mpl_val=True, add_single=True,
                    add_ref_area=True, add_bivariate=False, **kwargs):
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)
    coverage, area = area_coverage_CI(task=task, params_idx=params_idx, area_method=area_method, mpl_val=mpl_val, **kwargs)
    confidences = coverage.columns.levels[0]
    colors = plotly_default_colors(confidences.size)
    tickvals = np.hstack((0, confidences))
    ticktext = ['0'] + [f'<span style="color:{color}; font-weight:bold">{confidence}</span>' for confidence, color in zip(confidences, colors)]
    x = time_step_to_days(coverage.index.values)
    default_colors = plotly_default_colors(6)
    settings = ['TFT', 'RW', 'CRW', 'MP', 'Naive']
    colors = {s: c for s, c in zip(settings, default_colors)}

    titles = [f'Target coverage: {confidence}' for confidence in confidences]
    x_title = ' <br>Days'
    if magnitude == 'coverage' and metric != 'CI':
        if metric == 'rae':
            y_title = r'$\Huge{\displaystyle \frac{\alpha - \hat{\alpha}}{\alpha}}$'
        elif metric == 'arae':
            y_title = r'$\frac{|\alpha - \hat{\alpha}|}{\alpha}$<br> '
            warnings.warn("metric 'arae' requires C.I. to be recomputed.")
        else:
            raise ValueError(f"metric {metric} not recognized.")
    elif magnitude == 'area' and area_method == 'km':
        y_title = 'Area [km<sup>2</sup>]<br> '
    else:
        y_title = f'{magnitude.capitalize()}<br> '

    fig = get_subplots(cols=3, rows=1, simple_axes=True,
                       subplot_titles=titles,
                       x_title=x_title, y_title=y_title,
                       shared_yaxes=True,
                       # shared_yaxes=magnitude == 'area' or (magnitude == 'coverage' and metric == 'CI'),
                       )
    def _plot(df, dash, name, add_reference=False, label=True):
        line_specs=dict(dash=dash)
        color = colors[name]
        for col, confidence in enumerate(confidences, start=1):
            df_c = df[confidence]
            if magnitude == 'coverage':
                if metric == 'rae':
                    df_c = rae(1 - df_c, 1 - confidence)
                    CI = np.clip(np.vstack(df_c['CI'].apply(np.sort).values), None, 1)
                elif metric == 'CI':
                    CI = np.clip(np.vstack(df_c['CI'].values), 0, 1)
                elif metric == 'arae':
                    df_c = arae(1 - df_c, 1 - confidence)
                    CI = np.vstack(df_c['CI'].apply(np.sort).values)
                else:
                    raise ValueError(f"metric {metric} not recognized.")
            else:
                CI = np.clip(np.vstack(df_c['CI'].values), 0, None)
            y = df_c['mean'].values
            single_plot = plot_confidence_bands(fig=None, x=x, y=y, CI=CI, color=color, opacity=0.1,
                                                line_specs=line_specs,
                                                lw=8,
                                                label=None if (not label or col > 1) else name)

            # add to subplots
            for trace in single_plot.data:
                fig.add_trace(trace, row=1, col=col)

            if magnitude == 'coverage' and add_reference:
                if metric in ['rae', 'arae']:
                    fig.add_shape(type="line", x0=x.min(), y0=0, x1=x.max(), y1=0, line=dict(color='black', dash='dash', width=4), row=1, col=col)

                else:
                    fig.add_shape(type="line", x0=x.min(), y0=confidence, x1=x.max(), y1=confidence, line=dict(color='black', dash='dash', width=4), row=1, col=col)

        return


    _plot(coverage if magnitude == 'coverage' else area,
          'solid', 'TFT', add_reference=True)

    if add_bivariate:
        dist_kws = kwargs.copy()
        dist_kws.update(params.model_specs()['TFT_dist'])
        dist_kws['quantiles'] = 'all'
        dist_kws['max_train_days'] = 4
        del dist_kws['baseline']
        del dist_kws['ssm_model']
        coverage_d, area_d = area_coverage_CI(task=task, **dist_kws, mpl_val=mpl_val)
        colors['TFT[B]'] = 'gold'
        _plot(coverage_d if magnitude == 'coverage' else area_d,
              'dash', 'TFT[B]', add_reference=True)

    if add_single:
        coverage_single, area_single = area_coverage_CI(task=task, params_idx=params_idx, area_method=area_method, mpl_val=mpl_val, ID='all', **kwargs)
        colors['TFT[s]'] = 'cyan'
        _plot(coverage_single if magnitude == 'coverage' else area_single,
              'dash', 'TFT[s]')
    for model in ['rw', 'crw', 'mp']:
        df = ssm.eval_ssm_CI(task=task, model=model, magnitude=magnitude, area_method=area_method, se_val_fit=mpl_val)
        _plot(df, 'dash', model.upper())
    if add_baseline:
        if task == 'imputation':
            extra_specs = params.baseline_specs_imputation.copy()
        else:
            extra_specs = {}
        baseline_coverage, baseline_area = area_coverage_CI_baseline(task=task, params_idx=0, area_method=area_method, CI_mpl=mpl_val, **kwargs, **extra_specs)
        _plot(baseline_coverage if magnitude == 'coverage' else baseline_area,
              'dash', 'Naive')
    if magnitude == 'area' and add_ref_area:
        area_max, area_ref = ref_area_CI(task=task, params_idx=params_idx, mpl_val=False, **kwargs)
        colors['A<sub>ref,max</sub>'] = 'black'
        _plot(area_ref, 'dot', 'A<sub>ref,max</sub>', add_reference=False, label=True)
        _plot(area_max, 'dot', 'A<sub>ref,max</sub>', add_reference=False, label=False)

    fig.update_layout(margin=dict(b=150, l=150))
    for annotation in fig.layout.annotations:
        if annotation.text == x_title:
            annotation.y = 0
            annotation.yshift = 20
            annotation.font.size = 65
        elif annotation.text == y_title:
            annotation.x = 0.02 if ((magnitude != 'coverage') or (magnitude == 'coverage' and metric == 'CI')) else -0.01
            annotation.font.size = 65
        else:
            annotation.font.size = 50
    if magnitude == 'area':
        fig.update_layout(**mod_logaxes_expfmt(fig, axes=['y']),
                          yaxis_dtick=1)
    elif metric == 'CI':
        fig.update_layout(**mod_range(fig, ([0,1.05],), axes=['y']))
    elif metric == 'rae':
        fig.update_layout(**mod_range(fig, ([-5,1.1],), axes=['y']),
                          margin_l=170)
    return fig

@savefig
def model_comp_coverage(params_idx='best', metric='rae', mpl_val=True, **kwargs):
    return model_comp_plot(params_idx=params_idx, metric=metric, magnitude='coverage', mpl_val=mpl_val, **kwargs)

@savefig
def model_comp_area(params_idx='best', area_method='km', mpl_val=True, **kwargs):
    return model_comp_plot(params_idx=params_idx, magnitude='area', area_method=area_method, mpl_val=mpl_val, **kwargs)

@savedata
def conf_region_aggregate_CI(baseline=False, ssm_model=None, magnitude='coverage', confidence=0.5, metric=None, area_method='km', params_idx='best', CI_method='bca', **kwargs):
    """
    Compute the aggregate metric for the confidence region.

    metric: 'rae', 'arae', None ('CI')

    Returns:
        - DataFrame with the aggregate metric across animals and time steps.
    """
    if ssm_model is not None:
        df = ssm.eval_ssm(model=ssm_model, area_method=area_method, **kwargs)
        df = df.groupby(['time_step'])[f'{magnitude}_{confidence}'].apply(np.array)
        df = pd_utils.expand_sequences(df.to_frame()).T.droplevel(0)
        df.index.name = 'animal'
        df.columns.name = 'time_step'
    elif baseline:
        if magnitude == 'coverage':
            df = baseline_coverage_area(**kwargs)
            df = df[confidence].unstack(0)
        elif magnitude == 'area':
            df = quantile_baseline_area(area_method=area_method, **kwargs)
            df = df.swaplevel(0, -1).loc[confidence].unstack(-1)
    else:
        quantiles = kwargs.get('quantiles', 'exact')
        if quantiles == 'exact':
            if params_idx == 'best':
                params_idx, _ = quantile_best_model(**kwargs)
            df = quantile_coverages(params_idx=params_idx, area_method=area_method, **kwargs)[2]
        elif quantiles == 'all':
            del kwargs['quantiles']
            if params_idx == 'best':
                best_model_keys = ['task', 'density', 'rho']
                best_model_kws = {}
                for k in best_model_keys:
                    if k in kwargs:
                        best_model_kws[k] = kwargs[k]
                mpl_val = kwargs.get('mpl_val', True)
                best_model_kws['rho'] = mpl_val and best_model_kws.get('rho', True)
                specs = dist_best_model_specs(**best_model_kws, mpl_val=mpl_val)
                kwargs.update(specs)
                if mpl_val:
                    del kwargs['rho']
            else:
                kwargs['params_idx'] = params_idx
            df = dist_pr(**kwargs)
            df = df.astype(float) # np.float64 and bool -> float
        df = df.swaplevel(0, -1).loc[confidence, magnitude].unstack(-1)

    # DEPRECATED version:
    # Xs = df.stack(dropna=False).groupby('time_step').apply(np.array)
    # Xs = tuple(Xs)
    # out = custom_metrics._aggregate_CI_across_time_steps(Xs, nb_mean, labels='', R=int(1e4))

    # NEW VERSION
    df_input = df.stack().swaplevel().to_frame(magnitude)
    out = custom_metrics.aggregate_CI_across_time_steps(df_input, custom_metrics.nb_mean_not_nans, labels=[magnitude], R=int(1e4), CI_method=CI_method)

    sample_stat = out['sample_stat'].iloc[0]
    CI = out['CI'].iloc[0]
    return sample_stat, CI
    # if magnitude == 'coverage':
    #     return aggregate_metric_CI(df, metric=metric, target=confidence)
    # else:
    #     return aggregate_metric_CI(df)

def aggregate_summary(task='forecasting', metric='alpha', raw=False, simplify=True, CI_expansion=True, force_CI_expansion=True, divide_area=1, unit=True, area_exp_fmt=True):
    magnitude_to_specs = {'area': dict(func=conf_region_aggregate_CI, unit='km²'),
                          'coverage': dict(func=conf_region_aggregate_CI, unit='')}

    dfs = []

    # first process distance
    df = aggregate_CI_summary(point_prediction_aggregate_CI, task=task, ci_func=False) # point prediction does not use CI_expansion
    df = df.stack()
    magnitude_str = 'Distance [km]'
    df.index = pd.MultiIndex.from_tuples([(magnitude_str, *i) for i in df.index])
    df.index.names = ['magnitude', 'model', 'statistic']
    dfs.append(df)

    for confidence in tqdm(params.confidences):
        for magnitude, specs in magnitude_to_specs.items():
            df = aggregate_CI_summary(specs['func'], CI_expansion=CI_expansion, force_CI_expansion=force_CI_expansion, task=task,
                                      confidence=confidence, magnitude=magnitude)
            df = df.stack()
            if magnitude == 'coverage' and metric == 'alpha':
                df = 1 - df
                df = df.unstack()
                df['CI'] = df['CI'].apply(lambda x: x[::-1])
                df = df.stack()
                magnitude_str = metric
            else:
                magnitude_str = magnitude.capitalize()
            if specs['unit']:
                magnitude_str += f' [{specs["unit"]}]'
            df.index = pd.MultiIndex.from_tuples([(f'{magnitude_str} (α={round(1-confidence, 2)})', *i) for i in df.index])
            df.index.names = ['magnitude', 'model', 'statistic']
            dfs.append(df)
    df = pd.concat(dfs)

    if not raw:
        df = df.unstack(-1)
        df['CI'] = df['CI'].apply(np.squeeze)
        def process_row(row):
            if 'Area' in row.name[0]: # rounding
                if divide_area > 1:
                    row = row / divide_area
                if simplify:
                    num_without_exp = lambda f: f'{f:.2e}'.split('e')[0]
                    exp_low_str = f'{row.CI[0]:.2e}'.split('e')[1]
                    exp_high_str = f'{row.CI[1]:.2e}'.split('e')[1]
                    exp_low = int(exp_low_str)
                    exp_high = int(exp_high_str)
                    diff = exp_high - exp_low
                    if not diff:
                        return f'{num_without_exp(row.sample_stat)} [{num_without_exp(row.CI[0])}, {num_without_exp(row.CI[1])}] e{exp_low_str}'
                    else:
                        # everything in terms of exp_low
                        sample_stat = row.sample_stat / 10**exp_low
                        CI = row.CI / 10**exp_low
                        return f'{sample_stat:.1f} [{CI[0]:.1f}, {CI[1]:.1f}] e{exp_low_str}'
                elif area_exp_fmt:
                    return f'{row.sample_stat:.2e} [{row.CI[0]:.2e}, {row.CI[1]:.2e}]'
                else:
                    return f'{row.sample_stat:.2f} [{row.CI[0]:.2f}, {row.CI[1]:.2f}]'
            elif 'Distance' in row.name[0]:
                return f'{int(row.sample_stat)} [{int(row.CI[0])}, {int(row.CI[1])}]'
            else:
                return f'{row.sample_stat:.2f} [{row.CI[0]:.2f}, {row.CI[1]:.2f}]'
        df = df.apply(process_row, axis=1).unstack(0)

    cols = df.columns
    col_order = ([col for col in cols if not 'Area' in col and not 'Distance' in col]
             + [col for col in cols if 'Area' in col]
             + [col for col in cols if 'Distance' in col])
    model_order = [*params.model_specs().keys()]
    df = df.loc[model_order, col_order]
    df.index.name = df.index.name.capitalize()
    colname = df.columns.name.capitalize()
    if not unit:
        df.columns = [re.sub(r'\[.*\]', '', col) for col in df.columns]
    df.columns.name = colname
    return df

def quantile_val_loss(params_idx=0, quantiles='exact', cds='mercator', delete_missing=True, **kwargs):
    results = quantile_results(params_idx=params_idx, quantiles=quantiles, cds=cds, **kwargs)
    x = results['x_val']
    y_real = x['decoder_target']
    y_pred = results['y_pred_raw_val']
    Q = params.default_quantiles[quantiles]

    loss = QuantileLoss(quantiles=Q)
    loss_x = loss.loss(y_pred[0], y_real[0])
    loss_y = loss.loss(y_pred[1], y_real[1])
    if delete_missing:
        missing = get_missing_values(params_idx=params_idx, quantiles=quantiles, cds=cds, partition='val', **kwargs)
        missing = np.hstack(missing)
        loss_x = loss_x.reshape(-1, loss_x.shape[-1])[~missing] # join the first two dimensions
        loss_y = loss_y.reshape(-1, loss_y.shape[-1])[~missing]

    loss_x = loss_x.mean().item()
    loss_y = loss_y.mean().item()

    return pd.Series(dict(loss_x=loss_x, loss_y=loss_y), name=params_idx)

def eval_quantile_val_coverage(params_idx=0, criterion='rae', delete_missing=True, **kwargs):
    df = quantile_coverages(params_idx=params_idx, partition='val', delete_missing=delete_missing, **kwargs)[2]
    df = df['coverage']

    avg_cov = df.groupby(['time_step', 'confidence']).mean()
    avg_cov = avg_cov.unstack(1)
    confidences = avg_cov.columns.values

    avg_alpha = 1 - avg_cov.mean().values
    alpha_target = 1 - confidences

    if criterion == 'rae':
        error = np.abs(avg_alpha - alpha_target) / alpha_target
    elif criterion == 'diff':
        error = avg_alpha - alpha_target
    elif criterion == 'adiff':
        error = np.abs(avg_alpha - alpha_target)
    elif criterion == 'ae':
        error = (avg_alpha - alpha_target) / alpha_target
    elif criterion == 'std':
        error = np.std(avg_cov)
    elif criterion == 'Q':
        Q = custom_metrics.coverage_quality(avg_alpha, alpha_target=alpha_target) # 0 worst, 1 best
        error = 1 - Q
    else:
        raise ValueError(f"Invalid criterion: {criterion}")
    return pd.Series(error, index=confidences, name=params_idx)

@savedata
def quantile_best_model(magnitude='quality', criterion='Q', reduction='mean', force_CI_expansion=True, num_models=5, **kwargs):
    """
    Selects the best performing model on the validation set based on the criterion.
    """
    if 'ID' in kwargs and kwargs['ID'] is not None:
        del kwargs['ID']
        print("Ignoring ID. Choosing the best parameters for the whole dataset")

    del_kwargs = ['partition', 'ID', 'pretrained', 'eval_mode']
    for k in del_kwargs:
        if k in kwargs:
            del kwargs[k]

    mpl_val = kwargs.get('mpl_val', True)
    value = None
    direction = 'minimize'
    if not mpl_val and force_CI_expansion:
        print("Selecting best model accounting for CI expansion, despite having passed 'mpl_val=False'")
        kwargs['mpl_val'] = True
    else:
        kwargs['mpl_val'] = mpl_val
    if magnitude == 'val_loss':
        results = pd.concat([quantile_val_loss(params_idx=i, **kwargs) for i in range(num_models)], axis=1)
    elif magnitude == 'coverage':
        results = pd.concat([eval_quantile_val_coverage(params_idx=i, criterion=criterion, **kwargs) for i in range(num_models)], axis=1)
        magnitude = f'coverage error rate ({criterion})'
    elif magnitude == 'quality':
        results = pd.concat([getattr(quality_sample(params_idx=i, partition='val', **kwargs), reduction)() for i in range(num_models)], axis=1).T
        value = results[criterion]
        magnitude = criterion
        direction = 'maximize'
    else:
        raise ValueError(f"magnitude {magnitude} not recognized")

    if value is None:
        if isinstance(reduction, str):
            value = getattr(results, reduction)()
        elif callable(reduction):
            value = results.apply(reduction, axis=0)
        else:
            raise ValueError("reduction must be either a string or a callable")

    if direction == 'minimize':
        candidates = value[value.round(4) == value.round(4).min()]
    else:
        candidates = value[value.round(4) == value.round(4).max()]
    if len(candidates) > 1:
        print("Multiple candidates. Selecting minimum std in validation coverage.")
        std = pd.concat([eval_quantile_val_coverage(params_idx=i, criterion='std', **kwargs) for i in range(num_models)], axis=1).mean()
        best = std.loc[candidates.index].idxmin()
    else:
        if direction == 'minimize':
            best = value.idxmin()
        else:
            best = value.idxmax()
    best = int(best) # np.int64 -> int
    print(f"Quantile best model: {best}\nwith {magnitude}: {value[best]:.2f}")
    return best, value[best]

def q_to_polygon(y_quantile, conf_idx):
    """
    A --- B
    |     |
    C --- D
    """
    polygon = [[y_quantile[0, conf_idx[0]], y_quantile[1, conf_idx[1]]], # A
               y_quantile[:, conf_idx[1]], # B
               [y_quantile[0, conf_idx[1]], y_quantile[1, conf_idx[0]]], # D
                y_quantile[:, conf_idx[0]],  # C
               ]
    return np.array(polygon)

@savefig('i+c+step+cds+geo+text+title_type')
def trajectory_confidence_region(fig=None, plot_pr=True, results=None, PR_label=None, plot_rest=True, i=28, c=0.9, step=3, partition='test', cds='mercator', geo=True, projection_type='azimuthal equal area', title_type='step', legend=True, legend_orientation='v', text=False, n_obs=None, n_obs_lims=None, lw=None, ms=None, ms_p=None, lw_p=None, mlw=None, lc=None, lc_p=None, mlw_p=None, initial_loc=False, mpl_val=True, task='forecasting', **kwargs):
    """
    c = confidence
    projection_type: 'orthographic', 'azimuthal equal area'
    other title_type: 'cds', 'step', 'step_n'
    """
    geo_to_specs = {True: dict(lw=3, ms=10, ms_p=15, lw_p=3, mlw=2, lc='black', mlw_p=3, lc_p='black'),
                    False: dict(lw=3, ms=10, ms_p=15, lw_p=3, mlw=2, lc='black', mlw_p=3, lc_p='black')}
    specs = geo_to_specs[geo]
    local_specs = dict(lw=lw, ms=ms, ms_p=ms_p, lw_p=lw_p, mlw=mlw, lc=lc, mlw_p=mlw_p, lc_p=lc_p)
    for k, v in local_specs.items():
        if v is not None:
            specs[k] = v

    confidence_to_idxs = {0.5: [2, -3],
                          0.9: [1, -2],
                          0.95: [0, -1]}
    conf_idx = confidence_to_idxs[c]

    if results is None:
        results = quantile_results(cds=cds, task=task, **kwargs)
    x = results[f'x_{partition}']
    y_quantiles = results[f'y_pred_raw_{partition}']
    if mpl_val:
        print("Expanding quantiles according to mpl_val")
        mpls = TFT_mpl_val(cds=cds, task=task, delete_missing=True, **kwargs)
        y_quantiles = expand_TFT_quantiles(y_quantiles, mpls)
    y_quantile = y_quantiles[:,i]
    y_pred = results[f'y_pred_{partition}'][:, i]
    y_real = x['decoder_target']
    y_real = np.stack(y_real, axis=0)[:, i]
    y_observed = x['encoder_target']
    length = x['encoder_lengths'][i]
    y_observed = np.stack(y_observed, axis=0)[:, i, :length]
    targets = params.cds_to_cols[cds]
    if n_obs is None:
        n_obs = length
    if n_obs_lims is None:
        n_obs_lims = max(0, step)
    if task == 'imputation':
        # Obtain future coordinate
        encoder_variables = results['encoder_variables'] # correct way: dataset.reals_with_future.index('future_X'), however indices for future_X,Y coincide between results['encoder_variables'] and dataset.reals_with_future
        future_cds = [encoder_variables.index('future_X'), encoder_variables.index('future_Y')]
        x_cont = x['encoder_cont'][i]
        Xf, Yf = x_cont[..., future_cds].transpose(0, 1)
        def rescale(Z, k):
            avg, sigma = x['target_scale'][k][i]
            return Z*sigma + avg
        Xf = rescale(Xf, 0).numpy()
        Yf = rescale(Yf, 1).numpy()
        y_future = np.vstack((Xf, Yf))
        length_future = x['future_lengths'].numpy()[i]
        y_future = y_future[:, :length_future]
    else:
        y_future = None

    def compute_lims(y_quantile, y_pred, y_real, y_observed, offset_lon=0.5, offset_lat=0.25, y_future=None):
        offset = np.array([offset_lat, offset_lon])
        mins = np.array(([y_quantile[:, :n_obs_lims+1, conf_idx].min(axis=2).min(axis=1),
                          y_pred[:, :n_obs_lims+1].min(axis=1),
                          y_real[:, :n_obs_lims+1].min(axis=1),
                          y_observed[..., -n_obs:].min(axis=1)]
                         + ([] if y_future is None else [y_future.min(axis=1)])
                         )).min(axis=0)
        maxs = np.array(([y_quantile[:, :n_obs_lims+1, conf_idx].max(axis=2).max(axis=1),
                          y_pred[:, :n_obs_lims+1].max(axis=1),
                          y_real[:, :n_obs_lims+1].max(axis=1),
                          y_observed[..., -n_obs:].max(axis=1)]
                         + ([] if y_future is None else [y_future.max(axis=1)])
                         )).max(axis=0)
        mins -= offset
        maxs += offset

        lat_lim = [mins[0], maxs[0]]
        lon_lim = [mins[1], maxs[1]]
        return lat_lim, lon_lim

    # predicted_color = plotly_default_colors(1)[0]
    predicted_color = 'cyan'
    if geo:
        if cds == 'mercator':
            to_degrees = lambda y: np.stack(space.mercator_inv(*y), axis=0) * 180 / np.pi
            y_quantile = to_degrees(y_quantile)
            y_pred = to_degrees(y_pred)
            y_real = to_degrees(y_real)
            y_observed = to_degrees(y_observed)
            if y_future is not None:
                y_future = to_degrees(y_future)

        lat_lim, lon_lim = compute_lims(y_quantile, y_pred, y_real, y_observed, y_future=y_future)

        if fig is None:
            fig = get_figure(height=800, width=800)
        if plot_rest:
            fig.add_trace(go.Scattergeo(lat=y_observed[0], lon=y_observed[1], mode='lines+markers', name='Observed', marker=dict(color='dimgray', size=specs['ms'], line_width=specs['mlw'])))
            if y_future is not None:
                fig.add_trace(go.Scattergeo(lat=y_future[0], lon=y_future[1], showlegend=False, mode='lines+markers', marker=dict(color='dimgray', size=specs['ms'], line_width=specs['mlw'])))
            if initial_loc:
                fig.add_trace(go.Scattergeo(lat=[y_observed[0, 0]], lon=[y_observed[1, 0]], mode='markers', showlegend=False, marker=dict(color='dimgray', size=specs['ms_p'], line=dict(color='black', width=specs['mlw'])), line_width=specs['lw_p']))
        if step+1:
            if plot_pr:
                # prediction region
                p = q_to_polygon(y_quantile[:, step], conf_idx)
                p = np.vstack((p, p[0]))
                p = p[::-1]
                fig.add_trace(go.Scattergeo(
                    lat=p[:, 0],
                    lon=p[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(color='blue', width=0.8),
                    name = 'PR' if PR_label is None else PR_label,
                )
                )
            real_color = 'gold'
            # real_color = '#BA55D3'
            # real_color = '#D02090' # – Bright Cerise

            if plot_rest:
                # connect observed to real future location
                fig.add_trace(go.Scattergeo(lat=[y_observed[0, -1], y_real[0, 0]], lon=[y_observed[1, -1], y_real[1, 0]], mode='lines', showlegend=False, line=dict(color=real_color, width=specs['lw'])))
                fig.add_trace(go.Scattergeo(lat=y_real[0, :step+1], lon=y_real[1, :step+1], mode='lines+markers', showlegend=False, marker=dict(color=real_color, size=specs['ms'], line=dict(color=specs['lc'], width=specs['mlw'])), line_width=specs['lw']))
                fig.add_trace(go.Scattergeo(lat=[y_real[0, step]], lon=[y_real[1, step]], mode='markers', name='Real', marker=dict(color=real_color, size=specs['ms_p'], line=dict(color=specs['lc_p'], width=specs['mlw_p'])), line_width=specs['lw_p']))
                if text:
                    text_str = '$\Huge \displaystyle x_{{{}}}$'.format('n' if title_type == 'step_n' else step+1).replace('x', '\mathbf{x').replace('}', '}}')
                    fig.add_annotation(x=y_real[0, step], y=y_real[1, step], text=text_str, showarrow=False, xref='x', yref='y')
                # connect observed to predicted location
                fig.add_trace(go.Scattergeo(lat=[y_observed[0, -1], y_pred[0, 0]], lon=[y_observed[1, -1], y_pred[1, 0]], mode='lines', showlegend=False, line=dict(color=predicted_color, width=specs['lw'])))
                fig.add_trace(go.Scattergeo(lat=y_pred[0, :step+1], lon=y_pred[1, :step+1], mode='lines+markers', showlegend=False, marker=dict(color=predicted_color, size=specs['ms'], line=dict(color=specs['lc'], width=specs['mlw'])), line_width=specs['lw']))
                fig.add_trace(go.Scattergeo(lat=[y_pred[0, step]], lon=[y_pred[1, step]], mode='markers', name='Predicted', marker=dict(color=predicted_color, size=specs['ms_p'], line=dict(color=specs['lc_p'], width=specs['mlw_p'])), line_width=specs['lw_p']))
                if text:
                    text_str = '$\Huge \displaystyle x_{{{}}}$'.format('n' if title_type == 'step_n' else step+1).replace('x', '\mathbf{\hat{x}').replace('}$', '}}$')
                    fig.add_annotation(x=y_pred[0, step], y=y_pred[1, step], text=text_str, showarrow=False, xref='x', yref='y')
        light_gray = '#d3d3d3'
        fig.update_geos(lataxis_range=lat_lim, lonaxis_range=lon_lim,
                        projection_type=projection_type, showland=True, landcolor=light_gray,
                        )
        fig.update_layout(font_size=30,
                          legend_y=0.8,
                          title='Latitude - Longitude',
                          )
    else:
        fig = get_figure(xaxis_title=targets[0], yaxis_title=targets[1], title=cds.capitalize())
        fig.add_trace(go.Scatter(x=y_observed[0], y=y_observed[1], mode='lines+markers', name='Observed', marker=dict(color='dimgray', size=specs['ms']), line_width=specs['lw']))
        # initial point in blue
        fig.add_trace(go.Scatter(x=[y_observed[0, 0]], y=[y_observed[1, 0]], mode='markers',showlegend=False, marker=dict(color='dimgray', size=specs['ms_p'], line=dict(color=specs['lc'], width=3))))
        y_observed.shape
        if step+1:
            # prediction region
            p = q_to_polygon(y_quantile[:, step], conf_idx)
            fig.add_trace(go.Scatter(
                x=p[:, 0],
                y=p[:, 1],
                mode='lines+markers',
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.3)',
                line=dict(color='blue', width=0.8),
                name = 'PR',
            )
            )
            # connect observed to real future location
            fig.add_trace(go.Scatter(x=[y_observed[0, -1], y_real[0, 0]], y=[y_observed[1, -1], y_real[1, 0]], mode='lines', showlegend=False, line=dict(color='gold', width=specs['lw'])))
            # Real location
            fig.add_trace(go.Scatter(x=y_real[0, :step+1], y=y_real[1, :step+1], mode='lines+markers', showlegend=False, marker=dict(color='gold', size=specs['ms'], line=dict(color=specs['lc'], width=3)), line_width=specs['lw']))
            fig.add_trace(go.Scatter(x=[y_real[0, step]], y=[y_real[1, step]], mode='markers', name='Real', marker=dict(color='gold', size=specs['ms_p'], line=dict(color=specs['lc_p'], width=specs['mlw_p'])), line_width=specs['lw_p']))
            # connect observed to predicted location
            fig.add_trace(go.Scatter(x=[y_observed[0, -1], y_pred[0, 0]], y=[y_observed[1, -1], y_pred[1, 0]], mode='lines', showlegend=False, line=dict(color=predicted_color, width=specs['lw'])))
            # Predicted location
            fig.add_trace(go.Scatter(x=y_pred[0, :step+1], y=y_pred[1, :step+1], mode='lines+markers', showlegend=False, marker=dict(color=predicted_color, size=specs['ms'], line=dict(color=specs['lc'], width=3)), line_width=specs['lw']))
            fig.add_trace(go.Scatter(x=[y_pred[0, step]], y=[y_pred[1, step]], mode='markers', name='Predicted', marker=dict(color=predicted_color, size=specs['ms_p'], line=dict(color=specs['lc_p'], width=specs['mlw_p'])), line_width=specs['lw_p']))
        lat_lims, lon_lims = compute_lims(y_quantile, y_pred, y_real, y_observed, offset_lon=0, offset_lat=0)
        tickvals = np.linspace(lat_lims[0], lat_lims[1], 5)
        ticktext = [f'{int(v)}' for v in tickvals]
        fig.update_layout(xaxis_nticks=5,
                          yaxis=dict(nticks=5, tickvals=tickvals, ticktext=ticktext),
                          )

    if title_type == 'cds':
        if geo and cds == 'mercator':
            title = 'Latitude - Longitude'
        else:
            title = cds.capitalize()
    elif title_type == 'step':
        title = f't<sub>{step+1}</sub>'
    elif title_type == 'step_n':
        title = 't<sub>n</sub>'
    else:
        raise ValueError(f"Invalid title type: {title_type}")

    fig.update_layout(title=dict(text=title, x=0.5, xanchor='center', yanchor='top', y=0.95,
                                 font_size=40 if title_type == 'cds' else 60))
    if not legend:
        fig.update_layout(showlegend=False)
    elif legend_orientation == 'h':
        fig.update_layout(legend_orientation='h',
                          legend_x=0.5, legend_y=0.1)
    return fig

@savefig('i+c+step')
def trajectory_confidence_region_imputation(results=None, i=31, c=0.9, step=8, cds='mercator', geo=True, projection_type='azimuthal equal area', legend=True, legend_orientation='v', lw=None, ms=None, ms_p=None, lw_p=None, mlw=None, **kwargs):

    if results is None:
        results = quantile_results(**kwargs)
    x = results['x_test']
    confidence_to_idxs = {0.5: [2, -3],
                          0.9: [1, -2],
                          0.95: [0, -1]}
    conf_idx = confidence_to_idxs[c]
    y_quantile = results['y_pred_raw_test'][:,i][..., conf_idx] # quantile dimension last
    stack_targets = lambda t: torch.stack(t, axis=0).numpy()
    y_pred = results['y_pred_test'][:, i]
    y_real = stack_targets(x['decoder_target'])[:, i]
    y_encoder = stack_targets(x['encoder_target'])[:, i]
    targets = params.cds_to_cols[cds]

    # Obtain future coordinate
    encoder_variables = results['encoder_variables'] # correct way: dataset.reals_with_future.index('future_X'), however indices for future_X,Y coincide between results['encoder_variables'] and dataset.reals_with_future
    # past_cds = [encoder_variables.index('X'), encoder_variables.index('Y')]
    future_cds = [encoder_variables.index('future_X'), encoder_variables.index('future_Y')]
    x_cont = x['encoder_cont'][i]
    Xf, Yf = x_cont[..., future_cds].transpose(0, 1)
    def rescale(Z, k):
        avg, sigma = x['target_scale'][k][i]
        return Z*sigma + avg
    Xf = rescale(Xf, 0).numpy()
    Yf = rescale(Yf, 1).numpy()
    y_future = np.vstack((Xf, Yf))

    if cds == 'mercator':
        # (X, Y) -> (lat, lon)
        to_degrees = lambda y: np.stack(space.mercator_inv(*y), axis=0) * 180 / np.pi
        y_quantile = to_degrees(y_quantile)
        y_pred = to_degrees(y_pred)
        y_encoder = to_degrees(y_encoder)
        y_future = to_degrees(y_future)
        y_real = to_degrees(y_real)

    missing_to_nan = []
# missing_to_nan = 'decoder'
    if missing_to_nan == 'all':
        missing_to_nan = ['encoder', 'future', 'decoder']
    elif isinstance(missing_to_nan, str):
        missing_to_nan = [missing_to_nan]
    def set_missing_to_nan(part):
        missing = x[f'{part}_missing'][i]
        if part == 'decoder':
            y_pred[:, missing] = np.nan
            y_quantile[:, missing] = np.nan
            y_real[:, missing] = np.nan
        else:
            globals()[f'y_{part}'][:, missing] = np.nan
        return
    for part in missing_to_nan:
        set_missing_to_nan(part)

    def compute_lims(offset_lon=0.5, offset_lat=0.25):
        offset = np.array([offset_lat, offset_lon])
        arrs = [y_pred, y_real, y_encoder, y_future]
        mins = np.array(([y_quantile[:, :, conf_idx].min(axis=2).min(axis=1)]
                         + [a.min(axis=1) for a in arrs])).min(axis=0)
        maxs = np.array(([y_quantile[:, :, conf_idx].max(axis=2).max(axis=1)]
                         + [a.max(axis=1) for a in arrs])).max(axis=0)
        mins -= offset
        maxs += offset

        lat_lim = [mins[0], maxs[0]]
        lon_lim = [mins[1], maxs[1]]
        return lat_lim, lon_lim
    lat_lim, lon_lim = compute_lims()

    predicted_color = 'cyan'
    real_color = 'gold'
    geo_to_specs = {True: dict(lw=3, ms=15, ms_p=20, lw_p=3, mlw=2),
                    False: dict(lw=3, ms=15, ms_p=20, lw_p=3, mlw=2)}
    specs = geo_to_specs[geo]
    local_specs = dict(lw=lw, ms=ms, ms_p=ms_p, lw_p=lw_p, mlw=mlw)
    for k, v in local_specs.items():
        if v is not None:
            specs[k] = v

    fig = get_figure(height=800, width=800)
    y_pred_step = y_pred[:, ::step]
    y_real_step = y_real[:, ::step]

    # Observed
    fig.add_trace(go.Scattergeo(lat=y_encoder[0], lon=y_encoder[1], mode='lines+markers', name='Observed', showlegend=True, marker=dict(color='dimgray', size=specs['ms'], line_width=specs['mlw'])))
    fig.add_trace(go.Scattergeo(lat=y_future[0], lon=y_future[1], showlegend=False, mode='lines+markers', marker=dict(color='dimgray', size=specs['ms'], line_width=specs['mlw'])))
    # Observed - Predicted
    def observed_imputation_link(y, color):
        fig.add_trace(go.Scattergeo(lat=[y_encoder[0, -1], y[0, 0]], lon=[y_encoder[1, -1], y[1, 0]], mode='lines', showlegend=False, line=dict(color=color, width=specs['lw'])))
        fig.add_trace(go.Scattergeo(lat=[y[0, -1], y_future[0,0]], lon=[y[1, -1], y_future[1, 0]], mode='lines', showlegend=False, line=dict(color=color, width=specs['lw'])))
        return
    observed_imputation_link(y_pred_step, predicted_color)
    observed_imputation_link(y_real_step, real_color)
    # Predicted
    fig.add_trace(go.Scattergeo(lat=y_pred_step[0], lon=y_pred_step[1], mode='lines+markers', showlegend=True, name='Predicted', marker=dict(color=predicted_color, size=specs['ms'], line=dict(color='black', width=specs['mlw'])), line_width=specs['lw']))
    # Real
    fig.add_trace(go.Scattergeo(lat=y_real[0][::step], lon=y_real[1][::step], mode='lines+markers', showlegend=True, name='Real', marker=dict(color=real_color, size=specs['ms'], line=dict(color='black', width=specs['mlw'])), line_width=specs['lw']))

    def plot_PR():
        # prediction region
        for j, y_q in enumerate(y_quantile.transpose(1, 0, 2)[::step]): # time as first dimension
            p = q_to_polygon(y_q, [0,1])
            p = np.vstack((p, p[0]))
            p = p[::-1]
            fig.add_trace(go.Scattergeo(
                lat=p[:, 0],
                lon=p[:, 1],
                mode='lines+markers',
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='blue', width=0.8),
                name = 'PR',
                showlegend = j == 0
            ))

    light_gray = '#d3d3d3'
    fig.update_geos(lataxis_range=lat_lim, lonaxis_range=lon_lim,
                    projection_type=projection_type, showland=True, landcolor=light_gray,
                    )
    if not legend:
        fig.update_layout(showlegend=False)
    elif legend_orientation == 'h':
        fig.update_layout(legend_orientation='h',
                          legend_x=0.5, legend_y=0.1)
    elif legend_orientation == 'v':
        fig.update_layout(legend_orientation='v',
                          legend_y=0.8)
    plot_PR()
    return fig

def get_smallest_mpl(study):
    trials = study.best_trials
    values = np.round([trial.value for trial in trials], decimals=3)
    params = np.array([trial.params['mpl'] for trial in trials])
    valid = values == values.min()
    params_pruned = params[valid]
    values_pruned = values[valid]
    optimal = np.abs(params_pruned).argmin()
    return params_pruned[optimal], values_pruned[optimal]

@savedata
def TFT_mpl_val(params_idx=0, quantiles='exact', cds='mercator', delete_missing=True, conservative=False, iterative=False, **kwargs):
    """
    Optimizes the mpl parameter for the TFT model on the validation set.
    mpl artificially increases or decreases the predicted quantiles, resulting in a wider or narrower prediction regions.

    For multiple trajectories, minimizes the RMSE between the empirical and target CER averaged across all trajectories for each time step.
    For a single trajectory, minimizes the RMSE between the empirical and target CER for that trajectory across all time steps.
    """
    # TODO: maybe would be a good idea select the optimal mpl by maximizing the quality factor Q.
    ID = kwargs.get('ID', None)
    if ID is not None and ID != 'all': # single trajectory
        mpl_range = (-0.9, 90.)
        num_trials = 1000
    else:
        mpl_range = (-0.5, 5.)
        num_trials = 500

    if isinstance(quantiles, str):
        quantile_method = deepcopy(quantiles)
        quantiles = params.default_quantiles[quantiles]
    elif not isinstance(quantiles, (list, np.ndarray)):
        raise ValueError(f"quantiles should be a list or numpy array. Got {type(quantiles)}. Alternatively, it can be a string in ['exact', 'bonferroni'].")
    else:
        quantile_method = quantiles

    results = quantile_results(params_idx=params_idx, cds=cds, quantiles=quantile_method, **kwargs)
    x = results[f'x_val']
    y_quantiles = results[f'y_pred_quantiles_val']
    num_animals = y_quantiles.shape[1]
    stack_targets = lambda t: torch.stack(t, axis=0).numpy()
    decoder_targets = stack_targets(x[f"decoder_target"])
    decoder_lengths = x[f"decoder_lengths"].numpy()
    if quantile_method == 'bonferroni':
        get_alpha_joint = lambda alpha_each_target: alpha_each_target * 2
    elif quantile_method == 'exact':
        get_alpha_joint = lambda alpha_each_target: 1 - (1 - alpha_each_target)**2
    else:
        get_alpha_joint = lambda alpha_each_target: 1 - (1 - alpha_each_target)**2
    missing = get_missing_values(params_idx=params_idx, quantiles=quantile_method, cds=cds, partition='val', area_method='km',
                                 repeat_by_conf=False, **kwargs)
    missing = np.hstack(missing)

    midpoint = len(quantiles) // 2
    def expand_quantiles(y_quantiles, mpl):
        z = y_quantiles.copy()
        z_pred = z[..., midpoint][..., None] # median quantile
        z[..., :midpoint] -= np.abs(z[..., :midpoint] - z_pred) * mpl # lower bound
        z[..., midpoint+1:] += np.abs(z[..., midpoint+1:] - z_pred) * mpl # upper bound
        return z

    def compute_coverages(y_quantiles):
        coverage_region = {}
        for i, q in enumerate(quantiles):
            if i < len(quantiles) // 2:
                # Coverage each target
                alpha_each_target = 1 - (quantiles[-i-1] - q)
                # Coverage region
                alpha_joint = get_alpha_joint(alpha_each_target)
                c = np.round(1 - alpha_joint, decimals=3)
                c_joint = ((decoder_targets > y_quantiles[:, :, :, i])
                           & (decoder_targets < y_quantiles[:, :, :, -i-1])
                           ).all(axis=0)
                for animal, c_jointk in enumerate(c_joint):
                    pred_length = decoder_lengths[animal]
                    for time_step, c_joint_kt in enumerate(c_jointk):
                        if time_step < pred_length:
                            coverage_region[(time_step, animal, c)] = c_joint_kt
                        else:
                            coverage_region[(time_step, animal, c)] = np.nan

        coverage_region = pd.Series(coverage_region).unstack(-1)
        coverage_region.index.names = ['time_step', 'animal']
        return coverage_region

    # Conservative approach: prune functions return true if trial should be pruned
    # Error = real - target coverages
    if conservative == 'mean':
        prune_error = lambda error: error.mean() < 0
    elif conservative == 'median':
        prune_error = lambda error: error.median() < 0
    elif isinstance(conservative, float): # quantile
        prune_error = lambda error: error.quantile(conservative) < 0
    elif not conservative:
        prune_error = lambda error: False
    else:
        assert not conservative, f"Invalid value for 'conservative': {conservative}"

    def optimize_quantile_mpl(c=0.5):
        """
        Optimize the mpl parameter for the CI expected coverage.
        """
        def objective(trial):
            mpl = trial.suggest_float('mpl', *mpl_range)
            df_i = compute_coverages(expand_quantiles(y_quantiles, mpl=mpl))
            df_i = df_i[c]
            if delete_missing:
                df_i[missing] = np.nan
            if num_animals > 1:
                error = df_i.groupby('time_step').mean() - c
                if prune_error(error):
                    raise optuna.TrialPruned()
                else:
                    rmse = np.sqrt((error**2).mean())
            else:
                error = pd.Series([df_i.mean() - c])
                if prune_error(error):
                    raise optuna.TrialPruned()
                rmse = abs(error.iloc[0])
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=num_trials)
        return study


    # iterative search
    def eval_df(df, c=0.5):
        df = df[c]
        if delete_missing:
            df[missing] = np.nan
        if num_animals > 1:
            error = df.groupby('time_step').mean() - c
            if prune_error(error):
                rmse = np.nan
            else:
                rmse = np.sqrt((error**2).mean())
        else:
            error = pd.Series([df.mean() - c])
            if prune_error(error):
                rmse = np.nan
            else:
                rmse = abs(error.iloc[0])
        return rmse

    def mpl_iterative_search(c=0.5, width_cutoff=1e-3, rmse_width_cutoff=1e-3):
        df_orig = compute_coverages(y_quantiles)
        coverage_orig = df_orig.mean()
        overestimates = coverage_orig.loc[c] > c
        if overestimates:
            mpl_range = [-0.99, 0]
        else:
            mpl_range = [0, 10]
        mpl_lb = mpl_range[0]
        mpl_best = np.nan
        error_best = np.nan
        def search(mpl_range):
            mpl_width = mpl_range[1] - mpl_range[0]
            while mpl_width > width_cutoff:
                mid_mpl = (mpl_range[0] + mpl_range[1]) / 2
                df_low = compute_coverages(expand_quantiles(y_quantiles, mpl=mpl_range[0]))
                df_high = compute_coverages(expand_quantiles(y_quantiles, mpl=mpl_range[1]))
                rmse_low = eval_df(df_low, c=c)
                rmse_high = eval_df(df_high, c=c)
                rmse_width = abs(rmse_high - rmse_low)
                if rmse_width >= rmse_width_cutoff and (math.isnan(rmse_low) or rmse_low > rmse_high): # high is better
                    mpl_best = mpl_range[1]
                    error_best = rmse_high
                    if math.isnan(rmse_high):
                        # increase both bounds
                        mpl_range[0] -= mid_mpl
                        mpl_range[1] += mid_mpl
                    else:
                        mpl_range[0] = mid_mpl
                else: # low is better
                    mpl_best = mpl_range[0]
                    error_best = rmse_low
                    mpl_range[1] = mid_mpl
                mpl_width = mpl_range[1] - mpl_range[0]
            return mpl_best, error_best, mpl_range

        mpl_best, error_best, mpl_range = search(mpl_range)
        mpl_best2, error_best2, mpl_range2 = search([mpl_lb, mpl_best])
        while error_best2 < error_best:
            error_best = error_best2
            mpl_best = mpl_best2
            mpl_range = mpl_range2
            mpl_best2, error_best2, mpl_range2 = search([mpl_lb, mpl_best])
        out = pd.Series(dict(mpl=mpl_best, rmse=error_best), name=c)
        return out

    if iterative:
        mpls = pd.concat([mpl_iterative_search(c=c) for c in params.confidences], axis=1).T
    else:
        mpls = {}
        for c in tqdm(params.confidences):
            study = optimize_quantile_mpl(c=c)
            mpl, rmse = get_smallest_mpl(study)
            mpls[(c, 'mpl')] = mpl
            mpls[(c, 'rmse')] = rmse
        mpls = pd.Series(mpls).unstack(-1)
    return mpls

def expand_TFT_quantiles(y_quantiles, mpls):
    S = mpls['mpl'].iloc[::-1]
    z = y_quantiles.copy()
    midpoint = z.shape[-1] // 2
    z_pred = z[..., midpoint] # median quantile
    for i, (confidence, mpl) in enumerate(S.items()):
        idxs = [i, -i-1]
        z[..., idxs[0]] -= np.abs(z[..., idxs[0]] - z_pred) * mpl # lower bound
        z[..., idxs[1]] += np.abs(z[..., idxs[1]] - z_pred) * mpl # upper bound
    return z

def TFT_asymmetry(params_idx='best', **kwargs):
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(**kwargs)
    _, _2, y_quantiles, *_3 = load_quantile_results(params_idx=params_idx, **kwargs)
    z = y_quantiles.copy()
    midpoint = z.shape[-1] // 2
    z_pred = z[..., midpoint] # median quantile

    def compute_asymmetry_1D(z_low, z_high):
        max_length = np.fmax(z_low, z_high)
        min_length = np.fmin(z_low, z_high)
        return (max_length / min_length)

    def compute_asymmetry_2D(z_low, z_high):
        width_x, width_y =  z_high + z_low
        max_width = np.fmax(width_x, width_y)
        min_width = np.fmin(width_x, width_y)
        return (max_width / min_width)

    def is_right_tail_higher(z_low, z_high):
        return (z_high > z_low).astype(int)

    def is_X_wider(z_low, z_high):
        width_x, width_y =  z_high + z_low
        return (width_x > width_y).astype(int)

    asymmetry = {}
    for i, c in enumerate(params.confidences):
        lower = i
        upper = -i - 1
        z_low = z_pred - z[..., lower]
        z_high = z[..., upper] - z_pred
        asymmetry_1D = compute_asymmetry_1D(z_low, z_high)
        asymmetry[(c, '1D_x')] = asymmetry_1D[0]
        asymmetry[(c, '1D_y')] = asymmetry_1D[1]
        asymmetry[(c, '1D_x_right_higher')] = is_right_tail_higher(z_low[0], z_high[0])
        asymmetry[(c, '1D_y_right_higher')] = is_right_tail_higher(z_low[1], z_high[1])
        asymmetry[(c, '2D')] = compute_asymmetry_2D(z_low, z_high)
        asymmetry[(c, '2D_X_wider')] = is_X_wider(z_low, z_high)
    asymmetry = pd.Series(asymmetry).unstack()
    asymmetry.index.name = 'confidence'
    return asymmetry

@savedata
def TFT_asymmetry_CI(boot='bca', **kwargs):
    asymmetry = TFT_asymmetry(**kwargs)
    sample_stat = asymmetry.applymap(np.mean)
    computer = getattr(bootstrap, f'CI_{boot}')
    CI = asymmetry.applymap(lambda x: computer(x.astype(float), custom_metrics.nb_mean, R=int(1e5)))
    _preprocess = lambda df, name: df.reset_index().melt(id_vars='confidence', value_name=name, var_name='asymmetry').set_index(['confidence', 'asymmetry']).sort_index()
    df = pd.concat([_preprocess(sample_stat, 'sample_stat'),
                    _preprocess(CI, 'CI')], axis=1)
    return df

def baseline_asymmetry(quantiles='exact', conf_2D=True, **kwargs):
    """
    conf_2D: map confidence levels to the associated 2D confidences for the PR. This only affects the confidence index values.
    """
    baseline_lengths, _ = quantile_baseline_lengths(to_width=False, quantiles=quantiles, **kwargs)
    df = baseline_lengths.unstack(0)

    def asymmetry_1D(x):
        """
        x: semiwidths of the 1D prediction interval.
        """
        lengths = np.abs(x)
        min_length, max_length = np.sort(lengths)
        if min_length:
            return max_length / min_length
        else:
            return np.nan

    def is_right_tail_higher(x):
        lengths = np.abs(x)
        return lengths[1] > lengths[0]

    def asymmetry_2D(row):
        width_x = row.X[1] - row.X[0]
        width_y = row.Y[1] - row.Y[0]
        max_width = max(width_x, width_y)
        min_width = min(width_x, width_y)
        if min_width:
            return max_width / min_width
        else:
            return np.nan

    def is_X_wider(row):
        width_x = row.X[1] - row.X[0]
        width_y = row.Y[1] - row.Y[0]
        return width_x > width_y

    df['1D_x'] = df['X'].apply(asymmetry_1D)
    df['1D_y'] = df['Y'].apply(asymmetry_1D)
    df['1D_x_right_higher'] = df['X'].apply(is_right_tail_higher).astype(int)
    df['1D_y_right_higher'] = df['Y'].apply(is_right_tail_higher).astype(int)
    df['2D'] = df.apply(asymmetry_2D, axis=1)
    df['2D_X_wider'] = df.apply(is_X_wider, axis=1).astype(int)
    df = df[['1D_x', '1D_y', '1D_x_right_higher', '1D_y_right_higher', '2D', '2D_X_wider']]
    df.columns.name = 'asymmetry'
    if conf_2D:
        df = df.reset_index(level=-1)
        c = df['confidence']
        alpha_1D = 1 - c
        if quantiles == 'exact':
            alpha_2D = 1 - (1 - alpha_1D)**2

        else:
            alpha_2D = 2 * alpha_1D
        df['confidence'] = (1 - alpha_2D).round(2)
        df = df.set_index('confidence', append=True)
    return df

@savedata
def baseline_asymmetry_CI(boot='bca', **kwargs):
    df = baseline_asymmetry(**kwargs)

    out = {}
    cols = df.columns
    computer = getattr(bootstrap, f'CI_{boot}')
    for c, df_c in df.groupby('confidence'):
        for col in cols:
            X = df_c[col].dropna().values.astype(float)
            out[(c, col, 'sample_stat')] = X.mean()
            out[(c, col, 'CI')] = computer(X, custom_metrics.nb_mean)
    out = pd.Series(out).unstack()
    out.index.names = ['confidence', 'asymmetry']
    return out

def ssm_asymmetry(**kwargs):
    """
    1D asymmetries are always 1 for SSMs.
    """
    df = ssm.eval_ssm(**kwargs)
    def asymmetry_2D(row):
        x_semiwidth = row['x.se']
        y_semiwidth = row['y.se']
        min_width, max_width = np.sort([x_semiwidth, y_semiwidth])
        return max_width / min_width

    def proportion_x_greater(row):
        x_semiwidth = row['x.se']
        y_semiwidth = row['y.se']
        return (x_semiwidth > y_semiwidth)

    df['2D'] = df.apply(asymmetry_2D, axis=1)
    df['x_wider'] = df.apply(proportion_x_greater, axis=1).astype(int)
    df = df[['id', 'time_step', '2D', 'x_wider']].set_index(['id', 'time_step'])
    return df

@savedata
def ssm_asymmetry_CI(boot='bca', **kwargs):
    df = ssm_asymmetry(**kwargs)
    out = {}
    computer = getattr(bootstrap, f'CI_{boot}')
    for col in df.columns:
        X = df[col].values.astype(float)
        out[(col, 'sample_stat')] = X.mean()
        out[(col, 'CI')] = computer(X, custom_metrics.nb_mean, R=int(1e5))
    out = pd.Series(out).unstack()
    return out

@savedata
def error_analysis_dataset(task='forecasting', params_idx='best', partition='test', c='avg', mpl_val=True, eval_mode=None, **kwargs):
    if eval_mode is None:
        eval_kwargs = {}
    else:
        eval_kwargs = dict(eval_mode=eval_mode)
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)
    if partition == 'train' and eval_mode is None:
        results = quantile_results_train(task=task, params_idx=params_idx, **kwargs)
    else:
        results = quantile_results(task=task, params_idx=params_idx, **eval_kwargs, **kwargs)
    hp = get_hp(task=task, params_idx=params_idx, **kwargs)
    if 'max_train_days' in hp:
        dataset_kwargs = dict(max_train_days=hp['max_train_days'])
    else:
        dataset_keys = ['max_train_days', 'store_missing_idxs']
        if task == 'imputation':
            dataset_keys += ['expand_encoder_until_future_length', 'predict_shift']
        dataset_kwargs = {k:v for k, v in kwargs.items() if k in dataset_keys}
    x = results[f'x_{partition}']

    def compute_trj_properties(var_type):
        is_future =  var_type == 'future'
        if is_future:
            encoder, l = custom_metrics.compute_future_encoder(results, x=x, partition=partition)
            # Reverse ordering for future values
            S = pd.Series([encoder[:, i, :li][:, ::-1] for i, li in enumerate(l)], name='trajectory')
        else:
            stack_targets = lambda t: torch.stack(t, axis=0).numpy()
            encoder = stack_targets(x['encoder_target'])
            l = x['encoder_lengths']
            S = pd.Series([encoder[:, i, :li] for i, li in enumerate(l)], name='trajectory')

        trj_properties = []
        for f in custom_metrics.trj_properties_funcs:
            result = S.apply(f)
            name = f.__name__
            if is_future:
                name = f'future_{name}'
            result.name = name
            trj_properties.append(result)
        trj_properties = pd.concat(trj_properties, axis=1)

        missing_info = custom_metrics.missing_info(results, partition=partition, var_type=var_type)
        trj_properties = pd.concat([trj_properties, missing_info], axis=1)
        return trj_properties

    encoder_properties = compute_trj_properties('encoder')
    if task == 'imputation':
        future_properties = compute_trj_properties('future')
        encoder_future_diff_properties = encoder_properties - future_properties.values
        encoder_future_diff_properties.columns = [f'encoder_future_diff_{col}' for col in encoder_future_diff_properties.columns]
        gap_distance = custom_metrics.gap_distance(results=results, partition=partition)
        trj_properties = pd.concat([encoder_properties, future_properties, encoder_future_diff_properties, gap_distance], axis=1)
    else:
        trj_properties = encoder_properties

    asymmetry = TFT_asymmetry(task=task, params_idx=params_idx, partition=partition, **eval_kwargs, **kwargs)
    if c == 'avg':
        avg_asymmetry = []
        for conf in params.confidences:
            avg_asymmetry_conf = asymmetry.loc[conf].apply(np.mean, axis=1) # avg asymmetry for each trj, at target conf c
            df = pd_utils.expand_sequences(avg_asymmetry_conf.to_frame())[conf].T
            avg_asymmetry.append(df)
        avg_asymmetry = pd_utils.tuple_wise(*avg_asymmetry).applymap(np.mean)
    else:
        avg_asymmetry = asymmetry.loc[c].apply(np.mean, axis=1)
        avg_asymmetry = pd_utils.expand_sequences(avg_asymmetry.to_frame())[c].T

    cov_region = quantile_coverages(task=task, params_idx=params_idx, partition=partition, mpl_val=mpl_val, **eval_kwargs, **kwargs)[2]
    if c == 'avg':
        coverage = cov_region['coverage'].groupby(['confidence', 'animal']).mean()
        area = cov_region['area'].groupby(['confidence', 'animal']).mean()
        alpha = 1 - coverage
        cs = alpha.index.get_level_values(0)
        alpha_error = alpha - (1-cs)
        alpha_error_abs = alpha_error.abs()
        coverage_quality = (area / area.median()) * (1 + (alpha_error/ (1-cs)))
        def compute_area_growth(S):
            """
            Relative growth of the area when passing from 0.5 to 0.95 target coverage
            """
            return S.xs(0.95, level='confidence') / S.xs(0.5, level='confidence')
        area_growth = cov_region['area'].groupby(['time_step', 'animal'], group_keys=False).apply(compute_area_growth)

        dfs = pd.Series(dict(coverage=coverage, area=area, alpha_error=alpha_error, alpha_error_abs=alpha_error_abs, coverage_quality=coverage_quality, area_growth=area_growth))
        dfs = dfs.apply(lambda df: df.groupby('animal').mean())
        CI_properties = pd.concat([avg_asymmetry, dfs.T], axis=1)
    else:
        cov_region = cov_region.swaplevel(0, -1).loc[c]
        area = cov_region['area'].groupby('animal').mean()
        coverage = cov_region['coverage'].groupby('animal').mean()
        alpha = 1 - coverage
        alpha_error = alpha - (1 - c)
        alpha_error.name = 'alpha_error'
        alpha_error_abs = alpha_error.abs()
        alpha_error_abs.name = 'alpha_error_abs'
        coverage_quality = (area / area.median()) * (1 + (alpha_error/ (1-c)))
        coverage_quality.name = 'coverage_quality'
        CI_properties = pd.concat([avg_asymmetry, area, coverage, alpha_error, alpha_error_abs, coverage_quality], axis=1)

    feature_avg, categoricals = load.avg_features(task=task, partition=partition, **eval_kwargs, **dataset_kwargs)
    # all columns in cat to categorical
    categoricals = categoricals.apply(lambda x: x.astype('category'))
    if task == 'imputation':
        # diff features
        corr_methods = ['spearman', 'pearson']
        features = [f for f in feature_avg.columns if not any(m in f for m in corr_methods)]
        static_features = ['Weight', 'Length']
        continuous_features = [f for f in features if not any(s in f for s in static_features)]
        encoder = [f for f in continuous_features if not f.startswith('future')]
        future = [f'future_{f}' for f in encoder]
        encoder_future_diff_features = feature_avg[encoder] - feature_avg[future].values
        encoder_future_diff_features.columns = [f'encoder_future_diff_{col}' for col in encoder_future_diff_features.columns]

        # gap of avg locations
        gap_avg_loc = custom_metrics.gap_of_avg_locations(feature_avg)

        feature_avg = pd.concat([feature_avg, encoder_future_diff_features, gap_avg_loc], axis=1)


    # quality properties
    quality = quality_by_animal(task=task, params_idx=params_idx, partition=partition, mpl_val=mpl_val, **eval_kwargs, **kwargs)

    # distance error
    distance = point_prediction_errors(task=task, params_idx=params_idx, partition=partition, **eval_kwargs, **kwargs)
    distance = distance.mean(axis=1)
    distance.name = 'distance'

    data = pd.concat([trj_properties, CI_properties, feature_avg, categoricals, quality, distance], axis=1)
    # delete duplicated columns
    data = data.loc[:, ~data.columns.duplicated()]
    return data

def compute_quality_df(ssm_model=None, ref_method='cumulative-by-direction', ref_double=False, ref_R=int(1e4), ref_alpha=0.05, baseline=False, CI_method='bca', params_idx='best', quantiles='exact', cds='mercator', partition='test', area_method='km', delete_missing=True, mpl_val='best', ID=None, pretrained=False, recompute_mercator=False, eval_mode=None, task='forecasting', naive_pred='last-obs', naive_pred_lengths='last-obs', skip_computation=False,
                       s_q=1, dist_method='hull', rho=False, density='pchip', n_sample=int(1e5), exclude_me=False, optimize_spread=False, optimize_cmax=False, n_grid=100,
                       **kwargs):
    A_ref_kwargs = dict(method=ref_method, double=ref_double, R=ref_R, alpha=ref_alpha) # DEPRECATED
    if ssm_model is not None:
        A_ref_kwargs = {f'ref_{k}': v for k, v in A_ref_kwargs.items()}
        df = ssm.eval_ssm(model=ssm_model, area_method=area_method, partition=partition, se_val_fit=mpl_val, recompute_mercator=recompute_mercator, task=task, **A_ref_kwargs)
        df = ssm.reformat_df(df, magnitudes=['coverage', 'area', 'Q_area', 'area_ref'], to_float=True)
        return df
    else:
        specs = dict(quantiles=quantiles, cds=cds, partition=partition, area_method=area_method, delete_missing=delete_missing, **kwargs)
        if ID is not None:
            specs['ID'] = ID
            if pretrained:
                specs['pretrained'] = True
        if eval_mode is not None:
            specs['eval_mode'] = eval_mode
        if task != 'forecasting':
            specs['task'] = task
        if quantiles == 'all':
            specs['s_q'] = s_q
        specs_skip = specs.copy()
        specs_skip['skip_computation'] = skip_computation

        if baseline:
            naive_kwargs=dict(CI_mpl=mpl_val, CI_method=CI_method, naive_pred=naive_pred, naive_pred_lengths=naive_pred_lengths)
            specs_cvg = specs_skip.copy()
            del specs_cvg['area_method']
            coverages = baseline_coverage_area(**specs_cvg, **naive_kwargs)
            if isinstance(coverages, SavedataSkippedComputation):
                return coverages
            area_base = quantile_baseline_area(**specs_skip, **naive_kwargs)
            coverages = coverages.stack(dropna=False)
            coverages.name = 'coverage'
            area_base.name = 'area'
            coverages = pd.concat([coverages, area_base], axis=1)
        else:
            if quantiles == 'all':
                dist_specs = specs_skip.copy()
                dist_specs['n_sample'] = n_sample
                dist_specs['n_grid'] = n_grid
                dist_specs['method'] = dist_method
                for k in ['area_method', 'quantiles']:
                    del dist_specs[k]
                if cds == 'mercator':
                    del dist_specs['cds']
                if params_idx == 'best':
                    best_specs = dist_best_model_specs(task=task, mpl_val=mpl_val, rho=rho and mpl_val, density=density, optimize_spread=optimize_spread, optimize_cmax=optimize_cmax)
                    dist_specs.update(best_specs)
                else:
                    dist_specs['params_idx'] = params_idx
                    if rho:
                        dist_specs['rho'] = True
                        if mpl_val:
                            if optimize_spread:
                                dist_specs['optimize_spread'] = True
                            if optimize_cmax:
                                dist_specs['optimize_cmax'] = True
                    if density != 'pchip':
                        dist_specs['density'] = density
                        if exclude_me:
                            dist_specs['exclude_me'] = True
                if mpl_val == 'best':
                    raise ValueError("mpl_val='best' is not supported for 'all' quantiles. Pass mpl_val=True")
                coverages = dist_pr(**dist_specs, mpl_val=mpl_val)
                if isinstance(coverages, SavedataSkippedComputation):
                    return coverages
                else:
                    coverages = coverages.astype(float) # np.float64 and bool -> float
            else:
                if params_idx == 'best':
                    params_idx, _ = quantile_best_model(**specs)
                TFT_kwargs = dict(mpl_val=mpl_val, params_idx=params_idx)
                coverages = quantile_coverages(**specs_skip, **TFT_kwargs)
                if isinstance(coverages, SavedataSkippedComputation):
                    return coverages
                else:
                    coverages = coverages[2]

        results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(params_idx=0 if params_idx == 'best' else params_idx,
                                                                                                                               **specs)
        baseline, _, y_real = get_predictions(results=results, x=x, task=task, cds=cds, partition=partition, naive_pred='last-obs')
        df = pd_utils.tuple_wise(y_real, baseline)

        num_confidences = coverages.index.levels[-1].size
        days = time_step_to_days(coverages.index.get_level_values('time_step').values)
        A_max = custom_metrics.maximal_area(task=task, baseline=baseline, days=days)
        if isinstance(A_max, pd.Series):
            flatten_area_series = lambda A: np.repeat(np.vstack(A).T, # shape (time, animal)
                                                      num_confidences, axis=1).flatten() # match coverages index (sorted by time, for each animal 'num_confidences' values)
            A_max = flatten_area_series(A_max)
        coverages['area_max'] = A_max

        # OLD VERSION
        # A_ref = df.apply(custom_metrics.compute_reference_area, task=task, **A_ref_kwargs)
        # coverages['area_ref'] = flatten_area_series(A_ref)

        # NEW: variable for each confidence level
        A_ref = []
        for confidence in coverages.index.levels[-1]:
            alpha = 1 - confidence
            A = df.apply(custom_metrics.compute_reference_area, task=task, alpha=alpha).values
            A_ref.append(np.vstack(A).T)
        A_ref = np.stack(A_ref, axis=-1).flatten()
        coverages['area_ref'] = A_ref

        area_data = coverages[['area', 'area_ref', 'area_max']].values.astype(float).T
        coverages['Q_area'] = custom_metrics.area_quality(*area_data)
        coverages[coverages.area.isna()] = np.nan
        return coverages

def quality_by_animal(**kwargs):
    df = compute_quality_df(**kwargs)
    avg_by_t = df.groupby(['confidence', 'animal']).mean()
    avg_by_t['Q_alpha'] = custom_metrics.coverage_quality(1 - avg_by_t['coverage'],
                                                          1 - avg_by_t.index.get_level_values('confidence').values)
    result = avg_by_t.groupby('animal').mean()[['Q_area', 'Q_alpha']]
    result['Q'] = result.prod(axis=1)
    return result

def quality_sample(**kwargs):
    """
    Compute the quality metrics for the sample.
    """
    df = compute_quality_df(**kwargs)

    if isinstance(df, SavedataSkippedComputation):
        return df
    else:
        avg_by_animal = df.groupby(['confidence', 'time_step']).mean()
        avg_by_animal['Q_alpha'] = custom_metrics.coverage_quality(1 - avg_by_animal['coverage'],
                                                              1 - avg_by_animal.index.get_level_values('confidence').values)
        result = avg_by_animal.groupby('confidence').mean()[['Q_area', 'Q_alpha']]
        result['Q'] = result.prod(axis=1)
        return result

@savedata
def quality_CI_by_time_step(c=0.5, **kwargs):
    coverages = compute_quality_df(**kwargs)
    alpha_target = 1 - c
    df = coverages.dropna().swaplevel(0, -1).loc[c] # dropna for compatibility with missing values

    compute_Qs = custom_metrics.get_compute_Qs(alpha_target)

    outputs = ['Q_alpha', 'Q_area', 'Q']
    X = df[['coverage', 'Q_area']].values
    X[:,0] = 1 - X[:,0] # alpha
    time_steps = df.index.get_level_values('time_step').values
    results = {}
    for t in tqdm(np.unique(time_steps)):
        x_t = X[time_steps == t]
        sample_stat = compute_Qs(x_t)
        CI = bootstrap.CI_percentile(x_t, compute_Qs, R=int(1e5))
        for label, ss, ci in zip(outputs, sample_stat, CI):
            results[(label, t, 'sample_stat')] = ss
            results[(label, t, 'CI')] = ci
    results = pd.Series(results).unstack()
    return results

@savedata('all-R')
def quality_CI_aggregate(c=0.5, R=int(1e5), **kwargs):
    coverages = compute_quality_df(**kwargs)
    alpha_target = 1 - c
    compute_Qs = custom_metrics.get_compute_Qs(alpha_target)
    df = coverages.swaplevel(0, -1).loc[c]
    df['alpha'] = 1 - df['coverage']
    Xs = df[['alpha', 'Q_area']].groupby('time_step').apply(lambda x: x.values)
    Xs = tuple(Xs)
    output_metric = ['Q_alpha', 'Q_area', 'Q']
    return custom_metrics.aggregate_CI_across_time_steps(Xs, compute_Qs, output_metric, R=R, alpha=0.05)

@savedata
def _quality_CI_across_confidence(task='forecasting', ssm_model=None, baseline=False, params_idx='best', **kwargs):
    """
    DEPRECATED. Use quality_CI_across_confidence instead. Error: resampling should be at the trajectory level.
    """
    output_metric = ['Q_alpha', 'Q_area', 'Q']
    df = compute_quality_df(task=task, ssm_model=ssm_model, baseline=baseline, params_idx=params_idx, **kwargs)
    df = df.dropna()
    df['alpha'] = 1 - df['coverage']
    num_confidences = df.index.levels[-1].size
    Xs = df[['alpha', 'Q_area']].groupby('time_step').apply(lambda x: x.values.reshape(-1, num_confidences, 2)) # elements shape (num_animals, num_confidences, 2)
    Xs = tuple(Xs)

    compute_Qs = []
    for c in params.confidences:
        alpha_target = 1 - c
        compute_Qs.append(custom_metrics.get_compute_Qs(alpha_target)) # 1 for each column
        # resampling occurs at the animal level for each time step.
    def average_Q(x):
        """
        Average quality across target confidences
        """
        Qs = np.array([compute_Q(x[:,i]) for i, compute_Q in enumerate(compute_Qs)]) # shape (num_confidences, output)
        return Qs.mean(axis=0)
    return custom_metrics._aggregate_CI_across_time_steps(Xs, average_Q, labels=output_metric, alpha=0.05, R=int(1e4))

@savedata
def quality_CI_across_confidence(task='forecasting', ssm_model=None, baseline=False, params_idx='best', CI_method='bca', **kwargs):
    output_metric = ['Q_alpha', 'Q_area', 'Q']
    df = compute_quality_df(task=task, ssm_model=ssm_model, baseline=baseline, params_idx=params_idx, **kwargs)
    df = df.dropna()
    df['alpha'] = 1 - df['coverage']
    df_input = df[['alpha', 'Q_area']]
    def Q_avg(X):
        Q_alpha = np.array([custom_metrics.coverage_quality(np.nanmean(X[:, :, i, 0], axis=0), 1-c) for i, c in enumerate(params.confidences)]).T
        Q_area = np.nanmean(X[..., 1], axis=0)
        Q = Q_alpha * Q_area
        out = np.array([Q_alpha.mean(), Q_area.mean(), Q.mean()])
        return out
    return custom_metrics.aggregate_CI_across_time_steps(df_input, Q_avg, labels=output_metric, alpha=0.05, R=int(1e4), CI_method=CI_method)

@savedata
def quality_aggregate_CI_delete_seeds(task='forecasting', R=int(1e4), params_idx='best', n=200, seeds=range(5), **kwargs):
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)

    def preprocess_quality_df(**kwargs):
        df = compute_quality_df(**kwargs)
        df = df.dropna()
        df['alpha'] = 1 - df['coverage']
        df = df[['alpha', 'Q_area']]
        return df

    def Q_avg_seeds(X):
        seeds = X.shape[-1]
        Q_alpha = np.empty(seeds)
        Q_area = np.empty(seeds)
        Q = np.empty(seeds)
        for seed in range(seeds):
            X_s = X[..., seed]
            Q_alpha_s = np.array([custom_metrics.coverage_quality(np.nanmean(X_s[:, :, i, 0], axis=0), 1-c) for i, c in enumerate(params.confidences)]).T
            Q_area_s = np.nanmean(X_s[..., 1], axis=0)
            Q_s = Q_alpha_s * Q_area_s

            Q_alpha[seed] = Q_alpha_s.mean()
            Q_area[seed] = Q_area_s.mean()
            Q[seed] = Q_s.mean()

        # average across seeds
        out = np.array([Q_alpha.mean(), Q_area.mean(), Q.mean()])
        return out

    labels = ['Q_alpha', 'Q_area', 'Q']
    return custom_metrics.delete_seed_aggregate_CI(n=n, func=preprocess_quality_df, statistic=Q_avg_seeds, delete_seeds=seeds, R=R,
                                                   labels=labels,
                                                   params_idx=params_idx, task=task, **kwargs)

def quality_aggregate_CI_delete_seeds_summary(task, params_idx='best'):
    kwargs = params.TFT_specs[task]
    if params_idx == 'best':
        params_idx, _ = quantile_best_model(task=task, **kwargs)
    out = []
    for n in tqdm(params.n_to_delete_prob.keys()):
        df = quality_aggregate_CI_delete_seeds(task=task, n=n, params_idx=params_idx, **kwargs)
        df = df.swaplevel().loc['Q']
        out.append(df)
    # Add results for n=1 and n_max
    out_1 = quality_CI_across_confidence(task=task, **kwargs, ID='all', mpl_val=True)
    out_1 = out_1.loc['Q'].to_frame(name=1).T

    data = quantile_results_train(task=task, **kwargs, params_idx=params_idx)
    n_train_max = np.unique(data['x_train']['groups'].numpy()).size
    out_n = quality_CI_across_confidence(task=task, **kwargs, mpl_val=True)
    out_n = out_n.loc['Q'].to_frame(name=n_train_max).T

    df = pd.concat([out_n] + out + [out_1], axis=0)
    return df

@savefig('all-yaxis_type-xaxis_type-ylim')
def quality_across_n(task='forecasting', params_idx='best', ssm_model=None, yaxis_type=None, xaxis_type='log', ylim=[0, 1]):
    df = quality_aggregate_CI_delete_seeds_summary(task, params_idx=params_idx)

    if ssm_model is None:
        ssm_model = 'mp' if task == 'forecasting' else 'rw'
    out_ssm = quality_CI_across_confidence(task=task, ssm_model=ssm_model, baseline=False, mpl_val=True)
    n_min = df.index[-1]
    n_max = df.index[0]
    ssm_min = out_ssm.loc['Q'].to_frame(n_min).T
    ssm_max = ssm_min.copy()
    ssm_max.index = [n_max]
    df_ssm = pd.concat([ssm_min, ssm_max])

    fig = plot_confidence_bands(df=df, label='TFT', xaxis_title='Dataset size', yaxis_title='PR Quality', title_text=task.capitalize())
    plot_confidence_bands(fig=fig, df=df_ssm, label='SSM', color='black')
    fig.update_layout(xaxis_type=xaxis_type, yaxis_type=yaxis_type, yaxis_range=ylim)
    return fig

def main_results(mpl_val=True, task='forecasting', CI_method='bca'):
    out = []
    for model, specs in params.model_specs().items():
        if model in ['TFT_dist', 'TFT', 'TFT_single', 'Naive']:
            specs.update(params.TFT_specs[task])
            if model == 'Naive' and task == 'imputation':
                specs.update(params.baseline_specs_imputation)
        result = quality_CI_across_confidence(mpl_val=mpl_val, task=task, CI_method=CI_method, **specs)
        result = pd_utils.format_CI_results(result)
        result.name = model
        out.append(result)
    out = pd.concat(out, axis=1).T
    if mpl_val:
        # add distance
        df = aggregate_CI_summary(point_prediction_aggregate_CI, task=task, CI_method=CI_method, ci_func=False)
        df = df.applymap(np.round, decimals=1)
        df = pd_utils.format_CI_results(df)
        df = df.apply(lambda x: re.sub(r'\.(\d+)0', r'.\1', x)) # delete second decimal (leave first as it is)
        df.name = 'Distance [km]'
        out = pd.concat([df, out], axis=1)

    colmap = {'Q_alpha': '$Q_{\\alpha}$', 'Q_area': '$Q_{\\text{A}}$', 'Q': '$Q$'}
    index_map = {'TFT_single': 'TFT[s]', 'TFT_dist': 'TFT[B]'}
    out = out.rename(columns=colmap, index=index_map)
    return out

@savedata
def quantile_imputation_all(params_idx=0, partition='test', cds='mercator', quantiles='exact', **kwargs):
    """
    Returns:
    dict: A dictionary where the keys are the group IDs and the values are tuples containing the real ID, real data (numpy array), and imputed data (pandas DataFrame).

    The function first retrieves the results of the quantile calculation and loads the pretrained model. It then constructs a QuantileForecaster model and loads the pretrained model into it. The function then computes the predictions and constructs a DataFrame for the imputed data. The function then preprocesses the data for all animals in the given partition. Finally, the function returns a dictionary containing the imputed data for all animals.

    The preprocessing step consist in aligning the data with the missing steps and reconstructing the trajectories.
    """
    task = 'imputation'
    eval_mode = 'all'
    results = quantile_results(task=task, params_idx=params_idx, cds=cds, quantiles=quantiles, **kwargs)
    pretrained_model = results['state_dict']

    # build tft
    model_specs = params.quantile_best_params[task][cds][params_idx][1].copy()
    del model_specs['gradient_clip_val']

    tft = model.QuantileForecaster(quantiles=quantiles, cds=cds, model_specs=model_specs, task=task, eval_mode=eval_mode)
    tft.model.load_state_dict(pretrained_model)
    print("Computing predictions...")
    results = tft.get_results(partition=partition)
    print("Constructing imputation dataframe")
    x = results[f'x_{partition}']
    groups = x['groups'].squeeze()

    def preprocess_imputation_all(group_ID):
        is_idx = groups == group_ID
        stack_targets = lambda t: torch.stack(t, axis=0).numpy()
        encoder = stack_targets([t[is_idx] for t in x['encoder_target']])
        X, Y = encoder
        decoder_length = x['decoder_lengths'][is_idx].numpy()
        encoder_length_real = (X != 0).sum(axis=1)
        max_encoder_length = (x['encoder_target'][0] != 0).sum(axis=1).max().item()

        # Obtain future coordinates
        encoder_variables = results['encoder_variables'] # correct way: dataset.reals_with_future.index('future_X'), however indices for future_X,Y coincide between results['encoder_variables'] and dataset.reals_with_future
        # past_cds = [encoder_variables.index('X'), encoder_variables.index('Y')]
        future_cds = [encoder_variables.index('future_X'), encoder_variables.index('future_Y')]
        x_cont = x['encoder_cont'][is_idx]
        Xf, Yf = x_cont[..., future_cds].transpose(0, 2).transpose(1, 2)
        def rescale(Z, i):
            avg, sigma = x['target_scale'][i][is_idx].T
            return Z*sigma[:, None] + avg[:, None]
        Xf = rescale(Xf, 0).numpy()
        Yf = rescale(Yf, 1).numpy()

        # Obtain past coordinates
        def reconstruct_target(Z, Zf):
            Z = Z.copy()
            Zf = Zf.copy()
            trj_split_idx = np.hstack((0, encoder_length_real))
            less_than_max_encoder_length = np.where(trj_split_idx >= max_encoder_length)[0] - 1
            if less_than_max_encoder_length.size > 0:
                less_than_max_encoder_length = less_than_max_encoder_length[0]
            else:
                less_than_max_encoder_length = trj_split_idx.size
            trj_split_idx = trj_split_idx[:less_than_max_encoder_length+1]

            start = trj_split_idx[:-1].copy()
            end = trj_split_idx[1:].copy()
            start +=  np.hstack((0, decoder_length[:less_than_max_encoder_length]))[:-1]
            Z_reconstructed = []
            for Zi, start_i, end_i, decoder_length_i in zip(Z, start, end, decoder_length):
                Z_reconstructed.append(np.hstack((Zi[start_i:end_i], np.zeros((decoder_length_i)))))

            for i in range (less_than_max_encoder_length, Z.shape[0]):
                z0 = Z[i-1]
                z1 = Z[i]
                z0_last_observed = np.where(z0 != 0)[0][-1]
                shift = np.where(z1 == z0[z0_last_observed])[0]
                k = 1
                while not (np.diff(shift) == 1).all():
                    # previous value
                    shift = np.where(z1 == z0[z0_last_observed - k])[0]
                    k += 1
                if shift.size > 0:
                    shift = shift[-1]
                    z1 = z1[shift+1:]
                z1_last_observed = np.where(z1 != 0)[0][-1]
                z1 = z1[:z1_last_observed+1]
                Z_reconstructed.append(np.hstack((z1, np.zeros((decoder_length[i])))))

            Z_reconstructed = np.hstack(Z_reconstructed)
            missing_times = np.where(Z_reconstructed == 0)[0]

            # add observed future values
            zf = Zf[-2].copy()
            if less_than_max_encoder_length < Z.shape[0]:
                last_shift = np.where(zf == z1[-1])[0]
                if last_shift.size == 0:
                    zf = Zf[-3].copy()
                    last_shift = np.where(zf == z1[-1])[0]
                k = 1
                while not (np.diff(last_shift) == 1).all():
                    last_shift = np.where(zf == z1[-1-k])[0]
                    k += 1
                last_shift = last_shift[-1]
                zf = zf[last_shift+1:]
            # exclude last sequence of zeros
            is_repeated = np.hstack((False, np.diff(zf) == 0))
            zf[is_repeated] = 0
            last_observed_value = np.where(zf != 0)[0][-1]
            zf = zf[:last_observed_value+1]
            Z_reconstructed = np.hstack((Z_reconstructed, zf))
            Z_reconstructed[Z_reconstructed == 0] = np.nan
            return Z_reconstructed, missing_times

        # Real data
        X_r, missing_times = reconstruct_target(X, Xf)
        Y_r, _ = reconstruct_target(Y, Yf)
        # Group together consecutive missing times
        decoder_idxs = np.hstack((0, decoder_length)).cumsum()
        decoder_start = decoder_idxs[:-1]
        dec_end = decoder_idxs[1:]
        missing_times = [missing_times[start:end] for start, end in zip(decoder_start, dec_end)]

        # Imputed data
        X_q, Y_q = results[f'y_pred_raw_{partition}'][:, is_idx]
        X_pred, Y_pred = results[f'y_pred_{partition}'][:, is_idx]
        split_by_decoder_length = lambda X, name: pd.Series([X_i[:l] for l, X_i in zip(decoder_length, X)], name=name)
        X_q = split_by_decoder_length(X_q, 'X_q')
        Y_q = split_by_decoder_length(Y_q, 'Y_q')
        X_pred = split_by_decoder_length(X_pred, 'X_pred')
        Y_pred = split_by_decoder_length(Y_pred, 'Y_pred')
        df = pd.concat([X_q, Y_q, X_pred, Y_pred], axis=1)
        encoder_future_length = (np.diff(Xf) != 0).sum(axis=1) + 1
        df['time_idx'] = missing_times
        df['encoder_future_length'] = encoder_future_length
        df['encoder_length'] = encoder_length_real
        df['decoder_length'] = decoder_length
        df.index.name = 'imputation_ID'
        return X_r, Y_r, df

    dataset = getattr(tft, f'dataset_{partition}')
    real_IDs = dataset.transform_values(name='ID', values=groups, group_id=True, inverse=True)
    group_to_ID = pd.Series(real_IDs, index=groups.numpy()).to_dict()

    imputed_data = {}
    for group_ID in tqdm(groups.unique().numpy()):
        result = preprocess_imputation_all(group_ID=group_ID)
        ID = group_to_ID[group_ID]
        imputed_data[group_ID] = (ID, result)

    return imputed_data

def check_no_future_info_leak():
    kwargs = dict(predict_shift=112, max_train_days=3, store_missing_idxs=True, expand_encoder_until_future_length=True, task='imputation', epochs=200, params_idx=0)
    results = quantile_results(**kwargs)
    x = results['x_test']
    features = results['encoder_variables']
    future_cds = [features.index(f'future_{f}') for f in ['X', 'Y']]# correct way: dataset.reals_with_future.index('future_X'), however indices for future_X,Y coincide between results['encoder_variables'] and dataset.reals_with_future
    stack_targets = lambda t: torch.stack(t, axis=0).numpy()

    decoder_all = stack_targets(x['decoder_target'])
    poss_diff = defaultdict(list)
    for i in range(decoder_all.shape[1]):
        decoder = decoder_all[:, i]
        future = x['encoder_cont'][i, :, future_cds].numpy().T
        loc, scale = stack_targets(x['target_scale']).T[:, i][..., None] # shape (loc-scale, N, target)
        future = future*scale + loc
        dx = future[:, 0] - decoder[:, -1] # t_n+1 - t_n
        poss_diff['X'].append(dx[0])
        poss_diff['Y'].append(dx[1])
    poss_diff = pd.DataFrame(poss_diff) # does not have zeros (future is not leaked to decoder)

    # Ensuring correct reconstruction of future cds with loc and scale
    encoder_all = stack_targets(x['encoder_target'])

    encoder_cds = [features.index(f) for f in ['X', 'Y']]
    encoder_rec = x['encoder_cont'][:, :, encoder_cds].numpy()
    loc, scale = stack_targets(x['target_scale']).T[:, :,  None] # shape (loc-scale, N, target)
    encoder_rec = encoder_rec*scale + loc
    encoder_rec = encoder_rec.transpose(2, 0, 1)
    encoder_rec_diff = pd.DataFrame(np.abs(encoder_all - encoder_rec).sum(axis=-1).T) # has mostly zeros (encoder is correctly reconstructed from features, loc and scale)
    return poss_diff, encoder_rec_diff

@savedata
def best_max_train_days_imputation(partition='val', **kwargs):
    """
    Compares the effect of varying the encoder and future lengths in the quality of the predictions, i.e. varying the number of days used as input.
    """
    T = [1, 3, 4, 5, 7, 9, 14, 21, 56, 84]
    specs = dict(predict_shift=112, store_missing_idxs=True, expand_encoder_until_future_length=True, task='imputation', epochs=200, mpl_val=True, partition=partition,
                 skip_computation=True)
    specs.update(kwargs)
    num_models = 5
    out = []
    for max_train_days in tqdm(T):
        specs['max_train_days'] = max_train_days
        for params_idx in range(num_models):
            df = quality_sample(params_idx=params_idx, **specs)
            if isinstance(df, SavedataSkippedComputation):
                warnings.warn(f"Skipping max_train_days={max_train_days} and params_idx={params_idx}")
                continue
            else:
                df = df.mean()
                df['max_train_days'] = max_train_days
                df['params_idx'] = params_idx
                out.append(df)
    df = pd.concat(out, axis=1).T
    return df.sort_values('Q', ascending=False)

@savedata
def best_setting_imputation(partition='val', max_train_days=4, **kwargs):
    """
    Compares the effect of setting zero attention to missing values in the encoder and future vs not doing so.
    Settings:
    1. store_missing_idxs=True -> set zero attention to missing values in the encoder and future. Used outdated hyperparameter ranges (not enough hidden size, etc).
    2. store_missing_idxs=False -> do not set zero attention to missing values in the encoder and future.
    3. store_missing_idxs=True, mod_hp=dict(store_missing_idxs=False) -> set zero attention to missing values in the encoder and future, but use the best hyperparameters found for (2).
    """
    specs = dict(reverse_future=True, predict_shift=112, expand_encoder_until_future_length=True, task='imputation', epochs=200, mpl_val=True, partition=partition, max_train_days=max_train_days, **kwargs,
                  skip_computation=True)
    settings = [dict(store_missing_idxs=True),
                dict(store_missing_idxs=False),
                dict(store_missing_idxs=True, mod_hp=dict(store_missing_idxs=False)),
                dict(store_missing_idxs=True, mod_hp=dict(store_missing_idxs=False), decoder_missing_zero_loss=True),
                ]
    num_models = 5
    out = []
    pbar = tqdm(range(len(settings) * num_models))
    for setting in settings:
        setting_specs = specs.copy()
        setting_specs.update(setting)
        if not setting_specs['store_missing_idxs']:
            del setting_specs['store_missing_idxs']
        for params_idx in range(num_models):
            df = quality_sample(params_idx=params_idx, **specs)
            if isinstance(df, SavedataSkippedComputation):
                warnings.warn(f"Skipping max_train_days={max_train_days} and params_idx={params_idx}")
            else:
                df = df.mean()
                df['store_missing_idxs'] = setting['store_missing_idxs']
                df['mod_hp'] = setting.get('mod_hp', {})
                df['decoder_missing_zero_loss'] = setting.get('decoder_missing_zero_loss', False)
                df['params_idx'] = params_idx
                out.append(df)
            pbar.update(1)
    df = pd.concat(out, axis=1).T
    return df.sort_values('Q', ascending=False)

@savedata
def optimal_bw(task='forecasting', s_q=1, R=50, **load_kwargs):
    results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(task=task, quantiles='all', s_q=s_q, **load_kwargs)
    q = np.array(quantiles)
    q_series = y_quantiles.transpose(1, 2, 0, 3) # (N, time_step, target, q)
    q_series = q_series.astype(np.float64)
    N, time_steps, targets, num_quantiles = q_series.shape
    bw = {}
    for animal in tqdm(range(N)):
        for time_step in range(time_steps):
            for target in targets:
                x_q = q_series[animal, time_step, target]
                bw[(animal, time_step, target)] = distribution.optimize_bw(x_q, num_iter=R)
    bw = pd.Series(bw)
    bw.index.names = ['animal', 'time_step', 'target']
    return bw

def load_np_distributions_y_real(task='forecasting', s_q=1, delete_missing=True, **load_kwargs):
    results, *_ = load_quantile_results(task=task, quantiles='all', s_q=s_q, mpl_val=False, **load_kwargs)
    keys = ['partition', 'cds', 'naive_pred']
    target_kwargs = dict(task=task, results=results)
    for k in keys:
        if k in load_kwargs:
            target_kwargs[k] = load_kwargs[k]
    *_, y_real = get_predictions(**target_kwargs)
    y_real = custom_metrics.expand_sequences(y_real)
    if delete_missing:
        missing = get_missing_values(task=task, **load_kwargs, s_q=s_q, quantiles='all')
        missing = custom_metrics.expand_sequences(missing)
        y_real[missing] = np.nan
    return y_real

def load_np_distributions(task='forecasting', s_q=1, mpl_val=False, load_kwargs={}, return_target=False, delete_missing=True, rho=False, rho_eps=0.8, density='pchip', dp=0.001, **dist_kwargs):
    """
    Returns pandas Series of NonParametricDistributions for each trajectory.

    Index: (animal, time_step)
    """
    print(f"Estimating densities using the {density.upper()} method.")
    # monotonic_q = load_kwargs.get('monotonic_q', False)
    cds = load_kwargs.get('cds', 'mercator')
    dist_kwargs['cds'] = cds

    results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(task=task, quantiles='all', s_q=s_q, mpl_val=mpl_val, **load_kwargs)
    q = np.array(quantiles)
    q_series = y_quantiles.transpose(1, 2, 0, 3) # (N, time_step, target, q)
    q_series = q_series.astype(np.float64)
    N, time_steps, targets, num_quantiles = q_series.shape

    if rho:
        rho_pred, _, rho_q = rho_predictions(task=task, partition=load_kwargs.get('partition', 'test'))
        rho_pred = rho_pred.astype(np.float64)
        rho_spread = np.diff(rho_q[..., [0, -1]].astype(np.float64), axis=-1).squeeze()
        get_rho_kwargs = lambda animal, time_step: dict(rho=rho_pred[animal, time_step],
                                                 rho_spread=rho_spread[animal, time_step])
    else:
        get_rho_kwargs = lambda animal, time_step: dict(rho=0, rho_spread=0)

    if rho and rho_eps == 'best':
        method = dist_kwargs.pop('method', 'hull')
        mpl_load_kwargs = load_kwargs.copy()
        _ = mpl_load_kwargs.pop('partition', None)
        mpls = dist_pr_mpl_val(task=task, s_q=s_q, delete_missing=delete_missing, **mpl_load_kwargs, rho=True, method=method, skip_computation=True)
        if isinstance(mpls, SavedataSkippedComputation):
            raise ValueError("mpl_val not computed for rho=True.")
        else:
            rho_eps = mpls.loc[0.95, 'rho_eps']
            print(f"Loading data for optimal rho_eps={rho_eps}")

    density_kwargs = dict(density=density)
    if density == 'qrde':
        # Pre-compute beta weights
        beta_n = q.size + 2 - int(dist_kwargs.get('exclude_me', False))
        prob = np.arange(0, 1+dp, dp)
        beta_weights = distribution.hd_beta_weights(beta_n, prob)
        density_kwargs['beta_weights'] = beta_weights

    dist = {}
    for animal in tqdm(range(N)):
        for time_step in range(time_steps):
            # if monotonic_q: # ensures increasing quantiles by construction => All distributions are NonParametricBivariateDistributions
            #     x_q, y_q = q_series[animal, time_step]
            #     is_float_x, is_float_y = False, False
            # else:
            x_q, y_q = distribution.ensure_increasing_quantiles(q_series[animal, time_step])  # Ensure x_q and y_q are strictly increasing
            is_float_x, is_float_y = isinstance(x_q, float), isinstance(y_q, float)
            if is_float_x or is_float_y:
                if is_float_x and is_float_y:
                    d = distribution.NonParametricPointDistribution(x_q, y_q, **dist_kwargs)
                else:
                    d = distribution.NonParametricMixedDistribution(x_q, y_q, q, **dist_kwargs, **density_kwargs)
            else:
                d = distribution.NonParametricBivariateDistribution(x_q, y_q, q, **dist_kwargs, **density_kwargs, **get_rho_kwargs(animal, time_step), rho_eps=rho_eps)
            dist[(animal, time_step)] = d
    dist = pd.Series(dist)

    if delete_missing:
        missing = get_missing_values(task=task, **load_kwargs, s_q=s_q, quantiles='all')
        missing = custom_metrics.expand_sequences(missing)
        dist[missing] = np.nan

    if return_target:
        keys = ['partition', 'cds', 'naive_pred']
        target_kwargs = dict(task=task, results=results)
        for k in keys:
            if k in load_kwargs:
                target_kwargs[k] = load_kwargs[k]
        *_, y_real = get_predictions(**target_kwargs)
        y_real = custom_metrics.expand_sequences(y_real)
        if delete_missing:
            y_real[missing] = np.nan
        if cds != 'mercator' and dist_kwargs.get('to_mercator', False):
            r0 = load.reference_point()
            def y_real_to_mercator(x):
                if isinstance(x, float):
                    return x
                else:
                    return space.spherical_fixed_point_to_mercator(x[None], r0)[0]
            y_real = y_real.apply(y_real_to_mercator)
        return dist, y_real
    else:
        return dist

@savedata
def dist_pp(task='forecasting', s_q=1, load_kwargs={}, n_sample=int(1e5), n_grid=1000, mode='mean', mode_margin=0.1, mode_method='sample', mode_weighted=False, rho=False, **kwargs):
    if mode == 'mode':
        if mode_method == 'roots':
            if rho == True or kwargs.get('density', 'pchip') == 'qrde':
                sample_dist = True
                sample_pdf = False
                warnings.warn("mode_method='roots' not supported for rho=True or density='qrde'. Falls back to mode_method='sample' with mode_margin=0.", UserWarning)
            else:
                sample_dist = False
                sample_pdf = False
        elif mode_method == 'grid':
            sample_dist = False
            sample_pdf = True
        else: # sample
            sample_dist = True
            sample_pdf = False
    elif mode == 'median':
        rho = False
        sample_dist = False
        sample_pdf = False
    else: # mean
        rho = False
        sample_dist = True
        sample_pdf = False

    dist = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, n_sample=n_sample, n_grid=n_grid, mode_method=mode_method, mode_margin=mode_margin, mode_weighted=mode_weighted,
                                 sample_dist=sample_dist, sample_pdf=sample_pdf, **kwargs, return_target=False, delete_missing=False, to_mercator=True)
    pp = dist.apply(lambda x: x.pp(mode=mode))
    return pp

@savedata
def optimize_mode_pp(task='forecasting', s_q=1, delete_missing=True, mode_method='sample', rho=False, density='pchip', **load_kwargs):
    num_trials = 200

    if mode_method == 'sample':
        sample_dist = True
        sample_pdf = False
    else:
        sample_dist = False
        sample_pdf = True

    if rho:
        rho_kwargs = dict(rho=True, rho_eps='best')
        n_sample = int(5e4) # sampling with correlation is more costly
        n_grid = 500
    else:
        rho_kwargs = {}
        n_sample = int(1e5)
        n_grid = 1000

    load_kwargs['partition'] = 'val'

    dist, y_target = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, n_sample=n_sample, n_grid=n_grid, sample_dist=sample_dist, sample_pdf=sample_pdf, return_target=True, delete_missing=delete_missing, mode_method=mode_method, density=density, **rho_kwargs, to_mercator=True)

    to_lat_lon = lambda S: S.apply(lambda x: x if isinstance(x, float) else space.mercator_inv(*x)) # in radians
    y_target = to_lat_lon(y_target)

    if delete_missing:
        y_target = y_target.dropna()
        dist = dist.dropna()

    mode_margin_range = (0, 1)
    mode_weighted_range = [True, False]

    def objective(trial):
        mode_margin = trial.suggest_float("mode_margin", *mode_margin_range)
        mode_weighted = trial.suggest_categorical("mode_weighted", mode_weighted_range)

        pp_mode = dist.apply(lambda x: x.pp(mode='mode', mode_margin=mode_margin, mode_weighted=mode_weighted))
        pp_mode = to_lat_lon(pp_mode)

        df = pd_utils.vstack_wise(y_target.apply(lambda x: np.array(x)[:, None]),
                                  pp_mode.apply(lambda x: np.array(x)[:, None])).apply(np.squeeze)
        distance = df.apply(lambda x: space.great_circle_distance(*x)) * params.R_earth
        error_by_t = distance.groupby(level=1).mean()
        rmse = np.sqrt((error_by_t**2).mean())
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    results = study.trials_dataframe().sort_values('value')
    results = results[['value', 'params_mode_margin', 'params_mode_weighted']]
    results.columns = ['distance_rmse', 'mode_margin', 'mode_weighted']
    return results.head(5)

def pp_summary_dist(task='forecasting', s_q=5, params_idx=0, add_best_mode=True, rho=False, **extra_kws):
    specs = dict(task=task, s_q=s_q, params_idx=params_idx)
    specs.update(params.TFT_specs[task])
    specs.update(extra_kws)
    if task == 'forecasting':
        specs['max_train_days'] = 4
    specs['quantiles'] = 'all'

    if rho:
        pp_specs = dict(median=dict(dist_mode='median'),
                        mean=dict(dist_mode='mean'),
                        mode_roots=dict(dist_mode='mode', mode_method='roots', rho=True, rho_eps=0.1),
                        mode_roots_best_rho_eps=dict(dist_mode='mode', mode_method='roots', rho=True, rho_eps='best'))
        if add_best_mode:
            pp_specs['mode_margin'] = dict(dist_mode='mode', mode_method='sample', mode_margin='best', rho=True, rho_eps='best')
    else:
        pp_specs = dict(median=dict(dist_mode='median'),
                        mean=dict(dist_mode='mean'),
                        mode_roots=dict(dist_mode='mode', mode_method='roots'))
        if add_best_mode:
            pp_specs['mode_margin'] = dict(dist_mode='mode', mode_method='sample', mode_margin='best')
    out = []
    for method, kwargs in tqdm(pp_specs.items()):
        sample_stat, CI = point_prediction_aggregate_CI(**specs, **kwargs)
        out.append(pd.Series(dict(CI=CI, sample_stat=sample_stat), name=method))
    out = pd.concat(out, axis=1).T
    return out.sort_values('sample_stat')

@savedata('all-preload')
def dist_pr(task='forecasting', cs=[0.5, 0.9, 0.95], s_q=1, delete_missing=True, n_sample=int(1e5), n_grid=100, method='hull', partition='test', mpl_val=False, preload=True, **load_kwargs):
    if isinstance(cs, str) and cs == 'all':
        cs = params.cs_pr_plot
    rho = load_kwargs.pop('rho', False)
    rho_eps = load_kwargs.pop('rho_eps', 0.8)
    optimize_spread = load_kwargs.pop('optimize_spread', False)
    optimize_cmax = load_kwargs.pop('optimize_cmax', False)
    density = load_kwargs.pop('density', 'pchip')
    exclude_me = load_kwargs.pop('exclude_me', False)
    dist_kws = {}
    if exclude_me:
        dist_kws['exclude_me'] = True
    if rho:
        sample_dist = not mpl_val
        if method == 'contour':
            sample_pdf = not mpl_val
        else:
            sample_pdf = False
    else:
        sample_dist = True
        if method == 'contour':
            sample_pdf = True
        else:
            sample_pdf = False
    if not preload:
        sample_pdf = False
        sample_dist = False
    load_kwargs['partition'] = partition

    dist, y_real = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, return_target=True, n_sample=n_sample, n_grid=n_grid, sample_pdf=sample_pdf, sample_dist=sample_dist, delete_missing=delete_missing, rho=rho, rho_eps=rho_eps, density=density, to_mercator=True, **dist_kws)

    if mpl_val:
        print("Evaluating PR area and coverage using mpl_val")
        mpl_val_kwargs = load_kwargs.copy()
        del mpl_val_kwargs['partition']
        if density != 'pchip':
            mpl_val_kwargs['density'] = density
        if exclude_me:
            mpl_val_kwargs['exclude_me'] = True
        if rho:
            if optimize_spread:
                mpl_val_kwargs['optimize_spread'] = True
            if optimize_cmax:
                mpl_val_kwargs['optimize_cmax'] = True

        # Do not pass n_sample or n_grid. Load mpls using default values.
        mpls = dist_pr_mpl_val(task=task, target_cs=cs, s_q=s_q, delete_missing=delete_missing, method=method, **mpl_val_kwargs, rho=rho)
        c_to_mpl = mpls['mpl'].to_dict()
        c_max = 1 if density == 'qrde' else 0.98
        cds_c_max = None
        out = []
        for c, mpl in tqdm(c_to_mpl.items()):
            if rho:
                rho_eps = mpls.loc[c, 'rho_eps']
                if optimize_spread:
                    max_rho_spread = mpls.loc[c, 'max_rho_spread']
                else:
                    max_rho_spread = None
                dist = dist.apply(mod_rho_specs, rho_eps=rho_eps, method=method, max_rho_spread=max_rho_spread)
                if optimize_cmax:
                    c_max = mpls.loc[c, 'c_max']
                    if np.isnan(c_max):
                        c_max = 1 if density == 'qrde' else 0.98
                    # no NonParametricMixedDistribution in validation set
                    # dx = mpls.loc[c, 'dx']
                    # dy = mpls.loc[c, 'dy']
                    # dist = dist.apply(mod_dy_specs, dx=dx, dy=dy)
            if mpl > c_max:
                expand_mpl = mpl - c_max
                if cds_c_max is None or rho:
                    _, cds_c_max = distribution.eval_pr(dist, y_real, c_max, method)
                cds = distribution.expand_pr(cds_c_max, expand_mpl)
                df = distribution.eval_pr_from_cds(cds, y_real) # index (time_step, animal)
                # add confidence to index
                df['confidence'] = c
                df = df.set_index('confidence', append=True)
            else:
                out_, _ = distribution.eval_pr(dist, y_real, mpl, method)
                df = out_.unstack()
                df.index.names = ['time_step', 'animal', 'confidence']
                # replace confidence in index, from mpl -> c
                df = df.rename(index={mpl: c}, level='confidence')
            out.append(df)
        df = pd.concat(out, axis=0)
    else:
        print("Evaluating PR area and coverage")
        df = pd.concat([distribution.eval_pr(dist, y_real, c, method=method, del_after_compute=True)[0] for c in cs], axis=0)
        df = df.unstack()
        df.index.names = ['time_step', 'animal', 'confidence']

    df = df.sort_index()
    return df

def dist_pr_cds(cs=0.9, save=False, compute_jointly=True, **kwargs):
    if isinstance(cs, float):
        cs = [cs]
    elif isinstance(cs, str) and cs == 'all':
        cs = params.cs_pr_plot

    if len(cs) == 1:
        df = _dist_pr_cds(cs=cs[0], **kwargs, save=save)
        df = df.droplevel('confidence')
    else:
        if compute_jointly:
            df = _dist_pr_cds(cs=cs, **kwargs, save=save)
        else:
            out = []
            for c in cs:
                out.append(_dist_pr_cds(cs=c, **kwargs, save=save))
            df = pd.concat(out, axis=0)
    return df

@savedata(save=False) # do not save unless explicitely stated
def _dist_pr_cds(task='forecasting', cs=0.9, s_q=1, delete_missing=True, n_sample=int(1e5), n_grid=100, method='hull', partition='test', mpl_val=False, rho=False, density='pchip', **load_kwargs):
    if isinstance(cs, float):
        cs = [cs]
    sample_dist = True
    if method == 'contour':
        sample_pdf = True
    else:
        sample_pdf = False
    load_kwargs['partition'] = partition
    exclude_me = load_kwargs.pop('exclude_me', False)
    optimize_spread = load_kwargs.pop('optimize_spread', False)
    optimize_cmax = load_kwargs.pop('optimize_cmax', False)

    dist, y_real = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, return_target=True, n_sample=n_sample, n_grid=n_grid, sample_pdf=sample_pdf, sample_dist=sample_dist, delete_missing=delete_missing, to_mercator=True, density=density, rho=rho)

    if mpl_val:
        print("Evaluating PR area and coverage using mpl_val")
        mpl_val_kwargs = load_kwargs.copy()
        del mpl_val_kwargs['partition']
        if density != 'pchip':
            mpl_val_kwargs['density'] = density
        if exclude_me:
            mpl_val_kwargs['exclude_me'] = True
        if rho:
            if optimize_spread:
                mpl_val_kwargs['optimize_spread'] = True
            if optimize_cmax:
                mpl_val_kwargs['optimize_cmax'] = True
        # Do not pass n_sample or n_grid. Load mpls using default values.
        mpls = dist_pr_mpl_val(task=task, s_q=s_q, delete_missing=delete_missing, method=method, **mpl_val_kwargs, rho=rho, target_cs=cs)
        c_max = 1 if density == 'qrde' else 0.98
        cds_c_max = None
        out = []
        for c in tqdm(cs):
            if rho:
                rho_eps = mpls.loc[c, 'rho_eps']
                if optimize_spread:
                    max_rho_spread = mpls.loc[c, 'max_rho_spread']
                else:
                    max_rho_spread = None
                dist = dist.apply(mod_rho_specs, rho_eps=rho_eps, method=method, max_rho_spread=max_rho_spread)
                if optimize_cmax:
                    c_max = mpls.loc[c, 'c_max']
                    if np.isnan(c_max):
                        c_max = 1 if density == 'qrde' else 0.98
                    # dx = mpls.loc[c, 'dx']
                    # dy = mpls.loc[c, 'dy']
                    # dist = dist.apply(mod_dy_specs, dx=dx, dy=dy)

            mpl = mpls.loc[c, 'mpl']
            if mpl > c_max:
                expand_mpl = mpl - c_max
                if cds_c_max is None or rho:
                    _, cds_c_max = distribution.eval_pr(dist, y_real, c_max, method)
                cds = distribution.expand_pr(cds_c_max, expand_mpl)
                # add confidence to index
                cds = cds.to_frame(name='cds')
                cds['confidence'] = c
                cds = cds.set_index('confidence', append=True)['cds']
            else:
                _, cds = distribution.eval_pr(dist, y_real, mpl, method)
                cds.name = 'cds'
                # replace confidence in index, from mpl -> c
                cds.index.names = ['time_step', 'animal', 'confidence']
                cds = cds.rename(index={mpl: c}, level='confidence')
            out.append(cds)
        df = pd.concat(out, axis=0)
    else:
        print("Evaluating PR area and coverage")
        df = pd.concat([distribution.eval_pr(dist, y_real, c, method=method)[1] for c in cs], axis=0)
        df.index.names = ['time_step', 'animal', 'confidence']

    df = df.sort_index()
    return df

def mod_rho_specs(d, rho_eps=0.1, method='hull', max_rho_spread=None):
    if isinstance(d, distribution.NonParametricBivariateDistribution):
        d.rho_eps = rho_eps
        d.max_rho_spread = max_rho_spread
        if method == 'contour':
            pdf, l_x, l_y = d.get_pdf(rho_eps=rho_eps)
            d.pdf = pdf
            d.l_x = l_x
            d.l_y = l_y
        else:
            d.get_sample(rho_eps=rho_eps)
    return d

def mod_dy_specs(d, dx=10, dy=10):
    if isinstance(d, distribution.NonParametricMixedDistribution):
        d.dx = dx
        d.dy = dy
    return d

def dist_pr_mpl_val(task='forecasting', target_cs=[0.5, 0.9, 0.95], s_q=1, delete_missing=True, n_sample=None, n_grid=50, method='hull', rho=False, **load_kwargs):

    @savedata
    def _dist_pr_mpl_val(task='forecasting', target_c=0.9, s_q=1, delete_missing=True, n_sample=int(1e5), n_grid=50, method='hull', **load_kwargs):
        sample_dist = True
        if method == 'contour':
            sample_pdf = True
            n_trials = 100
        else:
            sample_pdf = False
            if method == 'hull':
                n_trials = 200
            else:
                n_trials = 100
        if 'partition' in load_kwargs:
            raise ValueError("partition should not be passed as a load_kwargs")
        load_kwargs['partition'] = 'val'
        density = load_kwargs.pop('density', 'pchip')
        exclude_me = load_kwargs.pop('exclude_me', False)

        dist, y_real = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, return_target=True, n_sample=n_sample, n_grid=n_grid, sample_pdf=sample_pdf, sample_dist=sample_dist, delete_missing=delete_missing, to_mercator=False, density=density, exclude_me=exclude_me)

        print("Evaluating predictive distributions for c=c_max")
        c_max = 1 if density == 'qrde' else 0.98
        _, cds_c_max = distribution.eval_pr(dist, y_real, c_max, method)
        mpl_range = (0, 5.)

        def optimize_dist_mpl(c=0.5):
            """
            Optimize the mpl parameter for the CR expected coverage.
            """
            def objective(trial):
                mpl = trial.suggest_float('mpl', *mpl_range)
                if mpl > c_max:
                    expand_mpl = mpl - c_max
                    cds = distribution.expand_pr(cds_c_max, expand_mpl)
                    df = distribution.eval_pr_from_cds(cds, y_real)
                else:
                    out, _ = distribution.eval_pr(dist, y_real, mpl, method)
                    df = out.unstack()
                    df.index.names = ['time_step', 'animal', 'confidence']

                error = df.dropna()['coverage'].groupby('time_step').mean() - c
                rmse = np.sqrt((error**2).mean())
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            return study

        mpls = {}
        study = optimize_dist_mpl(target_c)
        mpl, rmse = get_smallest_mpl(study)
        mpls[(target_c, 'mpl')] = mpl
        mpls[(target_c, 'rmse')] = rmse
        mpls = pd.Series(mpls).unstack(-1)
        return mpls

    @savedata
    def _dist_pr_mpl_val_rho(task='forecasting', target_c=0.9, s_q=1, delete_missing=True, n_sample=int(2e4), n_grid=50, method='hull', **load_kwargs):
        density = load_kwargs.pop('density', 'pchip')
        exclude_me = load_kwargs.pop('exclude_me', False)
        optimize_spread = load_kwargs.pop('optimize_spread', False)
        optimize_cmax = load_kwargs.pop('optimize_cmax', False)
        dist, y_real = load_np_distributions(task=task, s_q=s_q, load_kwargs=load_kwargs, return_target=True, delete_missing=delete_missing, density=density,
                                             n_sample=n_sample, n_grid=n_grid, sample_pdf=False, sample_dist=False, rho=True, to_mercator=True, exclude_me=exclude_me)

        not_valid = dist.apply(lambda x: not isinstance(x, distribution.NonParametricBivariateDistribution))
        dist[not_valid] = np.nan
        y_real[not_valid] = np.nan

        # Compute reference data
        results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(task=task, **load_kwargs, s_q=s_q, quantiles='all')
        partition = load_kwargs.get('partition', 'test')
        baseline, _, y_r = get_predictions(results=results, x=x, task=task, partition=partition, naive_pred='last-obs')
        df_ref = pd_utils.tuple_wise(y_r, baseline)
        out_index = y_real.swaplevel().sort_index().index

        time_steps = out_index.get_level_values(0)
        days = time_step_to_days(time_steps)
        flatten_area_series = lambda S: np.vstack(S).T.flatten()

        A_max = custom_metrics.maximal_area(task=task, baseline=baseline, days=days)
        if isinstance(A_max, pd.Series):
            A_max = flatten_area_series(A_max)
        A_ref = df_ref.apply(custom_metrics.compute_reference_area, task=task, alpha=1-target_c)
        A_ref = flatten_area_series(A_ref)
        ref_data = pd.DataFrame(dict(area_ref=A_ref, area_max=A_max), index=out_index)
        ref_data[y_real.isna().values] = np.nan

        # if method == 'hull':
        #     n_trials = 200
        # else:
        #     n_trials = 120
        n_trials = 200
        # if optimize_cmax:
        #     n_trials += 80

        mpl_range = (0., 5.)
        rho_eps_range = (0.1, 0.95)
        max_rho_spread_range = (0.1, 3 if task == 'imputation' else 6) # imputation is more accurate
        c_max_range = (0.8, 1)
        dx_range = (10, 500)
        dy_range = (10, 500)

        def optimize_params(c=0.5, dist=None):
            """
            Optimize mpl parameters for Q_alpha and Q_area.
            """
            def objective(trial, dist=dist):
                mpl = trial.suggest_float('mpl', *mpl_range)
                rho_eps = trial.suggest_float('rho_eps', *rho_eps_range)
                if optimize_spread:
                    max_rho_spread = trial.suggest_float('max_rho_spread', *max_rho_spread_range)
                else:
                    max_rho_spread = None
                if optimize_cmax and mpl > c_max_range[0]:
                    c_max = trial.suggest_float('c_max', *c_max_range)
                else:
                    c_max = 1 if density == 'qrde' else 0.98


                dist = dist.apply(mod_rho_specs, rho_eps=rho_eps, method=method, max_rho_spread=max_rho_spread)
                # no distribution in validation is NonParametricMixedDistribution
                # dx = trial.suggest_float('dx', *dx_range, log=True)
                # dy = trial.suggest_float('dy', *dy_range, log=True)
                # dist = dist.apply(mod_dy_specs, dx=dx, dy=dy)

                if mpl > c_max:
                    expand_mpl = mpl - c_max
                    _, cds_c_max = distribution.eval_pr(dist, y_real, c_max, method=method)
                    cds = distribution.expand_pr(cds_c_max, expand_mpl)
                    df = distribution.eval_pr_from_cds(cds, y_real)
                    df.index.names = ['time_step', 'animal']
                else:
                    out, _ = distribution.eval_pr(dist, y_real, mpl, method)
                    df = out.unstack()
                    df.index.names = ['time_step', 'animal', 'confidence']
                    df.index = df.index.droplevel('confidence')
                df['coverage'] = df['coverage'].astype(float)

                # Compute quality
                area_data = pd.concat((df['area'], ref_data[['area_ref', 'area_max']]), axis=1)
                area_data = area_data.values.astype(float).T
                df['Q_area'] = custom_metrics.area_quality(*area_data)
                avg_by_animal = df.groupby(['time_step']).mean()
                avg_by_animal['Q_alpha'] = custom_metrics.coverage_quality(1 - avg_by_animal['coverage'],
                                                                           1 - c)
                result = avg_by_animal[['Q_area', 'Q_alpha']]
                Q_alpha = result['Q_alpha'].mean()
                Q_area = result['Q_area'].mean()
                # prioritize Q_alpha
                if Q_alpha > 0.95:
                    return Q_area # in [0, 1]
                else:
                    return -1 # force optuna to discard the trial

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            return study

        study = optimize_params(target_c, dist)
        mpls = {}
        for k, v in study.best_params.items():
            mpls[(target_c, k)] = v
        mpls = pd.Series(mpls).unstack(-1)
        return mpls

    if rho:
        func = _dist_pr_mpl_val_rho
        if n_sample is None:
            n_sample = int(2e4)
    else:
        func = _dist_pr_mpl_val
        if n_sample is None:
            n_sample = int(1e5)

    out = []
    for target_c in target_cs:
        df = func(task=task, target_c=target_c, s_q=s_q, delete_missing=delete_missing, n_sample=n_sample, n_grid=n_grid, method=method, **load_kwargs)
        if isinstance(df, SavedataSkippedComputation):
            return SavedataSkippedComputation
        else:
            out.append(df)
    out = pd.concat(out, axis=0)
    out.index.name = 'confidence'
    return out

@savedata
def best_model_distribution(task='forecasting', mpl_val=True, **kwargs):
    specs = dict(quantiles='all', mpl_val=mpl_val, max_train_days=4, skip_computation=True, task=task)
    specs.update(**params.TFT_specs[task])
    specs.update(**kwargs)

    s_qs = [0.5, 1, 5, 10]
    dist_methods = ['hull', 'contour']
    num_models = 5
    partitions = ['val', 'test']
    pbar = tqdm(range(len(s_qs) * len(dist_methods) * num_models * len(partitions)))

    out = []
    not_computed = []
    for s_q in s_qs:
        specs['s_q'] = s_q
        for dist_method in dist_methods:
            specs['dist_method'] = dist_method
            for params_idx in range(num_models):
                specs['params_idx'] = params_idx
                for partition in partitions:
                    specs['partition'] = partition
                    df = quality_sample(**specs)
                    if isinstance(df, SavedataSkippedComputation):
                        warnings.warn(f"Skipping s_q={s_q}, dist_method={dist_method}, params_idx={params_idx}, partition={partition}")
                        not_computed.append(pd.Series(dict(s_q=s_q, dist_method=dist_method, params_idx=params_idx, partition=partition)))
                    else:
                        has_nans = df.isna().values.any()
                        df = df.mean()
                        if has_nans:
                            df[:] = np.nan
                            warnings.warn(f"NaNs found in s_q={s_q}, dist_method={dist_method}, params_idx={params_idx}, partition={partition}")
                        df['s_q'] = s_q
                        df['dist_method'] = dist_method
                        df['params_idx'] = params_idx
                        df['partition'] = partition
                        out.append(df)
                    pbar.update(1)
    pbar.close()

    df = pd.concat(out, axis=1).T
    df = df.set_index(['partition', 's_q', 'dist_method', 'params_idx'])
    val = df.loc['val']
    test = df.loc['test']
    val = val.sort_values('Q', ascending=False)
    test = test.loc[val.index.intersection(test.index)]
    if not_computed:
        not_computed = pd.concat(not_computed, axis=1).T

    return val, test, not_computed

def dist_best_model_specs(task='forecasting', mpl_val=True, density='qrde', rho=True, optimize_spread=False, optimize_cmax=False, **kwargs):
    extra_kwargs = {}
    if density != 'pchip':
        extra_kwargs['density'] = density
    if rho:
        extra_kwargs['rho'] = True
        if mpl_val:
            if optimize_spread:
                extra_kwargs['optimize_spread'] = True
            if optimize_cmax:
                extra_kwargs['optimize_cmax'] = True

    val, *_ = best_model_distribution(task=task, mpl_val=mpl_val, **kwargs, **extra_kwargs)
    best = val.reset_index().iloc[0]
    # if mpl_val:
    #     val_rho, *_ = best_model_distribution(task=task, mpl_val=mpl_val, rho=True, **kwargs, **extra_kwargs)
    #     best_rho = val_rho.reset_index().iloc[0]
    #     if best_rho['Q'] > best['Q']:
    #         best = best_rho
    #         best['rho'] = True
    params = [i for i in best.index if not i.startswith('Q')]
    params = best[params].to_dict()
    params.update(extra_kwargs)
    if 'dist_method' in params:
        params['method'] = params.pop('dist_method')
    print(f"Best parameters for distribution model: {params}")
    return params

@savefig
def plot_pr_comparison(task='forecasting', partition='test', i=28, step=3, s_q=5, c=0.9, params_idx=0, density='pchip'):
    load_kwargs = params.TFT_specs[task].copy()
    if task == 'forecasting':
        load_kwargs['max_train_days'] = 4
    load_kwargs['partition'] = partition
    cds = dist_pr_cds(task=task, s_q=s_q, params_idx=params_idx, **load_kwargs, mpl_val=True, n_sample=int(1e5), cs=c, density=density)
    y_real = load_np_distributions_y_real(task=task, s_q=s_q, params_idx=params_idx, **load_kwargs)

    params_idx_q, _ = quantile_best_model(task=task, **params.TFT_specs[task])
    kwargs_q = dict(task=task, **params.TFT_specs[task], params_idx=params_idx_q, partition=partition)
    results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(**kwargs_q, mpl_val=True)
    missing_q = get_missing_values(**kwargs_q)
    num_missing_pr = y_real.groupby(level=0).apply(lambda x: x.isna().sum())
    num_missing_q = missing_q.apply(np.sum)
    assert (num_missing_q == num_missing_pr).all(), "animals for distribution and quantile approaches do not match"

    fig = trajectory_confidence_region(**kwargs_q, i=i, c=c, mpl_val=True, return_fig=True, save=False, step=step)

    pr = cds.loc[step, i]
    to_degrees = lambda y: np.stack(space.mercator_inv(*y.T), axis=0) * 180 / np.pi
    pr = to_degrees(pr)
    pr = pr[:, ::-1]
    fig.add_trace(go.Scattergeo(
        lat=pr[0],
        lon=pr[1],
        mode='lines+markers',
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=0.8),
        name = 'PR (D)'
    ))

    pred_kwargs = load_kwargs.copy()
    pred_kwargs['params_idx'] = params_idx
    modes = {'me': ('median', 'magenta'), 'μ': ('mean', 'black'), 'mode': ('mode', 'green')}
    if density != 'pchip':
        pp_kwargs = dict(density=density)
    else:
        pp_kwargs = {}
    for label, (mode, color) in modes.items():
        y_pred = dist_pp(task=task, s_q=s_q, load_kwargs=pred_kwargs, mode=mode, **pp_kwargs)
        y_p = np.vstack(y_pred.loc[i])[:step+1]
        y_p = to_degrees(y_p)
        fig.add_trace(go.Scattergeo(lat=y_p[0, :step+1], lon=y_p[1, :step+1], mode='lines+markers', name=label, showlegend=True,
                                    marker=dict(color=color, size=10, line=dict(color='black', width=3)), line_width=3))

    return fig

def rho_predictions(task='forecasting', params_idx='best', partition='test', skip_computation=True):
    if params_idx == 'best':
        params_idx = rho_best_model(task=task)
    kwargs = params.TFT_specs[task].copy()
    kwargs['max_train_days'] = 4
    if task == 'forecasting':
        s_q = 10
    else:
        s_q = 5
    mod_hp = dict(quantiles='all', s_q=s_q)
    results = quantile_results(task=task, **kwargs, params_idx=params_idx, mod_hp=mod_hp, skip_computation=skip_computation, target='rho', quantiles='1D')
    if isinstance(results, SavedataSkippedComputation):
        return results
    else:
        y_pred = results[f'y_pred_{partition}']
        y_real = results[f'x_{partition}']['decoder_target'].numpy()
        y_q = results[f'y_pred_raw_{partition}']
        return y_pred, y_real, y_q

def eval_rho_predictions(**kwargs):
    out = rho_predictions(**kwargs)
    if isinstance(out, SavedataSkippedComputation):
        return pd.Series(dict(rmse=np.nan, rho_spread_rmse=np.nan))
    else:
        y_pred, y_real, y_q = out
        error = y_real - y_pred
        rmse = np.sqrt(np.mean(error**2, axis=1))
        rho_spread = np.diff(y_q[..., [0, -1]], axis=-1).squeeze()
        rho_spread_rmse = np.sqrt(np.mean(rho_spread**2, axis=1))
        S = pd.Series(dict(rmse=rmse.mean(), rho_spread_rmse=rho_spread_rmse.mean()))
        return S

@savedata
def rho_predictions_summary(num_models=5, **kwargs):
    df = pd.concat([eval_rho_predictions(params_idx=i, **kwargs) for i in range(num_models)], axis=1).T
    return df

def rho_best_model(task='forecasting'):
    df = rho_predictions_summary(task=task, partition='val')
    best = df['rmse'].idxmin()
    print(f"Rho best  model for task {task}: {best}")
    return best

def clip_polygon(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Clips polygon p1 using polygon p2 and returns the intersection area.

    Parameters:
        p1 (np.ndarray): First polygon as an (N, 2) array of (x, y) points.
        p2 (np.ndarray): Second polygon as an (M, 2) array of (x, y) points.

    Returns:
        np.ndarray: The clipped polygon as an array of (x, y) points.
                    Returns an empty array if there is no intersection.
    """
    poly1 = Polygon(p1)
    poly2 = Polygon(p2)

    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        return np.empty((0, 2))
    return np.array(intersection.exterior.coords)

def plot_bivariate_pr(fig, pr, colorscale='Hot_r', black_line=True, opacity=0.4, undersample=2, cbar=True):
    sorted_probs = pr.index.values[::-1] # highest to lowest
    if undersample > 1:
        sorted_probs = sorted_probs[::undersample]
    colorbar = (sorted_probs - sorted_probs.min()) / (sorted_probs.max() - sorted_probs.min())
    try:
        cmap = cm.get_cmap(colorscale)
    except:
        cmap = cm.get_cmap(colorscale.lower())
    colors = cmap(colorbar)
    line_color = colors.copy()
    fillcolor = colors.copy()
    fillcolor[:, -1] = opacity

    p_big = pr[0.95]
    for i, prob in enumerate(sorted_probs):
        fc = fillcolor[i]
        lc = line_color[i]
        poly = pr[prob]
        if i > 0:
            poly = clip_polygon(poly.T, p_big.T).T
            p_big = poly.copy()
        poly = poly[:, ::-1]
        lat, lon = poly

        fig.add_trace(go.Scattergeo(
            lat=list(lat) + [lat[0]],  # Close the polygon by repeating the first point
            lon=list(lon) + [lon[0]],  # Close the polygon by repeating the first point
            fill="toself",
            # fillcolor=to_hex(colors[i]),
            fillcolor=f"rgba{tuple(fc)}",
            line=dict(color="black" if black_line else to_hex(lc),
                      width=2),
            # line=dict(color="black", width=1),
            mode="lines",
            showlegend=False,
        ))

    if cbar:
        p0 = sorted_probs.min()
        pf = sorted_probs.max()
        # tickvals = [p0, 0.25, 0.5, 0.75, pf]
        tickvals = [p0, 0.5, pf]
        fig.add_trace(go.Scattergeo(
            lat=[None],
            lon=[None],
            mode='markers',
            marker=dict(
                colorscale=colorscale,
                showscale=True,
                # colorbar on the left side of the plot
                colorbar=dict(title="PR TFT[B]", tickvals=tickvals, len=0.6),
                cmin=p0,
                cmax=pf,
            ),
            showlegend=False
        ))
    return

def plot_pr_comparison_cds(task='forecasting', partition='test', method='hull'):
    specs = dist_best_model_specs(task=task, mpl_val=True, rho=True, optimize_spread=True, optimize_cmax=True)
    specs['partition'] = partition
    specs_pp = specs.copy()
    specs.update(params.TFT_specs[task])
    specs['max_train_days'] = 4
    specs_cds = specs.copy()
    del specs_pp['method']
    if task == 'imputation':
        specs_cds['params_idx'] = 0
        specs_cds['s_q'] = 5
    del specs_cds['method']

    cds = dist_pr_cds(task=task, **specs_cds, cs='all', mpl_val=True, method=method)
    return cds, specs

@savefig('all-cds-specs-results-verify_match')
def plot_pr_comparison_v2(cds=None, specs=None, results=None, task='forecasting', animal=28, step=3, partition='test', method='hull', legend=True, cbar=True, verify_match=True):
    if cds is None or specs is None:
        cds, specs = plot_pr_comparison_cds(task=task, partition=partition, method=method)
    pr = cds.loc[step, animal]
    if method == 'contour':
        def keep_most_relevant(pr_i):
            exterior, holes = pr_i
            return exterior[0]
        pr = pr.apply(keep_most_relevant)
    to_degrees = lambda y: np.stack(space.mercator_inv(*y.T), axis=0) * 180 / np.pi
    pr = pr.apply(to_degrees)
    s_q = specs['s_q']
    params_idx = specs['params_idx']

    # TFT rectangle
    params_idx_q, _ = quantile_best_model(task=task, **params.TFT_specs[task])
    kwargs_q = dict(task=task, **params.TFT_specs[task], params_idx=params_idx_q, partition=partition)
    if results is None:
        results, x, y_quantiles, decoder_targets, decoder_lengths, targets, quantiles, quantile_method = load_quantile_results(**kwargs_q, mpl_val=True)
    else:
        x = results[f'x_{partition}']

    if verify_match:
        load_kwargs = params.TFT_specs[task].copy()
        if task == 'forecasting':
            load_kwargs['max_train_days'] = 4
        load_kwargs['partition'] = partition
        y_real = load_np_distributions_y_real(task=task, s_q=s_q, params_idx=params_idx, **load_kwargs)

        missing_q = get_missing_values(x=x)
        num_missing_pr = y_real.groupby(level=0).apply(lambda x: x.isna().sum())
        num_missing_q = missing_q.apply(np.sum)
        assert (num_missing_q == num_missing_pr).all(), "animals for distribution and quantile approaches do not match"

    fig = trajectory_confidence_region(**kwargs_q, i=animal, c=0.95, mpl_val=True, return_fig=True, save=False, step=step, plot_rest=False, PR_label='PR TFT (p=0.95)', results=results)
    plot_bivariate_pr(fig, pr, 'Hot_r', cbar=cbar)
    fig = trajectory_confidence_region(fig=fig, **kwargs_q, i=animal, c=0.95, mpl_val=True, return_fig=True, save=False, step=step, plot_pr=False, ms_p=20, mlw_p=4, lc='#404040', lc_p='black', results=results)
    fig.update_layout(legend_orientation="h", legend=dict(x=0.5, y=-0.1))
    day = time_step_to_days(step+1)
    if day == int(day):
        day = int(day)
    fig.update_layout(title=dict(text="t = {} {}".format(day, 'day' if day == 1 else 'days'),
                                 x=0.5, y=0.9, xanchor='center', yanchor='top',
                                 font_size=40))
    # Adapt limits
    lat_min, lat_max = fig.layout.geo.lataxis.range
    lon_min, lon_max = fig.layout.geo.lonaxis.range
    offset_lon = 0.5
    offset_lat = 0.25
    lat_min = min(lat_min, pr.apply(lambda pr_i: pr_i[0].min()).min() - offset_lat)
    lat_max = max(lat_max, pr.apply(lambda pr_i: pr_i[0].max()).max() + offset_lat)
    lon_min = min(lon_min, pr.apply(lambda pr_i: pr_i[1].min()).min() - offset_lon)
    lon_max = max(lon_max, pr.apply(lambda pr_i: pr_i[1].max()).max() + offset_lon)
    fig.update_geos(lataxis_range=[lat_min, lat_max], lonaxis_range=[lon_min, lon_max])
    if not legend:
        fig.update_layout(showlegend=False)
    return fig

@savefig('all-cds-specs-results')
def plot_pr_comparison_trajectory(cds=None, specs=None, results=None, task='forecasting', partition='test', method='hull', animal=37, day_step=1, **kwargs):
    """
    Returns:
    - plotly.graph_objects.Figure: A Plotly figure object containing the subplots for each day.
    """
    args = dict(task=task, partition=partition, method=method)
    if day_step is None:
        if task == 'forecasting':
            days = np.arange(1, 4)
        else:
            days = np.array([1, 3, 6])
    else:
        days = np.arange(day_step, day_step*4, day_step)
    steps = days*4 - 1
    figs = []
    if cds is None or specs is None:
        cds, specs = plot_pr_comparison_cds(**args)
    if results is None:
        # TFT rectangle
        params_idx_q, _ = quantile_best_model(task=task, **params.TFT_specs[task])
        kwargs_q = dict(task=task, **params.TFT_specs[task], params_idx=params_idx_q, partition=partition)
        results = load_quantile_results(**kwargs_q, mpl_val=True)[0]
    for step in tqdm(steps):
        try:
            fig = plot_pr_comparison_v2(cds=cds, specs=specs, results=results, animal=animal, step=step, **args, **kwargs,
                                        legend=False, cbar=False, verify_match=False,
                                        return_fig=True, save=False)
        except:
            warnings.warn("No data available for the required step. Attempting with the subsequent step.", RuntimeWarning)
            fig = plot_pr_comparison_v2(cds=cds, specs=specs, results=results, animal=animal, step=step+1, **args, **kwargs,
                                        legend=False, cbar=False, verify_match=False,
                                        return_fig=True, save=False)
        figs.append(fig)

    # common range
    lat_range = [min(fig.layout.geo.lataxis.range[0] for fig in figs), max(fig.layout.geo.lataxis.range[1] for fig in figs)]
    lon_range = [min(fig.layout.geo.lonaxis.range[0] for fig in figs), max(fig.layout.geo.lonaxis.range[1] for fig in figs)]

    subplot_specs = [[dict(type='scattergeo')] * steps.size]
    subplots = get_subplots(cols=steps.size, rows=1,
                            subplot_titles=[f"Day {d}" for d in days],
                            height=450, width=1700,
                            specs=subplot_specs,
                            )
    for i, fig in enumerate(figs):
        for data in fig.data:
            data.showlegend = False
            subplots.add_trace(data, row=1, col=i+1)
    subplots.update_layout(margin=dict(l=10, b=10, r=10, t=55))
    subplots.update_geos(lataxis_range=lat_range, lonaxis_range=lon_range)
    return subplots
