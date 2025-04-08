import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import torch
import optuna
from numba import njit
import scipy.stats as ss
import re
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn._statistics import KDE
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

from tidypath.fmt import dict_to_id, hash_string
from phdu import SavedataSkippedComputation, geometry, bootstrap, pd_utils
import phdu.stats.corr as phdu_corr
from phdu import savedata, savefig, geometry
from phdu.plots.plotly_utils import CI_plot, CI_ss_plot, get_figure, plot_confidence_bands, mod_logaxes_expfmt, get_subplots, mod_range, fig_base_layout
from phdu.plots.base import color_std, plotly_default_colors
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.model_selection import StratifiedKFold, KFold
from . import model, distribution, params, ssm, custom_metrics, analysis
from .custom_metrics import nb_mean, nb_median, time_step_to_days, rae, arae, aggregate_metric_CI, aggregate_CI_summary
from .preprocessing import space, load
from .shap_plots_mod import summary_legacy as shap_summary
try:
    from phdu.stats.rtopy._helper import load_R_pkg, r, ro
except:
    pass


class ModelCheckpoint(xgb.callback.TrainingCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_score = float("inf")

    def after_iteration(self, model, epoch, evals_log):
        current_score = evals_log['val']['rmse'][-1]
        if current_score < self.best_score:
            self.best_score = current_score
            model.save_model(self.save_path)
        return False

def prune_by_nan_ptg(df, nan_threshold=0.6):
    nan_ptg = df.isna().mean()
    valid = nan_ptg < nan_threshold
    print(f"Pruning {(~valid).sum()} features with more than {nan_threshold:.0%} missing values")
    df = df.loc[:, valid]
    return df

def fill_inf(Z):
    numerical_with_infs = ((Z == np.inf).any()) & (Z.dtypes != 'category')
    numerical_with_infs = Z.columns[numerical_with_infs]
    for cat in numerical_with_infs:
        print(f"Filling infinite values for {cat} with maximum value")
        z = Z[cat]
        Z[cat] = z.replace(np.inf,
                           z[z < np.inf].max())
    return Z

def train_test_split(task='forecasting', ef_abs_diff=None, val_mode='train', train_mode='train', nan_threshold=0.6, fold_idx=None, target='Q', stratify=True, exclude_categorical=False, exclude_prediction_attrs=False, c=True, **kwargs):
    """
    c: if False, exclude correlation columns
    """
    data = analysis.error_analysis_dataset(partition='test', task=task, **kwargs)
    data = prune_by_nan_ptg(data, nan_threshold=nan_threshold)
    print("Data shape:", data.shape)
    target_keys = ['Q', 'distance', 'alpha', 'coverage']
    possible_targets = [c for c in data.columns if any(c.startswith(k) for k in target_keys)]
    possible_targets_extended = possible_targets + ['area']
    assert target in possible_targets_extended, "Invalid target"
    training_cols = [c for c in data.columns if not c in possible_targets]
    if 'area' in target:
        training_cols = [c for c in training_cols if not 'area' in c]
    if exclude_prediction_attrs:
        print("Excluding prediction attributes")
        training_cols = [c for c in training_cols if not c in params.error_analysis_prediction_attrs]
    if not c:
        print("Excluding correlation columns")
        corr_types = ['spearman', 'pearson']
        training_cols = [col for col in training_cols if not any(corr in col for corr in corr_types)]
    print("Training columns:", training_cols)

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    def add_data(mode, partition):
        if mode is not None:
            d = analysis.error_analysis_dataset(partition=partition, task=task, **kwargs)
            d = prune_by_nan_ptg(d)
            if mode == 'train':
                X_train.append(d[training_cols])
                y_train.append(d[target])
            elif mode == 'val':
                X_val.append(d[training_cols])
                y_val.append(d[target])
            elif mode == 'test':
                X_test.append(d[training_cols])
                y_test.append(d[target])
        return
    add_data(val_mode, 'val')
    add_data(train_mode, 'train')

    if fold_idx is not None:
        if stratify:
            if target == 'Q':
                bins = np.linspace(0, 1, 6)
                target_binned = pd.cut(data[target], bins, labels=False)
                split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(data, target_binned.values)
            else:
                target_binned = pd.qcut(data[target], 5, labels=False)
                split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(data, target_binned.values)
        else:
            split = KFold(n_splits=5, shuffle=True, random_state=42).split(data)
        for i, (train_idx, test_idx) in enumerate(split):
            if i == fold_idx:
                X_train.append(data.iloc[train_idx][training_cols])
                y_train.append(data.iloc[train_idx][target])
                X_test.append(data.iloc[test_idx][training_cols])
                y_test.append(data.iloc[test_idx][target])
                break
    else:
        X_test.append(data[training_cols])
        y_test.append(data[target])

    X_train = pd.concat(X_train, axis=0)
    y_train = pd.concat(y_train, axis=0)
    if not X_val:
        X_val = None
        y_val = None
    else:
        X_val = pd.concat(X_val, axis=0)
        y_val = pd.concat(y_val, axis=0)
    X_test = pd.concat(X_test, axis=0)
    y_test = pd.concat(y_test, axis=0)

    # Verify categorical dtypes are still present after concat
    def fix_dtype(X):
        datatypes = data.dtypes[training_cols]
        wrong_dtype = datatypes[datatypes != X.dtypes]
        if wrong_dtype.size > 0:
            for feature, dtype in wrong_dtype.items():
                X[feature] = X[feature].astype(dtype)
        return X
    X_train = fix_dtype(X_train)
    if X_val is not None:
        X_val = fix_dtype(X_val)
    X_test = fix_dtype(X_test)

    if exclude_categorical:
        numerical = X_train.dtypes != 'category'
        X_train = X_train.loc[:, numerical]
        if X_val is not None:
            X_val = X_val.loc[:, numerical]
        X_test = X_test.loc[:, numerical]


    X_train = fill_inf(X_train)
    if X_val is not None:
        X_val = fill_inf(X_val)
    X_test = fill_inf(X_test)

    if task == 'imputation' and ef_abs_diff is not None:
        assert ef_abs_diff in ['add', 'replace'], "Invalid ef_abs_diff"
        cols_diff = [c for c in X_train.columns if c.startswith("encoder_future_diff_")]
        cols_diff_abs = [c.replace("encoder_future_diff_", "encoder_future_diff_abs_") for c in cols_diff]
        print(f"Adding absolute difference of encoder_future_diff features: {len(cols_diff)}")
        X_train[cols_diff_abs] = X_train[cols_diff].abs()
        if X_val is not None:
            X_val[cols_diff_abs] = X_val[cols_diff].abs()
        X_test[cols_diff_abs] = X_test[cols_diff].abs()
        if ef_abs_diff == 'replace':
            print("Deleting original encoder_future_diff features")
            X_train = X_train.drop(columns=cols_diff)
            if X_val is not None:
                X_val = X_val.drop(columns=cols_diff)
            X_test = X_test.drop(columns=cols_diff)

    return X_train, X_val, X_test, y_train, y_val, y_test

def mRMR(X_train, y_train, threshold=0.05, return_score=False):
    if threshold == 0:
        return X_train.columns.tolist()
    # feature mutual information
    n_neighbors = 3
    X = X_train.copy()
    y = y_train.copy()
    feature = y_train.name
    categorical = X.columns[X.dtypes == 'category']
    for cat in categorical:
        if cat == 'season':
            X[cat] = X[cat].map({'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3})
        else:
            X[cat] = X[cat].cat.codes
    numerical_with_nans = (X.isna().any()) & (X.dtypes != 'category')
    numerical_with_nans = X.columns[numerical_with_nans]
    for cat in numerical_with_nans:
        print("Filling missing values for", cat)
        X[cat] = X[cat].fillna(X[cat].mean()) # fill with mean

    X = X.astype(np.float32)
    y = y.values.astype(np.float32)

    mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=n_neighbors, random_state=0)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    mi.name = feature

    # feature redundancy. MI between one feature and the rest
    redundancy = []
    for col in tqdm(X.columns):
        y = X[col].values.astype(np.float32)
        X_ = X.drop(columns=col)
        r = mutual_info_regression(X_, y, discrete_features=False, n_neighbors=n_neighbors, random_state=0)
        r = pd.Series(r, index=X_.columns)
        r.name = col
        redundancy.append(r)

    redundancy = pd.concat(redundancy, axis=1).T # rows = feature, columns = redundancy with other features
    redundancy.index.name = 'feature'
    redundancy.columns.name = 'redundancy'
    order = redundancy.index
    redundancy = redundancy.loc[order, order]

    def mrmr_step(mrmr, redundancy):
        """
        computes the mRMR score
        """
        mrmr_score = mi.drop(mrmr) - redundancy[mrmr].mean(axis=1)
        mrmr_score = mrmr_score.sort_values(ascending=False)
        return mrmr_score
    mrmr = [mi.idxmax()] # maximum relevance minimum redundancy
    for _ in range(mi.size - 1):
        mrmr_score = mrmr_step(mrmr, redundancy)
        mrmr.append(mrmr_score.idxmax())

    features_ordered = mi[mrmr]
    features_ordered /= mi.max()
    if return_score:
        return features_ordered
    else:
        cut = np.where(features_ordered < threshold)[0][0]
        selected = features_ordered.iloc[:cut].index.tolist()
        return selected

@savedata
def mrmr_selection(target='Q', exclude_prediction_attrs=False, c=True, task='forecasting', eval_mode=None, **kwargs):
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(target=target, c=c, exclude_prediction_attrs=exclude_prediction_attrs, task=task, eval_mode=eval_mode, **kwargs)
    features_mrmr = mRMR(X_train, y_train, return_score=True)
    return features_mrmr

def train_and_eval(X_train, y_train, X_test=None, y_test=None, X_val=None, y_val=None, xgb_only=False, xgb_params={}, save_key='', num_boost_round=None):
    assert X_test is not None or X_val is not None, "At least one of X_test or X_val must be provided"
    numeric = X_train.dtypes != 'category'
    def rmse(y, y_pred):
        return np.sqrt(np.mean((y - y_pred)**2))

    # xgboost
    D_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    objective = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    if xgb_params:
        xgb_params = {**xgb_params, **objective}
    else:
        default_xgb_params = {'tree_method': 'hist', 'eta': 0.1, 'max_depth': 20,
                              'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
                              'colsample_bylevel': 0.8, 'colsample_bynode': 0.8}
        xgb_params = {**default_xgb_params, **objective}
    if X_val is not None:
        if num_boost_round is None:
            num_boost_round = 2000
        D_val = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        best_model_path = f'best_xgb_model_{save_key}.json'
        clf_xgb = xgb.train(xgb_params, D_train, num_boost_round=num_boost_round, evals=[(D_val, 'val')], early_stopping_rounds=100, # let early stopping decide
                            callbacks=[ModelCheckpoint(best_model_path)])
        # load best model
        clf_xgb = xgb.Booster()
        clf_xgb.load_model(best_model_path)
        os.remove(best_model_path)
    else:
        if num_boost_round is None:
            num_boost_round = 500
        clf_xgb = xgb.train(xgb_params, D_train, num_boost_round=num_boost_round)
    if X_test is not None:
        D_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        y_pred = clf_xgb.predict(D_test)
        rmse_xgb = rmse(y_test.values, y_pred)
        rmse_baseline = rmse(y_test.values, y_train.mean())
    else:
        rmse_xgb = None
        rmse_baseline = None

    if not xgb_only:
        # DecisionTree
        if not numeric.all():
            Z_train = X_train.loc[:, numeric]
            Z_test = X_test.loc[:, numeric]
        else:
            Z_train = X_train
            Z_test = X_test
        def fill_nans(Z):
            numerical_with_nans = (Z.isna().any()) & (Z.dtypes != 'category')
            numerical_with_nans = Z.columns[numerical_with_nans]
            for cat in numerical_with_nans:
                print("Filling missing values for", cat)
                Z[cat] = Z[cat].fillna(Z[cat].mean()) # fill with mean
            return Z
        Z_train = fill_nans(Z_train)
        Z_test = fill_nans(Z_test)

        clf = DecisionTreeRegressor()
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)
        rmse_tree = rmse(y_test.values, y_pred)

        # linear regression
        # exclude categorical columns
        Z_train_sm = sm.add_constant(Z_train)
        model = sm.OLS(y_train, Z_train_sm).fit()
        Z_test_sm = sm.add_constant(Z_test, has_constant='add')
        y_pred = model.predict(Z_test_sm)
        rmse_lr = rmse(y_test.values, y_pred)

        models = dict(xgb=clf_xgb, tree=clf, lr=model)
        out = pd.Series(dict(xgb=rmse_xgb, tree=rmse_tree, lr=rmse_lr, baseline=rmse_baseline))
    else:
        out = pd.Series(dict(xgb=rmse_xgb, baseline=rmse_baseline))
        models = clf_xgb
    if X_val is not None:
        rmse_val = rmse(y_val, clf_xgb.predict(D_val))
    else:
        rmse_val = None
    return out, models, rmse_val

@savedata('all-timeout')
def xgb_optimal_hp(task='forecasting', c=True, target='Q', n_trials=1000, timeout=3600*20.0, eval_mode=None, exclude_prediction_attrs=False, **kwargs):
    specs = dict(target=target, c=c, task=task, eval_mode=eval_mode, exclude_prediction_attrs=exclude_prediction_attrs, **kwargs)
    X_train, X_val, _, y_train, y_val, _2 = train_test_split(val_mode='val', train_mode='train', fold_idx=None, exclude_categorical=False, params_idx='best', **specs)
    # features_mrmr = mRMR(X_train, y_train, return_score=True)
    features_mrmr = mrmr_selection(**specs)
    save_key = dict_to_id(specs)

    if torch.cuda.is_available():
        print("Using GPU for HP tuning")
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
    else:
        gpu_params = {}

    def objective(trial):
        X_cp_train = X_train.copy()
        X_cp_val = X_val.copy()
        # Define the hyperparameter search space
        mrmr_threshold = trial.suggest_uniform('mrmr_threshold', 0, 0.4)
        cut = np.where(features_mrmr < mrmr_threshold)[0]
        if cut.size > 0:
            cut = cut[0]
            selected = features_mrmr.iloc[:cut].index.tolist()
            X_cp_train = X_cp_train[selected]
            X_cp_val = X_cp_val[selected]

        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 2, 50),  # Broadened range
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.5),  # More room for exploration
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 20.0),  # Increased upper range
            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),  # Allow smaller subsample ratios
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),  # Lower boundary for more sparse trees
            'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.3, 1.0),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.3, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-4, 100.0),  # Increased regularization range
            'alpha': trial.suggest_loguniform('alpha', 1e-4, 100.0),  # Increased regularization range
            'gamma': trial.suggest_loguniform('gamma', 1e-4, 10.0),  # Controls tree pruning (new)
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])  # Growth policy for trees (optional)
        }

        # xgb_params = {
        #     'objective': 'reg:squarederror',
        #     'eval_metric': 'rmse',
        #     'max_depth': trial.suggest_int('max_depth', 3, 30),
        #     'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        #     'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10.0),
        #     'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        #     'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        #     'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1.0),
        #     'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.5, 1.0),
        #     'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        #     'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0)
        # }
        xgb_params = {**xgb_params, **gpu_params}

        # Train the model
        *_ , rmse_val = train_and_eval(X_cp_train, y_train, X_val=X_cp_val, y_val=y_val, xgb_only=True, xgb_params=xgb_params, save_key=save_key)

        return rmse_val

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study, features_mrmr

def feature_dim_reduction(target='distance', task='forecasting', c=True, exclude_prediction_attrs=True, eval_mode=None, ef_abs_diff=None):
    specs = locals()
    specs.update(params.TFT_specs[task])
    specs_dataset = specs.copy()
    specs_dataset['c'] = True

    X_train, *_ = train_test_split(val_mode='val', train_mode='train', fold_idx=None,  exclude_categorical=False, params_idx='best', **specs_dataset)
    num_original = X_train.shape[1]

    study, features_mrmr = xgb_optimal_hp(**specs)
    mrmr_threshold = study.best_params.pop('mrmr_threshold')
    cut = np.where(features_mrmr < mrmr_threshold)[0]
    if cut.size > 0:
        cut = cut[0]
        selected = features_mrmr.iloc[:cut].index.tolist()
        X_train = X_train[selected]
        num_pruned = X_train.shape[1]
    else:
        num_pruned = num_original

    return dict(original=num_original, pruned=num_pruned)

def train_optimal_hp(task='forecasting', target='Q', eval_mode=None, save_key=None, ef_abs_diff=None, **kwargs):
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(val_mode='val', train_mode='train', fold_idx=None, target=target, exclude_categorical=False, params_idx='best', eval_mode=eval_mode, task=task, ef_abs_diff=ef_abs_diff, **kwargs)
    if ef_abs_diff is not None:
        ef_kwargs = dict(ef_abs_diff=ef_abs_diff)
    else:
        ef_kwargs = {}
    study, features_mrmr = xgb_optimal_hp(task=task, target=target, eval_mode=eval_mode, **ef_kwargs, **kwargs)
    params = study.best_params
    mrmr_threshold = params.pop('mrmr_threshold')
    cut = np.where(features_mrmr < mrmr_threshold)[0]
    if cut.size > 0:
        cut = cut[0]
        selected = features_mrmr.iloc[:cut].index.tolist()
        X_train = X_train[selected]
        X_val = X_val[selected]
        X_test = X_test[selected]

    if save_key is None:
        save_key = dict_to_id(target=target, eval_mode=eval_mode, ef_abs_diff=ef_abs_diff, **kwargs)

    out, clf, rmse_val = train_and_eval(X_train, y_train,
                                        X_test=X_test, y_test=y_test,
                                        X_val=X_val, y_val=y_val, xgb_only=True, xgb_params=params, save_key=save_key,
                                        num_boost_round=int(1e6)) # let early stopping decide
    rmse_reduction = out.xgb/out.baseline - 1
    print(f"RMSE reduction: {rmse_reduction:.3%}")
    return out, clf, X_test, y_test

@savedata
def get_rmse_val_reduction(all_features=True, **specs):
    print("Computing RMSE reduction in validation set")
    _, clf = xgb_results(all_features=all_features, **specs)
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(val_mode='val', train_mode='train', fold_idx=None, exclude_categorical=False, params_idx='best', **specs)
    if not all_features:
        study, features_mrmr = xgb_optimal_hp(**specs)
        params = study.best_params
        mrmr_threshold = params.pop('mrmr_threshold')
        cut = np.where(features_mrmr < mrmr_threshold)[0]
        if cut.size > 0:
            cut = cut[0]
            selected = features_mrmr.iloc[:cut].index.tolist()
            X_train = X_train[selected]
            X_val = X_val[selected]
            X_test = X_test[selected]

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y - y_pred)**2))
    rmse_baseline_val = rmse(y_val.values, y_train.mean())
    D_val = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    rmse_val = rmse(y_val.values, clf.predict(D_val))
    rmse_val_reduction = rmse_val/rmse_baseline_val - 1
    return rmse_val_reduction

@savedata
def xgb_results(target='Q', c=True, task='forecasting', all_features=False, exclude_prediction_attrs=False, eval_mode=None, **kwargs):
    specs = dict(target=target, task=task, exclude_prediction_attrs=exclude_prediction_attrs, c=c, eval_mode=eval_mode, **kwargs)
    save_key = dict_to_id(all_features=all_features, **specs)
    if len(save_key) > 100:
        save_key = hash_string(save_key)
        print(f"Hashing save key. New length: {len(save_key)}")

    if all_features:
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(val_mode='val', train_mode='train', fold_idx=None, exclude_categorical=False, params_idx='best', **specs)
        out, clf, *_ = train_and_eval(X_train, y_train,
                                      X_test=X_test, y_test=y_test,
                                      X_val=X_val, y_val=y_val, save_key=save_key,
                                      num_boost_round=int(1e6)) # let early stopping decide
        # keep only xgb model
        clf = clf['xgb']
    else:
        out, clf, *_ = train_optimal_hp(save_key=save_key, **specs)
    out.name = target
    return out, clf

def xgb_summary(task='forecasting', targets=['Q', 'distance'], all_features=False, exclude_prediction_attrs=False, skip_computation=False, add_val_rmse_reduction=True, **kwargs):
    # previous default targets=['Q', 'distance', 'area', 'Q_alpha']
    specs = dict(task=task, all_features=all_features, exclude_prediction_attrs=exclude_prediction_attrs, **kwargs, skip_computation=skip_computation)
    specs_rmse_red = specs.copy()
    del specs_rmse_red['skip_computation']
    results = []
    val_rmse_reduction = []
    for target in targets:
        result = xgb_results(target=target, **specs)
        if not isinstance(result, SavedataSkippedComputation):
            results.append(result[0])
            if add_val_rmse_reduction:
                val_rmse_reduction.append(get_rmse_val_reduction(target=target, **specs_rmse_red))
    if results:
        results = pd.concat(results, axis=1).T
    else:
        results = pd.DataFrame()
    if add_val_rmse_reduction:
        val_rmse_reduction = pd.Series(val_rmse_reduction, index=results.index, name='val_rmse_reduction')
    return results, val_rmse_reduction

def xgb_summary_across_params(targets=['Q', 'distance', 'area', 'Q_alpha'], task='forecasting', skip_computation=None, add_val_rmse_reduction=True):
    """
    Returns the error reduction in the test set (xgb compared to baseline), for each target across different settings.
    """
    if skip_computation is None:
        if 'area' in targets or 'Q_alpha' in targets:
            skip_computation = True
            warnings.warn('Skipping computation')
        else:
            skip_computation = False
    kwargs = params.TFT_specs[task].copy()
    kwargs['task'] = task
    kwargs['targets'] = targets
    kwargs['skip_computation'] = skip_computation
    specs = {}
    result = []
    for c in [True, False]:
        specs['c'] = c
        for all_features in [True, False]:
            specs['all_features'] = all_features
            for exclude_prediction_attrs in [True, False]:
                specs['exclude_prediction_attrs'] = exclude_prediction_attrs
                if task == 'imputation':
                    ef_abs_diff_values = [None, 'add', 'replace']
                else:
                    ef_abs_diff_values = [None]
                for ef_abs_diff in ef_abs_diff_values:
                    if ef_abs_diff is not None and task == 'imputation':
                        specs['ef_abs_diff'] = ef_abs_diff
                    elif 'eb_abs_diff' in specs:
                        del specs['ef_abs_diff']
                    out, val_rmse_reduction = xgb_summary(**kwargs, **specs, add_val_rmse_reduction=add_val_rmse_reduction)
                    if not out.empty:
                        error_reduction = out['xgb']/out['baseline'] - 1
                        error_reduction = error_reduction.to_frame('error_reduction')
                        error_reduction.index.name = 'target'
                        if add_val_rmse_reduction:
                            error_reduction = pd.concat([error_reduction, val_rmse_reduction], axis=1)
                        error_reduction = error_reduction.reset_index()
                        error_reduction = error_reduction.assign(**specs)
                        result.append(error_reduction)
    result = pd.concat(result, axis=0, ignore_index=True)
    if add_val_rmse_reduction:
        result = result.sort_values('val_rmse_reduction')
    else:
        result = result.sort_values('error_reduction')
    result = result.reset_index(drop=True)
    return result

def get_shap_values(use_tqdm=True, task='forecasting', c=True, target='Q', all_features=False, return_y=False, eval_mode=None, exclude_prediction_attrs=False, **kwargs):
    args = locals()
    args.update(kwargs)
    del args['kwargs']
    del args['use_tqdm']

    specs = dict(target=target, task=task, eval_mode=eval_mode, exclude_prediction_attrs=exclude_prediction_attrs, c=c, **kwargs)
    if 'overwrite' in specs:
        del specs['overwrite']
    _, clf = xgb_results(all_features=all_features, **specs)

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(val_mode='val', train_mode='train', fold_idx=None,  exclude_categorical=False, params_idx='best', **specs)
    if not all_features:
        X_test = X_test[clf.feature_names]
    print(f"Number of features: {X_test.shape[1]}")

    if torch.cuda.is_available():
        clf.set_param({'device': 'cuda'})

    @savedata('all-return_y')
    def _get_shap_values(**kwargs):
        explainer = shap.TreeExplainer(clf)
        if use_tqdm:
            shap_interactions = []
            for i in tqdm(range(X_test.shape[0])):
                D_test = xgb.DMatrix(X_test.iloc[[i]], label=y_test.iloc[[i]], enable_categorical=True)
                shap_interactions.append(explainer.shap_interaction_values(D_test))
            shap_interactions = np.array(shap_interactions)
        else:
            D_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            shap_interactions = explainer.shap_interaction_values(D_test)
        return shap_interactions

    shap_interactions = _get_shap_values(**args)
    shap_interactions = shap_interactions.squeeze()
    shap_values = shap_interactions.sum(axis=1)
    if return_y:
        return shap_values, shap_interactions, X_test, y_test
    else:
        return shap_values, shap_interactions, X_test

def process_feature(F, task, plotter='matplotlib', suffix=True):
    f = deepcopy(F)
    corr_pattern = r"([^_]+)_([\w\s\(\)]+)-([\w\s\(\)]+)"
    match = re.match(corr_pattern, f)
    if match: # correlation
        corr_method, f1, f2 = match.groups()
        f1 = process_feature(f1, task, plotter=plotter)
        f2 = process_feature(f2, task, plotter=plotter)
        corr_str = corr_method[0].upper() # P for Pearson, S for Spearman
        if plotter == 'plotly':
            return f"ρ<sub>{corr_str}</sub>({f1}, {f2})"
        elif plotter == 'matplotlib':
            math_space = lambda s: s.replace(" ", r"\ ")
            return r'$\rho_{\mathrm{' + math_space(corr_str) + r'}}(\mathrm{' + math_space(f1) + r'}, \mathrm{' + math_space(f2) + '})$'
        else:
            raise ValueError(f"Invalid plotter: {plotter}")
    else:
        if f.startswith("encoder_future_diff_abs_"):
            f = f.replace("encoder_future_diff_abs_", "")
            f = process_feature(f, task, plotter=plotter, suffix=False)
            if plotter == 'matplotlib':
                f = r'$|\Delta_{\mathrm{p}\to\mathrm{f}}$ ' + f + r'$|$'
            elif plotter == 'plotly':
                f = f"|Δ<sub>p→f</sub> {f}|"
            else:
                raise ValueError(f"Invalid plotter: {plotter}")
        elif f.startswith('encoder_future_diff_'):
            f = f.replace("encoder_future_diff_", "")
            f = process_feature(f, task, plotter=plotter, suffix=False)
            if plotter == 'matplotlib':
                f = r'$\Delta_{\mathrm{p}\to\mathrm{f}}$ ' + f
            elif plotter == 'plotly':
                f = f"Δ<sub>p→f</sub> {f}"
            else:
                raise ValueError(f"Invalid plotter: {plotter}")
        else:
            has_future = f.startswith('future_')
            if has_future:
                f = f.replace('future_', '')
            if f.startswith('net_movement_'):
                f = f.replace('net_movement_', '')
                f = f.upper()
                if plotter == 'matplotlib':
                    f = r'$\Delta$' + f + r'$_{t_{1}\to t_{n}}$'
                elif plotter == 'plotly':
                    f = f"Δ{f}<sub>t₁→tₙ</sub>"
                else:
                    raise ValueError(f"Invalid plotter: {plotter}")
            elif f.endswith("_loc"):
                f = f.replace("_loc", "")
                f = params.error_analysis_feature_map.get(f, f) + " (mean)"
            elif f.endswith("_scale"):
                f = f.replace("_scale", "")
                f = params.error_analysis_feature_map.get(f, f) + " (std)"
            else:
                f = params.error_analysis_feature_map.get(f, f)
            if suffix:
                if has_future:
                    f += " (future)"
                elif task == 'imputation' and f not in params.error_analysis_prediction_attrs_mapped + params.error_analysis_main_targets:
                    f += " (past)"
            if f.startswith('Q_'):
                subindex = f.split('_')[1]
                smap = {'area': 'A'}
                if plotter == 'matplotlib':
                    smap['alpha'] = r'\alpha'
                    subindex = smap[subindex]
                    f = r'$Q_{' + subindex + r'}$'
                elif plotter == 'plotly':
                    smap['alpha'] = 'α'
                    subindex = smap[subindex]
                    f = f"Q<sub>{subindex}</sub>"
        return f

@savefig('all-xlims-mod_xlabel-max_display')
def shap_plot(task='forecasting', target='Q', max_display=10, eval_mode=None, mod_xlabel=True, add_int=True, fig_kwargs={}, xlims=None, **kwargs):
    specs = params.TFT_specs[task].copy()
    specs.update(kwargs)
    shap_values, shap_interactions, X_test = get_shap_values(task=task, target=target, eval_mode=eval_mode, **specs)
    if not add_int:
        # diagonal term in shap_interaction_values is the main effect
        shap_values = np.diagonal(shap_interactions, axis1=1, axis2=2)

    features = [process_feature(f, task) for f in X_test.columns]
    X_test.columns = features

    shap_summary(shap_values, X_test, max_display=max_display, show=False, plot_size=(8, 5), xlims=xlims, **fig_kwargs)
    fig = plt.gcf()
    # change all fonts to sans-serif
    plt.yticks(fontname='sans-serif', fontsize=16)
    plt.xticks(fontname='sans-serif', fontsize=14)
    # colorbar font
    cbar = fig.get_axes()[-1]
    cbar.tick_params(labelsize=14)
    cbar.set_ylabel('Feature value', fontsize=16, fontname='sans-serif')
    if mod_xlabel:
        target_map = {'distance': 'distance error'}
        target = target_map.get(target, target)
        # plt.xlabel(f'Change in predicted {target}', fontsize=22, fontname='sans-serif', labelpad=10)
        plt.title(f'Change in predicted {target}', fontsize=22, fontname='sans-serif', pad=20)
        plt.xlabel("")
    else:
        # change x-axis font
        xtext = fig.get_axes()[0].get_xlabel()
        plt.xlabel(xtext, fontsize=20, fontname='sans-serif', labelpad=10)
    # use only 5 ticks in x-axis
    fig.axes[0].xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.close()
    return fig

def preprocess_arules(num_bins=5):
    data = analysis.error_analysis_dataset(partition='test', params_idx='best')
    target = 'Q'
    target_keys = ['Q', 'alpha', 'coverage']
    possible_targets = [c for c in data.columns if any(c.startswith(k) for k in target_keys)]
    assert target in possible_targets, "Invalid target"
    training_cols = [c for c in data.columns if not c in possible_targets]

    def bin_numerical_variable(series):
        qs = np.linspace(0, 1, num_bins+1)
        bins = [series.quantile(q) for q in qs]
        bins_pruned = [bins[0]]
        p_lows = ['0']
        p_highs = []
        for i, b in enumerate(bins[1:], start=1):
            if b not in bins_pruned:
                q = int(qs[i] * 100)
                bins_pruned.append(b)
                p_lows.append(q)
                p_highs.append(q)
            elif i == len(bins) - 1:
                p_highs.append(100)
        labels = [f'p{low} - p{high}' for low, high in zip(p_lows, p_highs)]
        if len(bins_pruned) == len(labels):
            bins_pruned.append(bins_pruned[-1] + 1)

        return pd.cut(series, bins=bins_pruned, labels=labels, include_lowest=True)

    df_binned = data.copy()
    for column in df_binned.columns:
        if data[column].dtype != 'category':  # Exclude the target variable
            df_binned[column] = bin_numerical_variable(df_binned[column])
    df_apriori = df_binned[training_cols + ['Q', 'distance']] # targets 'Q' and 'distance'
    return df_apriori

@savedata
def arules(target='Q', mode='highest', min_support=0.05, min_conf=0.7, maxlen=7, exclude_targets=True, num_bins=5):
    """
    To be run at nuredduna (has the necessary packages).
    """
    load_R_pkg('arules')
    df_apriori = preprocess_arules(num_bins=num_bins)
    if exclude_targets:
        targets = ['Q', 'distance']
        exclude = [t for t in targets if t != target][0]
        df_apriori = df_apriori.drop(columns=exclude)
    if mode == 'highest':
        if num_bins == 5:
            target_str = f'"{target}=p80 - p100"'
        elif num_bins == 2:
            target_str = f'"{target}=p50 - p100"'
    elif mode == 'lowest':
        if num_bins == 5:
            target_str = f'"{target}=p0 - p20"'
        elif num_bins == 2:
            target_str = f'"{target}=p0 - p50"'
    else:
        raise ValueError("Invalid mode")

    ro.globalenv['df'] = df_apriori
    r("transactions <- as(df, 'transactions')")
    r(f"rules <- apriori(data=transactions, parameter=list(supp={min_support}, conf={min_conf}, maxlen={maxlen}), appearance=list(default='lhs', rhs={target_str}))")
    if maxlen > 1:
        print("Removing redundant rules")
        r("subsetrules <- which(colSums(is.subset(rules, rules)) > 1)") # get subset rules in vector
        r("rules_pruned <- rules[-subsetrules]")
        original_length, new_length = r("length(rules)")[0], r("length(rules_pruned)")[0]
        print(f"Original rules: {original_length}, pruned rules: {new_length}")
        df = r("rules_df <- as(rules_pruned, 'data.frame')")
    else:
        df = r("rules_df <- as(rules, 'data.frame')")
    return df.reset_index(drop=True)

def arules_summary(**kwargs):
    dfs = []
    for target in ['Q', 'distance']:
        for mode in ['highest', 'lowest']:
            for exclude_targets in [True, False]:
                df = arules(target=target, mode=mode, exclude_targets=exclude_targets, **kwargs)
                dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = df.sort_values('count', ascending=False).drop_duplicates().reset_index(drop=True)
    return df

@savedata
def corr_with_target(task='forecasting', target='distance', method='spearman', **kwargs):
    ef_abs_diff = kwargs.pop('ef_abs_diff', None)
    data = analysis.error_analysis_dataset(partition='test', params_idx='best', task=task, **kwargs)
    data = prune_by_nan_ptg(data, nan_threshold=0.6)
    data = fill_inf(data)
    if ef_abs_diff is not None:
        assert ef_abs_diff in ['add', 'replace'], "Invalid ef_abs_diff"
        cols_diff = [c for c in data.columns if c.startswith("encoder_future_diff_")]
        cols_diff_abs = [c.replace("encoder_future_diff_", "encoder_future_diff_abs_") for c in cols_diff]
        print(f"Adding absolute difference of encoder_future_diff features: {len(cols_diff)}")
        data[cols_diff_abs] = data[cols_diff].abs()
        if ef_abs_diff == 'replace':
            print("Deleting original encoder_future_diff features")
            data = data.drop(columns=cols_diff)

    c, p = phdu_corr.corr_pruned(data, col=target, method=method, alpha=None, ns_to_nan=False)
    c = c[target].dropna()
    p = p[target].dropna()

    target_keys = ['alpha', 'coverage'] # Q
    possible_targets = [c for c in data.columns if any(c.startswith(k) for k in target_keys)]
    training_cols = [c for c in data.columns if not c in possible_targets]
    common = set(training_cols).intersection(set(c.index)) - {target}
    c = c.loc[common]
    p = p.loc[common]
    order = c.abs().sort_values(ascending=False).index
    c = c[order]
    p = p[order]

    p_adj = ss.false_discovery_control(p, method='bh') # benjamini-hochberg. Correlations are transitive => the p-values are positively dependent
    df = pd.concat([c.to_frame('corr'),
                    pd.DataFrame(dict(p=p.values, p_adj=p_adj), index=p.index)], axis=1)
    alpha = 0.05
    df = df.assign(significant="no")
    df.loc[df.p < alpha, 'significant'] = 'yes - not corrected'
    df.loc[df.p_adj < alpha, 'significant'] = 'yes - corrected'
    return df

@savefig('all-height-min_corr-sort-fs-width')
def corr_plot(task='forecasting', target='distance', method='spearman', sort=True, height=900, width=900, min_corr=0.2, mc_corrected=False, fs=None, **kwargs):
    """
    mc_corrected: correct p-values for multiple comparisons
    """
    specs = params.TFT_specs[task].copy()
    df = corr_with_target(task=task, target=target, method=method, **kwargs, **specs)
    if min_corr:
        df = df[df['corr'].abs() >= min_corr]
    if sort:
        df = df.sort_values('corr')
    df.index = [process_feature(f, task, plotter='plotly') for f in df.index]
    fig = get_figure(xaxis_title=f'Correlation with {target}', yaxis_title='Feature', xaxis_range=[-1, 1],
                     margin=dict(t=0,b=0,l=0,r=0),  height=height, width=width)
    base_colors = plotly_default_colors(3)
    if task == 'forecasting':
        if mc_corrected:
            df = df[df['significant'] == 'yes - corrected']
            fig.add_bar(x=df['corr'].values, y=df.index.values, orientation='h')
        else:
            # cmap = {'no': 'gray',
            #         'yes - not corrected': base_color,
            #         'yes - corrected': 'gold'}
            # exclude non-significant
            df = df[df['significant'] != 'no']
            df['color'] = 'gray' # not significant under correction
            df = df.reset_index()
            df = df.rename(columns={'index': 'feature'})
            def get_var_type(x):
                if x in params.error_analysis_main_targets or x.startswith('Q') or 'area' in x:
                    return 'Score'
                else:
                    return 'Past'
            df['var_type'] = df['feature'].apply(get_var_type).values
            df.loc[df['var_type'] == 'Past', 'color'] = base_colors[0]
            df.loc[df['var_type'] == 'Score', 'color'] = 'black'

            fig.add_bar(x=df['corr'].values, y=df.feature.values, orientation='h', marker_color=df['color'])
    else: # imputation
        assert not mc_corrected, "mc_corrected (mandatory corrected p values) not implemented for imputation"
        df = df[df['significant'] != 'no']
        df = df.reset_index()
        df = df.rename(columns={'index': 'feature'})
        def get_var_type(x):
            if x.startswith('|Δ<sub>p→f</sub>'):
                return '|Future - Past|'
            elif 'future' in x:
                return 'Future'
            elif 'past' in x:
                return 'Past'
            else:
                return 'Score'
        df['var_type'] = df['feature'].apply(get_var_type).values

        def del_str(x):
            for key in ['|Δ<sub>p→f</sub> ', '|', ' (future)', ' (past)']:
                x = x.replace(key, '')
            return x
        df['feature'] = df['feature'].apply(del_str)

        # add nans to force maintaining the order
        df_past = df.query('var_type == "Past" & significant == "yes - corrected"')
        df_future = df.query('var_type == "Future" & significant == "yes - corrected"')
        df_diff = df.query('var_type == "|Future - Past|" & significant == "yes - corrected"')
        df_score = df.query('var_type == "Score" & significant == "yes - corrected"')
        df_not_corrected = df.query('significant == "yes - not corrected"')
        past_idxs = df_past.index.values.copy()
        future_idxs = df_future.index.values.copy()
        diff_idxs = df_diff.index.values.copy()
        df_past = pd.concat([df_past,
                             df_not_corrected.assign(corr=np.NaN),
                             df_future.assign(corr=np.NaN),
                             df_diff.assign(corr=np.NaN),
                             df_score.assign(corr=np.NaN),
                             ], axis=0, ignore_index=False).sort_index()
        df_future = pd.concat([df_future,
                               df_past.loc[past_idxs].assign(corr=np.NaN),
                               df_not_corrected.assign(corr=np.NaN),
                               df_diff.assign(corr=np.NaN),
                               df_score.assign(corr=np.NaN),
                               ], axis=0, ignore_index=False).sort_index()
        df_diff = pd.concat([df_diff,
                             df_past.loc[past_idxs].assign(corr=np.NaN),
                             df_future.loc[future_idxs].assign(corr=np.NaN),
                             df_not_corrected.assign(corr=np.NaN),
                             df_score.assign(corr=np.NaN),
                             ], axis=0, ignore_index=False).sort_index()
        df_score = pd.concat([df_score,
                              df_past.loc[past_idxs].assign(corr=np.NaN),
                              df_future.loc[future_idxs].assign(corr=np.NaN),
                              df_diff.loc[diff_idxs].assign(corr=np.NaN),
                              df_not_corrected.assign(corr=np.NaN),
                              ], axis=0, ignore_index=False).sort_index()

        plot_specs = {'Score': (df_score, 'black'),
                      'Past': (df_past, base_colors[0]),
                      'Future': (df_future, base_colors[1]),
                      '|Future - Past|': (df_diff, base_colors[2]),
                      }

        if not df_not_corrected.empty:
            plot_specs['Not significant'] = (df_not_corrected, 'gray')
        for var_type, (df_var, color) in plot_specs.items():
            fig.add_bar(x=df_var['corr'].values, y=df_var.feature.values, orientation='h', marker_color=color, name=var_type)

    if fs is not None:
        fig.update_layout(yaxis_tickfont_size=fs)
    return fig

def preprocess_distance_analysis(task='forecasting', delete_missing=True, partition='test'):
    kwargs = params.TFT_specs[task].copy()
    params_idx, _ = analysis.quantile_best_model(task=task, **kwargs)
    if partition == 'train':
        results = analysis.quantile_results_train(params_idx=params_idx, task=task, **kwargs)
    else:
        results = analysis.quantile_results(params_idx=params_idx, task=task, **kwargs)
    error = analysis.point_prediction_errors(partition=partition, params_idx=params_idx, task=task, **kwargs)
    error_avg = error.mean(axis=1)
    error_avg.name = 'distance'

    if 'max_train_days' in kwargs:
        max_train_days = kwargs['max_train_days']
        del kwargs['max_train_days']
    else:
        hp = analysis.get_hp(task=task, **kwargs, params_idx=params_idx)
        if 'max_train_days' in hp:
            max_train_days = hp['max_train_days']
        else:
            raise ValueError('max_train_days not found')

    x = results[f'x_{partition}']
    cds_encoder = custom_metrics.stack_targets(x['encoder_target'])
    cds_decoder = custom_metrics.stack_targets(x['decoder_target'])
    if task == 'imputation':
        cds_future = custom_metrics.extract_future_cds(results, x=x)

    if delete_missing:
        def is_missing(var_type, arr):
            lengths = x[f'{var_type}_lengths']
            nan_mask = analysis.create_mask(lengths.max(), lengths)
            intermediate_missing = x[f'{var_type}_missing']
            missing = nan_mask | intermediate_missing
            return missing.unsqueeze(0).expand(arr.shape)
        cds_encoder[is_missing('encoder', cds_encoder)] = np.nan
        cds_decoder[is_missing('decoder', cds_decoder)] = np.nan
        if task == 'imputation':
            cds_future[is_missing('future', cds_future)] = np.nan

    numericals, categoricals = load.avg_features(task=task, max_train_days=max_train_days, partition=partition, **kwargs)
    speed = pd.Series(list(cds_encoder.transpose(1,0,2)), name='speed').apply(custom_metrics.average_speed)
    training, *_ = load.load_dataset(task=task, **kwargs, max_train_days=max_train_days)
    bathymetry_rescaled = training.scalers['Bathymetry'].inverse_transform(numericals['Bathymetry_loc'].values.reshape(-1, 1))
    if task == 'imputation':
        speed_future = pd.Series(list(cds_future.transpose(1,0,2)), name='future_speed').apply(custom_metrics.average_speed)
        bathymetry_rescaled_future = training.scalers['Bathymetry'].inverse_transform(numericals['future_Bathymetry_loc'].values.reshape(-1, 1))
        return error_avg, bathymetry_rescaled, speed, cds_encoder, cds_decoder, cds_future, speed_future, bathymetry_rescaled_future
    else:
        return error_avg, bathymetry_rescaled, speed, cds_encoder, cds_decoder

@savedata
def bathymetry_speed_data(partition='train', task='forecasting'):
    distance_error, bathymetry_rescaled, speed, *_ = preprocess_distance_analysis(task=task, partition=partition)
    bathymetry_rescaled = bathymetry_rescaled.squeeze()
    valid = speed.values < 10
    speed = speed.values[valid]
    bathymetry_rescaled = bathymetry_rescaled[valid]
    distance_error = distance_error.values[valid]
    return bathymetry_rescaled, speed, distance_error # (x, y, z)

@savedata
def bathymetry_speed_train_data(nbins=10):
    error_avg, bathymetry_rescaled, speed, *_ = preprocess_distance_analysis(task='forecasting', partition='train')
    X = speed.copy()
    X.name = 'average_speed'
    X = X.to_frame()
    X['Bathymetry_loc'] = bathymetry_rescaled/1000 # km
    valid = X['average_speed'] < 10
    X = X[valid]
    def bin_data(col):
        bins = np.linspace(X[col].min(), X[col].max(), nbins+1)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        X[col] = pd.cut(X[col], bins=bins, labels=bin_centers, include_lowest=True)
        return bins
    bins_bat = bin_data('Bathymetry_loc')
    bins_speed = bin_data('average_speed')
    return X, [bins_speed, bins_bat]

@savedata
def bathymetry_speed_test_error(nbins=10):
    _, (bins_speed, bins_bat) = bathymetry_speed_train_data(nbins=nbins)
    error_avg, bathymetry_rescaled, speed, *_ = preprocess_distance_analysis(task='forecasting', partition='test')
    X = speed.copy()
    X.name = 'average_speed'
    X = X.to_frame()
    X['Bathymetry_loc'] = bathymetry_rescaled/1000 # km
    valid = X['average_speed'] < 10
    X = X[valid]
    cols = X.columns.tolist()
    def bin_data(col, bins):
        bin_centers = (bins[1:] + bins[:-1]) / 2
        X[col] = pd.cut(X[col], bins=bins, labels=bin_centers, include_lowest=True)
        return bins
    bins_bat = bin_data('Bathymetry_loc', bins_bat)
    bins_speed = bin_data('average_speed', bins_speed)
    X = pd.concat([X, error_avg[valid]], axis=1)
    X = X.groupby(cols).mean().reset_index()
    return X

@savefig('all-cmax')
def distance_error_heatmap(task='forecasting', cmax=150, **kwargs):
    error_avg, bathymetry_rescaled, speed, *_ = preprocess_distance_analysis(task=task, **kwargs)
    X = speed.copy()
    X.name = 'average_speed'
    X = X.to_frame()
    X['Bathymetry_loc'] = bathymetry_rescaled/1000 # km
    valid = X['average_speed'] < 10
    X = X[valid]
    cols = X.columns.tolist()
    for c in cols:
        bins = np.linspace(X[c].min(), X[c].max(), 11)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        X[c] = pd.cut(X[c], bins=bins, labels=bin_centers, include_lowest=True)
    X = pd.concat([X, error_avg[valid]], axis=1)
    X = X.groupby(cols).mean().reset_index()

    tickvals = np.arange(0, cmax+50, 50)
    ticktext = [str(t) for t in tickvals]
    ticktext[-1] +=  '+'
    fig = get_figure(xaxis_title='Bathymetry [km]', yaxis_title='Speed [km/h]', width=1100)
    fig.add_trace(go.Heatmap(x=X['Bathymetry_loc'], y=X['average_speed'], z=X['distance'],
                             colorscale='RdBu_r', zmin=0, zmax=cmax,
                             colorbar=dict(title='Distance error [km]', tickvals=tickvals, ticktext=ticktext, nticks=5)
                             ))
    return fig

@savefig('all-cmax-bathymetry_isoline-red_font-offset')
def distance_error_contour(task='forecasting', speed_filter=3, bathymetry_isoline=-1000, red_font=True, cmax=100, offset=2):
    error_avg, _, speed, cds_encoder, cds_decoder = preprocess_distance_analysis(task=task)

    to_degrees = lambda y: np.stack(space.mercator_inv(*y), axis=0) * 180 / np.pi
    cds_encoder = to_degrees(cds_encoder)
    cds_decoder = to_degrees(cds_decoder)
    lat_min, lon_min = np.nanmin(cds_encoder, axis=1).min(axis=1)
    lat_max, lon_max = np.nanmax(cds_encoder, axis=1).max(axis=1)
    lat_lim = [lat_min, lat_max]
    lon_lim = [lon_min, lon_max]

    bathymetry_data = np.genfromtxt('utils/data/BathymetryData.dat',
                                    skip_header=0,
                                    skip_footer=0,
                                    names=None,
                                    delimiter=' ')

    ground = bathymetry_data > 0
    bathymetry_data[ground] = 0

    lon_edges = np.arange(-180, 180.25, 0.25)
    lon_centers = 0.5 * (lon_edges[1:] + lon_edges[:-1])
    lat_edges = np.arange(90, -90.25, -0.25)
    lat_centers = 0.5 * (lat_edges[1:] + lat_edges[:-1])
    df = pd.DataFrame(bathymetry_data, index=lat_centers, columns=lon_centers)

    valid_lon = (lon_centers >= lon_lim[0] - offset) & (lon_centers <= lon_lim[1] + offset)
    valid_lat = (lat_centers >= lat_lim[0] - offset) & (lat_centers <= lat_lim[1] + offset)
    df = df.loc[valid_lat, valid_lon]
    df.index.name = 'lat'
    df.columns.name = 'lon'
    x = df.columns.values
    y = df.index.values
    z = df.values

    if speed_filter is not None:
        if speed_filter < 1:
            valid = speed < speed.quantile(speed_filter)
        else:
            valid = speed < speed_filter
        cds_encoder_fig = cds_encoder[:, valid]
        error_avg_fig = error_avg[valid]
    else:
        cds_encoder_fig = cds_encoder
        error_avg_fig = error_avg

    tickvals = np.arange(z.max(), z.min(), -2000)[::-1].astype(int)
    ticktext = [str(t) for t in tickvals]
    tickvals -= 400

    fig = get_figure(xaxis_title='Longitude', yaxis_title='Latitude', height=800, width=1400)
    fig.add_trace(go.Contour(z=z, x=x, y=y, colorscale='Blues_r',
                             colorbar=dict(title='Bathymetry [m]', len=0.5, tickvals=tickvals, ticktext=ticktext, y=1, yanchor='top'),
                             ))

    if bathymetry_isoline is not None:
        levels, contour_lines = geometry.extract_contour_polygons(x=x, y=y, z=z)
        valid = levels == bathymetry_isoline
        levels = levels[valid]
        contour_lines = np.array(contour_lines, dtype=object)[valid].tolist()

        for i, line in enumerate(contour_lines):
            fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines', line=dict(color='red', width=4), showlegend=i==0, name=str(bathymetry_isoline)))
        if red_font:
            fig.update_layout(legend=dict(font=dict(color='red')))

    # join all trajectories in single array for plot
    if cmax is None:
        cmax = error_avg_fig.quantile(0.9)
    cds_fig = []
    color = []
    nan_separator = np.ones((2, 1)) * np.nan
    append_nan = lambda x: np.hstack((x, nan_separator))
    for cds, error in zip(cds_encoder_fig.transpose(1, 0, 2), error_avg_fig): # (n, 2 , t)
        not_nan = (~np.isnan(cds)).all(axis=0)
        cds_fig.append(append_nan(cds[:, not_nan]))
        c = np.ones(not_nan.sum()) * error
        c = np.hstack((c, np.nan))
        color.append(c)
    lat, lon = np.hstack(cds_fig)
    color = np.hstack(color)
    color[color > cmax] = cmax
    tickvals = np.linspace(0, cmax, 5, dtype=int)
    ticktext = [str(t) for t in tickvals]
    ticktext[-1] += '+'
    # set error_avg as color
    fig.add_trace(go.Scatter(x=lon, y=lat, mode='lines+markers',
                             marker=dict(color=color, size=8, opacity=0.8,
                                         cmin=0,
                                         cmax=cmax,
                                         colorscale='Inferno_r', #'Reds',
                                         colorbar=dict(title='Distance error [km]', len=0.5,
                                                       y=0.5, yanchor='top',
                                                       tickvals=tickvals, ticktext=ticktext,
                                                       ),

                                         ),
                             line_width=2,
                             ))
    fig.update_layout(xaxis_range=[x.min(), x.max()], yaxis_range=[y.min(), y.max()])
    return fig

@savefig
def past_future_orientation_imputation():
    specs = dict(target='area', task='imputation', eval_mode=None, exclude_prediction_attrs=True, c=True, **params.TFT_specs['imputation'])
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(val_mode='val', train_mode='train', fold_idx=None,  exclude_categorical=False, params_idx='best', **specs)

    df = X_test[['encoder_future_diff_net_movement_x', 'net_movement_x', 'future_net_movement_x']]
    df['encoder_future_diff_abs_net_movement_x'] = df['encoder_future_diff_net_movement_x'].abs()
    df['encoder_future_prod'] = df['net_movement_x'] * df['future_net_movement_x']
    df['sign'] = np.sign(df['encoder_future_prod'])
    x = df['sign']
    y = df['encoder_future_diff_abs_net_movement_x']
    x_bins = np.unique(x)
    dp = 0.05
    y_bins = y.quantile(np.arange(0, 1 + dp, dp)).values

    fig = get_figure(xaxis_title="Past-future orientation (X)",
                     yaxis_title=process_feature('encoder_future_diff_abs_net_movement_x', task='imputation', plotter='plotly'),
                     width=900)
    fig.add_vline(x=0, line=dict(color='black', width=2))
    fig.add_vline(x=1, line=dict(color='black', width=2))
    H = np.histogram2d(x, y, bins=[x_bins, y_bins])[0]
    H /= H.sum()
    H[H == 0] = np.nan
    fig.add_trace(go.Heatmap(z=H.T, x=x_bins, y=y_bins, colorscale='Blues', zmin=0, zmax=dp, colorbar=dict(title='Probability')))
    for y_q in y_bins:
        fig.add_hline(y=y_q, line=dict(color='black', width=1))
    for p in [0.25, 0.75]:
        fig.add_hline(y=y.quantile(p), line=dict(color='red', width=6, dash='dot'))

    fig.update_layout(yaxis_type='log')
    fig.update_layout(**mod_logaxes_expfmt(fig, axes=['y']))
    fig.update_layout(xaxis_tickvals=[-0.5, 0.5],
                      xaxis_ticktext=['Opposite', 'Same'])
    return fig

@savefig
def imputation_heatmap(target='distance'):
    data = analysis.error_analysis_dataset(task='imputation', **params.TFT_specs['imputation'], partition='test')
    cols_diff = [c for c in data.columns if c.startswith("encoder_future_diff_")]
    cols_diff_abs = [c.replace("encoder_future_diff_", "encoder_future_diff_abs_") for c in cols_diff]
    print(f"Adding absolute difference of encoder_future_diff features: {len(cols_diff)}")
    data[cols_diff_abs] = data[cols_diff].abs()

    error = data[target]
    cmax = 50
    tickvals = np.arange(0, cmax+25, 25, dtype=int)
    if target == 'distance':
        X = data[['encoder_future_diff_abs_X_loc', 'encoder_future_diff_abs_Y_loc']]
        title = 'Distance error [km]'
        height = 650
    elif target == 'area':
        X = data[['encoder_future_diff_abs_net_movement_x', 'encoder_future_diff_abs_X_loc']]
        cmax *= 1000
        tickvals *= 1000
        title = 'Area [km²]'
        height = 750

    X.columns = [process_feature(f, 'imputation', 'plotly') for f in X.columns]
    cols = X.columns.tolist()
    for c in cols:
        bins = np.linspace(X[c].min(), X[c].max(), 11)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        X[c] = pd.cut(X[c], bins=bins, labels=bin_centers, include_lowest=True)
    X = pd.concat([X, error], axis=1)
    X = X.groupby(cols).mean().reset_index()

    q90, q100 = error.quantile(0.9), error.max()
    print(f"90th percentile: {q90}, max: {q100}")
    ticktext = [str(t) for t in tickvals]
    ticktext[-1] +=  '+'
    fig = get_figure(xaxis_title=f'{cols[0]} [km]', yaxis_title=f'{cols[1]} [km]', width=1000, height=height)
    fig.add_trace(go.Heatmap(x=X.iloc[:,0], y=X.iloc[:,1], z=X[target],
                             colorscale='RdBu_r', zmin=0, zmax=cmax,
                             colorbar=dict(title=title, tickvals=tickvals, ticktext=ticktext, len=0.8),
                             ))
    return fig


def bathymetry_speed_pmf(task='forecasting', partition='train', bw_adjust=1, cut=0, pmf='joint'):
    x, y, _ = bathymetry_speed_data(task=task, partition=partition)
    estimator = KDE(bw_adjust=bw_adjust, cut=cut)
    if pmf == 'x':
        return estimator(x)
    elif pmf == 'y':
        return estimator(y)
    elif pmf == 'joint':
        return estimator(x, y)
    else:
        raise ValueError(f'pmf={pmf} not recognized')

@savefig('task')
def bathymetry_speed_against_distance(task='forecasting', bw_adjust=1, cut=0, cumulative=False, labelsize=24, ticksize=16, cbar_ticksize=12, cbar_titlesize=14, pmf_mpl=1000):
    # change plt.rcParams font family to sans seriff
    plt.rcParams['font.family'] = 'sans-serif'


    X, Y, _ = bathymetry_speed_data(task=task, partition='train') # (bathymetry, speed, distance error)
    x, y, z = bathymetry_speed_data(task=task, partition='test')
    X /= 1000
    x /= 1000
    cmax = 100
    if cmax is not None:
        z[z > cmax] = cmax
    pmf_xy, _ = bathymetry_speed_pmf(task=task, partition='train', bw_adjust=bw_adjust, cut=cut, pmf='joint')
    pmf_xy *= pmf_mpl

    g = sns.jointplot(x=X, y=Y, kind='kde', cmap='Blues', fill=True,
                      xlim=(X.min(), X.max()), ylim=(Y.min(), Y.max()),
                      joint_kws=dict(bw_adjust=bw_adjust, cut=cut),
                      marginal_kws=dict(bw_adjust=bw_adjust, cut=cut),
                      )
    g.ax_joint.set_xlabel('Bathymetry [km]', fontsize=labelsize)
    g.ax_joint.set_ylabel('Speed [m/s]', fontsize=labelsize)
    if cumulative:
        # add marginal cumulatives
        g.plot_marginals(sns.kdeplot, color='orange', fill=False, cumulative=True)
    # plot test distance error
    g.ax_joint.scatter(x, y, c=z, cmap='Reds', alpha=0.7, edgecolors='k', linewidths=1)

    # Colorbar for KDE
    cbar_ax = g.fig.add_axes([0.96, .59, .03, .25])  # x, y, width, height
    cbar = plt.colorbar(g.ax_joint.collections[0], cax=cbar_ax)
    max_pmf = pmf_xy.max()
    # change colorbar limits
    cbar.norm.vmin = 0
    cbar.norm.vmax = max_pmf
    cbar.cmap = plt.cm.Blues
    cbar.draw_all()
    # ticks
    ticks = [0, np.round(max_pmf/2, 2), np.round(max_pmf, 2)]
    ticklabels = [str(t) for t in ticks]
    ticklabels[-1] += " " * 2
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels, fontsize=cbar_ticksize)
    if pmf_mpl == 1:
        cbar_label = 'Density (train)'
    else:
        cbar_label = f'Density (train) x {pmf_mpl}'
    cbar.set_label(cbar_label, fontsize=cbar_titlesize)

    # Colorbar for distance error
    cbar_ax2 = g.fig.add_axes([0.96, .175, .03, .25])  # x, y, width, height
    cbar2 = plt.colorbar(g.ax_joint.collections[-1], cax=cbar_ax2)
    max_distance = z.max()
    # change colorbar limits
    cbar2.norm.vmin = 0
    cbar2.norm.vmax = max_distance
    cbar2.cmap = plt.cm.Reds
    cbar2.draw_all()
    # ticks
    ticks = [0, round(max_distance/2), round(max_distance)]
    ticklabels = [str(t) for t in ticks]
    ticklabels[-1] += '+'
    cbar2.set_ticks(ticks)
    cbar2.set_ticklabels(ticklabels, fontsize=cbar_ticksize)
    cbar2.set_label('Distance (test) [km]', fontsize=cbar_titlesize)

    # set limits
    g.ax_joint.set_ylim(0, y.max()*1.1)
    g.ax_joint.xaxis.set_major_locator(plt.MaxNLocator(int(X.max() - X.min()) + 1))
    g.ax_joint.yaxis.set_major_locator(plt.MaxNLocator(round(y.max()) + 1))
    # set tick sizes
    g.ax_joint.tick_params(axis='both', which='major', labelsize=ticksize)
    return g.fig

def bathymetry_speed_pmf_corr_with_distance(task='forecasting'):
    """
    Calculate the correlation between the probability mass function (PMF) of (bathymetry, speed) in the training set (the available training data),
    and the distance error of the test set.

    Parameters:
    task (str): The task type, default is 'forecasting'.

    Returns:
    pd.DataFrame: A DataFrame containing the correlation coefficients and p-values for
                  Spearman and Pearson correlation methods.

    The function performs the following steps:
    1. Computes the joint PMF of bathymetry speed for the training set.
    2. Retrieves the (bathymetry, speed, distance error) from the test set.
    3. For each test data point, finds the closest corresponding PMF value from the training set.
    4. Calculates the correlation between the PMF values and the test set distance errors using
       Spearman and Pearson methods.
    5. Returns the correlation coefficients and p-values as a DataFrame.
    """
    pmf_xy, (supp_x, supp_y) = bathymetry_speed_pmf(task=task, partition='train', pmf='joint')
    x, y, z = bathymetry_speed_data(task=task, partition='test')

    pmf_z = [] # train pmf associated with the distance error of the test set
    for xi, yi in zip(x, y):
        closest_x = np.abs(supp_x - xi).argmin()
        closest_y = np.abs(supp_y - yi).argmin()
        pmf_closest = pmf_xy.T[closest_x, closest_y]
        pmf_z.append(pmf_closest)
    pmf_z = np.array(pmf_z)

    correlation = {}
    for method in ['spearman', 'pearson']:
        c = getattr(ss, f'{method}r')(pmf_z, z)
        correlation[(method, 'corr')] = c.statistic
        correlation[(method, 'p-value')] = c.pvalue
    correlation = pd.Series(correlation).unstack()
    return correlation
