import numpy as np
import pandas as pd
from pathlib import Path
import os
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

RootDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(RootDir)

from forecasting import analysis, params
import getopt
from tidypath import fmt
import phdu

import matplotlib
matplotlib.use('Agg')

# Default params
weather = 'all'
params_idx = 0
epochs = 200
cds = 'mercator'
cumulative = False
overwrite = False
joint_prediction = False
ID = None
pretrained = False
patience = 40
store_missing_idxs = False
max_train_days = 28
delete_prob = None
delete_seed = 0

mid_rmse = False
mid_weight = 2

quantiles = 'exact'
s_q = 1

rho = False

mod_hp = {}

monotonic_q = False

dive_data = False
add_z = True

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "w:i:m:e:c:O:C:j:I:p:P:M:D:S:q:Q:T:R:W:H:V:d:z:",
                               ["weather=", "params_idx=", "monotonic_q=", "epochs=", "cds=", "overwrite=", "cumulative=", "joint_prediction=", "ID=", "pretrained=", "patience=", "store_missing_idxs=", "delete_prob=", "delete_seed=", "quantiles=", "s_q=", "max_train_days=", "mid_rmse=", "mid_weight=", "mod_hp=", "rho=", "dive_data=", "add_z="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-i", "--params_idx"):
        params_idx = fmt.decoder(arg)
    elif opt in ("-m", "--monotonic_q"):
        monotonic_q = fmt.decoder(arg)
    elif opt in ("-e", "--epochs"):
        epochs = int(arg)
    elif opt in ("-c", "--cds"):
        cds = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-C", "--cumulative"):
        cumulative = fmt.decoder(arg)
    elif opt in ("-j", "--joint_prediction"):
        joint_prediction = fmt.decoder(arg)
    elif opt in ("-I", "--ID"):
        ID = fmt.decoder(arg)
    elif opt in ("-p", "--pretrained"):
        pretrained = fmt.decoder(arg)
    elif opt in ("-P", "--patience"):
        patience = int(arg)
    elif opt in ("-M", "--store_missing_idxs"):
        store_missing_idxs = fmt.decoder(arg)
    elif opt in ("-D", "--delete_prob"):
        delete_prob = fmt.decoder(arg)
    elif opt in ("-S", "--delete_seed"):
        delete_seed = fmt.decoder(arg)
    elif opt in ("-q", "--quantiles"):
        quantiles = arg
    elif opt in ("-Q", "--s_q"):
        s_q = fmt.decoder(arg)
    elif opt in ("-T", "--max_train_days"):
        max_train_days = int(arg)
    elif opt in ("-R", "--mid_rmse"):
        mid_rmse = fmt.decoder(arg)
    elif opt in ("-W", "--mid_weight"):
        mid_weight = fmt.decoder(arg)
    elif opt in ("-H", "--mod_hp"):
        # format: k1=v1,k2=v2,k3=v3 ...
        mod_hp = {k: fmt.decoder(v) for k, v in [kv.split('=') for kv in arg.split(',')]}
    elif opt in ("-V", "--rho"):
        rho = fmt.decoder(arg)
    elif opt in ("-d", "--dive_data"):
        dive_data = fmt.decoder(arg)
    elif opt in ("-z", "--add_z"):
        add_z = fmt.decoder(arg)

phdu.getopt_printer(opts)

kws = dict(weather=weather, params_idx=params_idx, epochs=epochs, cds=cds, overwrite=overwrite, cumulative=cumulative, joint_prediction=joint_prediction, ID=ID, pretrained=pretrained, patience=patience, store_missing_idxs=store_missing_idxs, delete_prob=delete_prob, delete_seed=delete_seed, quantiles=quantiles, s_q=s_q, max_train_days=max_train_days, mid_rmse=mid_rmse, mid_weight=mid_weight, mod_hp=mod_hp, rho=rho, monotonic_q=monotonic_q, dive_data=dive_data, add_z=add_z)
print(f"Params:\n{kws}")

input_kws = dict(weather=weather, epochs=epochs, cds=cds, overwrite=overwrite, patience=patience, mod_hp=mod_hp)
if store_missing_idxs:
    input_kws['store_missing_idxs'] = store_missing_idxs
if cumulative:
    input_kws['cumulative'] = cumulative
    input_kws['joint_prediction'] = joint_prediction
if delete_prob is not None:
    input_kws['delete_prob'] = delete_prob
    input_kws['delete_seed'] = delete_seed
if max_train_days != 28:
    input_kws['max_train_days'] = max_train_days
if quantiles != 'exact':
    input_kws['quantiles'] = quantiles
    input_kws['s_q'] = s_q
if mid_rmse:
    input_kws['mid_rmse'] = mid_rmse
    input_kws['mid_weight'] = mid_weight
if rho:
    input_kws['quantiles'] = '1D'
    input_kws['target'] = 'rho'
if monotonic_q:
    input_kws['monotonic_q'] = monotonic_q
if dive_data:
    input_kws['dive_data'] = dive_data
    input_kws['add_z'] = add_z
    input_kws['limit_train_batches'] = 400

print(f"\nInput kws:\n{input_kws}\n")


# if mode == 'distribution':
#     model_specs = params.distribution_best_params['forecasting'][cds][params_idx][1]
#     print(f"Model specs:\n{model_specs}")
#     num_mixtures = model_specs.pop('num_mixtures')
#     gradient_clip_val = model_specs.pop('gradient_clip_val')
#     analysis.train_and_store_distribution(model_specs=model_specs, num_mixtures=num_mixtures, gradient_clip_val=gradient_clip_val, **input_kws)

# elif mode == 'quantile':

analysis.quantile_results(**input_kws, params_idx=params_idx, verbose=1)

print("Done!")
