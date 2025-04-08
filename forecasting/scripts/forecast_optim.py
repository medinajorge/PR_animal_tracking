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

from forecasting import analysis
import getopt
from tidypath import fmt
import phdu

# Default params
weather = 'all'
seed = 0
overwrite = False
is_energy_score = False
cds = 'mercator'
cumulative = False
use_learning_rate_finder = True
joint_prediction = False

timeout = 17
prune_memory_error = False
extend = True
store_missing_idxs = True
max_train_days = 28

mid_rmse = False
mid_weight = 2

quantiles='exact'
s_q = 1

rho = False

monotonic_q = False

dive_data = False
add_z = True

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "w:s:m:e:O:c:C:l:j:t:P:E:M:R:W:q:Q:D:p:d:z:",
                               ["weather=", "seed=", "monotonic_q=", "energy_score=", "overwrite=", "cds=", "cumulative=", "learning_rate_finder=", "joint_prediction=", "timeout=", "prune_memory_error=", "extend=", "store_missing_idxs=", "mid_rmse=", "mid_weight=", "quantiles=", "s_q=", "max_train_days=", "rho=", "dive_data=", "add_z="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-w", "--weather"):
        weather = fmt.decoder(arg)
    elif opt in ("-s", "--seed"):
        seed = int(arg)
    elif opt in ("-m", "--monotonic_q"):
        monotonic_q = fmt.decoder(arg)
    elif opt in ("-e", "--energy_score"):
        is_energy_score = fmt.decoder(arg)
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-c", "--cds"):
        cds = arg
    elif opt in ("-C", "--cumulative"):
        cumulative = fmt.decoder(arg)
    elif opt in ("-l", "--learning_rate_finder"):
        use_learning_rate_finder = fmt.decoder(arg)
    elif opt in ("-j", "--joint_prediction"):
        joint_prediction = fmt.decoder(arg)
    elif opt in ("-t", "--timeout"):
        timeout = fmt.decoder(arg)
    elif opt in ("-P", "--prune_memory_error"):
        prune_memory_error = fmt.decoder(arg)
    elif opt in ("-E", "--extend"):
        extend = fmt.decoder(arg)
    elif opt in ("-M", "--store_missing_idxs"):
        store_missing_idxs = fmt.decoder(arg)
    elif opt in ("-R", "--mid_rmse"):
        mid_rmse = fmt.decoder(arg)
    elif opt in ("-W", "--mid_weight"):
        mid_weight = fmt.decoder(arg)
    elif opt in ("-q", "--quantiles"):
        quantiles = arg
    elif opt in ("-Q", "--s_q"):
        s_q = fmt.decoder(arg)
    elif opt in ("-D", "--max_train_days"):
        max_train_days = int(arg)
    elif opt in ("-p", "--rho"):
        rho = fmt.decoder(arg)
    elif opt in ("-d", "--dive_data"):
        dive_data = fmt.decoder(arg)
    elif opt in ("-z", "--add_z"):
        add_z = fmt.decoder(arg)

phdu.getopt_printer(opts)

kws = dict(weather=weather, seed=seed, overwrite=overwrite, cds=cds,
           timeout=timeout, prune_memory_error=prune_memory_error, extend=extend)
if is_energy_score:
    kws['is_energy_score'] = is_energy_score
if cumulative:
    kws['cumulative'] = cumulative
    kws['joint_prediction'] = joint_prediction
    if not use_learning_rate_finder:
        kws['use_learning_rate_finder'] = use_learning_rate_finder
if store_missing_idxs:
    kws['store_missing_idxs'] = store_missing_idxs
if mid_rmse:
    kws['mid_rmse'] = mid_rmse
    kws['mid_weight'] = mid_weight
if quantiles != 'exact':
    kws['quantiles'] = quantiles
    kws['s_q'] = s_q
if max_train_days != 28:
    kws['max_train_days'] = max_train_days
if rho:
    kws['quantiles'] = '1D'
    kws['target'] = 'rho'
if monotonic_q:
    kws['monotonic_q'] = monotonic_q
if dive_data:
    kws['dive_data'] = dive_data
    kws['add_z'] = add_z

print(f"Params:\n{kws}")

study = analysis.optimal_hyperparameters(**kws)

print("Done!")
